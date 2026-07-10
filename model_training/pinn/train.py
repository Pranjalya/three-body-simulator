"""
Training Pipeline for Three-Body Physics-Informed Neural Network (PINN).
Defines loss calculations combining coordinates data loss (MAE) and physics residual
loss (MSE), scheduling weights dynamically, and managing gradient optimization.
"""

import argparse
import time
import os
import torch
import torch.nn as nn
import numpy as np
from model import get_model
from dataset import get_dataloaders

def compute_physics_loss(model, inputs, eps=1e-8):
    """
    Computes the physics residual loss for the planar 3-body system.
    inputs: shape (batch_size, 7) - [x1_0, z1_0, x2_0, z2_0, x3_0, z3_0, t]
    """
    # Clone inputs and set requires_grad=True on the time component t
    init_pos = inputs[:, :6].clone()
    t = inputs[:, 6:7].clone().requires_grad_(True)
    x_in = torch.cat([init_pos, t], dim=1)
    
    # Forward pass
    u_pred = model(x_in)  # shape (batch_size, 12)
    
    # Compute derivative of each output component w.r.t t
    # u_pred: [x1, z1, x2, z2, x3, z3, vx1, vz1, vx2, vz2, vx3, vz3]
    du_dt = torch.zeros_like(u_pred)
    for i in range(12):
        grad_outputs = torch.zeros_like(u_pred)
        grad_outputs[:, i] = 1.0
        grad = torch.autograd.grad(
            outputs=u_pred,
            inputs=t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        du_dt[:, i] = grad[:, 0]
        
    pos = u_pred[:, :6]
    vel = u_pred[:, 6:]
    
    dpos_dt = du_dt[:, :6]
    dvel_dt = du_dt[:, 6:]
    
    # 1. Kinematic residual: dpos/dt - vel
    residual_pos = dpos_dt - vel
    
    # 2. Dynamic residual: dvel/dt - acc
    r1 = pos[:, 0:2]
    r2 = pos[:, 2:4]
    r3 = pos[:, 4:6]
    
    r12 = r1 - r2
    r13 = r1 - r3
    r23 = r2 - r3
    
    # Euclidean distance
    d12 = torch.norm(r12, p=2, dim=1, keepdim=True)
    d13 = torch.norm(r13, p=2, dim=1, keepdim=True)
    d23 = torch.norm(r23, p=2, dim=1, keepdim=True)
    
    # Clamp distances to prevent division by zero near singularity
    d12 = torch.clamp(d12, min=eps)
    d13 = torch.clamp(d13, min=eps)
    d23 = torch.clamp(d23, min=eps)
    
    a1 = - r12 / (d12**3) - r13 / (d13**3)
    a2 = r12 / (d12**3) - r23 / (d23**3)
    a3 = r13 / (d13**3) + r23 / (d23**3)
    
    acc = torch.cat([a1, a2, a3], dim=1)
    residual_vel = dvel_dt - acc
    
    # Combine all residuals
    residual = torch.cat([residual_pos, residual_vel], dim=1)
    
    loss_physics = torch.mean(residual**2)
    return loss_physics, u_pred

def main():
    parser = argparse.ArgumentParser(description="Train a PINN/DNN model for the three-body problem.")
    parser.add_argument("--data_path", type=str, default="three_body_data.npz", help="Path to the dataset file.")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "standard"], help="Model architecture.")
    parser.add_argument("--depth", type=int, default=12, help="Network depth.")
    parser.add_argument("--width", type=int, default=256, help="Network width.")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function.")
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs.")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size.")
    parser.add_argument("--lr", type=float, default=7.5e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay L2 regularization.")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping norm threshold.")
    parser.add_argument("--use_pinn", type=str, default="true", choices=["true", "false"], help="Enable physics loss.")
    parser.add_argument("--alpha_init", type=float, default=0.001, help="Initial alpha coefficient.")
    parser.add_argument("--alpha_final", type=float, default=0.75, help="Final alpha coefficient.")
    parser.add_argument("--save_path", type=str, default="best_model.pt", help="Path to save the best model checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load loaders
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file {args.data_path} not found. Please run generate_data.py first.")
        
    train_loader, val_loader = get_dataloaders(args.data_path, batch_size=args.batch_size, seed=args.seed)
    print(f"Data loaders created. Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Build model
    model = get_model(
        model_type=args.model_type,
        input_dim=7,
        output_dim=12,
        hidden_dim=args.width,
        depth=args.depth,
        activation_name=args.activation
    ).to(device)
    
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Reduce learning rate by factor of 0.7 if validation loss plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=5
    )
    
    use_pinn = args.use_pinn.lower() == "true"
    
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = 10
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        
        # Calculate current alpha (linear scheduler)
        if use_pinn:
            if args.epochs > 1:
                alpha = args.alpha_init + (args.alpha_final - args.alpha_init) * (epoch / (args.epochs - 1))
            else:
                alpha = args.alpha_final
        else:
            alpha = 0.0
            
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            if use_pinn:
                loss_physics, u_pred = compute_physics_loss(model, inputs)
                loss_data = torch.mean(torch.abs(u_pred - targets))  # MAE Loss
                loss = loss_data + alpha * loss_physics
            else:
                u_pred = model(inputs)
                loss_data = torch.mean(torch.abs(u_pred - targets))
                loss_physics = torch.tensor(0.0, device=device)
                loss = loss_data
                
            loss.backward()
            
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
            optimizer.step()
            
            epoch_data_loss += loss_data.item() * inputs.size(0)
            epoch_physics_loss += loss_physics.item() * inputs.size(0)
            epoch_total_loss += loss.item() * inputs.size(0)
            
        num_train_samples = len(train_loader.dataset)
        epoch_data_loss /= num_train_samples
        epoch_physics_loss /= num_train_samples
        epoch_total_loss /= num_train_samples
        
        # Evaluation
        model.eval()
        val_data_loss = 0.0
        val_physics_loss = 0.0
        val_total_loss = 0.0
        
        for val_inputs, val_targets in val_loader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            
            with torch.no_grad():
                val_u_pred = model(val_inputs)
                v_data_loss = torch.mean(torch.abs(val_u_pred - val_targets))
                val_data_loss += v_data_loss.item() * val_inputs.size(0)
                
            if use_pinn:
                # We compute physics loss without grad w.r.t parameters
                # but we need gradients w.r.t inputs, so we temporarily enable grads.
                # Since we don't call backward, optimizer weights won't be updated.
                with torch.enable_grad():
                    v_phys_loss, _ = compute_physics_loss(model, val_inputs)
                val_physics_loss += v_phys_loss.item() * val_inputs.size(0)
                
        num_val_samples = len(val_loader.dataset)
        val_data_loss /= num_val_samples
        val_physics_loss /= num_val_samples
        val_total_loss = val_data_loss + alpha * val_physics_loss
        
        # Update learning rate scheduler
        scheduler.step(val_data_loss)
        
        # Print logs
        if (epoch + 1) % 1 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
                  f"Train Data (MAE): {epoch_data_loss:.6f} | "
                  f"Train Phys: {epoch_physics_loss:.6f} | "
                  f"Val Data (MAE): {val_data_loss:.6f} | "
                  f"Val Phys: {val_physics_loss:.6f} | "
                  f"Alpha: {alpha:.4f} | LR: {current_lr:.2e}")
                  
        # Check for early stopping
        if val_data_loss < best_val_loss:
            best_val_loss = val_data_loss
            patience_counter = 0
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_data_loss,
                'args': args
            }, args.save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered! No improvement in validation loss for {early_stopping_patience} epochs.")
                break
                
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds. Best validation MAE loss: {best_val_loss:.6f}")
    print(f"Best model saved to {args.save_path}")

if __name__ == "__main__":
    main()
