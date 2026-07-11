"""
Inference and ONNX Export Script for Three-Body PINN Model.
Loads a trained checkpoint, runs predictions for arbitrary initial conditions
and times, and exports the model to ONNX format for browser use.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def load_pinn_model(model_path, device="cpu"):
    """
    Loads model checkpoint and reconstructs the architecture.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    from model import get_model
    model = get_model(
        model_type=args.model_type,
        input_dim=7,
        output_dim=12,
        hidden_dim=args.width,
        depth=args.depth,
        activation_name=args.activation
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {args.model_type.upper()} model from {model_path} (trained for {checkpoint['epoch']} epochs).")
    return model, args

def predict_trajectory(model, p2_init, t_max=10.0, dt=0.0390625, device="cpu"):
    """
    Predicts the full 3-body trajectory given initial position of body 2.
    Body 1 is fixed at (1.0, 0.0), Body 3 is determined by center of mass (origin).
    """
    p1_init = np.array([1.0, 0.0])
    p3_init = - p1_init - p2_init
    
    # Combined initial position of shape (6,)
    init_pos = np.concatenate([p1_init, p2_init, p3_init])
    
    t = np.arange(0, t_max + dt / 2.0, dt)
    num_steps = len(t)
    
    # Construct inputs of shape (num_steps, 7)
    init_pos_repeated = np.repeat(init_pos[np.newaxis, :], num_steps, axis=0)
    t_expanded = t[:, np.newaxis]
    inputs = np.concatenate([init_pos_repeated, t_expanded], axis=1)
    
    inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u_pred_t = model(inputs_t)
        u_pred = u_pred_t.cpu().numpy()  # shape (num_steps, 12)
        
    return t, u_pred

def export_to_onnx(model, output_onnx_path):
    """
    Exports the PyTorch model to ONNX format with dynamic batch sizing.
    """
    # Create a dummy input matching the model input shape (batch_size, 7)
    dummy_input = torch.randn(1, 7)
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported model to ONNX: {output_onnx_path}")

def plot_prediction(t, u_pred, save_path="prediction_plot.png"):
    """
    Plots the predicted 3-body orbits in the X-Z plane.
    """
    # u_pred: shape (num_steps, 12) -> columns 0-5 are positions
    x1, z1 = u_pred[:, 0], u_pred[:, 1]
    x2, z2 = u_pred[:, 2], u_pred[:, 3]
    x3, z3 = u_pred[:, 4], u_pred[:, 5]
    
    plt.figure(figsize=(8, 8))
    plt.plot(x1, z1, 'b-', label='Body 1 (Fixed Start)')
    plt.plot(x2, z2, 'r-', label='Body 2')
    plt.plot(x3, z3, 'g-', label='Body 3')
    
    # Mark starting positions
    plt.scatter([x1[0], x2[0], x3[0]], [z1[0], z2[0], z3[0]], color='black', marker='o', s=60, label='Start', zorder=5)
    
    plt.title("Predicted 3-Body Orbits (X-Z Plane)")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved orbit trajectory plot to {save_path}")

def print_integration_info():
    """
    Prints input/output tensor structure information to help with system integration.
    """
    info = """
=============================================================================
             INTEGRATION SPECIFICATIONS (INPUT / OUTPUT TENSORS)
=============================================================================

This model expects a 7-dimensional input vector and outputs a 12-dimensional state.

📥 INPUT TENSOR STRUCTURE (Shape: [batch_size, 7])
--------------------------------------------------
Each input row contains the initial positions of the three bodies and the time step:
  - Input[0] : Body 1 initial X position (fixed at 1.0)
  - Input[1] : Body 1 initial Z position (fixed at 0.0)
  - Input[2] : Body 2 initial X position
  - Input[3] : Body 2 initial Z position
  - Input[4] : Body 3 initial X position (dependent: -1.0 - Body 2 initial X)
  - Input[5] : Body 3 initial Z position (dependent: -Body 2 initial Z)
  - Input[6] : Target Time step 't' (value in range [0.0, 10.0])

📤 OUTPUT TENSOR STRUCTURE (Shape: [batch_size, 12])
----------------------------------------------------
Each output row contains the predicted positions and velocities at target time 't':
  - Output[0]  : Body 1 predicted X position
  - Output[1]  : Body 1 predicted Z position
  - Output[2]  : Body 2 predicted X position
  - Output[3]  : Body 2 predicted Z position
  - Output[4]  : Body 3 predicted X position
  - Output[5]  : Body 3 predicted Z position
  - Output[6]  : Body 1 predicted X velocity
  - Output[7]  : Body 1 predicted Z velocity
  - Output[8]  : Body 2 predicted X velocity
  - Output[9]  : Body 2 predicted Z velocity
  - Output[10] : Body 3 predicted X velocity
  - Output[11] : Body 3 predicted Z velocity

=============================================================================
"""
    print(info)

def main():
    print_integration_info()
    
    parser = argparse.ArgumentParser(description="Predict and export PINN 3-body trajectories.")
    parser.add_argument("--model_path", type=str, default="best_pinn_model.pt", help="Path to checkpoint.")
    parser.add_argument("--x2", type=float, default=-0.2, help="Initial X coordinate of Body 2.")
    parser.add_argument("--z2", type=float, default=0.5, help="Initial Z coordinate of Body 2.")
    parser.add_argument("--t_max", type=float, default=10.0, help="Max time to predict.")
    parser.add_argument("--dt", type=float, default=0.0390625, help="Timestep size.")
    parser.add_argument("--export_onnx", type=str, default="pinn_model.onnx", help="Path to save ONNX model (empty string to skip export).")
    parser.add_argument("--plot_path", type=str, default="predicted_orbit.png", help="Path to save predicted orbits plot.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint file {args.model_path} not found.")
        
    model, model_args = load_pinn_model(args.model_path, device)
    
    # Run prediction
    p2_init = np.array([args.x2, args.z2])
    print(f"Running trajectory inference starting at Body 2 = ({args.x2}, {args.z2})...")
    t, u_pred = predict_trajectory(model, p2_init, args.t_max, args.dt, device)
    
    # Print a sample prediction at t_max
    print(f"\nFinal State at t={t[-1]:.4f}:")
    print(f"Body 1 Position: ({u_pred[-1, 0]:.4f}, {u_pred[-1, 1]:.4f}) | Velocity: ({u_pred[-1, 6]:.4f}, {u_pred[-1, 7]:.4f})")
    print(f"Body 2 Position: ({u_pred[-1, 2]:.4f}, {u_pred[-1, 3]:.4f}) | Velocity: ({u_pred[-1, 8]:.4f}, {u_pred[-1, 9]:.4f})")
    print(f"Body 3 Position: ({u_pred[-1, 4]:.4f}, {u_pred[-1, 5]:.4f}) | Velocity: ({u_pred[-1, 10]:.4f}, {u_pred[-1, 11]:.4f})")
    
    # Save Plot
    if args.plot_path:
        plot_prediction(t, u_pred, args.plot_path)
        
    # Export to ONNX
    if args.export_onnx:
        # ONNX export requires CPU model evaluation
        cpu_model, _ = load_pinn_model(args.model_path, "cpu")
        export_to_onnx(cpu_model, args.export_onnx)

if __name__ == "__main__":
    main()
