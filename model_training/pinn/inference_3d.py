"""
Inference and ONNX Export Script for 3D Three-Body PINN Model.
Loads a trained 3D checkpoint, runs predictions for arbitrary initial conditions
and times, plots 3D orbits, and exports the model to ONNX format for browser use.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def load_pinn_model_3d(model_path, device="cpu"):
    """
    Loads 3D model checkpoint and reconstructs the architecture.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    from model import get_model
    # 3D: input_dim=10, output_dim=18
    model = get_model(
        model_type=args.model_type,
        input_dim=10,
        output_dim=18,
        hidden_dim=args.width,
        depth=args.depth,
        activation_name=args.activation
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {args.model_type.upper()} 3D model from {model_path} (trained for {checkpoint['epoch']} epochs).")
    return model, args

def predict_trajectory_3d(model, p2_init, t_max=10.0, dt=0.0390625, device="cpu"):
    """
    Predicts the full 3D 3-body trajectory given initial position of body 2.
    Body 1 is fixed at (1.0, 0.0, 0.0), Body 3 is determined by center of mass (origin).
    """
    p1_init = np.array([1.0, 0.0, 0.0])
    p3_init = - p1_init - p2_init
    
    # Combined initial position of shape (9,)
    init_pos = np.concatenate([p1_init, p2_init, p3_init])
    
    t = np.arange(0, t_max + dt / 2.0, dt)
    num_steps = len(t)
    
    # Construct inputs of shape (num_steps, 10)
    init_pos_repeated = np.repeat(init_pos[np.newaxis, :], num_steps, axis=0)
    t_expanded = t[:, np.newaxis]
    inputs = np.concatenate([init_pos_repeated, t_expanded], axis=1)
    
    inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u_pred_t = model(inputs_t)
        u_pred = u_pred_t.cpu().numpy()  # shape (num_steps, 18)
        
    return t, u_pred

def export_to_onnx_3d(model, output_onnx_path):
    """
    Exports the 3D PyTorch model to ONNX format with dynamic batch sizing.
    """
    # Create a dummy input matching the 3D model input shape (batch_size, 10)
    dummy_input = torch.randn(1, 10)
    
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
    print(f"Successfully exported 3D model to ONNX: {output_onnx_path}")

def plot_prediction_3d(t, u_pred, save_path="prediction_plot_3d.png"):
    """
    Plots the predicted 3-body orbits in 3D space.
    """
    # u_pred: shape (num_steps, 18) -> columns 0-8 are positions
    x1, y1, z1 = u_pred[:, 0], u_pred[:, 1], u_pred[:, 2]
    x2, y2, z2 = u_pred[:, 3], u_pred[:, 4], u_pred[:, 5]
    x3, y3, z3 = u_pred[:, 6], u_pred[:, 7], u_pred[:, 8]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x1, y1, z1, 'b-', label='Body 1 (Fixed Start)')
    ax.plot(x2, y2, z2, 'r-', label='Body 2')
    ax.plot(x3, y3, z3, 'g-', label='Body 3')
    
    # Mark starting positions
    ax.scatter([x1[0], x2[0], x3[0]], [y1[0], y2[0], y3[0]], [z1[0], z2[0], z3[0]], 
               color='black', marker='o', s=60, label='Start', zorder=5)
    
    ax.set_title("Predicted 3-Body Orbits (3D Space)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.legend()
    
    # Force equal aspect ratio
    max_range = np.array([x1.max()-x1.min(), y1.max()-y1.min(), z1.max()-z1.min(),
                          x2.max()-x2.min(), y2.max()-y2.min(), z2.max()-z2.min(),
                          x3.max()-x3.min(), y3.max()-y3.min(), z3.max()-z3.min()]).max()
    mid_x = (x1.mean() + x2.mean() + x3.mean()) / 3
    mid_y = (y1.mean() + y2.mean() + y3.mean()) / 3
    mid_z = (z1.mean() + z2.mean() + z3.mean()) / 3
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved 3D orbit trajectory plot to {save_path}")

def print_integration_info_3d():
    """
    Prints input/output tensor structure information to help with 3D system integration.
    """
    info = """
=============================================================================
             INTEGRATION SPECIFICATIONS (3D INPUT / OUTPUT TENSORS)
=============================================================================

This model expects a 10-dimensional input vector and outputs an 18-dimensional state.

📥 INPUT TENSOR STRUCTURE (Shape: [batch_size, 10])
--------------------------------------------------
Each input row contains the initial positions of the three bodies in 3D and the time step:
  - Input[0] : Body 1 initial X position (fixed at 1.0)
  - Input[1] : Body 1 initial Y position (fixed at 0.0)
  - Input[2] : Body 1 initial Z position (fixed at 0.0)
  - Input[3] : Body 2 initial X position
  - Input[4] : Body 2 initial Y position
  - Input[5] : Body 2 initial Z position
  - Input[6] : Body 3 initial X position (dependent: -1.0 - Body 2 initial X)
  - Input[7] : Body 3 initial Y position (dependent: -Body 2 initial Y)
  - Input[8] : Body 3 initial Z position (dependent: -Body 2 initial Z)
  - Input[9] : Target Time step 't' (value in range [0.0, 10.0])

📤 OUTPUT TENSOR STRUCTURE (Shape: [batch_size, 18])
----------------------------------------------------
Each output row contains the predicted 3D positions and 3D velocities at target time 't':
  - Output[0..2]   : Body 1 predicted position (X, Y, Z)
  - Output[3..5]   : Body 2 predicted position (X, Y, Z)
  - Output[6..8]   : Body 3 predicted position (X, Y, Z)
  - Output[9..11]  : Body 1 predicted velocity (Vx, Vy, Vz)
  - Output[12..14] : Body 2 predicted velocity (Vx, Vy, Vz)
  - Output[15..17] : Body 3 predicted velocity (Vx, Vy, Vz)

=============================================================================
"""
    print(info)

def main():
    print_integration_info_3d()
    
    parser = argparse.ArgumentParser(description="Predict and export 3D PINN 3-body trajectories.")
    parser.add_argument("--model_path", type=str, default="best_pinn_model_3d.pt", help="Path to checkpoint.")
    parser.add_argument("--x2", type=float, default=-0.2, help="Initial X coordinate of Body 2.")
    parser.add_argument("--y2", type=float, default=0.2, help="Initial Y coordinate of Body 2.")
    parser.add_argument("--z2", type=float, default=0.5, help="Initial Z coordinate of Body 2.")
    parser.add_argument("--t_max", type=float, default=10.0, help="Max time to predict.")
    parser.add_argument("--dt", type=float, default=0.0390625, help="Timestep size.")
    parser.add_argument("--export_onnx", type=str, default="pinn_model_3d.onnx", help="Path to save ONNX model (empty string to skip export).")
    parser.add_argument("--plot_path", type=str, default="predicted_orbit_3d.png", help="Path to save predicted orbits plot.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint file {args.model_path} not found.")
        
    model, model_args = load_pinn_model_3d(args.model_path, device)
    
    # Run prediction
    p2_init = np.array([args.x2, args.y2, args.z2])
    print(f"Running 3D trajectory inference starting at Body 2 = ({args.x2}, {args.y2}, {args.z2})...")
    t, u_pred = predict_trajectory_3d(model, p2_init, args.t_max, args.dt, device)
    
    # Print a sample prediction at t_max
    print(f"\nFinal State at t={t[-1]:.4f}:")
    print(f"Body 1 Position: ({u_pred[-1, 0]:.4f}, {u_pred[-1, 1]:.4f}, {u_pred[-1, 2]:.4f})")
    print(f"Body 2 Position: ({u_pred[-1, 3]:.4f}, {u_pred[-1, 4]:.4f}, {u_pred[-1, 5]:.4f})")
    print(f"Body 3 Position: ({u_pred[-1, 6]:.4f}, {u_pred[-1, 7]:.4f}, {u_pred[-1, 8]:.4f})")
    
    # Save Plot
    if args.plot_path:
        plot_prediction_3d(t, u_pred, args.plot_path)
        
    # Export to ONNX
    if args.export_onnx:
        # ONNX export requires CPU model evaluation
        cpu_model, _ = load_pinn_model_3d(args.model_path, "cpu")
        export_to_onnx_3d(cpu_model, args.export_onnx)

if __name__ == "__main__":
    main()
