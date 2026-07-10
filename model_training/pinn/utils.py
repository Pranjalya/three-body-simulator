"""
Evaluation and Visualization Utilities for Three-Body Orbits.
Implements energy calculation (kinetic + potential), trajectory plotting,
and checkpoint loading for model verification and visualization.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def calculate_energy(state, m1=1.0, m2=1.0, m3=1.0, G=1.0):
    """
    Computes total energy (kinetic + potential) of the 3-body system.
    state: shape (12,) or (12, N)
    """
    if len(state.shape) == 1:
        # Single time step
        r1 = state[0:2]
        r2 = state[2:4]
        r3 = state[4:6]
        v1 = state[6:8]
        v2 = state[8:10]
        v3 = state[10:12]
        
        # Kinetic Energy
        ke = 0.5 * (m1 * np.sum(v1**2) + m2 * np.sum(v2**2) + m3 * np.sum(v3**2))
        
        # Potential Energy
        d12 = np.linalg.norm(r1 - r2)
        d13 = np.linalg.norm(r1 - r3)
        d23 = np.linalg.norm(r2 - r3)
        
        pe = -G * (m1 * m2 / d12 + m1 * m3 / d13 + m2 * m3 / d23)
    else:
        # Time-series states
        r1 = state[0:2, :]
        r2 = state[2:4, :]
        r3 = state[4:6, :]
        v1 = state[6:8, :]
        v2 = state[8:10, :]
        v3 = state[10:12, :]
        
        ke = 0.5 * (m1 * np.sum(v1**2, axis=0) + m2 * np.sum(v2**2, axis=0) + m3 * np.sum(v3**2, axis=0))
        
        d12 = np.linalg.norm(r1 - r2, axis=0)
        d13 = np.linalg.norm(r1 - r3, axis=0)
        d23 = np.linalg.norm(r2 - r3, axis=0)
        
        pe = -G * (m1 * m2 / d12 + m1 * m3 / d13 + m2 * m3 / d23)
        
    return ke + pe, ke, pe

def evaluate_and_plot(model_path, data_path, sim_idx, save_dir="plots"):
    """
    Evaluates the model on a specific simulation and plots the trajectory and energy.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
    
    # Load data
    data = np.load(data_path)
    y0 = data['y0'][sim_idx]
    y_true = data['y'][sim_idx]  # (12, num_steps)
    t = data['t']
    
    num_steps = len(t)
    
    # Prepare model inputs
    init_pos = y0[:6]
    init_pos_repeated = np.repeat(init_pos[np.newaxis, :], num_steps, axis=0)
    t_expanded = t[:, np.newaxis]
    inputs = np.concatenate([init_pos_repeated, t_expanded], axis=1)
    
    inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_t = model(inputs_t)
        y_pred = y_pred_t.cpu().numpy().T  # (12, num_steps)
        
    # Compute MAE
    mae = np.mean(np.abs(y_pred - y_true))
    print(f"Simulation {sim_idx} MAE: {mae:.6f}")
    
    # Compute energies
    energy_true, ke_true, pe_true = calculate_energy(y_true)
    energy_pred, ke_pred, pe_pred = calculate_energy(y_pred)
    
    # Plot Trajectories
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Trajectory plot (X-Z plane)
    axs[0].plot(y_true[0, :], y_true[1, :], 'b-', label='Body 1 Ground Truth')
    axs[0].plot(y_pred[0, :], y_pred[1, :], 'b--', label='Body 1 Predicted')
    
    axs[0].plot(y_true[2, :], y_true[3, :], 'r-', label='Body 2 Ground Truth')
    axs[0].plot(y_pred[2, :], y_pred[3, :], 'r--', label='Body 2 Predicted')
    
    axs[0].plot(y_true[4, :], y_true[5, :], 'g-', label='Body 3 Ground Truth')
    axs[0].plot(y_pred[4, :], y_pred[5, :], 'g--', label='Body 3 Predicted')
    
    # Mark initial positions
    axs[0].scatter([y_true[0, 0], y_true[2, 0], y_true[4, 0]], 
                    [y_true[1, 0], y_true[3, 0], y_true[5, 0]], 
                    color='black', marker='o', s=50, label='Initial Position', zorder=5)
                    
    axs[0].set_title(f"3-Body Trajectories (XZ Plane) - Sim {sim_idx}\nMAE: {mae:.4f}")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Z")
    axs[0].grid(True)
    axs[0].legend(fontsize='small')
    axs[0].axis('equal')
    
    # Energy Plot
    axs[1].plot(t, energy_true, 'k-', label='True Total Energy')
    axs[1].plot(t, energy_pred, 'k--', label='Predicted Total Energy')
    axs[1].plot(t, ke_true, 'r-', alpha=0.5, label='True Kinetic Energy')
    axs[1].plot(t, ke_pred, 'r--', alpha=0.5, label='Predicted Kinetic Energy')
    axs[1].plot(t, pe_true, 'b-', alpha=0.5, label='True Potential Energy')
    axs[1].plot(t, pe_pred, 'b--', alpha=0.5, label='Predicted Potential Energy')
    
    axs[1].set_title("Conservation of Energy Check")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Energy")
    axs[1].grid(True)
    axs[1].legend(fontsize='small')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"sim_{sim_idx}_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Saved comparison plot to {plot_path}")
    
    # Print energy deviation
    true_dev = np.std(energy_true) / np.abs(np.mean(energy_true))
    pred_dev = np.std(energy_pred) / np.abs(np.mean(energy_pred))
    print(f"True Energy Standard Deviation (relative): {true_dev:.6e}")
    print(f"Predicted Energy Standard Deviation (relative): {pred_dev:.6e}")
    
    return mae, plot_path
