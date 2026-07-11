"""
GPU-Accelerated Data Generation Script for 3D Three-Body Problem Simulation.
Uses PyTorch to run vectorized Runge-Kutta 4th order (RK4) integration of
thousands of trajectories in parallel on the GPU.
"""

import argparse
import numpy as np
import torch
import time
from tqdm import tqdm

def deriv_gpu(y):
    """
    Equations of motion for the 3D three-body problem with equal masses.
    y: shape (B, 18) containing state vectors
    """
    r1 = y[:, 0:3]
    r2 = y[:, 3:6]
    r3 = y[:, 6:9]
    v1 = y[:, 9:12]
    v2 = y[:, 12:15]
    v3 = y[:, 15:18]

    r12 = r1 - r2
    r13 = r1 - r3
    r23 = r2 - r3

    d12_sq = torch.sum(r12**2, dim=1, keepdim=True)
    d13_sq = torch.sum(r13**2, dim=1, keepdim=True)
    d23_sq = torch.sum(r23**2, dim=1, keepdim=True)

    d12 = torch.sqrt(d12_sq)
    d13 = torch.sqrt(d13_sq)
    d23 = torch.sqrt(d23_sq)

    # Use a small eps to avoid NaN in division if collided
    eps = 1e-12
    a1 = - r12 / (d12_sq * d12 + eps) - r13 / (d13_sq * d13 + eps)
    a2 = r12 / (d12_sq * d12 + eps) - r23 / (d23_sq * d23 + eps)
    a3 = r13 / (d13_sq * d13 + eps) + r23 / (d23_sq * d23 + eps)

    return torch.cat([v1, v2, v3, a1, a2, a3], dim=1)

def rk4_step(y, dt):
    k1 = deriv_gpu(y)
    k2 = deriv_gpu(y + 0.5 * dt * k1)
    k3 = deriv_gpu(y + 0.5 * dt * k2)
    k4 = deriv_gpu(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def generate_initial_conditions_batch(batch_size):
    """
    Generate initial positions for particle 2 uniformly in a unit 3D sphere.
    Enforces that initial distances are >= 1e-2.
    """
    candidates = []
    while len(candidates) < batch_size:
        p2 = np.random.randn(3)
        norm = np.linalg.norm(p2)
        if norm > 1e-5:
            p2 = p2 / norm
            
        s = np.random.uniform(0, 1.0)
        p2 = s * p2
        
        p1 = np.array([1.0, 0.0, 0.0])
        p3 = - p1 - p2
        
        d12 = np.linalg.norm(p1 - p2)
        d13 = np.linalg.norm(p1 - p3)
        d23 = np.linalg.norm(p2 - p3)
        
        if d12 >= 1e-2 and d13 >= 1e-2 and d23 >= 1e-2:
            y0 = np.zeros(18)
            y0[0:3] = p1
            y0[3:6] = p2
            y0[6:9] = p3
            candidates.append(y0)
            
    return np.array(candidates)

def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated batched data generator for 3D 3-body simulation.")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of successful simulations to generate.")
    parser.add_argument("--output_path", type=str, default="three_body_data_3d.npz", help="Path to save the generated dataset.")
    parser.add_argument("--t_max", type=float, default=10.0, help="Max integration time.")
    parser.add_argument("--dt", type=float, default=0.0390625, help="Time step.")
    parser.add_argument("--substeps", type=int, default=20, help="Number of RK4 substeps per dt.")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for parallel integration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Generating {args.num_simulations} successful 3D simulations (batch size: {args.batch_size}, substeps: {args.substeps})...")

    t_eval = np.arange(0, args.t_max + args.dt / 2.0, args.dt)
    num_steps = len(t_eval)
    h = args.dt / args.substeps

    successful_y0 = []
    successful_y = []

    pbar = tqdm(total=args.num_simulations, desc="Successful Orbits")
    total_attempted = 0
    start_time = time.time()

    while len(successful_y0) < args.num_simulations:
        needed = args.num_simulations - len(successful_y0)
        current_batch_size = min(args.batch_size, max(needed * 2, 100))
        
        y0_np = generate_initial_conditions_batch(current_batch_size)
        total_attempted += current_batch_size

        # Move to GPU
        y0_t = torch.tensor(y0_np, dtype=torch.float32, device=device)
        
        # Keep history on GPU (92 MB VRAM for 5000 batch size, extremely small)
        y_history_gpu = torch.zeros((num_steps, current_batch_size, 18), dtype=torch.float32, device=device)
        y_history_gpu[0] = y0_t

        current_y = y0_t
        collided = torch.zeros(current_batch_size, dtype=torch.bool, device=device)

        # Integrate in batch
        with torch.no_grad():
            for step in range(1, num_steps):
                for _ in range(args.substeps):
                    current_y = rk4_step(current_y, h)
                    
                # Check collision (min distance < 1e-5) at the end of each time-step
                r1 = current_y[:, 0:3]
                r2 = current_y[:, 3:6]
                r3 = current_y[:, 6:9]
                d12 = torch.norm(r1 - r2, dim=1)
                d13 = torch.norm(r1 - r3, dim=1)
                d23 = torch.norm(r2 - r3, dim=1)
                collided |= (d12 < 1e-5) | (d13 < 1e-5) | (d23 < 1e-5)

                # Store step to GPU history
                y_history_gpu[step] = current_y

        # Move final results to CPU (only 1 copy operation)
        y_history_cpu = y_history_gpu.cpu().numpy()
        collided_cpu = collided.cpu().numpy()
        success_indices = np.where(~collided_cpu)[0]

        for idx in success_indices:
            if len(successful_y0) < args.num_simulations:
                successful_y0.append(y0_np[idx])
                # Original format targets: shape (18, num_steps)
                successful_y.append(y_history_cpu[:, idx, :].T)
                pbar.update(1)

    pbar.close()

    # Save exactly matching original format
    y0_all = np.array(successful_y0) # (num_simulations, 18)
    y_all = np.array(successful_y)   # (num_simulations, 18, num_steps)
    
    np.savez_compressed(
        args.output_path,
        y0=y0_all,
        y=y_all,
        t=t_eval
    )

    elapsed_time = time.time() - start_time
    print(f"Data generation complete in {elapsed_time:.2f} seconds. Saved to {args.output_path}")
    print(f"Successful/Attempted: {len(successful_y0)}/{total_attempted} ({len(successful_y0)/total_attempted*100:.1f}%)")

if __name__ == "__main__":
    main()
