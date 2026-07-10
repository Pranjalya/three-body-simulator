"""
Data Generation Script for Planar 3-Body Problem Simulation.
Uses Scipy's DOP853 high-precision integrator to generate non-colliding
unitary mass orbits under gravity, based on Algorithm 1 from the paper.
"""

import argparse
import multiprocessing
import numpy as np
from scipy.integrate import solve_ivp
import time
import os
from tqdm import tqdm

def check_collision(y, threshold=1e-4):
    """
    Check if any two bodies are too close (collision or near-collision).
    y: array of shape (12,) or (12, N)
    """
    if len(y.shape) == 1:
        r1 = y[0:2]
        r2 = y[2:4]
        r3 = y[4:6]
        d12 = np.linalg.norm(r1 - r2)
        d13 = np.linalg.norm(r1 - r3)
        d23 = np.linalg.norm(r2 - r3)
        return d12 < threshold or d13 < threshold or d23 < threshold
    else:
        # Vectorized check across all time steps
        r1 = y[0:2, :]
        r2 = y[2:4, :]
        r3 = y[4:6, :]
        d12 = np.linalg.norm(r1 - r2, axis=0)
        d13 = np.linalg.norm(r1 - r3, axis=0)
        d23 = np.linalg.norm(r2 - r3, axis=0)
        return np.any(d12 < threshold) or np.any(d13 < threshold) or np.any(d23 < threshold)

def three_body_deriv(t, y):
    """
    Equations of motion for the planar three-body problem with equal masses.
    y: [x1, z1, x2, z2, x3, z3, vx1, vz1, vx2, vz2, vx3, vz3]
    """
    r1 = y[0:2]
    r2 = y[2:4]
    r3 = y[4:6]
    v1 = y[6:8]
    v2 = y[8:10]
    v3 = y[10:12]

    r12 = r1 - r2
    r13 = r1 - r3
    r23 = r2 - r3

    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)

    # If they are extremely close, the force calculation will blow up.
    # solve_ivp will handle this if we return large derivatives, but raising an error
    # allows us to immediately stop and discard this simulation.
    if d12 < 1e-5 or d13 < 1e-5 or d23 < 1e-5:
        raise ValueError("Collision detected during integration.")

    a1 = - r12 / (d12**3) - r13 / (d13**3)
    a2 = r12 / (d12**3) - r23 / (d23**3)
    a3 = r13 / (d13**3) + r23 / (d23**3)

    return np.concatenate([v1, v2, v3, a1, a2, a3])

def simulate_one(args):
    """
    Worker function to simulate a single trajectory.
    """
    idx, p2_init, t_max, dt = args
    # p2_init: (x2, z2)
    # p1 is fixed at (1.0, 0.0)
    p1_init = np.array([1.0, 0.0])
    p3_init = - p1_init - p2_init

    # Initial state vector: [x1, z1, x2, z2, x3, z3, vx1, vz1, vx2, vz2, vx3, vz3]
    y0 = np.zeros(12)
    y0[0:2] = p1_init
    y0[2:4] = p2_init
    y0[4:6] = p3_init
    # Velocities are initialized to 0

    t_eval = np.arange(0, t_max + dt / 2.0, dt)

    try:
        sol = solve_ivp(
            three_body_deriv,
            [0, t_max],
            y0,
            method='DOP853',
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10
        )
        if sol.status == 0 and len(sol.t) == len(t_eval):
            # Return initial condition and the full trajectory
            # sol.y shape is (12, 257)
            return {
                'idx': idx,
                'success': True,
                'y0': y0,
                't': sol.t,
                'y': sol.y
            }
    except Exception:
        pass
    
    return {'idx': idx, 'success': False}

def generate_initial_condition():
    """
    Algorithm 1 from the paper.
    """
    theta = np.random.uniform(0, np.pi / 2.0)
    x = -np.minimum(0.5, np.cos(theta))
    z = np.sin(theta)
    p = np.array([x, z])
    s = np.random.uniform(0, 1.0)
    p2 = s * p
    return p2

def main():
    parser = argparse.ArgumentParser(description="Parallel data generator for 3-body simulation.")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of successful simulations to generate.")
    parser.add_argument("--output_path", type=str, default="three_body_data.npz", help="Path to save the generated dataset.")
    parser.add_argument("--t_max", type=float, default=10.0, help="Max integration time.")
    parser.add_argument("--dt", type=float, default=0.0390625, help="Time step.")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="Number of cores to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"Generating {args.num_simulations} successful simulations using {args.cores} cores...")
    start_time = time.time()

    successful_simulations = []
    total_attempted = 0

    # We will generate in batches to parallelize efficiently
    batch_size = max(args.cores * 4, 100)

    pbar = tqdm(total=args.num_simulations, desc="Successful simulations")
    try:
        while len(successful_simulations) < args.num_simulations:
            needed = args.num_simulations - len(successful_simulations)
            # Generate candidates (we request more than needed because ~20% fail)
            num_candidates = max(int(needed * 1.5), batch_size)
            
            tasks = []
            for i in range(num_candidates):
                p2 = generate_condition_retry()
                sim_idx = total_attempted + i
                tasks.append((sim_idx, p2, args.t_max, args.dt))

            total_attempted += num_candidates

            # Run in parallel
            with multiprocessing.Pool(processes=args.cores) as pool:
                # We use imap_unordered for streaming results and updating the progress bar in real-time
                for res in pool.imap_unordered(simulate_one, tasks):
                    if res['success']:
                        successful_simulations.append(res)
                        pbar.update(1)
                        if len(successful_simulations) == args.num_simulations:
                            break
            
            pbar.set_postfix(attempted=total_attempted)
    finally:
        pbar.close()

    # Sort by initial index to keep order deterministic given seed
    successful_simulations.sort(key=lambda x: x['idx'])

    # Format arrays for saving
    # y0_all: shape (num_simulations, 12)
    # y_all: shape (num_simulations, 12, num_time_steps)
    # t_all: shape (num_time_steps,)
    y0_all = np.array([sim['y0'] for sim in successful_simulations])
    y_all = np.array([sim['y'] for sim in successful_simulations])
    t_all = successful_simulations[0]['t']

    # Save to file
    np.savez_compressed(
        args.output_path,
        y0=y0_all,
        y=y_all,
        t=t_all
    )

    elapsed = time.time() - start_time
    print(f"Data generation complete. Saved to {args.output_path}")
    print(f"Total time: {elapsed:.2f} seconds. Successful/Attempted: {len(successful_simulations)}/{total_attempted} ({len(successful_simulations)/total_attempted*100:.1f}%)")

def generate_condition_retry():
    """
    Generate initial condition and check that they aren't in collision at t=0.
    """
    while True:
        p2 = generate_initial_condition()
        p1 = np.array([1.0, 0.0])
        p3 = - p1 - p2
        d12 = np.linalg.norm(p1 - p2)
        d13 = np.linalg.norm(p1 - p3)
        d23 = np.linalg.norm(p2 - p3)
        if d12 >= 1e-4 and d13 >= 1e-4 and d23 >= 1e-4:
            return p2

if __name__ == "__main__":
    main()
