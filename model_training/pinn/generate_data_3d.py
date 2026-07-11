"""
Data Generation Script for 3D Three-Body Problem Simulation.
Uses Scipy's DOP853 high-precision integrator to generate non-colliding
unitary mass orbits under gravity in full 3D space.
"""

import argparse
import multiprocessing
import numpy as np
from scipy.integrate import solve_ivp
import time
import os
from tqdm import tqdm

def check_collision_3d(y, threshold=1e-4):
    """
    Check if any two bodies are too close in 3D space.
    y: array of shape (18,) or (18, N)
    """
    if len(y.shape) == 1:
        r1 = y[0:3]
        r2 = y[3:6]
        r3 = y[6:9]
        d12 = np.linalg.norm(r1 - r2)
        d13 = np.linalg.norm(r1 - r3)
        d23 = np.linalg.norm(r2 - r3)
        return d12 < threshold or d13 < threshold or d23 < threshold
    else:
        r1 = y[0:3, :]
        r2 = y[3:6, :]
        r3 = y[6:9, :]
        d12 = np.linalg.norm(r1 - r2, axis=0)
        d13 = np.linalg.norm(r1 - r3, axis=0)
        d23 = np.linalg.norm(r2 - r3, axis=0)
        return np.any(d12 < threshold) or np.any(d13 < threshold) or np.any(d23 < threshold)

def three_body_deriv_3d(t, y):
    """
    Equations of motion for the 3D three-body problem with equal masses.
    y: [x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
    """
    r1 = y[0:3]
    r2 = y[3:6]
    r3 = y[6:9]
    v1 = y[9:12]
    v2 = y[12:15]
    v3 = y[15:18]

    r12 = r1 - r2
    r13 = r1 - r3
    r23 = r2 - r3

    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)

    if d12 < 1e-5 or d13 < 1e-5 or d23 < 1e-5:
        raise ValueError("Collision detected during integration.")

    a1 = - r12 / (d12**3) - r13 / (d13**3)
    a2 = r12 / (d12**3) - r23 / (d23**3)
    a3 = r13 / (d13**3) + r23 / (d23**3)

    return np.concatenate([v1, v2, v3, a1, a2, a3])

def simulate_one_3d(args):
    """
    Worker function to simulate a single 3D trajectory.
    """
    idx, p2_init, t_max, dt = args
    # p2_init: (x2, y2, z2)
    p1_init = np.array([1.0, 0.0, 0.0])
    p3_init = - p1_init - p2_init

    # Initial state vector: [x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
    y0 = np.zeros(18)
    y0[0:3] = p1_init
    y0[3:6] = p2_init
    y0[6:9] = p3_init

    t_eval = np.arange(0, t_max + dt / 2.0, dt)

    try:
        sol = solve_ivp(
            three_body_deriv_3d,
            [0, t_max],
            y0,
            method='DOP853',
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10
        )
        if sol.status == 0 and len(sol.t) == len(t_eval):
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

def generate_initial_condition_3d():
    """
    Generate initial positions for particle 2 uniformly in a unit 3D sphere.
    """
    while True:
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
            return p2

def main():
    parser = argparse.ArgumentParser(description="Parallel data generator for 3D 3-body simulation.")
    parser.add_argument("--num_simulations", type=int, default=100, help="Number of successful simulations to generate.")
    parser.add_argument("--output_path", type=str, default="three_body_data_3d.npz", help="Path to save the generated dataset.")
    parser.add_argument("--t_max", type=float, default=10.0, help="Max integration time.")
    parser.add_argument("--dt", type=float, default=0.0390625, help="Time step.")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="Number of cores to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"Generating {args.num_simulations} successful 3D simulations using {args.cores} cores...")
    start_time = time.time()

    successful_simulations = []
    total_attempted = 0
    batch_size = max(args.cores * 4, 100)

    pbar = tqdm(total=args.num_simulations, desc="3D Simulations")
    try:
        while len(successful_simulations) < args.num_simulations:
            needed = args.num_simulations - len(successful_simulations)
            num_candidates = max(int(needed * 1.5), batch_size)
            
            tasks = []
            for i in range(num_candidates):
                p2 = generate_initial_condition_3d()
                sim_idx = total_attempted + i
                tasks.append((sim_idx, p2, args.t_max, args.dt))

            total_attempted += num_candidates

            with multiprocessing.Pool(processes=args.cores) as pool:
                for res in pool.imap_unordered(simulate_one_3d, tasks):
                    if res['success']:
                        successful_simulations.append(res)
                        pbar.update(1)
                        if len(successful_simulations) == args.num_simulations:
                            break
            
            pbar.set_postfix(attempted=total_attempted)
    finally:
        pbar.close()

    successful_simulations.sort(key=lambda x: x['idx'])

    y0_all = np.array([sim['y0'] for sim in successful_simulations])
    y_all = np.array([sim['y'] for sim in successful_simulations])
    t_all = successful_simulations[0]['t']

    np.savez_compressed(
        args.output_path,
        y0=y0_all,
        y=y_all,
        t=t_all
    )

    elapsed = time.time() - start_time
    print(f"Data generation complete. Saved to {args.output_path}")
    print(f"Total time: {elapsed:.2f} seconds. Successful/Attempted: {len(successful_simulations)}/{total_attempted} ({len(successful_simulations)/total_attempted*100:.1f}%)")

if __name__ == "__main__":
    main()
