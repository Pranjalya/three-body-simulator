"""
Dataset Loader for Three-Body Orbits.
Implements vectorized data loading to prepare inputs and targets for
the non-autoregressive training setup. Split is performed at simulation level.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ThreeBodyDataset(Dataset):
    def __init__(self, data_path, sim_indices=None):
        data = np.load(data_path)
        y0 = data['y0']  # (num_simulations, 12)
        y = data['y']    # (num_simulations, 12, num_steps)
        t = data['t']    # (num_steps,)
        
        if sim_indices is not None:
            y0 = y0[sim_indices]
            y = y[sim_indices]
            
        self.num_sims = y0.shape[0]
        self.num_steps = len(t)
        
        # Determine dimension dynamically based on y0 shape (18 for 3D, 12 for 2D)
        if y0.shape[1] == 18:
            pos_dim = 9
            input_dim = 10
            output_dim = 18
        else:
            pos_dim = 6
            input_dim = 7
            output_dim = 12
            
        # Extract initial positions
        init_pos = y0[:, :pos_dim]
        
        # Vectorized expansion of initial positions across all time steps
        init_pos_repeated = np.repeat(init_pos[:, np.newaxis, :], self.num_steps, axis=1)
        
        # Vectorized expansion of time steps across all simulations
        t_grid = np.repeat(t[np.newaxis, :, np.newaxis], self.num_sims, axis=0)
        
        # Concat positions and time
        inputs_np = np.concatenate([init_pos_repeated, t_grid], axis=-1)
        
        # Transpose targets to shape (num_sims, num_steps, output_dim)
        targets_np = np.transpose(y, (0, 2, 1))
        
        # Reshape to 2D matrices
        self.inputs = torch.tensor(inputs_np.reshape(-1, input_dim), dtype=torch.float32)
        self.targets = torch.tensor(targets_np.reshape(-1, output_dim), dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_dataloaders(data_path, batch_size=5000, train_ratio=0.95, seed=42):
    # Load data to get total number of simulations
    data = np.load(data_path)
    num_simulations = data['y0'].shape[0]
    
    # Shuffle and split simulation indices (not individual points)
    # to guarantee validation evaluates on unseen trajectories
    np.random.seed(seed)
    indices = np.arange(num_simulations)
    np.random.shuffle(indices)
    
    split_idx = int(num_simulations * train_ratio)
    # Ensure at least 1 validation simulation if the dataset is small (e.g. for testing)
    if split_idx == num_simulations and num_simulations > 1:
        split_idx = num_simulations - 1
        
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = ThreeBodyDataset(data_path, sim_indices=train_indices)
    val_dataset = ThreeBodyDataset(data_path, sim_indices=val_indices)
    
    # Enable pin_memory for faster host-to-device transfers on CUDA GPUs
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    return train_loader, val_loader
