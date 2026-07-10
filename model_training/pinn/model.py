"""
PyTorch Neural Network Models for the Three-Body Problem.
Defines Standard DNN feedforward architectures and ResNetDNN architectures
featuring residual blocks to stabilize training under physics-informed losses.
"""

import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU()):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = self.activation(out + x)
        return out

class ResNetDNN(nn.Module):
    def __init__(self, input_dim=7, output_dim=12, hidden_dim=256, depth=12, activation=nn.ReLU()):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        
        # Number of hidden layers is depth - 2.
        # Each ResNetBlock has 2 hidden layers.
        if (depth - 2) % 2 != 0:
            raise ValueError("For ResNetDNN, (depth - 2) must be even to pack into ResNetBlocks of 2 layers each.")
        num_blocks = (depth - 2) // 2
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResNetBlock(hidden_dim, activation))
        self.blocks = nn.Sequential(*blocks)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.activation(self.input_layer(x))
        out = self.blocks(out)
        out = self.output_layer(out)
        return out

class StandardDNN(nn.Module):
    def __init__(self, input_dim=7, output_dim=12, hidden_dim=256, depth=12, activation=nn.ReLU()):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_model(model_type, input_dim=7, output_dim=12, hidden_dim=256, depth=12, activation_name='relu'):
    # Select activation function
    act_dict = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU()
    }
    activation = act_dict.get(activation_name.lower(), nn.ReLU())
    
    if model_type.lower() == 'resnet':
        return ResNetDNN(input_dim, output_dim, hidden_dim, depth, activation)
    elif model_type.lower() == 'standard':
        return StandardDNN(input_dim, output_dim, hidden_dim, depth, activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
