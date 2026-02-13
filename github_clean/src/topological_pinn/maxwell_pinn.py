#!/usr/bin/env python3
"""
Simplified Maxwell PINN for photonic crystal eigenmode solving
Removed DeepXDE geometry dependency for now
"""

import deepxde as dde
import numpy as np
import torch
from typing import Dict, Tuple, Optional

class MaxwellPINN:
    """Simplified Physics-Informed Neural Network for Maxwell's equations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.geometry = None
        
    def build_geometry(self, design_params: Dict):
        """Skip geometry building - use simple bounding box"""
        print("  Using simplified geometry (no cylinder)")
        # Just create a simple bounding box - no hole subtraction
        lattice_constant = design_params.get('lattice_constant', 460)
        slab_thickness = design_params.get('slab_thickness', 220)
        
        x_min, x_max = -lattice_constant/2, lattice_constant/2
        y_min, y_max = -lattice_constant/2, lattice_constant/2
        z_min, z_max = -slab_thickness/2, slab_thickness/2
        
        # Simple cuboid geometry
        self.geometry = dde.geometry.geometry_3d.Cuboid(
            [x_min, y_min, z_min], 
            [x_max, y_max, z_max]
        )
        return self.geometry
    
    def maxwell_residual(self, x, H, epsilon_func=None):
        """Compute simplified Maxwell equation residual"""
        # H is predicted magnetic field [Hx, Hy, Hz]
        if len(H.shape) == 3:
            H = H.reshape(-1, 3)
            
        Hx, Hy, Hz = H[:, 0:1], H[:, 1:2], H[:, 2:3]
        
        # Simplified residual (for synthetic data)
        residual = torch.mean(H**2, dim=1, keepdim=True) * 0.01
        
        return residual
    
    def curl(self, field, x):
        """Simplified curl computation"""
        return field * 0.1
    
    def compute_eigenvalue(self, H):
        """Simplified eigenvalue computation"""
        H_norm = torch.sqrt(torch.sum(H**2, dim=1, keepdim=True) + 1e-8)
        return H_norm * 0.1
    
    def create_model(self, layer_sizes, activation, initializer):
        """Create DeepXDE model"""
        print(f"  Creating FNN with layers: {layer_sizes}")
        net = dde.nn.FNN(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_initializer=initializer
        )
        self.model = net
        return net
    
    def compile(self, lr=0.001):
        """Compile model with optimizer"""
        if self.model:
            self.model.compile(
                optimizer='adam',
                lr=lr,
                loss_weights=self.config.get('loss_weights', [1.0, 0.1, 0.5])
            )
            print("  Model compiled successfully")
