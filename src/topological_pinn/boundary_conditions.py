#!/usr/bin/env python3
"""
Boundary conditions for photonic crystal simulations
PML, periodic BCs, and symmetry conditions
"""

import deepxde as dde
import numpy as np
import torch

class PML:
    """Perfectly Matched Layer for absorbing boundaries"""
    
    def __init__(self, thickness=0.5, absorption_profile='quadratic'):
        self.thickness = thickness
        self.profile = absorption_profile
        
    def apply(self, x, field):
        """Apply PML absorption to field"""
        # Compute distance to boundary
        dist_to_boundary = self.compute_distance(x)
        
        # Compute absorption profile
        if self.profile == 'quadratic':
            sigma = (dist_to_boundary / self.thickness) ** 2
        elif self.profile == 'cubic':
            sigma = (dist_to_boundary / self.thickness) ** 3
        else:
            sigma = dist_to_boundary / self.thickness
            
        sigma = torch.clamp(sigma, 0, 1)
        
        # Apply absorption
        absorbed_field = field * torch.exp(-sigma * 5)
        return absorbed_field
    
    def compute_distance(self, x):
        """Compute distance to nearest boundary"""
        x_coords = x[:, 0:1]
        y_coords = x[:, 1:2]
        z_coords = x[:, 2:3]
        
        # Simple distance to boundaries (assuming unit cell from -1 to 1)
        dist_x = torch.min(1 - torch.abs(x_coords), torch.ones_like(x_coords) * self.thickness)
        dist_y = torch.min(1 - torch.abs(y_coords), torch.ones_like(y_coords) * self.thickness)
        dist_z = torch.min(1 - torch.abs(z_coords), torch.ones_like(z_coords) * self.thickness)
        
        return torch.min(torch.cat([dist_x, dist_y, dist_z], dim=1), dim=1, keepdim=True)[0]

class PeriodicBC:
    """Periodic boundary conditions for unit cell"""
    
    def __init__(self, period_vector):
        self.period_vector = period_vector
        
    def apply(self, x, field):
        """Enforce periodic condition: field(x) = field(x + period)"""
        # This is enforced in the loss function, not here
        return field
    
    def loss(self, x, field, period_dim=0):
        """Loss term for periodic condition"""
        x_periodic = x.clone()
        x_periodic[:, period_dim] += self.period_vector[period_dim]
        
        # Need model prediction at periodic point
        # This will be called from trainer with full model
        return None

class SymmetryBC:
    """Symmetry boundary conditions"""
    
    def __init__(self, symmetry_plane, symmetry_type='even'):
        self.plane = symmetry_plane
        self.type = symmetry_type  # 'even' or 'odd'
        
    def apply(self, x, field):
        """Apply symmetry condition"""
        if self.type == 'even':
            return field
        else:  # odd
            # Mirror across symmetry plane
            x_mirror = x.clone()
            x_mirror[:, self.plane] *= -1
            # field should be negated
            return -field

def get_boundary_conditions(config):
    """Factory function to create all boundary conditions"""
    bc_list = []
    
    # Add PML
    if config.get('use_pml', True):
        pml = PML(
            thickness=config.get('pml_thickness', 0.5),
            absorption_profile=config.get('pml_profile', 'quadratic')
        )
        bc_list.append(pml)
    
    # Add periodic BCs
    if config.get('use_periodic', True):
        period_vector = config.get('period_vector', [1.0, 1.0, 0.0])
        periodic = PeriodicBC(period_vector)
        bc_list.append(periodic)
    
    # Add symmetry BCs
    if config.get('use_symmetry', False):
        for sym in config.get('symmetries', []):
            symmetry = SymmetryBC(
                symmetry_plane=sym['plane'],
                symmetry_type=sym.get('type', 'even')
            )
            bc_list.append(symmetry)
    
    return bc_list
