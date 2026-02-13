#!/usr/bin/env python3
"""
Create complete dataset from raw designs
Adds synthetic properties for training
"""

import h5py
import numpy as np
from pathlib import Path

def main():
    # Paths
    raw_path = Path("/home/raviraja/projects/topological_photonics_nature/data/topological/raw_designs.h5")
    output_path = Path("/home/raviraja/projects/topological_photonics_nature/data/topological/dataset_complete.h5")
    
    print(f"Reading raw designs from: {raw_path}")
    print(f"Creating complete dataset at: {output_path}")
    
    # Check if raw file exists
    if not raw_path.exists():
        print("ERROR: raw_designs.h5 not found!")
        print("Please run Phase 1 dataset generation first.")
        return
    
    # Open raw file
    with h5py.File(raw_path, 'r') as f_in:
        designs = f_in['designs'][:]
        print(f"Found {len(designs)} designs")
        
        # Generate synthetic properties for training
        num_designs = len(designs)
        
        # Create synthetic properties array
        # Each row: [chern_number, bandgap, thermal_drift, q_factor, loss]
        properties = np.zeros((num_designs, 5))
        
        # Generate realistic synthetic data
        for i, design in enumerate(designs):
            twist_angle = design[0] * 2.0 + 0.5  # Scale to 0.5-2.5 range
            
            # Chern number: 1 around magic angle, 0 elsewhere
            if 1.0 < twist_angle < 1.3:
                properties[i, 0] = 1  # Chern = 1
            elif 1.8 < twist_angle < 2.2:
                properties[i, 0] = 2  # Chern = 2
            else:
                properties[i, 0] = 0  # Chern = 0
            
            # Bandgap: peaks at magic angle
            properties[i, 1] = 50 + 30 * np.exp(-(twist_angle - 1.12)**2 / 0.1)
            
            # Thermal drift: minimum at magic angle
            properties[i, 2] = 0.05 + 0.2 * np.abs(twist_angle - 1.12)
            
            # Q-factor: related to bandgap
            properties[i, 3] = 10000 * (properties[i, 1] / 50)
            
            # Loss: inversely related to Q
            properties[i, 4] = 1.0 / (properties[i, 3] / 10000)
    
    # Write to output file
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('designs', data=designs, compression='gzip')
        f_out.create_dataset('properties', data=properties, compression='gzip')
        f_out.attrs['num_designs'] = num_designs
        f_out.attrs['description'] = 'Complete dataset with synthetic properties'
    
    print(f"✅ Created dataset_complete.h5 with {num_designs} designs")
    print(f"   Properties: Chern, Bandgap, Thermal Drift, Q-factor, Loss")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()
