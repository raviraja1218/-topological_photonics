#!/usr/bin/env python3
"""
Generate realistic synthetic dataset for topological photonic crystals
Creates designs with physical properties based on theoretical models
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
import time

def generate_designs(num_designs, seed=42):
    """Generate design parameters with realistic distributions"""
    np.random.seed(seed)
    
    # Parameter ranges (physical units)
    twist_angle = np.random.uniform(0.5, 2.5, num_designs)
    lattice_constant = np.random.uniform(400, 500, num_designs)
    hole_radius = np.random.uniform(80, 120, num_designs)
    slab_thickness = np.random.uniform(200, 240, num_designs)
    layer_separation = np.random.uniform(50, 200, num_designs)
    
    # Stack into design matrix
    designs = np.column_stack([
        twist_angle,
        lattice_constant,
        hole_radius,
        slab_thickness,
        layer_separation
    ])
    
    return designs

def compute_chern_number(twist_angle, lattice_constant):
    """Compute Chern number based on twist angle"""
    # Magic angle around 1.12 degrees
    magic_angle = 1.12
    angle_diff = np.abs(twist_angle - magic_angle)
    
    if angle_diff < 0.15:
        return 1  # Topological phase around magic angle
    elif 1.8 < twist_angle < 2.2:
        return 2  # Higher order topological phase
    else:
        return 0  # Trivial phase

def compute_bandgap(twist_angle, lattice_constant, hole_radius):
    """Compute bandgap in meV"""
    magic_angle = 1.12
    angle_diff = np.abs(twist_angle - magic_angle)
    
    # Base bandgap depends on lattice constant and hole radius
    base_gap = 40 + (lattice_constant - 400) * 0.1 + (hole_radius - 80) * 0.2
    
    # Enhancement at magic angle
    enhancement = 30 * np.exp(-angle_diff**2 / 0.02)
    
    # Add some noise
    noise = np.random.randn() * 3
    
    return base_gap + enhancement + noise

def compute_thermal_drift(twist_angle):
    """Compute thermal drift in pm/K"""
    magic_angle = 1.12
    angle_diff = np.abs(twist_angle - magic_angle)
    
    # Minimum at magic angle
    drift = 0.03 + 0.2 * angle_diff + 0.01 * np.random.randn()
    
    return max(0.01, drift)  # Never below 0.01

def compute_q_factor(bandgap, twist_angle):
    """Compute quality factor"""
    # Higher bandgap generally means higher Q
    base_q = 5000 + bandgap * 200
    
    # Enhancement at magic angle
    magic_angle = 1.12
    angle_diff = np.abs(twist_angle - magic_angle)
    enhancement = 50000 * np.exp(-angle_diff**2 / 0.03)
    
    # Add realistic variations
    noise = np.random.randn() * 2000
    
    return base_q + enhancement + noise

def compute_loss(q_factor):
    """Compute propagation loss in dB/cm"""
    # Loss inversely proportional to Q
    return 1.0 / (q_factor / 10000) * (0.8 + 0.2 * np.random.rand())

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--num_designs', type=int, default=5000,
                       help='Number of designs to generate')
    parser.add_argument('--output', type=str, required=True,
                       help='Output HDF5 file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_designs} synthetic designs...")
    start_time = time.time()
    
    # Generate designs
    designs = generate_designs(args.num_designs, args.seed)
    
    # Compute properties for each design
    properties = np.zeros((args.num_designs, 5))
    
    for i in range(args.num_designs):
        twist = designs[i, 0]
        lattice = designs[i, 1]
        radius = designs[i, 2]
        
        # Compute physical properties
        chern = compute_chern_number(twist, lattice)
        bandgap = compute_bandgap(twist, lattice, radius)
        drift = compute_thermal_drift(twist)
        q_factor = compute_q_factor(bandgap, twist)
        loss = compute_loss(q_factor)
        
        properties[i] = [chern, bandgap, drift, q_factor, loss]
        
        if (i+1) % 1000 == 0:
            print(f"  Processed {i+1}/{args.num_designs} designs")
    
    # Save to HDF5
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('designs', data=designs, compression='gzip')
        f.create_dataset('properties', data=properties, compression='gzip')
        
        f.attrs['num_designs'] = args.num_designs
        f.attrs['generation_time'] = time.time()
        f.attrs['seed'] = args.seed
        f.attrs['description'] = 'Synthetic dataset with realistic physical properties'
        f.attrs['property_names'] = [
            'chern_number',
            'bandgap_meV',
            'thermal_drift_pm_per_K',
            'q_factor',
            'loss_dB_per_cm'
        ]
    
    elapsed = time.time() - start_time
    print(f"\n✅ Dataset generation complete in {elapsed:.1f} seconds")
    print(f"   File: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Chern numbers: {np.bincount(properties[:,0].astype(int))}")
    print(f"   Bandgap range: {properties[:,1].min():.1f} - {properties[:,1].max():.1f} meV")
    print(f"   Thermal drift: {properties[:,2].min():.3f} - {properties[:,2].max():.3f} pm/K")
    print(f"   Q-factor range: {properties[:,3].min():.0f} - {properties[:,3].max():.0f}")
    
    # Find magic angle candidates
    magic_idx = np.where((properties[:,0] == 1) & (properties[:,2] < 0.05))[0]
    print(f"\n🎯 Found {len(magic_idx)} magic angle candidates (Chern=1, drift<0.05 pm/K)")

if __name__ == '__main__':
    main()
