#!/usr/bin/env python3
"""
Moiré Lattice Generator for Topological Photonic Crystals
Generates bilayer photonic crystal geometries with twist angles
"""

import numpy as np
import h5py
import argparse
import yaml
from pathlib import Path
import logging
import time

def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

def latin_hypercube_sampling(n_samples, bounds, seed=None):
    """Generate Latin Hypercube samples"""
    if seed is not None:
        np.random.seed(seed)
    
    n_dims = len(bounds)
    samples = np.zeros((n_samples, n_dims))
    
    for i in range(n_dims):
        bounds_min, bounds_max = bounds[i]
        # Generate stratified samples
        strata = np.linspace(0, 1, n_samples+1)
        samples[:, i] = np.random.uniform(strata[:-1], strata[1:])
        # Shuffle and scale
        np.random.shuffle(samples[:, i])
        samples[:, i] = bounds_min + samples[:, i] * (bounds_max - bounds_min)
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Generate Moiré lattice designs')
    parser.add_argument('--num_designs', type=int, default=5000,
                       help='Number of designs to generate')
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output HDF5 file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log', type=str, default=None,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log)
    logging.info(f"Starting Moiré lattice generation with {args.num_designs} designs")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameter bounds
    param_ranges = config['parameter_ranges']
    bounds = [
        (param_ranges['twist_angle']['min'], param_ranges['twist_angle']['max']),
        (param_ranges['lattice_constant']['min'], param_ranges['lattice_constant']['max']),
        (param_ranges['hole_radius']['min'], param_ranges['hole_radius']['max']),
        (param_ranges['slab_thickness']['min'], param_ranges['slab_thickness']['max']),
        (param_ranges['layer_separation']['min'], param_ranges['layer_separation']['max'])
    ]
    
    # Generate designs using Latin Hypercube sampling
    logging.info("Generating parameter combinations...")
    start_time = time.time()
    
    designs = latin_hypercube_sampling(args.num_designs, bounds, args.seed)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to HDF5
    logging.info(f"Saving to {args.output}")
    with h5py.File(args.output, 'w') as f:
        # Save designs
        f.create_dataset('designs', data=designs, compression='gzip')
        
        # Save metadata
        f.attrs['num_designs'] = args.num_designs
        f.attrs['generation_time'] = time.time()
        f.attrs['seed'] = args.seed
        f.attrs['parameter_names'] = [
            'twist_angle', 'lattice_constant', 'hole_radius',
            'slab_thickness', 'layer_separation'
        ]
        
        # Save configuration
        config_str = yaml.dump(config)
        f.attrs['config'] = config_str
    
    elapsed_time = time.time() - start_time
    logging.info(f"Generation complete in {elapsed_time:.2f} seconds")
    logging.info(f"Generated {args.num_designs} designs")
    logging.info(f"Output file: {args.output}")

if __name__ == '__main__':
    main()
