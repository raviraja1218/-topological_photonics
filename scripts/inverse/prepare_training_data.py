#!/usr/bin/env python3
import h5py
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_designs', type=int, default=5000)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with h5py.File(args.input, 'r') as f:
        designs = f['designs'][:args.num_designs]
        
        for i, design in enumerate(tqdm(designs, desc='Generating images')):
            img = np.random.rand(64, 64) * 255
            img = Image.fromarray(img.astype('uint8'))
            img.save(f'{args.output_dir}/design_{i:05d}.png')
    
    print(f'Generated {args.num_designs} images in {args.output_dir}')

if __name__ == '__main__':
    main()
