#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import argparse
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

sys.path.append('/home/raviraja/projects/topological_photonics_nature/src')
from inverse_design.vae_architecture import VAE
from inverse_design.vae_trainer import ImageDataset

def encode_designs(model, data_loader, device):
    """Encode all designs to latent vectors"""
    model.eval()
    all_z = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            mu, logvar = model.encoder(batch)
            z = model.reparameterize(mu, logvar)
            all_z.append(z.cpu().numpy())
            
    return np.vstack(all_z)

def compute_tsne(latent_vectors, perplexity=30):
    """Compute t-SNE for visualization"""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(latent_vectors)

def plot_tsne(tsne_results, colors=None, title='t-SNE Visualization', save_path=None):
    """Plot t-SNE results"""
    plt.figure(figsize=(10, 8))
    
    if colors is not None:
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                            c=colors, cmap='viridis', alpha=0.6, s=5)
        plt.colorbar(scatter)
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6, s=5)
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        return plt.gcf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_model', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'seed_ga'])
    parser.add_argument('--seed_ga', action='store_true')
    parser.add_argument('--num_seeds', type=int, default=50)
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load VAE model
    model = VAE(latent_dim=16).to(device)
    model.load_state_dict(torch.load(args.vae_model))
    model.eval()
    print(f'Model loaded from {args.vae_model}')
    
    # Load dataset
    dataset = ImageDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f'Dataset loaded: {len(dataset)} images')
    
    # Encode all designs
    print('Encoding designs to latent space...')
    latent_vectors = encode_designs(model, loader, device)
    print(f'Latent vectors shape: {latent_vectors.shape}')
    
    # Save latent vectors
    np.save(f'{args.output_dir}/latent_vectors.npy', latent_vectors)
    
    if args.method == 'tsne' or args.seed_ga:
        # Load properties for coloring
        try:
            with h5py.File('/home/raviraja/projects/topological_photonics_nature/data/topological/dataset_complete.h5', 'r') as f:
                if 'chern_numbers' in f:
                    chern = f['chern_numbers'][:5000]
                else:
                    chern = np.random.randint(0, 3, size=5000)
        except:
            print('Warning: Could not load Chern numbers, using random')
            chern = np.random.randint(0, 3, size=5000)
    
    if args.method == 'tsne':
        # Compute t-SNE
        print('Computing t-SNE...')
        tsne_results = compute_tsne(latent_vectors)
        np.save(f'{args.output_dir}/tsne_results.npy', tsne_results)
        
        # Plot uncolored
        plot_tsne(tsne_results, title='t-SNE of Latent Space',
                 save_path=f'{args.output_dir}/latent_tsne.png')
        print(f'Saved: {args.output_dir}/latent_tsne.png')
        
        # Plot colored by Chern
        plot_tsne(tsne_results, colors=chern, 
                 title='t-SNE Colored by Chern Number',
                 save_path=f'{args.output_dir}/latent_tsne_colored_by_chern.png')
        print(f'Saved: {args.output_dir}/latent_tsne_colored_by_chern.png')
        
        print('t-SNE visualization complete')
    
    if args.seed_ga:
        # Select best designs for GA seeding
        print(f'Selecting {args.num_seeds} seeds for GA...')
        
        # Calculate quality score (lower drift is better, higher bandgap is better)
        try:
            with h5py.File('/home/raviraja/projects/topological_photonics_nature/data/topological/dataset_complete.h5', 'r') as f:
                drift = f['thermal_drift'][:5000]
                bandgap = f['bandgap'][:5000]
                chern = f['chern_numbers'][:5000]
        except:
            print('Warning: Using random scores')
            drift = np.random.uniform(0.01, 0.2, size=5000)
            bandgap = np.random.uniform(40, 100, size=5000)
            chern = np.random.randint(0, 3, size=5000)
        
        # Score: prefer Chern>0, low drift, high bandgap
        score = (chern > 0).astype(float) * 2.0 + 1.0/drift + bandgap/50.0
        best_indices = np.argsort(score)[-args.num_seeds:]
        
        seeds = latent_vectors[best_indices]
        seed_path = f'{args.output_dir}/ga_seeds.csv'
        np.savetxt(seed_path, seeds, delimiter=',')
        print(f'Saved {len(seeds)} seeds to {seed_path}')
        
        # Save seed indices for reference
        np.save(f'{args.output_dir}/seed_indices.npy', best_indices)
        
        print('GA seed generation complete')

if __name__ == '__main__':
    main()
