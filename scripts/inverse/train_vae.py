#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

sys.path.append('/home/raviraja/projects/topological_photonics_nature/src')
from inverse_design.vae_architecture import VAE
from inverse_design.vae_trainer import ImageDataset, train_vae

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{log_dir}/vae_training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_dir)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create dataset
    dataset = ImageDataset(args.data_dir)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = VAE(latent_dim=config['model']['latent_dim']).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Model created with {num_params} parameters')
    
    # Train
    history = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        device=device,
        save_dir=args.output_dir
    )
    
    # Save final model
    torch.save(model.state_dict(), f'{args.output_dir}/vae_model.pth')
    
    # Save history as CSV
    df = pd.DataFrame(history)
    df.to_csv(f'{args.output_dir}/vae_training_history.csv', index=False)
    
    # Print final stats
    final_loss = history['train_loss'][-1]
    logging.info(f'Training complete. Final loss: {final_loss:.4f}')
    print(f'✅ Training complete! Final loss: {final_loss:.4f}')
    print(f'📊 History saved to: {args.output_dir}/vae_training_history.csv')
    print(f'💾 Model saved to: {args.output_dir}/vae_model.pth')

if __name__ == '__main__':
    main()
