#!/usr/bin/env python3
"""
Trainer for Maxwell PINN
Handles data loading, training loops, checkpointing
"""

import os
import sys
import json
import yaml
import h5py
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.topological_pinn.maxwell_pinn import MaxwellPINN
from src.topological_pinn.boundary_conditions import get_boundary_conditions
from src.topological_pinn.topological_loss import TopologicalLoss

class PINNTrainer:
    """Training orchestration for Maxwell PINN"""
    
    def __init__(self, config_path: str, data_path: str, output_dir: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard_logs')
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Load data
        self.train_data, self.val_data = self.load_data()
        
        # Topological loss
        self.topological_loss = TopologicalLoss(
            target_chern=self.config.get('target_chern', 1),
            weight=self.config.get('topology_weight', 0.3)
        )
        
    def load_data(self):
        """Load training and validation data from HDF5"""
        print(f"Loading data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load designs and properties
            designs = f['designs'][:]
            properties = f['properties'][:] if 'properties' in f else None
            
            # Split into train/val (80/20)
            n_samples = len(designs)
            indices = np.random.permutation(n_samples)
            split = int(0.8 * n_samples)
            
            train_indices = indices[:split]
            val_indices = indices[split:]
            
            train_data = designs[train_indices]
            val_data = designs[val_indices]
            
            if properties is not None:
                train_props = properties[train_indices]
                val_props = properties[val_indices]
            else:
                train_props = None
                val_props = None
        
        print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        return (train_data, train_props), (val_data, val_props)
    
    def build_model(self):
        """Build the PINN model"""
        print("Building Maxwell PINN model...")
        
        # Get architecture from config
        layer_sizes = self.config['model']['layers']
        activation = self.config['model']['activation']
        initializer = self.config['model']['initializer']
        
        # Create model
        pinn = MaxwellPINN(self.config)
        
        # Build geometry (using first design as reference)
        design_params = {
            'lattice_constant': self.train_data[0][1] * 400 + 400,  # Scale to nm
            'hole_radius': self.train_data[0][2] * 40 + 80,  # Scale to nm
            'slab_thickness': self.train_data[0][3] * 40 + 200,  # Scale to nm
        }
        pinn.build_geometry(design_params)
        
        # Create neural network
        net = pinn.create_model(layer_sizes, activation, initializer)
        
        self.model = net
        print("Model built successfully")
        
        return self.model
    
    def train(self, epochs: int = 1000, save_every: int = 50):
        """Main training loop"""
        print(f"Starting training for {epochs} epochs")
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate']
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
        
        # Training history
        history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'maxwell_residual': [],
            'topo_loss': [],
            'lr': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Training step
            train_loss, train_maxwell, train_topo = self.train_epoch(epoch)
            
            # Validation step
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to history
            history['epoch'].append(epoch)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['maxwell_residual'].append(train_maxwell)
            history['topo_loss'].append(train_topo)
            history['lr'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/maxwell_residual', train_maxwell, epoch)
            self.writer.add_scalar('Metrics/topological_loss', train_topo, epoch)
            self.writer.add_scalar('Training/learning_rate', current_lr, epoch)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"maxwell_res={train_maxwell:.4f}, topo_loss={train_topo:.4f}, lr={current_lr:.6f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)
                
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
        
        # Save training history
        self.save_history(history)
        
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        return history
    
    def train_epoch(self, epoch):
        """Single training epoch"""
        self.model.train()
        
        # Get batch
        batch_indices = np.random.choice(len(self.train_data[0]), size=self.config['training']['batch_size'])
        batch_x = self.train_data[0][batch_indices]
        
        # Convert to tensors
        x_tensor = torch.tensor(batch_x, dtype=torch.float32, device=self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        y_pred = self.model(x_tensor)
        
        # Compute losses
        mse_loss = torch.mean((y_pred - torch.zeros_like(y_pred)) ** 2)  # Simplified
        topo_loss = self.topological_loss(self.model, x_tensor, None)
        
        # Maxwell residual (simplified)
        maxwell_res = torch.mean(y_pred ** 2) * 0.1
        
        # Total loss
        total_loss = mse_loss + maxwell_res + topo_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), maxwell_res.item(), topo_loss.item()
    
    def validate(self):
        """Validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Get validation batch
            val_indices = np.random.choice(len(self.val_data[0]), size=min(32, len(self.val_data[0])))
            val_x = self.val_data[0][val_indices]
            x_tensor = torch.tensor(val_x, dtype=torch.float32, device=self.device)
            
            # Forward pass
            y_pred = self.model(x_tensor)
            
            # Compute loss
            mse_loss = torch.mean((y_pred - torch.zeros_like(y_pred)) ** 2)
            val_loss = mse_loss.item()
        
        return val_loss
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        if is_best:
            filename = self.output_dir / 'models' / 'best_model.pth'
        else:
            filename = self.output_dir / 'models' / f'checkpoint_epoch_{epoch}.pth'
        
        # Create directory if needed
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")
    
    def save_history(self, history):
        """Save training history to CSV"""
        import pandas as pd
        df = pd.DataFrame(history)
        df.to_csv(self.output_dir / 'training' / 'training_history.csv', index=False)
        print(f"Saved training history to {self.output_dir}/training/training_history.csv")

def main():
    parser = argparse.ArgumentParser(description='Train Maxwell PINN')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PINNTrainer(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Build model
    trainer.build_model()
    
    # Train
    trainer.train(epochs=args.epochs)

if __name__ == '__main__':
    main()
