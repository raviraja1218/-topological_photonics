#!/usr/bin/env python3
"""
Trainer for Maxwell PINN - FIXED VERSION
"""

import os
import sys
import yaml
import h5py
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.topological_pinn.maxwell_pinn import MaxwellPINN
from src.topological_pinn.topological_loss import TopologicalLoss

class PINNTrainer:
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
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'training').mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard_logs')
        
        # Load data
        self.train_data, self.val_data = self.load_data()
        
        # Topological loss
        self.topological_loss = TopologicalLoss(
            target_chern=1,
            weight=0.3
        )
        
    def load_data(self):
        """Load training and validation data from HDF5"""
        print(f"Loading data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            designs = f['designs'][:]
            properties = f['properties'][:] if 'properties' in f else None
            
            print(f"Loaded {len(designs)} designs")
            
            # Split into train/val (80/20)
            n_samples = len(designs)
            indices = np.random.RandomState(42).permutation(n_samples)
            split = int(0.8 * n_samples)
            
            train_indices = indices[:split]
            val_indices = indices[split:]
            
            train_data = designs[train_indices]
            val_data = designs[val_indices]
            
            if properties is not None:
                train_props = properties[train_indices]
                val_props = properties[val_indices]
            else:
                train_props = np.zeros((len(train_data), 1))
                val_props = np.zeros((len(val_data), 1))
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        return (train_data, train_props), (val_data, val_props)
    
    def build_model(self):
        """Build the PINN model"""
        print("Building Maxwell PINN model...")
        
        layer_sizes = self.config['model']['layers']
        activation = self.config['model']['activation']
        initializer = self.config['model']['initializer']
        
        # Create model
        pinn = MaxwellPINN(self.config)
        
        # Build geometry with default params
        design_params = {
            'lattice_constant': 460.0,
            'hole_radius': 100.0,
            'slab_thickness': 220.0,
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
        
        # Simple optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate']
        )
        
        # Training history
        history = {'epoch': [], 'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Training step
            train_loss = self.train_epoch()
            
            # Validation step
            val_loss = self.validate()
            
            # Log
            history['epoch'].append(epoch)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
        
        self.save_history(history)
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    
    def train_epoch(self):
        """Single training epoch"""
        self.model.train()
        
        # Get batch
        batch_size = self.config['training']['batch_size']
        batch_indices = np.random.choice(len(self.train_data[0]), size=min(batch_size, len(self.train_data[0])))
        batch_x = self.train_data[0][batch_indices]
        
        # Convert to tensor
        x_tensor = torch.tensor(batch_x, dtype=torch.float32, device=self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        y_pred = self.model(x_tensor)
        
        # Simple loss
        loss = torch.mean(y_pred ** 2)
        
        # Add topological loss
        topo_loss = self.topological_loss(self.model, x_tensor, None)
        total_loss = loss + topo_loss
        
        # Backward
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def validate(self):
        """Validation step"""
        self.model.eval()
        with torch.no_grad():
            val_indices = np.random.choice(len(self.val_data[0]), size=min(32, len(self.val_data[0])))
            val_x = self.val_data[0][val_indices]
            x_tensor = torch.tensor(val_x, dtype=torch.float32, device=self.device)
            y_pred = self.model(x_tensor)
            val_loss = torch.mean(y_pred ** 2).item()
        return val_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        filename = self.output_dir / 'models' / ('best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, filename)
    
    def save_history(self, history):
        """Save training history"""
        import pandas as pd
        df = pd.DataFrame(history)
        df.to_csv(self.output_dir / 'training' / 'training_history.csv', index=False)
        print(f"Saved training history to {self.output_dir}/training/training_history.csv")

def main():
    parser = argparse.ArgumentParser(description='Train Maxwell PINN')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=1000)
    
    args = parser.parse_args()
    
    trainer = PINNTrainer(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device
    )
    
    trainer.build_model()
    trainer.train(epochs=args.epochs)

if __name__ == '__main__':
    main()
