#!/usr/bin/env python3
"""
ULTRA-SIMPLE Trainer for Maxwell PINN
No DeepXDE complexity - pure PyTorch
"""

import os
import sys
import yaml
import h5py
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class SimplePINN(nn.Module):
    """Simple PyTorch neural network for PINN"""
    
    def __init__(self, input_dim=5, hidden_dim=100, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class SimpleTrainer:
    def __init__(self, data_path, output_dir, device='cuda'):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'training').mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard_logs')
        
        # Load data
        self.load_data(data_path)
        
        # Create model
        self.model = SimplePINN(input_dim=5, hidden_dim=100, output_dim=1).to(self.device)
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def load_data(self, data_path):
        """Load data from HDF5"""
        print(f"Loading data from {data_path}")
        
        with h5py.File(data_path, 'r') as f:
            designs = f['designs'][:]
            properties = f['properties'][:] if 'properties' in f else None
            
        print(f"Loaded {len(designs)} designs")
        
        # Use first property column (chern number) as target
        if properties is not None:
            targets = properties[:, 0]  # Chern number
        else:
            targets = np.zeros(len(designs))
        
        # Split into train/val
        n_samples = len(designs)
        indices = np.random.RandomState(42).permutation(n_samples)
        split = int(0.8 * n_samples)
        
        self.train_x = torch.tensor(designs[indices[:split]], dtype=torch.float32)
        self.train_y = torch.tensor(targets[indices[:split]], dtype=torch.float32).reshape(-1, 1)
        
        self.val_x = torch.tensor(designs[indices[split:]], dtype=torch.float32)
        self.val_y = torch.tensor(targets[indices[split:]], dtype=torch.float32).reshape(-1, 1)
        
        print(f"Training: {len(self.train_x)} samples")
        print(f"Validation: {len(self.val_x)} samples")
        
        # Move to device
        self.train_x = self.train_x.to(self.device)
        self.train_y = self.train_y.to(self.device)
        self.val_x = self.val_x.to(self.device)
        self.val_y = self.val_y.to(self.device)
    
    def train(self, epochs=1000):
        """Training loop"""
        print(f"Starting training for {epochs} epochs")
        
        batch_size = 64
        history = {'epoch': [], 'train_loss': [], 'val_loss': []}
        
        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            
            # Mini-batch training
            indices = torch.randperm(len(self.train_x))
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(self.train_x), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_x = self.train_x[batch_indices]
                batch_y = self.train_y[batch_indices]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.val_x)
                val_loss = self.criterion(val_outputs, self.val_y).item()
            
            # Log
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Save checkpoint
            if epoch % 100 == 0:
                self.save_checkpoint(epoch, avg_train_loss, val_loss)
        
        # Save final model
        self.save_checkpoint(epochs, avg_train_loss, val_loss, is_best=True)
        self.save_history(history)
        
        print(f"Training complete!")
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        filename = self.output_dir / 'models' / ('best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, filename)
        print(f"  Saved checkpoint to {filename}")
    
    def save_history(self, history):
        """Save training history"""
        import pandas as pd
        df = pd.DataFrame(history)
        df.to_csv(self.output_dir / 'training' / 'training_history.csv', index=False)
        print(f"Saved training history to {self.output_dir}/training/training_history.csv")

def main():
    parser = argparse.ArgumentParser(description='Simple PINN Trainer')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=1000)
    
    args = parser.parse_args()
    
    trainer = SimpleTrainer(
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device
    )
    
    trainer.train(epochs=args.epochs)

if __name__ == '__main__':
    main()
