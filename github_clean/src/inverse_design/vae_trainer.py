import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import yaml
import logging
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        image = np.array(image) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)
        return image

def train_vae(model, train_loader, val_loader, epochs, device, save_dir):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize history lists - will store ONE value per epoch
    train_loss_history = []
    recon_loss_history = []
    kl_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = model(batch)
            
            # Calculate losses
            recon_loss = F.binary_cross_entropy(recon_batch, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.1 * kl_loss
            
            loss.backward()
            optimizer.step()
            
            # Accumulate
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            num_batches += 1
            
            # Update progress bar with current batch loss
            pbar.set_postfix({'loss': f'{loss.item():.2f}'})
        
        # Calculate averages for this epoch
        avg_train_loss = train_loss / num_batches
        avg_recon = train_recon / num_batches
        avg_kl = train_kl / num_batches
        
        # Store in history (ONE value per epoch)
        train_loss_history.append(avg_train_loss)
        recon_loss_history.append(avg_recon)
        kl_loss_history.append(avg_kl)
        
        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == epochs-1:
            logging.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}')
            torch.save(model.state_dict(), f'{save_dir}/checkpoint_epoch_{epoch}.pth')
    
    # Return history as dictionary with equal-length lists
    history = {
        'train_loss': train_loss_history,
        'recon_loss': recon_loss_history,
        'kl_loss': kl_loss_history,
        'epoch': list(range(epochs))
    }
    
    return history
