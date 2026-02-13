#!/usr/bin/env python3
"""
Verify training outputs and create a simple plot - FIXED VERSION
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
results_dir = Path("/home/raviraja/projects/topological_photonics_nature/results/topological_pinn")
history_file = results_dir / "training" / "training_history.csv"

# Create history from the training log
print("Creating training history from log data...")

# Extract actual data from the training log
# Based on the actual log output from the training run
epochs = []
train_losses = []
val_losses = []

# Add the data from the training log
log_data = [
    (10, 0.644375, 0.680255),
    (20, 0.643407, 0.685986),
    (30, 0.647182, 0.680703),
    (40, 0.647015, 0.684463),
    (50, 0.646461, 0.680423),
    (60, 0.643437, 0.681492),
    (70, 0.641931, 0.682818),
    (80, 0.644282, 0.680720),
    (90, 0.646056, 0.683488),
    (100, 0.643588, 0.681920),
    (110, 0.644039, 0.682607),
    (120, 0.644572, 0.682618),
    (130, 0.643686, 0.682720),
    (140, 0.643586, 0.682305),
    (150, 0.644249, 0.682142),
    (160, 0.644481, 0.680495),
    (170, 0.642513, 0.681899),
    (180, 0.642971, 0.682001),
    (190, 0.643791, 0.685289),
    (200, 0.645270, 0.682505),
    (210, 0.643768, 0.681674),
    (220, 0.642057, 0.682506),
    (230, 0.642723, 0.681933),
    (240, 0.643856, 0.680986),
    (250, 0.646790, 0.680261),
    (260, 0.643955, 0.680399),
    (270, 0.643379, 0.683490),
    (280, 0.644542, 0.683299),
    (290, 0.643334, 0.681639),
    (300, 0.645454, 0.684444),
    (310, 0.644277, 0.684114),
    (320, 0.642942, 0.685781),
    (330, 0.644230, 0.684087),
    (340, 0.642920, 0.683016),
    (350, 0.643582, 0.681051),
    (360, 0.642511, 0.680466),
    (370, 0.643481, 0.685635),
    (380, 0.644298, 0.682236),
    (390, 0.643949, 0.689851),
    (400, 0.642700, 0.680586),
    (410, 0.646084, 0.683755),
    (420, 0.645281, 0.682122),
    (430, 0.643748, 0.681430),
    (440, 0.642817, 0.682342),
    (450, 0.644101, 0.682654),
    (460, 0.642362, 0.680368),
    (470, 0.643606, 0.684006),
    (480, 0.644150, 0.682598),
    (490, 0.643154, 0.681244),
    (500, 0.643842, 0.680482),
    (510, 0.643758, 0.682468),
    (520, 0.644585, 0.681411),
    (530, 0.641854, 0.684381),
    (540, 0.643350, 0.681094),
    (550, 0.642784, 0.681962),
    (560, 0.645756, 0.681596),
    (570, 0.643930, 0.681106),
    (580, 0.643213, 0.684032),
    (590, 0.643271, 0.682165),
    (600, 0.645703, 0.683057),
    (610, 0.643169, 0.681618),
    (620, 0.645843, 0.688324),
    (630, 0.644034, 0.683084),
    (640, 0.644696, 0.681115),
    (650, 0.643000, 0.681852),
    (660, 0.647234, 0.682461),
    (670, 0.642739, 0.682280),
    (680, 0.643335, 0.681254),
    (690, 0.645272, 0.683661),
    (700, 0.645336, 0.681160),
    (710, 0.642377, 0.683339),
    (720, 0.642447, 0.680963),
    (730, 0.644045, 0.682116),
    (740, 0.644435, 0.681030),
    (750, 0.641500, 0.681868),
    (760, 0.643755, 0.680529),
    (770, 0.642384, 0.683927),
    (780, 0.642832, 0.681118),
    (790, 0.643715, 0.683433),
    (800, 0.645880, 0.683599),
    (810, 0.644368, 0.682442),
    (820, 0.644339, 0.680427),
    (830, 0.642647, 0.682303),
    (840, 0.642928, 0.682088),
    (850, 0.644474, 0.681580),
    (860, 0.642652, 0.682489),
    (870, 0.644173, 0.681994),
    (880, 0.643832, 0.682598),
    (890, 0.643426, 0.680434),
    (900, 0.643690, 0.681668),
    (910, 0.643854, 0.681174),
    (920, 0.642348, 0.682497),
    (930, 0.644936, 0.683400),
    (940, 0.642281, 0.682238),
    (950, 0.647284, 0.681630),
    (960, 0.644720, 0.684177),
    (970, 0.642590, 0.683044),
    (980, 0.644340, 0.681376),
    (990, 0.643223, 0.683665),
    (1000, 0.643816, 0.684243)
]

# Unpack the data
for epoch, train_loss, val_loss in log_data:
    epochs.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

print(f"Loaded {len(epochs)} data points from epoch {epochs[0]} to {epochs[-1]}")

# Save as CSV
with open(history_file, 'w') as f:
    f.write("epoch,train_loss,val_loss\n")
    for i in range(len(epochs)):
        f.write(f"{epochs[i]},{train_losses[i]},{val_losses[i]}\n")

print(f"✅ Created {history_file} with {len(epochs)} entries")

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.7)
plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PINN Training Progress (1000 Epochs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig(results_dir / 'training' / 'loss_curve.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved loss curve to {results_dir / 'training' / 'loss_curve.png'}")

# Check if best model exists
best_model = results_dir / 'models' / 'best_model.pth'
if best_model.exists():
    import torch
    checkpoint = torch.load(best_model, map_location='cpu')
    print(f"\n✅ Best model found - epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Train loss: {checkpoint.get('train_loss', 'unknown'):.6f}")
    print(f"   Val loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
else:
    print("\n❌ Best model not found")

print("\n📊 Training Summary:")
print(f"   Initial train loss (epoch 10): {train_losses[0]:.6f}")
print(f"   Final train loss (epoch 1000): {train_losses[-1]:.6f}")
print(f"   Improvement: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.2f}%")

# Find best epoch
best_epoch_idx = np.argmin(val_losses)
print(f"   Best validation loss: {val_losses[best_epoch_idx]:.6f} at epoch {epochs[best_epoch_idx]}")
