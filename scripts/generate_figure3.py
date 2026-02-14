import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

fig = plt.figure(figsize=(12, 10))

# Panel A: VAE Architecture
ax1 = plt.subplot(2, 2, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')

# Encoder
rect1 = plt.Rectangle((1, 2), 2, 2, fill=True, color='lightblue', ec='black')
ax1.add_patch(rect1)
ax1.text(2, 3, 'Encoder\n256→128→64', ha='center', va='center')

# Latent
rect2 = plt.Rectangle((4, 2.5), 2, 1, fill=True, color='lightgreen', ec='black')
ax1.add_patch(rect2)
ax1.text(5, 3, 'Latent\n(16)', ha='center', va='center')

# Decoder
rect3 = plt.Rectangle((7, 2), 2, 2, fill=True, color='lightcoral', ec='black')
ax1.add_patch(rect3)
ax1.text(8, 3, 'Decoder\n64→128→256', ha='center', va='center')

# Arrows
ax1.annotate('', xy=(3, 3), xytext=(4, 3), arrowprops=dict(arrowstyle='->'))
ax1.annotate('', xy=(6, 3), xytext=(7, 3), arrowprops=dict(arrowstyle='->'))

ax1.set_title('a) VAE Architecture', fontweight='bold')

# Panel B: t-SNE Latent Space
ax2 = plt.subplot(2, 2, 2)
tsne_img = mpimg.imread('results/inverse_design/latent/latent_tsne_colored_by_chern.png')
ax2.imshow(tsne_img)
ax2.axis('off')
ax2.set_title('b) Latent Space t-SNE', fontweight='bold')

# Panel C: Reconstructions
ax3 = plt.subplot(2, 2, 3)
ax3.axis('off')
# Create 3x2 grid of original/reconstructed
for i in range(3):
    for j in range(2):
        img = np.random.rand(64, 64)  # Placeholder
        ax3.imshow(img, extent=[j*2, j*2+1, 2-i, 2-i+1], cmap='gray')
ax3.set_title('c) Original vs Reconstructed', fontweight='bold')

# Panel D: Latent Interpolation
ax4 = plt.subplot(2, 2, 4)
# Create interpolation images
for i in range(5):
    img = np.random.rand(64, 64)  # Placeholder
    ax4.imshow(img, extent=[i, i+1, 0, 1], cmap='gray')
ax4.set_xlim(0, 5)
ax4.set_ylim(0, 1)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_title('d) Latent Space Interpolation', fontweight='bold')

plt.tight_layout()
plt.savefig('paper/figures/Figure3_Combined.png', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/Figure3_Combined.eps', format='eps', bbox_inches='tight')
print("✅ Figure 3 created")
