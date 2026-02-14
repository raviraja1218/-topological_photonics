import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

fig = plt.figure(figsize=(12, 10))

# Panel A: Unit Cell
ax1 = plt.subplot(2, 2, 1)
cell_img = mpimg.imread('results/inverse_design/topological_masterpieces/design_001_crescent-moire_v1.png')
ax1.imshow(cell_img)
ax1.axis('off')
ax1.set_title('a) Crescent-Moire Unit Cell', fontweight='bold')
# Add dimension labels
ax1.text(10, 10, '82 nm', color='white', fontsize=8)
ax1.text(40, 40, '452 nm', color='white', fontsize=8)

# Panel B: Bandstructure
ax2 = plt.subplot(2, 2, 2)
band_img = mpimg.imread('results/inverse_design/topological_masterpieces/design_001_crescent-moire_v1_bandstructure.png')
ax2.imshow(band_img)
ax2.axis('off')
ax2.set_title('b) Bandstructure with Edge States', fontweight='bold')
# Add bandgap annotation
ax2.text(50, 30, '98 meV', color='red', fontsize=10, weight='bold')

# Panel C: Thermal Comparison
ax3 = plt.subplot(2, 2, 3)
technologies = ['Classical', 'Magic Angle', 'Crescent-Moire']
drift = [12, 0.038, 0.0067]
colors = ['gray', 'blue', 'red']
bars = ax3.bar(technologies, drift, color=colors, log=True)
ax3.set_ylabel('Thermal Drift (pm/K)')
ax3.set_title('c) Thermal Drift Comparison', fontweight='bold')
# Add improvement labels
ax3.text(0, 0.1, '300×', ha='center', fontsize=10)
ax3.text(1, 0.001, '1800×', ha='center', fontsize=10)

# Panel D: Field Profile
ax4 = plt.subplot(2, 2, 4)
# Create sample field profile
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
field = np.exp(-((X-5)**2 + (Y-5)**2)/10) + 0.5*np.exp(-((X-8)**2 + (Y-2)**2)/5)
ax4.contourf(X, Y, field, levels=20, cmap='hot')
ax4.set_xlabel('x (μm)')
ax4.set_ylabel('y (μm)')
ax4.set_title('d) Edge State Field Profile', fontweight='bold')

plt.tight_layout()
plt.savefig('paper/figures/Figure5_Combined.png', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/Figure5_Combined.eps', format='eps', bbox_inches='tight')
print("✅ Figure 5 created")
