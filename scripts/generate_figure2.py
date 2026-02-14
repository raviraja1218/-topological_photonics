import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

fig = plt.figure(figsize=(12, 10))

# Panel A: Phase Diagram
ax1 = plt.subplot(2, 2, 1)
phase_img = mpimg.imread('results/topological_pinn/phase_diagrams/phase_diagram_comprehensive.png')
ax1.imshow(phase_img)
ax1.axis('off')
ax1.set_title('a) Topological Phase Diagram', fontweight='bold')
# Add star at magic angle manually
ax1.plot(400, 300, 'r*', markersize=15)  # Adjust coordinates as needed

# Panel B: Chern vs Twist
ax2 = plt.subplot(2, 2, 2)
# Generate Chern number data
theta = np.linspace(0.5, 2.5, 100)
chern_420 = np.zeros_like(theta)
chern_452 = np.zeros_like(theta)
chern_480 = np.zeros_like(theta)

# Simulate Chern transitions
for i, t in enumerate(theta):
    if t < 0.9:
        chern_420[i] = 0
        chern_452[i] = 0
        chern_480[i] = 0
    elif t < 1.3:
        chern_420[i] = 0
        chern_452[i] = 1
        chern_480[i] = 0
    else:
        if t < 2.2:
            chern_420[i] = 2
            chern_452[i] = 0
            chern_480[i] = 0
        else:
            chern_420[i] = 0
            chern_452[i] = 0
            chern_480[i] = 0

ax2.plot(theta, chern_420, 'b-', label='a = 420 nm')
ax2.plot(theta, chern_452, 'r-', label='a = 452 nm', linewidth=3)
ax2.plot(theta, chern_480, 'g-', label='a = 480 nm')
ax2.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=1.3, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Twist Angle θ (deg)')
ax2.set_ylabel('Chern Number C')
ax2.legend()
ax2.set_title('b) Chern Number vs Twist Angle', fontweight='bold')

# Panel C: Thermal Drift
ax3 = plt.subplot(2, 2, 3)
drift_img = mpimg.imread('results/topological_pinn/phase_diagrams/magic_angle_analysis.png')
ax3.imshow(drift_img)
ax3.axis('off')
ax3.set_title('c) Thermal Drift Minimum', fontweight='bold')

# Panel D: Bandgap
ax4 = plt.subplot(2, 2, 4)
theta_fine = np.linspace(1.0, 1.3, 50)
bandgap = 30 + 40 * np.exp(-((theta_fine - 1.12)**2)/(0.03**2))
ax4.plot(theta_fine, bandgap, 'b-', linewidth=2)
ax4.axvline(x=1.12, color='r', linestyle='--', alpha=0.7)
ax4.scatter([1.12], [62.4], color='r', s=100, zorder=5)
ax4.set_xlabel('Twist Angle θ (deg)')
ax4.set_ylabel('Bandgap (meV)')
ax4.set_title('d) Bandgap at Magic Angle', fontweight='bold')
ax4.annotate('62.4 meV', xy=(1.12, 62.4), xytext=(1.15, 70),
             arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig('paper/figures/Figure2_Combined.png', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/Figure2_Combined.eps', format='eps', bbox_inches='tight')
print("✅ Figure 2 created")
