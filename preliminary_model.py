import numpy as np
import matplotlib.pyplot as plt

# Beam parameters
E = 200e9          # Young's modulus (Pa)
rho = 7850         # Density (kg/m^3)
b = 0.02           # Beam width (m)
h = 0.003          # Beam thickness (m)

A = b * h
I = b * h**3 / 12

# First mode constant for a cantilever beam
beta1 = 1.875104068711961

# Plot 1: First natural frequency vs beam length
L_vals = np.linspace(0.15, 0.60, 10)  # Beam length range (m)

f1_vals = (beta1**2 / (2 * np.pi * L_vals**2)) * np.sqrt(E * I / (rho * A))

plt.figure(figsize=(7, 4.5))
plt.plot(L_vals * 1000, f1_vals, marker='o')
plt.xlabel("Beam length, L (mm)")
plt.ylabel("First natural frequency, f₁ (Hz)")
plt.title("Preliminary simulation: effect of beam length on first natural frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prelim_frequency_vs_length.png", dpi=300)
plt.close()

# Plot 2: Displacement time history near resonance
L = 0.30  # Beam length (m)

f1 = (beta1**2 / (2 * np.pi * L**2)) * np.sqrt(E * I / (rho * A))
omega_n = 2 * np.pi * f1

zeta = 0.02
omega = 0.95 * omega_n
r = omega / omega_n

# SDOF amplitude ratio approximation
amp_ratio = 1 / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
X_static = 1e-3
X = X_static * amp_ratio

t = np.linspace(0, 0.25, 2000)
x = X * np.sin(omega * t)

plt.figure(figsize=(7, 4.5))
plt.plot(t, x * 1000)
plt.xlabel("Time (s)")
plt.ylabel("Tip displacement (mm)")
plt.title("Preliminary simulation: harmonic tip response near resonance")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("prelim_displacement_time_history.png", dpi=300)
plt.close()

# Plot 3: Simple cantilever beam schematic
fig, ax = plt.subplots(figsize=(8, 2.8))

# Beam
ax.plot([0, 6], [0, 0], linewidth=8, color='black')

# Fixed wall
ax.fill_betweenx([-0.6, 0.6], -0.15, 0.0, color='grey')
for y in np.linspace(-0.6, 0.6, 9):
    ax.plot([-0.28, 0], [y - 0.1, y + 0.1], linewidth=1, color='black')

# Harmonic load arrow
ax.arrow(6, 0.8, 0, -0.65, head_width=0.18, head_length=0.12,
        length_includes_head=True, color='red')
ax.text(6.1, 0.82, "Harmonic load", fontsize=10, va='bottom')

# Dimension line
ax.annotate("", xy=(0, -0.5), xytext=(6, -0.5), arrowprops=dict(arrowstyle="<->"))
ax.text(3, -0.72, "Beam length, L", ha="center", va="top", fontsize=10)

# Fixed support label
ax.text(0.1, 0.4, "Fixed support", fontsize=10)

ax.set_xlim(-0.5, 6.6)
ax.set_ylim(-1.0, 1.2)
ax.axis("off")

plt.tight_layout()
plt.savefig("cantilever_beam_schematic.png", dpi=300, bbox_inches="tight")
plt.close()

# Sample result
print(f"First natural frequency at L = 300 mm: {f1:.2f} Hz")
print(f"Maximum near-resonant displacement: {np.max(np.abs(x)) * 1000:.3f} mm")