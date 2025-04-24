import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sympy as sp

# Symbolic definition for φ(x, t)
x_sym, t_sym = sp.symbols('x t')
phi_expr = sp.sin(x_sym) * sp.cos(t_sym) + 0.3 * sp.sin(2 * x_sym * t_sym) + 0.1 * sp.cos(3 * x_sym + 0.5 * t_sym**2)

# Convert symbolic expression to numerical function
phi_func = sp.lambdify((x_sym, t_sym), phi_expr, 'numpy')
dphi_dx_func = sp.lambdify((x_sym, t_sym), sp.diff(phi_expr, x_sym), 'numpy')
dphi_dt_func = sp.lambdify((x_sym, t_sym), sp.diff(phi_expr, t_sym), 'numpy')

# High resolution grid
x = np.linspace(-10, 10, 800)
t = np.linspace(0, 20, 800)
X, T = np.meshgrid(x, t)

# Evaluate φ and gradients
phi = phi_func(X, T)
dphi_dx = dphi_dx_func(X, T)
dphi_dt = dphi_dt_func(X, T)

# Second-order derivatives with smoothing
d2phi_dx2 = np.gradient(dphi_dx, x[1] - x[0], axis=1)
d2phi_dt2 = np.gradient(dphi_dt, t[1] - t[0], axis=0)
d2phi_dx2 = gaussian_filter(d2phi_dx2, sigma=1)
d2phi_dt2 = gaussian_filter(d2phi_dt2, sigma=1)

# Advanced entropy sensor
gradient_magnitude = np.sqrt(dphi_dx**2 + dphi_dt**2)
curvature_tensor = np.sqrt(d2phi_dx2**2 + d2phi_dt2**2)
anisotropy = np.abs(d2phi_dx2 - d2phi_dt2) / (1e-5 + np.abs(d2phi_dx2 + d2phi_dt2))
S = np.log1p(gradient_magnitude**2) + 0.5 * np.log1p(curvature_tensor + anisotropy)

# Nonlinear potential
V_phi = 0.25 * phi**4 - 0.5 * phi**2

# Spatial-temporal coupling function
alpha = 0.05 + 0.05 * np.sin(0.2 * X + 0.1 * T**1.5) * np.cos(0.3 * T)

# Entropy-augmented Lagrangian
L = 0.5 * (dphi_dx**2 + dphi_dt**2) - V_phi + alpha * S * phi**2

# === Visualization ===
fig, ax = plt.subplots(1, 2, figsize=(18, 7))

# Plot entropy sensor
entropy_plot = ax[0].imshow(S, extent=[x.min(), x.max(), t.min(), t.max()],
                            aspect='auto', origin='lower', cmap='magma')
ax[0].set_title('Entropy Sensor S(x,t)', fontsize=14)
ax[0].set_xlabel('x')
ax[0].set_ylabel('t')
fig.colorbar(entropy_plot, ax=ax[0], label='Entropy Magnitude')
ax[0].contour(X, T, phi, levels=10, colors='white', linewidths=0.5)

# Plot Lagrangian
lagrangian_plot = ax[1].imshow(L, extent=[x.min(), x.max(), t.min(), t.max()],
                               aspect='auto', origin='lower', cmap='viridis')
ax[1].set_title('Entropy-Augmented Lagrangian L(x,t)', fontsize=14)
ax[1].set_xlabel('x')
ax[1].set_ylabel('t')
fig.colorbar(lagrangian_plot, ax=ax[1], label='Lagrangian Density')
ax[1].contour(X, T, alpha, levels=5, colors='black', linestyles='dotted', linewidths=0.6)

plt.tight_layout()
plt.show()
