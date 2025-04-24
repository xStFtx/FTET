import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sympy as sp
import argparse

# === Field Definitions ===
def get_phi_functions(field_type='default'):
    x_sym, t_sym = sp.symbols('x t')
    if field_type == 'default':
        phi_expr = sp.sin(x_sym) * sp.cos(t_sym) + 0.3 * sp.sin(2 * x_sym * t_sym) + 0.1 * sp.cos(3 * x_sym + 0.5 * t_sym**2)
    elif field_type == 'soliton':
        phi_expr = 2 / sp.cosh(x_sym - 0.5 * t_sym)
    elif field_type == 'gaussian':
        phi_expr = sp.exp(-((x_sym - 2)**2 + (t_sym - 5)**2)/5)
    elif field_type == 'random':
        # Not symbolic, will handle in main
        phi_expr = None
    else:
        raise ValueError(f"Unknown field_type: {field_type}")
    if phi_expr is not None:
        phi_func = sp.lambdify((x_sym, t_sym), phi_expr, 'numpy')
        dphi_dx_func = sp.lambdify((x_sym, t_sym), sp.diff(phi_expr, x_sym), 'numpy')
        dphi_dt_func = sp.lambdify((x_sym, t_sym), sp.diff(phi_expr, t_sym), 'numpy')
    else:
        phi_func = dphi_dx_func = dphi_dt_func = None
    return phi_func, dphi_dx_func, dphi_dt_func

# === Field and Derivatives ===
def compute_field_and_derivatives(x, t, phi_func, dphi_dx_func, dphi_dt_func, field_type='default'):
    X, T = np.meshgrid(x, t)
    if field_type == 'random':
        np.random.seed(42)
        phi = np.random.normal(0, 0.2, size=X.shape)
        dphi_dx = np.gradient(phi, x[1] - x[0], axis=1)
        dphi_dt = np.gradient(phi, t[1] - t[0], axis=0)
    else:
        phi = phi_func(X, T)
        dphi_dx = dphi_dx_func(X, T)
        dphi_dt = dphi_dt_func(X, T)
    d2phi_dx2 = np.gradient(dphi_dx, x[1] - x[0], axis=1)
    d2phi_dt2 = np.gradient(dphi_dt, t[1] - t[0], axis=0)
    d2phi_dx2 = gaussian_filter(d2phi_dx2, sigma=1)
    d2phi_dt2 = gaussian_filter(d2phi_dt2, sigma=1)
    return X, T, phi, dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2

# === Entropy Sensor ===
def compute_entropy_sensor(dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2):
    gradient_magnitude = np.sqrt(dphi_dx**2 + dphi_dt**2)
    curvature_tensor = np.sqrt(d2phi_dx2**2 + d2phi_dt2**2)
    anisotropy = np.abs(d2phi_dx2 - d2phi_dt2) / (1e-5 + np.abs(d2phi_dx2 + d2phi_dt2))
    S = np.log1p(gradient_magnitude**2) + 0.5 * np.log1p(curvature_tensor + anisotropy)
    return S

# === Lagrangian ===
def compute_lagrangian(phi, dphi_dx, dphi_dt, S, X, T, alpha_amp):
    V_phi = 0.25 * phi**4 - 0.5 * phi**2
    alpha = alpha_amp + 0.05 * np.sin(0.2 * X + 0.1 * T**1.5) * np.cos(0.3 * T)
    L = 0.5 * (dphi_dx**2 + dphi_dt**2) - V_phi + alpha * S * phi**2
    return L, alpha

# === Visualization ===
def plot_fields(x, t, X, T, S, L, phi, alpha, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    entropy_plot = ax[0].imshow(S, extent=[x.min(), x.max(), t.min(), t.max()],
                                aspect='auto', origin='lower', cmap='magma')
    ax[0].set_title('Entropy Sensor S(x,t)', fontsize=14)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('t')
    fig.colorbar(entropy_plot, ax=ax[0], label='Entropy Magnitude')
    ax[0].contour(X, T, phi, levels=10, colors='white', linewidths=0.5)
    lagrangian_plot = ax[1].imshow(L, extent=[x.min(), x.max(), t.min(), t.max()],
                                   aspect='auto', origin='lower', cmap='viridis')
    ax[1].set_title('Entropy-Augmented Lagrangian L(x,t)', fontsize=14)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('t')
    fig.colorbar(lagrangian_plot, ax=ax[1], label='Lagrangian Density')
    ax[1].contour(X, T, alpha, levels=5, colors='black', linestyles='dotted', linewidths=0.6)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()

# === Diagnostics ===
def print_diagnostics(S, L):
    print("--- Diagnostics ---")
    print(f"Entropy Sensor S: min={S.min():.3f}, max={S.max():.3f}, mean={S.mean():.3f}, std={S.std():.3f}")
    print(f"Lagrangian L: min={L.min():.3f}, max={L.max():.3f}, mean={L.mean():.3f}, std={L.std():.3f}")

# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Field-Theoretic Entropy Tensor Simulation")
    parser.add_argument('--xlim', type=float, nargs=2, default=[-10, 10], help='x range')
    parser.add_argument('--tlim', type=float, nargs=2, default=[0, 20], help='t range')
    parser.add_argument('--nx', type=int, default=800, help='number of x points')
    parser.add_argument('--nt', type=int, default=800, help='number of t points')
    parser.add_argument('--alpha', type=float, default=0.05, help='base alpha amplitude')
    parser.add_argument('--field', type=str, default='default', choices=['default', 'soliton', 'gaussian', 'random'], help='initial field type')
    parser.add_argument('--save', type=str, default=None, help='save figure to file')
    args = parser.parse_args()

    x = np.linspace(args.xlim[0], args.xlim[1], args.nx)
    t = np.linspace(args.tlim[0], args.tlim[1], args.nt)
    phi_func, dphi_dx_func, dphi_dt_func = get_phi_functions(args.field)
    X, T, phi, dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2 = compute_field_and_derivatives(
        x, t, phi_func, dphi_dx_func, dphi_dt_func, args.field)
    S = compute_entropy_sensor(dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2)
    L, alpha = compute_lagrangian(phi, dphi_dx, dphi_dt, S, X, T, args.alpha)
    print_diagnostics(S, L)
    plot_fields(x, t, X, T, S, L, phi, alpha, save_path=args.save)

if __name__ == "__main__":
    main()
