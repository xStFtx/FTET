import numpy as np
from scipy.ndimage import gaussian_filter

def leapfrog_time_evolution(phi0, phi1, dx, dt, nt, V_prime_func=None, noise_amp=0.0):
    """
    Leapfrog time evolution for the field φ(x, t).
    Args:
        phi0: initial field at t=0, shape (nx,)
        phi1: field at t=dt, shape (nx,)
        dx: spatial grid spacing
        dt: time step
        nt: number of time steps
        V_prime_func: function for dV/dφ, nonlinear potential
        noise_amp: amplitude of added noise per step
    Returns:
        phi_t: array, shape (nt, nx), field at each time step
    """
    nx = phi0.shape[0]
    phi_t = np.zeros((nt, nx))
    phi_t[0] = phi0
    phi_t[1] = phi1
    for i in range(1, nt-1):
        lap = (np.roll(phi_t[i], -1) - 2*phi_t[i] + np.roll(phi_t[i], 1)) / dx**2
        Vp = V_prime_func(phi_t[i]) if V_prime_func is not None else 0
        noise = noise_amp * np.random.randn(nx)
        phi_t[i+1] = 2*phi_t[i] - phi_t[i-1] + dt**2 * (lap - Vp) + noise
        # Optionally smooth to avoid numerical instabilities
        phi_t[i+1] = gaussian_filter(phi_t[i+1], sigma=0.5)
    return phi_t
