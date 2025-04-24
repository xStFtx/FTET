# core.py: Core simulation logic for field evolution
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Callable, Tuple, Optional

def get_phi_functions(field_type: str = 'default', custom_expr: Optional[str] = None):
    import sympy as sp
    x_sym, t_sym = sp.symbols('x t')
    if custom_expr:
        phi_expr = sp.sympify(custom_expr)
    elif field_type == 'default':
        phi_expr = sp.sin(x_sym) * sp.cos(t_sym) + 0.3 * sp.sin(2 * x_sym * t_sym) + 0.1 * sp.cos(3 * x_sym + 0.5 * t_sym**2)
    elif field_type == 'soliton':
        phi_expr = 2 / sp.cosh(x_sym - 0.5 * t_sym)
    elif field_type == 'gaussian':
        phi_expr = sp.exp(-((x_sym - 2)**2 + (t_sym - 5)**2)/5)
    elif field_type == 'random':
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

def compute_entropy_sensor(dphi_dx, dphi_dt, d2phi_dx2, d2phi_dt2):
    gradient_magnitude = np.sqrt(dphi_dx**2 + dphi_dt**2)
    curvature_tensor = np.sqrt(d2phi_dx2**2 + d2phi_dt2**2)
    anisotropy = np.abs(d2phi_dx2 - d2phi_dt2) / (1e-5 + np.abs(d2phi_dx2 + d2phi_dt2))
    S = np.log1p(gradient_magnitude**2) + 0.5 * np.log1p(curvature_tensor + anisotropy)
    return S

def compute_lagrangian(phi, dphi_dx, dphi_dt, S, X, T, alpha_amp):
    V_phi = 0.25 * phi**4 - 0.5 * phi**2
    alpha = alpha_amp + 0.05 * np.sin(0.2 * X + 0.1 * T**1.5) * np.cos(0.3 * T)
    L = 0.5 * (dphi_dx**2 + dphi_dt**2) - V_phi + alpha * S * phi**2
    return L, alpha
