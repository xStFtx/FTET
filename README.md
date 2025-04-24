# Entropy-Augmented Lagrangian Field Simulation

This simulation explores a quantum scalar field φ(x,t) augmented by an advanced entropy sensor S(x,t).

## Features
- Symbolic definition for φ(x, t) with nonlinear and higher-harmonic terms
- Analytical derivatives (via sympy) for accuracy
- Advanced entropy sensor: includes gradient, curvature, and anisotropy
- Nonlinear potential and spatial-temporal entropy coupling
- High-resolution grid and high-quality visualization
- Modular code for easy experimentation
- **Command-line interface**: Choose field type, grid size, coupling, and output file; supports evolution and animation options
- **Diagnostics**: Prints entropy and Lagrangian statistics
- **Cross-platform**: Works on Linux, macOS, and Windows (Python 3.8+)
- **Saving figures/animations**: Optionally saves the plot as an image (e.g. output.png) or animation (e.g. output.mp4)

## Overview
- Scalar field: φ(x,t) = sin(x) * cos(t) + 0.3 sin(2 x t) + 0.1 cos(3 x + 0.5 t²)
- Entropy sensor: S(x,t) = log1p(|∇φ|²) + 0.5 log1p(curvature + anisotropy)
- Lagrangian:
  
  L(x,t) = ½(∂φ/∂t)² + ½(∂φ/∂x)² - V(φ) + α * S(x,t) * φ²
  
  Where V(φ) = ¼φ⁴ - ½φ² (nonlinear), and α(x,t) is a spatial-temporal coupling.

## Output
- Left plot: Entropy Sensor S(x,t)
- Right plot: Entropy-Augmented Lagrangian L(x,t)
- Prints summary statistics for entropy and Lagrangian
- Optionally saves the plot as an image (e.g. output.png)

## Run Instructions

1. **Install requirements in a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Run the simulation**
   ```bash
   python main.py [options]
   ```
   Example with animation:
   ```bash
   python main.py --field soliton --xlim -20 20 --nx 400 --evolve --frames 300 --dt 0.02 --anim soliton.mp4
   ```
   For all options, use:
   ```bash
   python main.py --help
   ```

## CLI Arguments
- `--xlim`: x range (default: -10 10)
- `--tlim`: t range (default: 0 20)
- `--nx`: number of x points (default: 800)
- `--nt`: number of t points (default: 800)
- `--alpha`: base alpha amplitude (default: 0.05)
- `--field`: initial field type (`default`, `soliton`, `gaussian`, `random`)
- `--save`: save figure to file (e.g. output.png)
- `--evolve`: evolve field and produce animation
- `--frames`: number of animation frames (used with --evolve)
- `--dt`: time step for evolution (used with --evolve)
- `--anim`: filename for animation output (e.g. output.mp4)

## Field Types
- `default`: Mixed harmonics (original demo)
- `soliton`: Localized soliton profile
- `gaussian`: Gaussian pulse
- `random`: Random noise field

## Extensions
- Couple to other fields
- Add chaotic dynamics
- Extend to curved spacetime
- Add time evolution or animation

## Animation Output
Animation output: If `--anim` is specified, an animation is saved as an `.mp4` file (not tracked by git by default).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Repository
This project is modular and ready for extension. See `main.py` for the main entry point and function definitions. See `simulation.py` for additional simulation logic and animation handling.

---

Feel free to contribute or suggest improvements!
