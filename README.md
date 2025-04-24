# Entropy-Augmented Lagrangian Field Simulation

This simulation explores a quantum scalar field φ(x,t) augmented by an entropy sensor S(x,t).

## Overview

- Scalar field: φ(x,t) = sin(x) * cos(t)
- Entropy sensor: S(x,t) = log(1 + (∂φ/∂x)² + (∂φ/∂t)²)
- Lagrangian:
  
  L(x,t) = ½(∂φ/∂t)² + ½(∂φ/∂x)² - V(φ) + α * S(x,t) * φ²
  
  Where V(φ) = ½φ² (harmonic potential), and α is a coupling constant.

## Output

- Left plot: Entropy Sensor S(x,t)
- Right plot: Entropy-Augmented Lagrangian L(x,t)

## Run Instructions

Make sure Python 3 with numpy and matplotlib is installed. Then run:

```bash
python main.py
```

## Extensions

- Couple to other fields
- Add chaotic dynamics
- Extend to curved spacetime
