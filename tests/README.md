# Test Suite for Entropy-Augmented Lagrangian Field Simulation

This folder is intended for automated tests of the simulation code.

## How to Run Tests

1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Run all tests with:
   ```bash
   python -m unittest discover tests
   ```

## Adding Tests
- Place test files in this folder, named as `test_*.py`.
- Use the `unittest` module or your preferred testing framework.
- Ensure your tests are self-contained and reproducible.
