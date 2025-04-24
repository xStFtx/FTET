# Example Usage

## Basic Run
```bash
python main.py --field default --save myplot.png
```

## Soliton Animation
```bash
python main.py --field soliton --xlim -20 20 --nx 400 --evolve --frames 300 --dt 0.02 --anim soliton.mp4
```

## Help
```bash
python main.py --help
```

# Sample Output

## Diagnostics
```
--- Diagnostics ---
Entropy Sensor S: min=0.000, max=2.433, mean=0.239, std=0.271
Lagrangian L: min=-1.994, max=1.071, mean=0.019, std=0.273
Animation saved to output/soliton.mp4
```

## Output Directory
All images and animations are saved in the `output/` directory.
