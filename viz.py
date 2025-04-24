# viz.py: Visualization and animation utilities
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

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
        out_dir = os.path.join(os.path.dirname(save_path) or '.', 'output')
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.basename(save_path)
        save_path = os.path.join('output', filename) if not save_path.startswith('output' + os.sep) else save_path
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()

def animate_field(x, phi_t, interval=30, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(phi_t[0][None, :], aspect='auto', extent=[x.min(), x.max(), 0, 1], cmap='plasma', vmin=phi_t.min(), vmax=phi_t.max())
    ax.set_title('Field φ(x) Evolution')
    ax.set_xlabel('x')
    ax.set_ylabel('time (frame)')
    cb = fig.colorbar(im, ax=ax)
    def update(frame):
        im.set_data(phi_t[frame][None, :])
        ax.set_ylabel(f'time step {frame}')
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(phi_t), blit=True, interval=interval)
    if save_path:
        out_dir = os.path.join(os.path.dirname(save_path) or '.', 'output')
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.basename(save_path)
        save_path = os.path.join('output', filename) if not save_path.startswith('output' + os.sep) else save_path
        ani.save(save_path, writer='ffmpeg', dpi=150)
        print(f"Animation saved to {save_path}")
    plt.close(fig)
