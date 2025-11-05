"""
nn_training_animation.py

Educational Matplotlib animation illustrating neural network training:
- decreasing training / validation loss
- increasing accuracy
- simple schematic network diagram with "activating" connections.

The animation uses synthetic curves (no heavy ML frameworks),
so it runs with only NumPy + Matplotlib.

Author: Svetlana Romanova (SvetLuna)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, animation


def generate_training_curves(n_epochs: int = 100, seed: int = 42):
    """
    Generate synthetic training & validation loss and accuracy curves.

    Loss ~ exponential decay + noise
    Accuracy ~ rising logistic curve + noise
    """
    rng = np.random.default_rng(seed)
    epochs = np.arange(n_epochs)

    # Loss curves (start higher, decay downwards)
    train_loss = 1.5 * np.exp(-epochs / 22.0) + 0.05 * rng.normal(size=n_epochs)
    val_loss = 1.8 * np.exp(-epochs / 25.0) + 0.07 * rng.normal(size=n_epochs)
    train_loss = np.clip(train_loss, 0.02, None)
    val_loss = np.clip(val_loss, 0.02, None)

    # Accuracy curves (start ~0.5, go up to ~0.98)
    base_acc = 0.5 + 0.45 * (1.0 - np.exp(-epochs / 25.0))
    train_acc = base_acc + 0.03 * rng.normal(size=n_epochs)
    val_acc = base_acc - 0.02 + 0.04 * rng.normal(size=n_epochs)
    train_acc = np.clip(train_acc, 0.45, 0.99)
    val_acc = np.clip(val_acc, 0.4, 0.99)

    return epochs, train_loss, val_loss, train_acc, val_acc


def create_figure():
    """
    Create Matplotlib figure with a grid for:
      - top: schematic NN
      - bottom left: loss curves
      - bottom right: accuracy curves
    """
    plt.style.use("default")
    fig = plt.figure(figsize=(10, 6), dpi=120)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3])

    ax_net = fig.add_subplot(gs[0, :])
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_acc = fig.add_subplot(gs[1, 1])

    fig.suptitle("Neural Network Training — Loss, Accuracy and Schematic", fontsize=14, weight="bold")

    return fig, ax_net, ax_loss, ax_acc


def init_network_axes(ax):
    """
    Draw a simple 3-layer fully connected network schematic.

    Returns: list of line artists to be updated during animation.
    """
    ax.set_axis_off()

    # Layer x-positions
    x_input, x_hidden, x_output = 0.1, 0.5, 0.9

    # y-coordinates of nodes in each layer
    y_input = np.linspace(0.2, 0.8, 4)
    y_hidden = np.linspace(0.2, 0.8, 5)
    y_output = np.array([0.5])

    # Draw nodes
    node_style = dict(marker="o", markersize=12, linestyle="None", color="#4C72B0")
    ax.plot(np.full_like(y_input, x_input), y_input, **node_style)
    ax.plot(np.full_like(y_hidden, x_hidden), y_hidden, **node_style)
    ax.plot(np.full_like(y_output, x_output), y_output, **node_style)

    # Draw connections (store Line2D objects)
    connection_lines = []

    # Input → Hidden
    for yi in y_input:
        for yh in y_hidden:
            line, = ax.plot([x_input, x_hidden], [yi, yh],
                            color="gray", alpha=0.15, linewidth=1.2)
            connection_lines.append(line)

    # Hidden → Output
    for yh in y_hidden:
        line, = ax.plot([x_hidden, x_output], [yh, y_output[0]],
                        color="gray", alpha=0.15, linewidth=1.4)
        connection_lines.append(line)

    # Layer labels
    ax.text(x_input, 0.9, "Input layer", ha="center", va="center", fontsize=9)
    ax.text(x_hidden, 0.9, "Hidden layer", ha="center", va="center", fontsize=9)
    ax.text(x_output, 0.9, "Output layer", ha="center", va="center", fontsize=9)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    return connection_lines


def init_loss_axes(ax_loss, epochs, train_loss, val_loss):
    ax_loss.set_title("Loss vs Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlim(0, epochs[-1])
    ax_loss.set_ylim(0, max(train_loss[0], val_loss[0]) * 1.1)

    (line_train_loss,) = ax_loss.plot([], [], label="Train loss", color="#D9534F")
    (line_val_loss,) = ax_loss.plot([], [], label="Val loss", color="#F0AD4E")
    point_loss, = ax_loss.plot([], [], "o", color="black", markersize=4)

    ax_loss.legend(loc="upper right", fontsize=8)

    return line_train_loss, line_val_loss, point_loss


def init_acc_axes(ax_acc, epochs, train_acc, val_acc):
    ax_acc.set_title("Accuracy vs Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlim(0, epochs[-1])
    ax_acc.set_ylim(0.4, 1.0)

    (line_train_acc,) = ax_acc.plot([], [], label="Train acc", color="#5CB85C")
    (line_val_acc,) = ax_acc.plot([], [], label="Val acc", color="#5BC0DE")
    point_acc, = ax_acc.plot([], [], "o", color="black", markersize=4)

    ax_acc.legend(loc="lower right", fontsize=8)

    return line_train_acc, line_val_acc, point_acc


def build_animation(n_epochs: int = 120, interval_ms: int = 80):
    """
    Create Matplotlib FuncAnimation object.

    Returns (fig, anim) so the caller can either show() or save() the animation.
    """
    epochs, train_loss, val_loss, train_acc, val_acc = generate_training_curves(n_epochs=n_epochs)

    fig, ax_net, ax_loss, ax_acc = create_figure()

    conn_lines = init_network_axes(ax_net)
    line_train_loss, line_val_loss, point_loss = init_loss_axes(ax_loss, epochs, train_loss, val_loss)
    line_train_acc, line_val_acc, point_acc = init_acc_axes(ax_acc, epochs, train_acc, val_acc)

    def init():
        # Empty initial state
        line_train_loss.set_data([], [])
        line_val_loss.set_data([], [])
        point_loss.set_data([], [])

        line_train_acc.set_data([], [])
        line_val_acc.set_data([], [])
        point_acc.set_data([], [])

        # Faint connections
        for ln in conn_lines:
            ln.set_alpha(0.1)
        return (
            line_train_loss,
            line_val_loss,
            point_loss,
            line_train_acc,
            line_val_acc,
            point_acc,
            *conn_lines,
        )

    def update(frame: int):
        # Curves up to current epoch
        e = epochs[: frame + 1]

        line_train_loss.set_data(e, train_loss[: frame + 1])
        line_val_loss.set_data(e, val_loss[: frame + 1])
        point_loss.set_data(epochs[frame], train_loss[frame])

        line_train_acc.set_data(e, train_acc[: frame + 1])
        line_val_acc.set_data(e, val_acc[: frame + 1])
        point_acc.set_data(epochs[frame], train_acc[frame])

        # Light "activation" of random subset of connections
        # strength grows with training progress
        progress = frame / max(1, (len(epochs) - 1))
        k_active = max(1, int(progress * len(conn_lines)))
        active_idx = np.linspace(0, len(conn_lines) - 1, k_active).astype(int)

        for i, ln in enumerate(conn_lines):
            if i in active_idx:
                ln.set_alpha(0.4 + 0.4 * progress)
                ln.set_linewidth(1.4 + 0.8 * progress)
                ln.set_color("#4C72B0")
            else:
                ln.set_alpha(0.08)
                ln.set_linewidth(1.0)
                ln.set_color("gray")

        return (
            line_train_loss,
            line_val_loss,
            point_loss,
            line_train_acc,
            line_val_acc,
            point_acc,
            *conn_lines,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(epochs),
        interval=interval_ms,
        blit=True,
    )

    return fig, anim


def main(save_mp4: bool = False, filename: str = "nn_training_animation.mp4"):
    """
    Entry point for command-line use.

    If save_mp4 is True, tries to save the animation as MP4 (requires ffmpeg).
    Otherwise simply shows the animation window.
    """
    fig, anim = build_animation()

    if save_mp4:
        print(f"[INFO] Saving animation to {filename!r} (requires ffmpeg installed)...")
        try:
            anim.save(filename, writer="ffmpeg", dpi=150, bitrate=2400)
            print("[INFO] Done.")
        except Exception as exc:  # noqa: BLE001
            print("[WARN] Could not save MP4:", exc)
            print("       You can still view the animation interactively.")
            plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    # Example:
    #   python -m src.nn_training_animation        → just show animation
    #   python -m src.nn_training_animation save   → also save MP4
    import sys

    save_flag = len(sys.argv) > 1 and sys.argv[1].lower().startswith("save")
    main(save_mp4=save_flag)
