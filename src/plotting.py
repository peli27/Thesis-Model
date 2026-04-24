from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_frequency_vs_length(
    lengths_m: np.ndarray,
    freqs_hz: np.ndarray,
    save_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(lengths_m * 1000.0, freqs_hz, marker="o")
    plt.xlabel("Beam length, L (mm)")
    plt.ylabel("First natural frequency, f₁ (Hz)")
    plt.title("Preliminary simulation: effect of beam length on first natural frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_displacement_time_history(
    time_s: np.ndarray,
    displacement_m: np.ndarray,
    save_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(time_s, displacement_m * 1000.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Tip displacement (mm)")
    plt.title("Preliminary simulation: harmonic tip response near resonance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cantilever_schematic(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))

    # Beam
    ax.plot([0, 6], [0, 0], linewidth=8)

    # Fixed wall
    ax.fill_betweenx([-0.6, 0.6], -0.15, 0.0)
    for y in np.linspace(-0.6, 0.6, 9):
        ax.plot([-0.28, 0], [y - 0.1, y + 0.1], linewidth=1)

    # Load arrow
    ax.arrow(
        6, 0.8, 0, -0.65,
        head_width=0.18,
        head_length=0.12,
        length_includes_head=True,
    )
    ax.text(6.1, 0.82, "Harmonic load", fontsize=10, va="bottom")

    # Dimension line
    ax.annotate(
        "",
        xy=(0, -0.5),
        xytext=(6, -0.5),
        arrowprops=dict(arrowstyle="<->"),
    )
    ax.text(3, -0.72, "Beam length, L", ha="center", va="top", fontsize=10)
    ax.text(0.1, 0.4, "Fixed support", fontsize=10)

    ax.set_xlim(-0.5, 6.6)
    ax.set_ylim(-1.0, 1.2)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()