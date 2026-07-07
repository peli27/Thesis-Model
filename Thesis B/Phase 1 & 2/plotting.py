import os
import matplotlib.pyplot as plt


FIGURE_FOLDER = "figures"


def save_figure(filename):
    """Saves the current matplotlib figure into the figure folder."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_FOLDER, filename), dpi=300)
    plt.close()


def clean_float_for_filename(value):
    """Converts a float into a filename-safe string."""
    return str(value).replace(".", "_")


# ------ Frequency validation plots ------


def plot_frequency_vs_length(L_values, f1_values):
    plt.figure(figsize=(8, 5))
    plt.plot(L_values, f1_values)

    plt.xlabel("Beam Length, L (m)")
    plt.ylabel("First Natural Frequency, f1 (Hz)")
    plt.title("First Natural Frequency vs Beam Length")
    plt.grid(True)

    save_figure("frequency_vs_length.png")


def plot_frequency_vs_thickness(h_values, f1_values):
    plt.figure(figsize=(8, 5))
    plt.plot(h_values * 1000, f1_values)

    plt.xlabel("Beam Thickness, h (mm)")
    plt.ylabel("First Natural Frequency, f1 (Hz)")
    plt.title("First Natural Frequency vs Beam Thickness")
    plt.grid(True)

    save_figure("frequency_vs_thickness.png")


def plot_frequency_vs_youngs_modulus(E_values, f1_values):
    plt.figure(figsize=(8, 5))
    plt.plot(E_values / 1e9, f1_values)

    plt.xlabel("Young's Modulus, E (GPa)")
    plt.ylabel("First Natural Frequency, f1 (Hz)")
    plt.title("First Natural Frequency vs Young's Modulus")
    plt.grid(True)

    save_figure("frequency_vs_youngs_modulus.png")


## ------ Mode shape validation plots ------


def plot_mode_shapes(x_values, mode_shapes, mode_numbers):
    plt.figure(figsize=(8, 5))

    for phi, mode in zip(mode_shapes, mode_numbers):
        plt.plot(x_values, phi, label=f"Mode {mode}")

    plt.xlabel("Normalised Beam Position, x/L")
    plt.ylabel("Normalised Mode Shape, phi(x)")
    plt.title("Simply Supported Beam Mode Shapes")
    plt.grid(True)
    plt.legend()

    save_figure("mode_shapes.png")


# ------ SDOF validation plots ------


def plot_time_response(t, x, v, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(t, x)

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("SDOF Displacement Time Response")
    plt.grid(True)

    save_figure(f"{filename}_displacement.png")

    plt.figure(figsize=(8, 5))
    plt.plot(t, v)

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("SDOF Velocity Time Response")
    plt.grid(True)

    save_figure(f"{filename}_velocity.png")


def plot_phase_portrait(x, v, filename):
    plt.figure(figsize=(6, 6))
    plt.plot(x, v)

    plt.xlabel("Displacement (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("SDOF Phase Portrait")
    plt.grid(True)

    save_figure(f"{filename}_phase_portrait.png")


# ------ Multi-mode beam plots ------


def plot_modal_contributions(
    t,
    modal_responses,
    filename="modal_contributions",
    ylabel="Modal displacement, q_n(t) (m)",
    title="Individual Modal Contributions",
):
    """
    Plots individual modal coordinates against time.

    This is not the physical beam displacement at a point. The physical
    displacement is reconstructed using the mode shape values at x/L.
    """
    plt.figure(figsize=(10, 6))

    for i, response in enumerate(modal_responses):
        plt.plot(t, response, label=f"Mode {i + 1}")

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    save_figure(f"{filename}.png")


def plot_point_displacement(t, displacement, x_position, filename=None):
    """Plots physical beam displacement at a selected x/L location."""
    if filename is None:
        x_label = clean_float_for_filename(x_position)
        filename = f"point_displacement_x_{x_label}"

    plt.figure(figsize=(10, 6))
    plt.plot(t, displacement)

    plt.xlabel("Time (s)")
    plt.ylabel("Beam displacement, w(x,t) (m)")
    plt.title(f"Beam Displacement at x/L = {x_position}")
    plt.grid(True)

    save_figure(f"{filename}.png")


def plot_point_velocity(t, velocity, x_position, filename=None):
    """Plots physical beam velocity at a selected x/L location."""
    if filename is None:
        x_label = clean_float_for_filename(x_position)
        filename = f"point_velocity_x_{x_label}"

    plt.figure(figsize=(10, 6))
    plt.plot(t, velocity)

    plt.xlabel("Time (s)")
    plt.ylabel("Beam velocity, w_dot(x,t) (m/s)")
    plt.title(f"Beam Velocity at x/L = {x_position}")
    plt.grid(True)

    save_figure(f"{filename}.png")