import matplotlib.pyplot as plt

def plot_frequency_vs_length(L_values, f1_values):
    plt.figure(figsize=(8, 5))
    plt.plot(L_values, f1_values)

    plt.xlabel("Beam Length, L (m)")
    plt.ylabel("First Natural Frequency, f1 (Hz)")
    plt.title("First Natural Frequency vs Beam Length")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/frequency_vs_length.png", dpi=300)
    plt.close()


def plot_frequency_vs_thickness(h_values, f1_values):
    plt.figure(figsize=(8, 5))
    plt.plot(h_values * 1000, f1_values)

    plt.xlabel("Beam Thickness, h (mm)")
    plt.ylabel("First Natural Frequency, f1 (Hz)")
    plt.title("First Natural Frequency vs Beam Thickness")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/frequency_vs_thickness.png", dpi=300)
    plt.close()


def plot_frequency_vs_youngs_modulus(E_values, f1_values):
    plt.figure(figsize=(8, 5))
    plt.plot(E_values / 1e9, f1_values)

    plt.xlabel("Young's Modulus, E (GPa)")
    plt.ylabel("First Natural Frequency, f1 (Hz)")
    plt.title("First Natural Frequency vs Young's Modulus")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/frequency_vs_youngs_modulus.png", dpi=300)
    plt.close()


def plot_mode_shapes(x_values, mode_shapes, mode_numbers):
    plt.figure(figsize=(8, 5))

    for phi, mode in zip(mode_shapes, mode_numbers):
        plt.plot(x_values, phi, label=f"Mode {mode}")

    plt.xlabel("Normalised Beam Position, x/L")
    plt.ylabel("Normalised Mode Shape, φ(x)")
    plt.title("Simply-Supported Beam Mode Shapes")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/mode_shapes.png", dpi=300)
    plt.close()


def plot_time_response(t, x, v, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(t, x)

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Displacement Time Response")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}_displacement.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(t, v)

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity Time Response")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}_velocity.png", dpi=300)
    plt.close()


def plot_phase_portrait(x, v, filename):
    plt.figure(figsize=(6, 6))
    plt.plot(x, v)

    plt.xlabel("Displacement (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Phase Portrait")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}_phase_portrait.png", dpi=300)
    plt.close()


def plot_modal_contributions(t, modal_responses):
    plt.figure(figsize=(10, 6))

    for i, response in enumerate(modal_responses):
        plt.plot(t, response, label=f"Mode {i + 1}")

    plt.xlabel("Time (s)")
    plt.ylabel("Modal Amplitude")
    plt.title("Individual Modal Responses")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/modal_contributions.png", dpi=300)
    plt.close()


def plot_total_response(t, total_response):
    plt.figure(figsize=(10, 6))

    plt.plot(t, total_response)

    plt.xlabel("Time (s)")
    plt.ylabel("Total Response")
    plt.title("Combined Multi-Mode Response")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/total_multimode_response.png", dpi=300)
    plt.close()


def plot_point_displacement(t, displacement, x_position):
    plt.figure(figsize=(10, 6))

    plt.plot(t, displacement)

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title(f"Beam Displacement at x/L = {x_position}")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        f"figures/point_displacement_x_{x_position}.png",
        dpi=300
    )

    plt.close()