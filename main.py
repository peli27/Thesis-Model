from pathlib import Path

from src.beam_properties import get_default_beam, first_natural_frequency
from src.frequency_analysis import frequency_vs_length
from src.response_analysis import simulate_sdof_harmonic_response
from src.plotting import (
    plot_frequency_vs_length,
    plot_displacement_time_history,
    plot_cantilever_schematic,
)


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    beam = get_default_beam()

    # Plot 1: Natural frequency vs beam length
    lengths_m, freqs_hz = frequency_vs_length(
        beam=beam,
        l_min=0.15,
        l_max=0.60,
        n_points=10,
    )
    plot_frequency_vs_length(
        lengths_m=lengths_m,
        freqs_hz=freqs_hz,
        save_path=results_dir / "prelim_frequency_vs_length.png",
    )

    # Plot 2: Harmonic response near resonance
    length_m = 0.30
    f1_hz = first_natural_frequency(beam=beam, length_m=length_m)
    t, x = simulate_sdof_harmonic_response(
        natural_frequency_hz=f1_hz,
        damping_ratio=0.02,
        forcing_frequency_ratio=0.95,
        forcing_amplitude_n=1.0,
        mass_kg=0.05,
        duration_s=0.25,
        n_points=2000,
    )
    plot_displacement_time_history(
        time_s=t,
        displacement_m=x,
        save_path=results_dir / "prelim_displacement_time_history.png",
    )

    # Plot 3: Beam schematic
    plot_cantilever_schematic(
        save_path=results_dir / "cantilever_beam_schematic.png",
    )

    print(f"First natural frequency at L = 300 mm: {f1_hz:.2f} Hz")
    print(f"Maximum simulated displacement: {abs(x).max() * 1000:.3f} mm")


if __name__ == "__main__":
    main()