"""
Phase 1 validation script for Thesis B.

This script checks the physics-based simply supported beam simulator before it
is used to generate the Phase 2 synthetic dataset.

It produces:
1. Natural frequency trend plots
2. Simply supported mode shape plots
3. SDOF validation response plots
4. Multi-mode beam response plots under harmonic excitation
"""

import numpy as np

from beam_properties import get_beam_properties
from frequency_analysis import natural_frequencies
from mode_shapes import simply_supported_mode_shape
from multi_mode_response import (
    damped_free_response,
    reconstruct_displacement_at_point,
    reconstruct_velocity_at_point,
    simulate_harmonic_beam_response,
    undamped_free_response,
)
from plotting import (
    plot_frequency_vs_length,
    plot_frequency_vs_thickness,
    plot_frequency_vs_youngs_modulus,
    plot_mode_shapes,
    plot_modal_contributions,
    plot_phase_portrait,
    plot_point_displacement,
    plot_point_velocity,
    plot_time_response,
)
from response_analysis import sdof_response


# Global validation settings
NUMBER_OF_MODES = 4
OBSERVATION_POINT = 0.65
SIMULATION_TIME = 2.0
TIME_STEP = 0.002
DAMPING_RATIO = 0.02


# Natural frequency validation studies
def run_frequency_validation():
    """Generates natural frequency trend plots."""

    # Study 1: Frequency vs beam length
    L_values = np.linspace(0.1, 1.0, 50)
    f1_length_values = []

    for L in L_values:
        beam = get_beam_properties()
        beam["L"] = L

        frequencies = natural_frequencies(beam, modes=1)
        f1_length_values.append(frequencies[0])

    plot_frequency_vs_length(L_values, f1_length_values)

    # Study 2: Frequency vs beam thickness
    h_values = np.linspace(0.001, 0.01, 50)
    f1_thickness_values = []

    for h in h_values:
        beam = get_beam_properties()
        beam["h"] = h
        beam["A"] = beam["b"] * beam["h"]
        beam["I"] = beam["b"] * beam["h"] ** 3 / 12

        frequencies = natural_frequencies(beam, modes=1)
        f1_thickness_values.append(frequencies[0])

    plot_frequency_vs_thickness(h_values, f1_thickness_values)

    # Study 3: Frequency vs Young's modulus
    E_values = np.linspace(50e9, 300e9, 50)
    f1_E_values = []

    for E in E_values:
        beam = get_beam_properties()
        beam["E"] = E

        frequencies = natural_frequencies(beam, modes=1)
        f1_E_values.append(frequencies[0])

    plot_frequency_vs_youngs_modulus(E_values, f1_E_values)


# Mode shape validation
def run_mode_shape_validation():
    """Plots the first four simply supported sine mode shapes."""

    x_values = np.linspace(0, 1, 200)
    mode_numbers = [1, 2, 3, 4]

    mode_shapes = []

    for mode in mode_numbers:
        phi = simply_supported_mode_shape(x_values, mode)
        mode_shapes.append(phi)

    plot_mode_shapes(x_values, mode_shapes, mode_numbers)


# SDOF validation cases
def run_sdof_validation():
    """
    Runs simple SDOF validation cases.

    These are not the main thesis simulator. They are used only to confirm
    expected vibration behaviour before moving to the multi-mode beam model.
    """

    beam = get_beam_properties()
    f1 = natural_frequencies(beam, modes=1)[0]
    omega1 = 2 * np.pi * f1

    m = 1.0
    k = m * omega1 ** 2
    c_damped = 2 * DAMPING_RATIO * m * omega1
    t_span = (0, SIMULATION_TIME)

    def zero_force(t):
        return 0.0

    # Case 1: Undamped free vibration
    t, x, v = sdof_response(
        m=m,
        c=0.0,
        k=k,
        force_function=zero_force,
        t_span=t_span,
        initial_conditions=[0.001, 0.0],
    )

    plot_time_response(t, x, v, "sdof_undamped_free_vibration")
    plot_phase_portrait(x, v, "sdof_undamped_free_vibration")

    # Case 2: Damped free vibration
    t, x, v = sdof_response(
        m=m,
        c=c_damped,
        k=k,
        force_function=zero_force,
        t_span=t_span,
        initial_conditions=[0.001, 0.0],
    )

    plot_time_response(t, x, v, "sdof_damped_free_vibration")
    plot_phase_portrait(x, v, "sdof_damped_free_vibration")

    # Case 3: Damped harmonic forced vibration
    F0 = 1.0
    forcing_frequency = 0.8 * f1
    forcing_omega = 2 * np.pi * forcing_frequency

    def harmonic_force(t):
        return F0 * np.sin(forcing_omega * t)

    t, x, v = sdof_response(
        m=m,
        c=c_damped,
        k=k,
        force_function=harmonic_force,
        t_span=t_span,
        initial_conditions=[0.0, 0.0],
    )

    plot_time_response(t, x, v, "sdof_damped_harmonic_excitation")
    plot_phase_portrait(x, v, "sdof_damped_harmonic_excitation")


# Multi-mode beam validation (Thesis response case)
def run_multimode_validation():
    """Runs multi-mode beam response validation and harmonic excitation case."""

    beam = get_beam_properties()
    frequencies = natural_frequencies(beam, modes=NUMBER_OF_MODES)
    t = np.arange(0, SIMULATION_TIME + TIME_STEP, TIME_STEP)

    # Validation case: damped free vibration using initial modal amplitudes
    amplitudes = [0.001, 0.0004, 0.0002, 0.0001]

    modal_displacements_free, modal_velocities_free = damped_free_response(
        frequencies=frequencies,
        amplitudes=amplitudes,
        damping_ratio=DAMPING_RATIO,
        t=t,
        return_velocity=True,
    )

    free_displacement = reconstruct_displacement_at_point(
        modal_displacements_free,
        OBSERVATION_POINT,
    )

    free_velocity = reconstruct_velocity_at_point(
        modal_velocities_free,
        OBSERVATION_POINT,
    )

    plot_modal_contributions(
        t,
        modal_displacements_free,
        filename="modal_contributions_damped_free",
        title="Damped Free Vibration Modal Contributions",
    )

    plot_point_displacement(
        t,
        free_displacement,
        OBSERVATION_POINT,
        filename="damped_free_displacement_x_0_65",
    )

    plot_point_velocity(
        t,
        free_velocity,
        OBSERVATION_POINT,
        filename="damped_free_velocity_x_0_65",
    )

    # Main case: harmonic distributed loading
    force_amplitude = 5.0              # N/m
    forcing_frequency = 0.8 * frequencies[0]

    response = simulate_harmonic_beam_response(
        beam=beam,
        frequencies=frequencies,
        damping_ratio=DAMPING_RATIO,
        force_amplitude=force_amplitude,
        forcing_frequency=forcing_frequency,
        t=t,
        x_position=OBSERVATION_POINT,
        number_of_modes=NUMBER_OF_MODES,
    )

    plot_modal_contributions(
        t,
        response["modal_displacements"],
        filename="modal_contributions_harmonic_forced",
        title="Harmonic Forced Vibration Modal Contributions",
    )

    plot_point_displacement(
        t,
        response["displacement"],
        OBSERVATION_POINT,
        filename="harmonic_forced_displacement_x_0_65",
    )

    plot_point_velocity(
        t,
        response["velocity"],
        OBSERVATION_POINT,
        filename="harmonic_forced_velocity_x_0_65",
    )

    print("\n--- Multi-mode harmonic response summary ---")
    print(f"Observation point x/L: {OBSERVATION_POINT}")
    print(f"Force amplitude: {force_amplitude:.3f} N/m")
    print(f"Excitation frequency: {forcing_frequency:.3f} Hz")
    print(f"First natural frequency: {frequencies[0]:.3f} Hz")
    print(f"Peak displacement: {response['peak_displacement']:.6e} m")
    print(f"RMS displacement: {response['rms_displacement']:.6e} m")


# ---------------------------------------------------------------------
# Run all Phase 1 checks

if __name__ == "__main__":
    
    run_frequency_validation()
    run_mode_shape_validation()
    run_sdof_validation()
    run_multimode_validation()

    print("\nPhase 1 validation plots generated successfully.")