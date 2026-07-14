"""
Physics-guided feature calculations.

This script calculates physics-derived quantities for the
physics-guided neural network surrogate.

The features are based on the simply supported Euler-Bernoulli
beam model used by the existing physics simulator.
"""

import sys
from pathlib import Path

import numpy as np

PHASE_1_2_DIR = (Path(__file__).resolve().parent.parent/ "Phase 1 & 2")

sys.path.insert(0, str(PHASE_1_2_DIR))

from frequency_analysis import natural_frequencies
from multi_mode_response import modal_mass


def build_beam_from_parameters(L, b, h, E, rho):
    """
    Creates the beam dictionary required by the existing physics functions.

    Parameters
    ----------
    L : float
        Beam length in metres.
    b : float
        Beam width in metres.
    h : float
        Beam thickness in metres.
    E : float
        Young's modulus in Pa.
    rho : float
        Material density in kg/m^3.

    Returns
    -------
    dict
        Beam properties including area A and second moment of area I.
    """

    parameters = {
        "L": L,
        "b": b,
        "h": h,
        "E": E,
        "rho": rho,
    }

    for name, value in parameters.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    beam = {
        "L": float(L),
        "b": float(b),
        "h": float(h),
        "E": float(E),
        "rho": float(rho),
    }

    beam["A"] = beam["b"] * beam["h"]
    beam["I"] = beam["b"] * beam["h"] ** 3 / 12

    return beam


def calculate_modal_stiffness(modal_mass_value, natural_frequency):
    """
    Calculates modal stiffness.

    k_n = M_n * omega_n^2

    where:
        M_n     = modal mass
        omega_n = natural angular frequency
    """

    if modal_mass_value <= 0:
        raise ValueError("Modal mass must be greater than zero.")

    if natural_frequency <= 0:
        raise ValueError("Natural frequency must be greater than zero.")

    omega_n = 2 * np.pi * natural_frequency

    modal_stiffness = modal_mass_value * omega_n**2

    return modal_stiffness


def calculate_frequency_ratio(excitation_frequency, natural_frequency):
    """
    Calculates the excitation-to-natural frequency ratio.

    r = f_excitation / f_natural
    """

    if excitation_frequency < 0:
        raise ValueError("Excitation frequency cannot be negative.")

    if natural_frequency <= 0:
        raise ValueError("Natural frequency must be greater than zero.")

    frequency_ratio = excitation_frequency / natural_frequency

    return frequency_ratio


def calculate_resonance_proximity(frequency_ratio):
    """
    Calculates proximity to resonance.

    resonance_proximity = |1 - r|

    A value close to zero indicates excitation close to resonance.
    """

    resonance_proximity = abs(1.0 - frequency_ratio)

    return resonance_proximity


def calculate_physics_features(
    L,
    b,
    h,
    E,
    rho,
    excitation_frequency,
):
    """
    Calculates all physics-guided features for one beam sample.

    Returns
    -------
    dict
        Physics-derived features used by the hybrid neural network.
    """

    # Build beam properties
    beam = build_beam_from_parameters(
        L=L,
        b=b,
        h=h,
        E=E,
        rho=rho,
    )

    # First natural frequency
    first_natural_frequency = float(
        natural_frequencies(
            beam,
            modes=1,
        )[0]
    )

    # Modal mass
    modal_mass_value = float(
        modal_mass(beam)
    )

    # First-mode modal stiffness
    modal_stiffness = calculate_modal_stiffness(
        modal_mass_value=modal_mass_value,
        natural_frequency=first_natural_frequency,
    )

    # Excitation-to-natural frequency ratio
    frequency_ratio = calculate_frequency_ratio(
        excitation_frequency=excitation_frequency,
        natural_frequency=first_natural_frequency,
    )

    # Distance from first-mode resonance
    resonance_proximity = calculate_resonance_proximity(
        frequency_ratio=frequency_ratio,
    )

    return {
        "first_natural_frequency": first_natural_frequency,
        "modal_stiffness": modal_stiffness,
        "modal_mass": modal_mass_value,
        "frequency_ratio": frequency_ratio,
        "resonance_proximity": resonance_proximity,
    }


if __name__ == "__main__":

    features = calculate_physics_features(
        L=2.0,
        b=0.04,
        h=0.01,
        E=206e9,
        rho=7850,
        excitation_frequency=10.0,
    )

    print("\n--- Physics-Guided Features ---")

    for feature_name, feature_value in features.items():
        print(f"{feature_name}: {feature_value:.6e}")