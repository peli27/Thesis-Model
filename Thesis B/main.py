import numpy as np

from beam_properties import get_beam_properties
from frequency_analysis import natural_frequencies
from plotting import (
    plot_frequency_vs_length,
    plot_frequency_vs_thickness,
    plot_frequency_vs_youngs_modulus
)

from mode_shapes import simply_supported_mode_shape
from plotting import plot_mode_shapes

from response_analysis import sdof_response
from plotting import plot_time_response, plot_phase_portrait

from multi_mode_response import (
    undamped_free_response,
    damped_free_response,
    damped_harmonic_response,
    combine_modal_responses,
    reconstruct_displacement_at_point
)

from plotting import plot_modal_contributions, plot_total_response, plot_point_displacement


# Load baseline beam
beam = get_beam_properties()

## Study 1: Frequency vs Length
L_values = np.linspace(0.1, 1.0, 50)
f1_length_values = []

for L in L_values:
    beam = get_beam_properties()
    beam["L"] = L

    frequencies = natural_frequencies(beam, modes=1)
    f1_length_values.append(frequencies[0])

plot_frequency_vs_length(L_values, f1_length_values)

## Study 2: Frequency vs Thickness
h_values = np.linspace(0.001, 0.01, 50)
f1_thickness_values = []

for h in h_values:
    beam = get_beam_properties()
    beam["h"] = h
    beam["A"] = beam["b"] * beam["h"]
    beam["I"] = beam["b"] * beam["h"]**3 / 12

    frequencies = natural_frequencies(beam, modes=1)
    f1_thickness_values.append(frequencies[0])

plot_frequency_vs_thickness(h_values, f1_thickness_values)

## Study 3: Frequency vs Young's Modulus
E_values = np.linspace(50e9, 300e9, 50)
f1_E_values = []

for E in E_values:
    beam = get_beam_properties()
    beam["E"] = E

    frequencies = natural_frequencies(beam, modes=1)
    f1_E_values.append(frequencies[0])

plot_frequency_vs_youngs_modulus(E_values, f1_E_values)

# ---------------------------------------------------
# Step 4: Visualise Beam Mode Shapes

x_values = np.linspace(0, 1, 200)

mode_numbers = [1, 2, 3, 4]

mode_shapes = []

for mode in mode_numbers:
    phi = simply_supported_mode_shape(x_values, mode)
    mode_shapes.append(phi)

plot_mode_shapes(x_values, mode_shapes, mode_numbers)

# --------------------------------------------------
# Step 5: Single Degree of Freedom Response
# Free damped vibration

beam = get_beam_properties()

frequencies = natural_frequencies(beam, modes=1)
f1 = frequencies[0]

omega1 = 2 * np.pi * f1

m = 1.0
k = m * omega1**2

damping_ratio = 0.02

t_span = (0, 2)

## Case 1: Undamped Free Vibration
# c = 0, F(t) = 0

c_undamped = 0.0
initial_conditions = [0.001, 0.0]

def zero_force(t):
    return 0.0

t, x, v = sdof_response(
    m,
    c_undamped,
    k,
    zero_force,
    t_span,
    initial_conditions
)

plot_time_response(t, x, v, "undamped_free_vibration")
plot_phase_portrait(x, v, "undamped_free_vibration")


## Case 2: Damped Free Vibration
# c > 0, F(t) = 0

c_damped = 2 * damping_ratio * m * omega1
initial_conditions = [0.001, 0.0]

t, x, v = sdof_response(
    m,
    c_damped,
    k,
    zero_force,
    t_span,
    initial_conditions
)

plot_time_response(t, x, v, "damped_free_vibration")
plot_phase_portrait(x, v, "damped_free_vibration")


## Case 3: Damped Harmonic Forced Vibration
# c > 0, F(t) = F0 sin(omega t)

F0 = 1.0

forcing_frequency = 0.8 * f1
forcing_omega = 2 * np.pi * forcing_frequency

def harmonic_force(t):
    return F0 * np.sin(forcing_omega * t)

initial_conditions = [0.0, 0.0]

t, x, v = sdof_response(
    m,
    c_damped,
    k,
    harmonic_force,
    t_span,
    initial_conditions
)

plot_time_response(t, x, v, "damped_harmonic_excitation")
plot_phase_portrait(x, v, "damped_harmonic_excitation")

# --------------------------------------------------
# Task 7: Multi-Mode Beam Response

beam = get_beam_properties()

frequencies = natural_frequencies(beam, modes=4)

t = np.linspace(0, 2, 5000)

amplitudes = [0.001, 0.0004, 0.0002, 0.0001]

damping_ratio = 0.02

# Case 1
modal_responses_c1 = undamped_free_response(frequencies, amplitudes, t)
total_response_c1 = combine_modal_responses(modal_responses_c1)

# Case 2
modal_responses_c2 = damped_free_response(frequencies, amplitudes, damping_ratio, t)
total_response_c2 = combine_modal_responses(modal_responses_c2)

# Case 3
modal_responses_c3 = damped_harmonic_response(
    frequencies=frequencies,
    force_amplitudes=[1.0, 0.4, 0.2, 0.1],
    damping_ratio=damping_ratio,
    t=t,
    forcing_frequency=frequencies[0]
)

total_response_c3 = combine_modal_responses(modal_responses_c3)

plot_modal_contributions(t, modal_responses_c2) # Plot modal contributions for Case 2
plot_total_response(t, total_response_c2)

# Figure C: Midpoint Beam Displacement
# --------------------------------------------------

x_position = 0.5

midpoint_displacement = reconstruct_displacement_at_point(
    modal_responses_c2, # Plot modal response for Case 2
    x_position
)

plot_point_displacement(
    t,
    midpoint_displacement,
    x_position
)