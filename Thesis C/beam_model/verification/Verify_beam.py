"""
Verification of the nonlinear multimodal beam model.

Checks:
1. natural frequency check
2. linear-limit check
3. nonlinear-response check
4. solver verification
5. paper benchmark
6. modal behaviour
7. modal convergence
8. numerical sensitivity
9. response-regime sweep
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ----- PROJECT PATHS -----

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from beam_model.nonlinear_beam import simulate_beam
from beam_model.verification.paper_benchmark import (
    run_paper_benchmark,
    analyse_time_windows,
    paper_wP,
    paper_modal_amplitudes
)

# ----- VERIFICATION SETTINGS -----

# Beam properties from the base paper
L = 2.0
b = 0.04
h = 0.01
E = 206e9
rho = 7850

# Thesis model damping and default number of modes
zeta = 0.01
N = 7

# Convergence criterion
convergence_tolerance = 1.0       # [%]


print("NONLINEAR BEAM VERIFICATION")


## ----- CHECK 1: NATURAL FREQUENCIES -----
# Compare the model frequencies against the analytical Euler-Bernoulli simply-supported beam equation.

print("\n========================================")
print("CHECK 1: NATURAL FREQUENCIES")

frequency_result = simulate_beam(L=L, b=b, h=h, E=E, rho=rho, zeta=zeta, N=N)
model_frequencies = frequency_result["frequencies"]

A = b * h
I = b * h**3 / 12
modes = np.arange(1, N + 1)
analytical_frequencies = (modes * np.pi / L)**2 * np.sqrt(E * I / (rho * A)) / (2 * np.pi)
frequency_errors = np.abs(model_frequencies - analytical_frequencies) / analytical_frequencies * 100

print("\nMode | Model [Hz] | Analytical [Hz] | Difference [%]")
for i in range(N):
    print(f"{i+1:4d} | {model_frequencies[i]:10.4f} | {analytical_frequencies[i]:15.4f} | {frequency_errors[i]:12.5f}")

print(f"\nMaximum frequency difference = {np.max(frequency_errors):.5f} %")


## ----- CHECK 2: LINEAR LIMIT CHECK -----
# At very small displacement, the nonlinear model should behave almost exactly like the linear model.

print("\n========================================")
print("CHECK 2: LINEAR LIMIT CHECK")

small_force = 0.1
small_linear = simulate_beam(pressure_amplitude=small_force, use_nonlinearity=False)
small_nonlinear = simulate_beam(pressure_amplitude=small_force, use_nonlinearity=True)
small_difference = abs(small_nonlinear["peak_displacement"] - small_linear["peak_displacement"]) / small_linear["peak_displacement"] * 100

print("\nLinear peak:", small_linear["peak_displacement"], "m")
print("Nonlinear peak:", small_nonlinear["peak_displacement"], "m")
print("Difference:", small_difference, "%")


## ----- CHECK 3: NONLINEAR RESPONSE -----
# Increase the forcing and confirm that geometric nonlinearity changes the response.

print("\n========================================")
print("CHECK 3: NONLINEAR RESPONSE")

large_force = 15.0
large_linear = simulate_beam(pressure_amplitude=large_force, use_nonlinearity=False)
large_nonlinear = simulate_beam(pressure_amplitude=large_force, use_nonlinearity=True)
large_difference = abs(large_nonlinear["peak_displacement"] - large_linear["peak_displacement"]) / large_linear["peak_displacement"] * 100

print("\nLinear peak:", large_linear["peak_displacement"], "m")
print("Nonlinear peak:", large_nonlinear["peak_displacement"], "m")
print("Difference:", large_difference, "%")

plt.figure(figsize=(10, 5))
plt.plot(large_linear["time"], large_linear["displacement"], label="Linear")
plt.plot(large_nonlinear["time"], large_nonlinear["displacement"], label="Nonlinear")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("Linear vs Nonlinear Beam Response")
plt.legend()
plt.grid()


## ----- CHECK 4: SOLVE_IVP VS RK4 -----
# Compare scipy's solve_ivp result against an independently implemented fourth-order Runge-Kutta solver.

print("\n========================================")
print("CHECK 4: SOLVE_IVP VS RK4")

solver_result = simulate_beam(pressure_amplitude=5.0, use_nonlinearity=True, t_end=1.0, dt=0.0002)
time = solver_result["time"]
beam_equations = solver_result["beam_equations"]
dt = time[1] - time[0]
y_rk4 = np.zeros((2 * N, len(time)))

for i in range(len(time) - 1):
    t_now = time[i]
    y_now = y_rk4[:, i]
    k1 = beam_equations(t_now, y_now)
    k2 = beam_equations(t_now + dt / 2, y_now + dt * k1 / 2)
    k3 = beam_equations(t_now + dt / 2, y_now + dt * k2 / 2)
    k4 = beam_equations(t_now + dt, y_now + dt * k3)
    y_rk4[:, i + 1] = y_now + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

phi = np.sin(modes * np.pi * 0.65)
w_rk4 = phi @ y_rk4[:N]
w_solve_ivp = solver_result["displacement"]
solver_difference = np.max(np.abs(w_solve_ivp - w_rk4))
relative_solver_difference = solver_difference / np.max(np.abs(w_solve_ivp)) * 100

print("\nMaximum absolute difference:", solver_difference, "m")
print("Maximum relative difference:", relative_solver_difference, "%")

plt.figure(figsize=(10, 5))
plt.plot(time, w_solve_ivp, label="solve_ivp")
plt.plot(time, w_rk4, "--", label="RK4")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("solve_ivp vs Independent RK4")
plt.legend()
plt.grid()


## ----- CHECK 5: BASE-PAPER BENCHMARK -----
# Check the Galerkin reconstruction and independently run the paper's nondimensional deterministic model.

print("\n========================================")
print("CHECK 5: BASE-PAPER BENCHMARK")

# ----- CHECK 5A: GALERKIN COEFFICIENTS -----

print("\n5A: Galerkin reconstruction")

my_phi = np.sin(modes * np.pi * 0.65)
paper_phi = np.array([0.8910065241, -0.8090169946, -0.1564344645, 0.9510565161, -0.7071067782, -0.3090169934, 0.9876883396])
phi_differences = np.abs(my_phi - paper_phi)

print("\nMode | Our coefficient | Paper coefficient | Difference")
for i in range(N):
    print(f"{i+1:4d} | {my_phi[i]:15.10f} | {paper_phi[i]:17.10f} | {phi_differences[i]:.3e}")

maximum_phi_difference = np.max(phi_differences)
print("\nMaximum coefficient difference:", maximum_phi_difference)

# ----- CHECK 5B: DYNAMIC NONDIMENSIONAL BENCHMARK -----

print("\n5B: Dynamic nondimensional benchmark")
print("Running benchmark to nondimensional time t = 200. This is the slowest verification check...")

paper_result = run_paper_benchmark(t_end=200, dt=0.02)

print("\nMaximum displacement during benchmark:", paper_result["max_w_P"])
print("Paper reported long-time maximum:", paper_wP)

print("\nMaximum modal amplitudes during benchmark:")
for i in range(N):
    print(f"Mode {i+1}: {paper_result['modal_amplitudes'][i]:.6f}")

print("\nPublished first-three modal maxima:")
for i in range(3):
    print(f"Mode {i+1}: {paper_modal_amplitudes[i]:.5f}")

# ----- CHECK 5C: LONG-TIME RESPONSE DEVELOPMENT -----
# Examine whether the first three modes and total response change over successive long-time windows.

print("\n5C: Long-time response development")

window_results, mode_2_growth, mode_3_growth = analyse_time_windows(paper_result)

for window in window_results:
    print(f"\nTime {window['start']} - {window['end']}:")
    print(f"Maximum w_P = {window['max_w']:.6f}")
    print(f"Mode 1 maximum = {window['max_modes'][0]:.6f}")
    print(f"Mode 2 maximum = {window['max_modes'][1]:.6f}")
    print(f"Mode 3 maximum = {window['max_modes'][2]:.6f}")

print("\nLong-time modal development:")
print(f"Mode 2 late/early amplitude ratio = {mode_2_growth:.3f}")
print(f"Mode 3 late/early amplitude ratio = {mode_3_growth:.3f}")
print("Note: exact chaotic trajectories are not expected to overlap point-for-point; this benchmark checks the published formulation and response development.")

plt.figure(figsize=(10, 5))
plt.plot(paper_result["time"], paper_result["w_P"])
plt.xlabel("Nondimensional time")
plt.ylabel("Nondimensional displacement")
plt.title("Base-Paper Long-Time Benchmark Response")
plt.grid()

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(paper_result["time"], paper_result["q_history"][i, :], label=f"Mode {i+1}")
plt.xlabel("Nondimensional time")
plt.ylabel("Nondimensional modal displacement")
plt.title("Base-Paper Long-Time Modal Responses")
plt.legend()
plt.grid()


## ----- CHECK 6: MULTIMODAL BEHAVIOUR -----
# Test the dimensional thesis model near the 1st, 3rd and 5th natural frequencies.

print("\n========================================")
print("CHECK 6: MULTIMODAL BEHAVIOUR")

test_cases = {"Near Mode 1": 1, "Near Mode 3": 3, "Near Mode 5": 5}

for case_name, target_mode in test_cases.items():
    excitation_frequency = 0.95 * model_frequencies[target_mode - 1]
    modal_result = simulate_beam(pressure_amplitude=15.0, excitation_frequency=excitation_frequency, use_nonlinearity=True, N=7, t_end=5, dt=0.0005)
    modal_amplitudes = np.max(np.abs(modal_result["q_history"]), axis=1)
    dominant_mode = np.argmax(modal_amplitudes) + 1

    print(f"\n{case_name}")
    print(f"Excitation frequency = {excitation_frequency:.4f} Hz")
    print(f"Dominant mode = Mode {dominant_mode}")
    for i in range(N):
        print(f"Mode {i+1}: {modal_amplitudes[i]:.6e} m")

    plt.figure(figsize=(8, 5))
    plt.bar(modes, modal_amplitudes)
    plt.xlabel("Mode number")
    plt.ylabel("Maximum modal amplitude [m]")
    plt.title(f"Modal Response - {case_name}")
    plt.grid(axis="y")


## ----- CHECK 7: MODAL CONVERGENCE -----
# Determine the minimum truncation order needed to match a 9-mode reference within the chosen tolerance.

print("\n========================================")
print("CHECK 7: MODAL CONVERGENCE")

mode_counts = [1, 2, 3, 5, 7, 9]
convergence_cases = {"Near Mode 1": 0.95 * model_frequencies[0], "Near Mode 3": 0.95 * model_frequencies[2], "Near Mode 5": 0.95 * model_frequencies[4]}

for case_name, excitation_frequency in convergence_cases.items():
    print(f"\n----- {case_name} -----")
    print(f"Excitation frequency = {excitation_frequency:.4f} Hz")

    peak_results = []
    rms_results = []

    for number_of_modes in mode_counts:
        convergence_result = simulate_beam(N=number_of_modes, pressure_amplitude=15.0, excitation_frequency=excitation_frequency, use_nonlinearity=True, t_end=5, dt=0.0005)
        peak_results.append(convergence_result["peak_displacement"])
        rms_results.append(convergence_result["rms_displacement"])
        print(f"{number_of_modes} modes: Peak = {convergence_result['peak_displacement']:.6e} m, RMS = {convergence_result['rms_displacement']:.6e} m")

    reference_peak = peak_results[-1]
    reference_rms = rms_results[-1]
    minimum_converged_modes = None

    print("\nDifference from 9-mode reference:")
    for i, number_of_modes in enumerate(mode_counts):
        peak_difference = abs(peak_results[i] - reference_peak) / reference_peak * 100
        rms_difference = abs(rms_results[i] - reference_rms) / reference_rms * 100
        print(f"{number_of_modes} modes: Peak difference = {peak_difference:.4f} %, RMS difference = {rms_difference:.4f} %")

        if minimum_converged_modes is None and peak_difference < convergence_tolerance and rms_difference < convergence_tolerance:
            minimum_converged_modes = number_of_modes

    print(f"Minimum order within {convergence_tolerance:.1f}% of the 9-mode reference = {minimum_converged_modes} modes")

    plt.figure(figsize=(8, 5))
    plt.plot(mode_counts, peak_results, marker="o", label="Peak displacement")
    plt.plot(mode_counts, rms_results, marker="o", label="RMS displacement")
    plt.xlabel("Number of modes")
    plt.ylabel("Displacement [m]")
    plt.title(f"Modal Convergence - {case_name}")
    plt.legend()
    plt.grid()


## ----- CHECK 8: NUMERICAL SENSITIVITY -----
# Tighten the solver tolerances and output spacing and confirm that the response changes very little.

print("\n========================================")
print("CHECK 8: NUMERICAL SENSITIVITY")

normal_result = simulate_beam(rtol=1e-8, atol=1e-10, dt=0.002)
tight_result = simulate_beam(rtol=1e-10, atol=1e-12, dt=0.001)
numerical_difference = abs(normal_result["peak_displacement"] - tight_result["peak_displacement"]) / tight_result["peak_displacement"] * 100

print("\nNormal settings peak:", normal_result["peak_displacement"], "m")
print("Tighter settings peak:", tight_result["peak_displacement"], "m")
print("Difference:", numerical_difference, "%")


## ----- CHECK 9: RESPONSE REGIME SWEEP -----
# Vary excitation frequency and forcing amplitude to confirm distinct low-response, resonance and stronger nonlinear regimes.

print("\n========================================")
print("CHECK 9: RESPONSE REGIME SWEEP")

frequency_ratios = np.array([0.50, 0.70, 0.85, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50])
forcing_levels = [0.1, 5.0, 15.0, 30.0]
first_frequency = model_frequencies[0]

plt.figure(figsize=(9, 6))

for force in forcing_levels:
    response_peaks = []
    used_frequencies = []

    print(f"\nForcing amplitude = {force} Pa")

    for ratio in frequency_ratios:
        excitation_frequency = ratio * first_frequency
        regime_result = simulate_beam(pressure_amplitude=force, excitation_frequency=excitation_frequency, use_nonlinearity=True, t_end=10)
        response_peaks.append(regime_result["peak_displacement"])
        used_frequencies.append(regime_result["excitation_frequency"])
        print(f"Ratio = {ratio:.2f}, Requested = {excitation_frequency:.3f} Hz, Used = {regime_result['excitation_frequency']:.3f} Hz, Peak = {regime_result['peak_displacement']:.6e} m")

    response_peaks = np.array(response_peaks)
    used_frequencies = np.array(used_frequencies)
    frequency_input_error = np.max(np.abs(used_frequencies - frequency_ratios * first_frequency))
    response_variation = np.max(response_peaks) / np.min(response_peaks)
    peak_ratio = frequency_ratios[np.argmax(response_peaks)]

    print(f"Maximum frequency-input error = {frequency_input_error:.3e} Hz")
    print(f"Maximum/minimum response ratio = {response_variation:.3f}")
    print(f"Largest response occurs at frequency ratio = {peak_ratio:.2f}")

    plt.plot(frequency_ratios, response_peaks, marker="o", label=f"{force} Pa")

plt.xlabel("Excitation frequency / first natural frequency")
plt.ylabel("Peak displacement [m]")
plt.title("Nonlinear Beam Response Regimes")
plt.legend()
plt.grid()


print("VERIFICATION RUN COMPLETE")

plt.show()
