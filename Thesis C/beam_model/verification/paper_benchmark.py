"""
Dynamic benchmark of the nonlinear beam model from Su et al. (2026).

- independently reproduce the deterministic, no-RPE nonlinear system described in the base paper
- use the paper's nondimensional equations
- use the paper's seven-mode Galerkin model
- use the paper's initial conditions
- reconstruct displacement at x/L = 0.65

This is used only as a verification benchmark.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ----- PAPER PARAMETERS -----

# Material
Q11 = 206e9       # Elastic modulus [Pa]
rho = 7850        # Density [kg/m^3]

# Geometry
L = 2.0           # Beam length [m]
b = 0.04          # Beam width [m]
h = 0.01          # Beam thickness [m]

# Harmonic excitation
q_A = 15.0        # Excitation amplitude [Pa]
q_f = 2000.0      # Excitation frequency used by paper

# Number of modes
N = 7

# Observation location
x_ratio = 0.65


# ----- NONDIMENSIONAL PARAMETERS -----
# Su et al. nondimensionalise the beam before applying Galerkin discretisation.

tau = np.sqrt(Q11 * h**2 / (12 * rho * L**4))

# Parameters from Eq. (17) of the paper
A = h**2 / (12 * L**2)
B = Q11 * h**2 / (2 * L**4 * rho * tau**2)
C = Q11 * h**2 / (12 * L**4 * rho * tau**2)


# ----- MODE NUMBERS -----

modes = np.arange(1, N + 1)
k = modes * np.pi


# ----- PAPER INITIAL CONDITIONS -----
# Taken directly from Eq. (43).

q0 = np.array([0.8000, 0.0080, 0.0005, 0.0020, 0.0010, 0.0007, 0.0004])
q_dot0 = np.array([0.80, 0.69, 0.52, 0.43, 0.34, 0.28, 0.15])
y0 = np.concatenate((q0, q_dot0))


# ----- PAPER REFERENCE RESULTS -----
# Long-time deterministic values reported by Su et al.

paper_wP = 2.35538
paper_modal_amplitudes = np.array([2.45934, 0.79082, 0.15898])


# ----- NONDIMENSIONAL HARMONIC LOAD -----
# Eq. (16): q_non = q / (rho * h^2 * tau^2)
# Eq. (41): q = q_A * sin(2*pi*q_f*t/tau)

def q_non(t):
    q = q_A * np.sin(2 * np.pi * q_f * t / tau)
    return q / (rho * h**2 * tau**2)


# ----- PAPER EQUATIONS OF MOTION -----
# Compact implementation of the seven Galerkin equations in Appendix A.

def paper_equations(t, y):

    # Modal displacements and velocities
    q = y[:N]
    q_dot = y[N:]

    # Nonlinear modal coupling: sum((j*pi)^2 * q_j^2)
    modal_sum = np.sum(k**2 * q**2)

    # Modal inertia, nonlinear term and linear bending term
    modal_inertia = 0.5 * (1 + A * k**2)
    nonlinear_term = B * k**2 * q * modal_sum / 4
    linear_term = C * k**4 * q / 2

    # Uniform load projection: odd modes directly forced, even modes zero
    modal_force = q_non(t) * (1 - (-1)**modes) / (modes * np.pi)

    # Modal acceleration
    q_ddot = (-nonlinear_term - linear_term + modal_force) / modal_inertia

    return np.concatenate((q_dot, q_ddot))


# ----- RUN PAPER BENCHMARK -----

def run_paper_benchmark(t_end=200.0, dt=0.02, rtol=1e-6, atol=1e-8):

    if t_end <= 0 or dt <= 0:
        raise ValueError("t_end and dt must both be greater than zero.")

    number_of_points = int(round(t_end / dt)) + 1
    time = np.linspace(0, t_end, number_of_points)

    solution = solve_ivp(paper_equations, [0, t_end], y0, t_eval=time, method="DOP853", rtol=rtol, atol=atol)

    if not solution.success:
        raise RuntimeError(solution.message)

    # Modal displacement histories: shape = (7, number of time points)
    q_history = solution.y[:N, :]
    phi = np.sin(modes * np.pi * x_ratio)
    w_P = phi @ q_history

    if q_history.shape != (N, len(solution.t)):
        raise ValueError(f"Unexpected q_history shape: {q_history.shape}")
    if w_P.shape != (len(solution.t),):
        raise ValueError(f"Unexpected w_P shape: {w_P.shape}")
    if not np.all(np.isfinite(q_history)) or not np.all(np.isfinite(w_P)):
        raise ValueError("Benchmark produced NaN or infinite values.")

    max_w_P = np.max(np.abs(w_P))
    modal_amplitudes = np.max(np.abs(q_history), axis=1)

    return {
        "time": solution.t,
        "q_history": q_history,
        "w_P": w_P,
        "max_w_P": max_w_P,
        "modal_amplitudes": modal_amplitudes,
        "phi": phi
    }


# ----- LONG-TIME RESPONSE DEVELOPMENT -----

def analyse_time_windows(result, time_windows=None):

    if time_windows is None:
        time_windows = [(0, 50), (50, 100), (100, 150), (150, 200)]

    window_results = []

    for start_time, end_time in time_windows:
        mask = (result["time"] >= start_time) & (result["time"] < end_time)

        if not np.any(mask):
            raise ValueError(f"No saved time points found between {start_time} and {end_time}.")

        window_w = result["w_P"][mask]
        window_q = result["q_history"][:, mask]
        max_w = np.max(np.abs(window_w))
        max_modes = np.max(np.abs(window_q), axis=1)

        window_results.append({"start": start_time, "end": end_time, "max_w": max_w, "max_modes": max_modes})

    early = window_results[0]
    late = window_results[-1]
    mode_2_growth = late["max_modes"][1] / early["max_modes"][1]
    mode_3_growth = late["max_modes"][2] / early["max_modes"][2]

    return window_results, mode_2_growth, mode_3_growth


# ----- RUN WHEN THIS FILE IS EXECUTED DIRECTLY -----

if __name__ == "__main__":

    print("***BASE-PAPER DYNAMIC BENCHMARK***")
    print("\nNondimensional parameters:")
    print("tau =", tau)
    print("A =", A)
    print("B =", B)
    print("C =", C)

    print("\nRunning long-time benchmark to t = 200...")
    result = run_paper_benchmark(t_end=200, dt=0.02)

    print("\nPaper initial conditions:")
    for i in range(N):
        print(f"Mode {i+1}: q(0) = {q0[i]:.4f}, q_dot(0) = {q_dot0[i]:.4f}")

    print("\nMode-shape coefficients at x/L = 0.65:")
    for i in range(N):
        print(f"Mode {i+1}: {result['phi'][i]:.10f}")

    print("\nMaximum displacement during benchmark:", result["max_w_P"])
    print("Paper reported long-time maximum:", paper_wP)

    print("\nMaximum modal amplitudes during benchmark:")
    for i in range(N):
        print(f"Mode {i+1}: {result['modal_amplitudes'][i]:.6f}")

    print("\nPaper reported first-three modal maxima:")
    for i in range(3):
        print(f"Mode {i+1}: {paper_modal_amplitudes[i]:.5f}")

    window_results, mode_2_growth, mode_3_growth = analyse_time_windows(result)

    print("\nLong-time response development:")
    for window in window_results:
        print(f"\nTime {window['start']} - {window['end']}:")
        print(f"Maximum w_P = {window['max_w']:.6f}")
        print(f"Mode 1 maximum = {window['max_modes'][0]:.6f}")
        print(f"Mode 2 maximum = {window['max_modes'][1]:.6f}")
        print(f"Mode 3 maximum = {window['max_modes'][2]:.6f}")

    print("\nLong-time modal development:")
    print(f"Mode 2 late/early amplitude ratio = {mode_2_growth:.3f}")
    print(f"Mode 3 late/early amplitude ratio = {mode_3_growth:.3f}")

    plt.figure(figsize=(10, 5))
    plt.plot(result["time"], result["w_P"])
    plt.xlabel("Nondimensional time")
    plt.ylabel("Nondimensional displacement")
    plt.title("Base-Paper Benchmark Response at x/L = 0.65")
    plt.grid()

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(result["time"], result["q_history"][i, :], label=f"Mode {i+1}")
    plt.xlabel("Nondimensional time")
    plt.ylabel("Nondimensional modal displacement")
    plt.title("Base-Paper Benchmark Modal Responses")
    plt.legend()
    plt.grid()

    plt.show()
