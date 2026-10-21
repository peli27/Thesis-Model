"""
Nonlinear multimodal simply-supported beam model.

Physics:
- Euler-Bernoulli beam kinematics
- von Karman geometric nonlinearity
- simply-supported boundary conditions
- Galerkin modal discretisation
- harmonic uniformly distributed transverse loading
- optional modal viscous damping
- rotary inertia term retained from Su et al.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ----- BEAM INPUT PARAMETERS -----
# Values come directly from the base paper by Su et al. (2020).

def simulate_beam(
    L=2.0,              # beam length [m]
    b=0.04,             # beam width [m]
    h=0.01,             # beam thickness [m]
    E=206e9,            # Young's modulus [Pa]
    rho=7850,           # density [kg/m^3]
    zeta=0.01,          # 1% damping
    pressure_amplitude=15.0,  # harmonic pressure amplitude [Pa]
    excitation_frequency=None,
    N=7,                # Number of vibration modes (The paper uses seven modes after its modal convergence study.)
    use_nonlinearity=True,
    t_end=10,
    dt=0.002,
    y0=None,
    rtol=1e-8,
    atol=1e-10
):

# ----- BEAM PROPERTIES -----

    # Cross-sectional area
    A = b * h

    # Second moment of area
    I = b * h**3 / 12


# ----- VIBRATION MODES -----

    # Mode numbers [1, 2, 3, 4, 5, 6, 7]
    modes = np.arange(1, N + 1)

    # Wave number for each mode
    k = modes * np.pi / L

    # Modal mass (Rotary inertia term included in the model mass calculation)
    M = rho * (A + I * k**2) * L / 2

    # Modal stiffness
    K = E * I * k**4 * L / 2

    # Natural angular frequencies [rad/s]
    omega_n = np.sqrt(K / M)

    # Natural frequencies [Hz]
    f_n = omega_n / (2 * np.pi)


## ----- DAMPING AND EXCITATION -----
    # Modal damping
    C = 2 * zeta * omega_n * M

    # For the first test, excite close to the first natural frequency, this puts it close to resonance
    if excitation_frequency is None:
        excitation_frequency = 0.95 * f_n[0]


## ----- MODAL FORCE -----
# How strongly the uniform load excites each mode
    force_participation = (L * (1 - (-1)**modes) / (modes * np.pi))

    def modal_force(t):

        # Harmonic pressure
        pressure = (pressure_amplitude * np.sin(2 * np.pi * excitation_frequency * t))

        # Convert pressure [Pa] into line load [N/m]
        line_load = b * pressure

        # Convert the distributed load into forces for each mode
        F = line_load * force_participation

        return F

## ----- VON KARMAN NONLINEAR FORCE -----
# Applying the von Kármán strain assumption used in the base paper.

    def nonlinear_force(q):

        # Turn nonlinearity off for a linear comparison
        if not use_nonlinearity:
            return np.zeros(N)

        # Measure how much total stretching is caused by vibration
        stretching = np.sum(k**2 * q**2)

        # Nonlinear force acting on every mode
        F_nonlinear = (E * A * L / 8 * k**2 * q * stretching)

        return F_nonlinear


## ----- BEAM EQUATIONS OF MOTION -----

    def beam_equations(t, y):

        # First N values are modal displacements
        q = y[:N]

        # Last N values are modal velocities
        q_dot = y[N:]

        # External harmonic force
        F = modal_force(t)

        # Nonlinear force
        F_NL = nonlinear_force(q)

        # Calculate modal acceleration
        q_ddot = (
            F
            - C * q_dot
            - K * q
            - F_NL
        ) / M

        # solve_ivp requires first-order equations
        return np.concatenate((q_dot, q_ddot))

# ----- INITIAL CONDITIONS -----
# If no initial conditions, start every mode from zero displacement and zero velocity.
    if y0 is None:
        y0 = np.zeros(2 * N)

## ----- RUNNING THE SIMULATION -----
# Numerically solves all seven coupled vibration equations through time.

# Number of saved time points
    number_of_points = int(round(t_end / dt)) + 1

    # Time values
    t = np.linspace(0, t_end, number_of_points)

    # Numerically solve the coupled modal equations
    solution = solve_ivp(
        beam_equations,
        [0, t_end],
        y0,
        t_eval=t,
        method="DOP853",
        rtol=rtol,
        atol=atol
    )

    # Stop if the numerical solver failed
    if not solution.success:
        raise RuntimeError(solution.message)

## ----- PHYSICAL BEAM DISPLACEMENT -----
# Reconstruct the physical displacement of the beam from the modal displacements.

    # Extract modal displacement histories
    q_history = solution.y[:N]

    # Measure the beam at x/L = 0.65 (Following base paper)
    x_ratio = 0.65

    # Mode shape values at this location
    phi = np.sin(modes * np.pi * x_ratio)

    # Add all modal contributions together. w contains physical displacement of the beam over time at the specified location.
    w = phi @ q_history

# ----- RESPONSE VALUES -----
    # Mainly ignore the first half of the simulation and measure the later response.

    start = len(w) // 2

    w_steady = w[start:]

    # Peak displacement
    peak_displacement = np.max(np.abs(w_steady))

    # RMS displacement
    rms_displacement = np.sqrt(np.mean(w_steady**2))

# ----- RETURN RESULTS -----
    return {
        "time": solution.t,
        "q_history": q_history,
        "displacement": w,
        "peak_displacement": peak_displacement,
        "rms_displacement": rms_displacement,
        "frequencies": f_n,
        "modes": modes,
        "modal_mass": M,
        "modal_stiffness": K,
        "excitation_frequency": excitation_frequency,
        "beam_equations": beam_equations
    } 


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # ----- DEFAULT BEAM INPUT PARAMETERS -----
    # Geometry and material values come from the base paper by Su et al. (2026).

    # Geometry
    L = 2.0
    b = 0.04
    h = 0.01

    # Material
    E = 206e9
    rho = 7850

    # Damping
    zeta = 0.01

    # Loading
    pressure_amplitude = 15.0

    # Number of vibration modes
    N = 7

    # Include geometric nonlinearity?
    use_nonlinearity = True


    # ----- RUN SIMULATION -----

    result = simulate_beam(
        L=L,
        b=b,
        h=h,
        E=E,
        rho=rho,
        zeta=zeta,
        pressure_amplitude=pressure_amplitude,
        N=N,
        use_nonlinearity=use_nonlinearity
    )


    # ----- PRINT NATURAL FREQUENCIES -----

    print("\nNatural frequencies:")

    for i in range(N):
        print(f"Mode {i+1}: {result['frequencies'][i]:.3f} Hz")

    print("\nExcitation frequency:", round(result["excitation_frequency"], 3), "Hz")

    print("\nSimulation complete")


    # ----- PHYSICAL BEAM DISPLACEMENT -----

    time = result["time"]
    w = result["displacement"]
    q_history = result["q_history"]


    # ----- PLOT PHYSICAL RESPONSE -----

    plt.figure(figsize=(10, 5))
    plt.plot(time, w)
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.title("Nonlinear Beam Response at x/L = 0.65")
    plt.grid()
    plt.show()


    # ----- PLOT MODAL RESPONSES -----
    # Shows how each mode contributes to the overall response.

    plt.figure(figsize=(10, 6))
    
    for i in range(N):
        plt.plot(time, q_history[i], label=f"Mode {i+1}")

    plt.xlabel("Time [s]")
    plt.ylabel("Modal displacement [m]")
    plt.title("Individual Modal Responses")
    plt.legend()
    plt.grid()
    plt.show()


    # ----- PRINT RESPONSE VALUES -----

    print("\nPeak displacement:", result["peak_displacement"], "m")
    print("RMS displacement:", result["rms_displacement"], "m")