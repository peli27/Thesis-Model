import numpy as np
from scipy.integrate import solve_ivp


def simulate_sdof_harmonic_response(
    natural_frequency_hz: float,
    damping_ratio: float,
    forcing_frequency_ratio: float,
    forcing_amplitude_n: float,
    mass_kg: float,
    duration_s: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solving the damped forced SDOF equation:
        m x'' + c x' + k x = F0 sin(omega t)
    """
    omega_n = 2.0 * np.pi * natural_frequency_hz
    omega = forcing_frequency_ratio * omega_n

    k = mass_kg * omega_n**2
    c = 2.0 * damping_ratio * mass_kg * omega_n

    def ode(t: float, y: np.ndarray) -> list[float]:
        x, x_dot = y
        force = forcing_amplitude_n * np.sin(omega * t)
        x_ddot = (force - c * x_dot - k * x) / mass_kg
        return [x_dot, x_ddot]

    t_eval = np.linspace(0.0, duration_s, n_points)
    y0 = [0.0, 0.0]

    sol = solve_ivp(
        ode,
        t_span=(0.0, duration_s),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
    )

    return sol.t, sol.y[0]