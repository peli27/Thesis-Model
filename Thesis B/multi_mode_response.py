import numpy as np
from scipy.integrate import solve_ivp


def undamped_free_response(frequencies, amplitudes, t):
    """
    Case 1: Undamped free vibration

    q_n(t) = A_n cos(omega_n t)
    """

    modal_responses = []

    for f, A in zip(frequencies, amplitudes):
        omega_n = 2 * np.pi * f

        q = A * np.cos(omega_n * t)

        modal_responses.append(q)

    return np.array(modal_responses)


def damped_free_response(frequencies, amplitudes, damping_ratio, t):
    """
    Case 2: Damped free vibration

    q_n(t) = A_n exp(-zeta omega_n t) cos(omega_d t)
    """

    modal_responses = []

    for f, A in zip(frequencies, amplitudes):
        omega_n = 2 * np.pi * f
        omega_d = omega_n * np.sqrt(1 - damping_ratio**2)

        q = A * np.exp(-damping_ratio * omega_n * t) * np.cos(omega_d * t)

        modal_responses.append(q)

    return np.array(modal_responses)


def damped_harmonic_response(
    frequencies,
    force_amplitudes,
    damping_ratio,
    t,
    forcing_frequency
):
    """
    Case 3: Damped harmonic excitation

    q_ddot + 2*zeta*omega_n*q_dot + omega_n^2*q = F_n sin(omega_f t)

    Assumes modal mass = 1 for each mode.
    """

    modal_responses = []

    omega_f = 2 * np.pi * forcing_frequency

    for f, F in zip(frequencies, force_amplitudes):
        omega_n = 2 * np.pi * f

        def modal_ode(ti, y):
            q = y[0]
            q_dot = y[1]

            q_ddot = (
                F * np.sin(omega_f * ti)
                - 2 * damping_ratio * omega_n * q_dot
                - omega_n**2 * q
            )

            return [q_dot, q_ddot]

        initial_conditions = [0, 0]

        solution = solve_ivp(
            modal_ode,
            [t[0], t[-1]],
            initial_conditions,
            t_eval=t
        )

        q = solution.y[0]
        modal_responses.append(q)

    return np.array(modal_responses)


def combine_modal_responses(modal_responses):
    """
    Combines all modal responses using modal superposition.
    """

    total_response = np.sum(modal_responses, axis=0)

    return total_response


def reconstruct_displacement_at_point(modal_responses, x_position):
    """
    Reconstructs physical beam displacement at a selected point.

    For a simply-supported beam:
        w(x,t) = sum(phi_i(x) * q_i(t))
        phi_i(x) = sin(i*pi*x)

    x_position is normalised:
        x = 0     left support
        x = 0.5   midpoint
        x = 1     right support
    """
    number_of_modes = modal_responses.shape[0]

    displacement = np.zeros_like(modal_responses[0])

    for i in range(number_of_modes):
        mode_number = i + 1

        phi = np.sin(mode_number * np.pi * x_position)

        displacement += phi * modal_responses[i]

    return displacement