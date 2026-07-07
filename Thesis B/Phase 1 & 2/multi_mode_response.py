import numpy as np
from scipy.integrate import solve_ivp


def modal_shape_at_point(mode_number, x_position):
    """
    Returns the simply supported mode shape value at a selected point.

    Parameters
    ----------
    mode_number : int
        Mode number, starting from 1.
    x_position : float
        Normalised beam position x/L.
        0 = left support, 0.5 = midpoint, 1 = right support.

    Returns
    -------
    float
        Mode shape value phi_n(x).
    """
    return np.sin(mode_number * np.pi * x_position)


def modal_mass(beam):
    """
    Calculates the modal mass for unnormalised sine modes.

    For a simply supported beam with phi_n(x) = sin(n*pi*x/L):
        M_n = integral_0^L rho*A*phi_n^2 dx = rho*A*L/2

    This is the same for every mode when the cross-section is uniform.
    """
    rho = beam["rho"]
    A = beam["A"]
    L = beam["L"]

    return rho * A * L / 2


def uniform_distributed_modal_forces(force_amplitude, beam, number_of_modes):
    """
    Projects a uniform harmonic distributed load onto the sine modes.

    The physical load is assumed to be:
        p(x, t) = p0 sin(omega_f t)

    where p0 is the distributed load amplitude in N/m.

    The modal force amplitude is:
        Q_n = integral_0^L p0 * sin(n*pi*x/L) dx

    For a uniform load, even modes receive zero modal force because of symmetry, while odd modes are excited.

    Parameters
    ----------
    force_amplitude : float
        Distributed load amplitude p0 in N/m.
    beam : dict
        Beam property dictionary containing L.
    number_of_modes : int
        Number of modes used in the modal expansion.

    Returns
    -------
    np.ndarray
        Generalised modal force amplitudes Q_n.
    """
    L = beam["L"]
    modal_forces = []

    for mode_number in range(1, number_of_modes + 1):
        coefficient = 1 - np.cos(mode_number * np.pi)
        Q_n = force_amplitude * L * coefficient / (mode_number * np.pi)
        modal_forces.append(Q_n)

    return np.array(modal_forces)


def reconstruct_response_at_point(modal_responses, x_position):
    """
    Reconstructs a physical beam response at a selected point.

    This can be used for either displacement or velocity:
        w(x,t)     = sum phi_n(x) * q_n(t)
        w_dot(x,t) = sum phi_n(x) * q_dot_n(t)

    Parameters
    ----------
    modal_responses : np.ndarray
        Array with shape (number_of_modes, number_of_time_steps).
    x_position : float
        Normalised beam position x/L.

    Returns
    -------
    np.ndarray
        Reconstructed physical response at the selected beam location.
    """
    modal_responses = np.asarray(modal_responses)
    number_of_modes = modal_responses.shape[0]

    response = np.zeros_like(modal_responses[0])

    for i in range(number_of_modes):
        mode_number = i + 1
        phi = modal_shape_at_point(mode_number, x_position)
        response += phi * modal_responses[i]

    return response


def reconstruct_displacement_at_point(modal_displacements, x_position):
    """
    Reconstructs beam displacement at a selected normalised position x/L.
    """
    return reconstruct_response_at_point(modal_displacements, x_position)


def reconstruct_velocity_at_point(modal_velocities, x_position):
    """
    Reconstructs beam velocity at a selected normalised position x/L.
    """
    return reconstruct_response_at_point(modal_velocities, x_position)


## ------ Validation response cases ------
def undamped_free_response(frequencies, amplitudes, t, return_velocity=False):
    """
    Case 1: undamped free vibration.

    q_n(t) = A_n cos(omega_n t)

    Checks that the model produces sustained oscillation when damping and external forcing are absent.
    """
    modal_displacements = []
    modal_velocities = []

    for f, A in zip(frequencies, amplitudes):
        omega_n = 2 * np.pi * f

        q = A * np.cos(omega_n * t)
        q_dot = -A * omega_n * np.sin(omega_n * t)

        modal_displacements.append(q)
        modal_velocities.append(q_dot)

    modal_displacements = np.array(modal_displacements)
    modal_velocities = np.array(modal_velocities)

    if return_velocity:
        return modal_displacements, modal_velocities

    return modal_displacements


def damped_free_response(frequencies, amplitudes, damping_ratio, t, return_velocity=False):
    """
    Case 2: damped free vibration.

    q_n(t) = A_n exp(-zeta*omega_n*t) cos(omega_d*t)

    Checks that the model produces a decaying response when damping is present.
    """
    modal_displacements = []
    modal_velocities = []

    for f, A in zip(frequencies, amplitudes):
        omega_n = 2 * np.pi * f
        omega_d = omega_n * np.sqrt(1 - damping_ratio**2)

        exponential_decay = np.exp(-damping_ratio * omega_n * t)

        q = A * exponential_decay * np.cos(omega_d * t)

        q_dot = A * exponential_decay * (
            -damping_ratio * omega_n * np.cos(omega_d * t)
            - omega_d * np.sin(omega_d * t)
        )

        modal_displacements.append(q)
        modal_velocities.append(q_dot)

    modal_displacements = np.array(modal_displacements)
    modal_velocities = np.array(modal_velocities)

    if return_velocity:
        return modal_displacements, modal_velocities

    return modal_displacements


## Main response case: damped harmonic excitation
def damped_harmonic_response(
    frequencies,
    force_amplitudes,
    damping_ratio,
    t,
    forcing_frequency,
    beam=None,
    initial_conditions=None,
    return_velocity=False,
):
    """
    Case 3: damped harmonic multi-mode response.

    Modal equation:
        q_n_ddot + 2*zeta*omega_n*q_n_dot + omega_n^2*q_n
        = modal_force_term_n * sin(omega_f*t)

    Parameters
    ----------
    frequencies : array-like
        Natural frequencies in Hz.
    force_amplitudes : float or array-like
        If beam is provided and this is a scalar: uniform load amplitude in N/m.
        If beam is not provided: modal forcing terms used directly.
    damping_ratio : float
        Modal damping ratio.
    t : np.ndarray
        Time array.
    forcing_frequency : float
        Harmonic excitation frequency in Hz.
    beam : dict, optional
        Beam property dictionary. Required for physically projected uniform load.
    initial_conditions : array-like, optional
        Either [q0, qdot0] for all modes, or an array of shape (modes, 2).
    return_velocity : bool
        If True, returns both modal displacement and modal velocity arrays.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Modal displacement histories, and optionally modal velocity histories.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    number_of_modes = len(frequencies)

    omega_f = 2 * np.pi * forcing_frequency

    # Convert the forcing input into a mass-normalised modal force term.
    if beam is not None:
        M_n = modal_mass(beam)

        if np.isscalar(force_amplitudes):
            modal_forces = uniform_distributed_modal_forces(
                force_amplitude=force_amplitudes,
                beam=beam,
                number_of_modes=number_of_modes,
            )
        else:
            modal_forces = np.asarray(force_amplitudes, dtype=float)

        modal_force_terms = modal_forces / M_n
    else:
        modal_force_terms = np.asarray(force_amplitudes, dtype=float)

    if modal_force_terms.size != number_of_modes:
        raise ValueError(
            "force_amplitudes must either be a scalar with beam properties "
            "or contain one value per vibration mode."
        )

    # Set initial conditions for each mode.
    if initial_conditions is None:
        initial_conditions = np.zeros((number_of_modes, 2))
    else:
        initial_conditions = np.asarray(initial_conditions, dtype=float)

        if initial_conditions.shape == (2,):
            initial_conditions = np.tile(initial_conditions, (number_of_modes, 1))

        if initial_conditions.shape != (number_of_modes, 2):
            raise ValueError(
                "initial_conditions must be [q0, qdot0] or have shape "
                "(number_of_modes, 2)."
            )

    modal_displacements = []
    modal_velocities = []

    for i, f in enumerate(frequencies):
        omega_n = 2 * np.pi * f
        modal_force_term = modal_force_terms[i]
        y0 = initial_conditions[i]

        def modal_ode(ti, y):
            q = y[0]
            q_dot = y[1]

            q_ddot = (
                modal_force_term * np.sin(omega_f * ti)
                - 2 * damping_ratio * omega_n * q_dot
                - omega_n**2 * q
            )

            return [q_dot, q_ddot]

        solution = solve_ivp(
            modal_ode,
            [t[0], t[-1]],
            y0,
            t_eval=t,
            rtol=1e-7,
            atol=1e-9,
        )

        if not solution.success:
            raise RuntimeError(
                f"Modal response solver failed for mode {i + 1}: "
                f"{solution.message}"
            )

        modal_displacements.append(solution.y[0])
        modal_velocities.append(solution.y[1])

    modal_displacements = np.array(modal_displacements)
    modal_velocities = np.array(modal_velocities)

    if return_velocity:
        return modal_displacements, modal_velocities

    return modal_displacements


def simulate_harmonic_beam_response(
    beam,
    frequencies,
    damping_ratio,
    force_amplitude,
    forcing_frequency,
    t,
    x_position=0.65,
    number_of_modes=None,
):
    """
    Simulates the beam response at a selected observation point.
    This is the function for Phase 2 dataset generation.

    Parameters
    ----------
    beam : dict
        Beam property dictionary containing E, rho, L, A and I.
    frequencies : array-like
        Natural frequencies in Hz.
    damping_ratio : float
        Modal damping ratio.
    force_amplitude : float
        Uniform distributed harmonic load amplitude in N/m.
    forcing_frequency : float
        Excitation frequency in Hz.
    t : np.ndarray
        Time array.
    x_position : float
        Normalised observation location x/L. Default is 0.65 to match the
        base-paper-inspired setup.
    number_of_modes : int, optional
        Number of modes to use. If omitted, all supplied frequencies are used.

    Returns
    -------
    dict
        Dictionary containing modal displacement, modal velocity, reconstructed
        displacement, reconstructed velocity, peak displacement and RMS displacement.
    """
    frequencies = np.asarray(frequencies, dtype=float)

    if number_of_modes is not None:
        frequencies = frequencies[:number_of_modes]

    modal_displacements, modal_velocities = damped_harmonic_response(
        frequencies=frequencies,
        force_amplitudes=force_amplitude,
        damping_ratio=damping_ratio,
        t=t,
        forcing_frequency=forcing_frequency,
        beam=beam,
        return_velocity=True,
    )

    displacement = reconstruct_displacement_at_point(
        modal_displacements,
        x_position,
    )

    velocity = reconstruct_velocity_at_point(
        modal_velocities,
        x_position,
    )

    peak_displacement = np.max(np.abs(displacement))
    rms_displacement = np.sqrt(np.mean(displacement**2))

    return {
        "modal_displacements": modal_displacements,
        "modal_velocities": modal_velocities,
        "displacement": displacement,
        "velocity": velocity,
        "peak_displacement": peak_displacement,
        "rms_displacement": rms_displacement,
        "x_position": x_position,
    }