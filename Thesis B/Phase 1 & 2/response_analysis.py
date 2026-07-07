import numpy as np
from scipy.integrate import solve_ivp


def sdof_response(m, c, k, force_function, t_span, initial_conditions):
    """
    Solves a single degree of freedom vibration system.

    m*x_ddot + c*x_dot + k*x = F(t)
    """

    def equation_of_motion(t, y):
        x = y[0]
        v = y[1]

        F = force_function(t)

        a = (F - c * v - k * x) / m

        return [v, a]

    t_eval = np.linspace(t_span[0], t_span[1], 2000)

    solution = solve_ivp(
        equation_of_motion,
        t_span,
        initial_conditions,
        t_eval=t_eval
    )

    return solution.t, solution.y[0], solution.y[1]