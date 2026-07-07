import numpy as np


def natural_frequencies(beam, modes=4):
    """
    Calculates the natural frequencies of a simply supported Euler-Bernoulli beam.
    """
    
    E = beam["E"]
    rho = beam["rho"]
    L = beam["L"]
    A = beam["A"]
    I = beam["I"]

    n = np.arange(1, modes + 1)

    frequencies = ((n * np.pi) ** 2 / (2 * np.pi * L**2)) * np.sqrt(E * I / (rho * A))

    return frequencies