import numpy as np

def simply_supported_mode_shape(x, mode_number):
    """
    Calculates the mode shape for a simply-supported beam.

    Parameters:
        x: position along the beam, normalised from 0 to 1
        mode_number: vibration mode number, e.g. 1, 2, 3

    Returns:
        phi: mode shape value at each x
    """

    phi = np.sin(mode_number * np.pi * x)

    return phi