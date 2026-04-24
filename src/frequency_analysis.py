import numpy as np
from src.beam_properties import Beam, first_natural_frequency


def frequency_vs_length(
    beam: Beam,
    l_min: float,
    l_max: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    lengths_m = np.linspace(l_min, l_max, n_points)
    freqs_hz = np.array(
        [first_natural_frequency(beam, length_m) for length_m in lengths_m]
    )
    return lengths_m, freqs_hz