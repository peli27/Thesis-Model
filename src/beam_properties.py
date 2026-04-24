from dataclasses import dataclass
import numpy as np


@dataclass
class Beam:
    youngs_modulus_pa: float
    density_kg_m3: float
    width_m: float
    thickness_m: float

    @property
    def area_m2(self) -> float:
        return self.width_m * self.thickness_m

    @property
    def second_moment_m4(self) -> float:
        return self.width_m * self.thickness_m**3 / 12.0


def get_default_beam() -> Beam:
    return Beam(
        youngs_modulus_pa=200e9,   # Pa
        density_kg_m3=7850.0,      # kg/m^3
        width_m=0.02,              # m
        thickness_m=0.003,         # m
    )


def first_natural_frequency(beam: Beam, length_m: float) -> float:
    """
    First bending natural frequency of a cantilever beam
    from Euler-Bernoulli beam theory.
    """
    beta1 = 1.875104068711961
    e = beam.youngs_modulus_pa
    i = beam.second_moment_m4
    rho = beam.density_kg_m3
    a = beam.area_m2

    return (beta1**2 / (2.0 * np.pi * length_m**2)) * np.sqrt(e * i / (rho * a))