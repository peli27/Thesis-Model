def get_beam_properties():
    """
    Returns the baseline simply supported Euler-Bernoulli beam properties.
    """
    
    beam = {
        "E": 206e9,
        "rho": 7850,
        "L": 2.0,
        "b": 0.04,
        "h": 0.01
    }

    beam["A"] = beam["b"] * beam["h"]
    beam["I"] = beam["b"] * beam["h"]**3 / 12

    return beam

