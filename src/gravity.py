"""
gravity.py

Gravitational acceleration model for the meteoroid trajectory
simulation.

Baseline model: spherical Earth, inverse-square falloff with altitude.
    g(h) = g0 * (R_E / (R_E + h))^2

A non-spherical (J2-corrected) model can be added later as a second
function with the same signature, following the same pattern as
atmosphere.py's exponential/tabulated dispatcher, for the gravity-model
sensitivity comparison in the project brief.
"""

import numpy as np

G0 = 9.80665       # standard gravitational acceleration at h=0, m/s^2
R_E = 6371000.0    # mean Earth radius, m


def spherical_gravity(h):
    """
    Spherical Earth gravity model: g(h) = g0 * (R_E / (R_E + h))^2.

    Parameters
    ----------
    h : float or array_like
        Altitude above sea level, in metres. Expected domain: h >= 0.

    Returns
    -------
    float or ndarray
        Gravitational acceleration in m/s^2 (magnitude, directed
        toward Earth's centre).
    """
    h = np.asarray(h, dtype=float)
    return G0 * (R_E / (R_E + h)) ** 2


def gravity(h, model="spherical"):
    """
    Select and evaluate a gravity model by name.

    Parameters
    ----------
    h : float or array_like
        Altitude above sea level, in metres.
    model : str, optional
        Which model to use. Currently only "spherical" is implemented;
        this dispatcher exists so dynamics.py can call gravity(h, model=...)
        without changes once a J2-corrected model is added. Default
        "spherical".

    Returns
    -------
    float or ndarray
        Gravitational acceleration in m/s^2.

    Raises
    ------
    ValueError
        If `model` is not a recognised option.
    """
    if model == "spherical":
        return spherical_gravity(h)
    else:
        raise ValueError(f"Unknown gravity model '{model}'. Use 'spherical'.")


if __name__ == "__main__":
    # Quick manual sanity check when running this file directly.
    test_altitudes = [0, 10000, 21500, 50000, 85000]
    for h in test_altitudes:
        print(f"h={h:>6} m  g={gravity(h):.6f} m/s^2")