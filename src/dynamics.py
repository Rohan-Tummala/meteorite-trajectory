"""
dynamics.py

Right-hand-side (RHS) function for the meteoroid atmospheric entry
trajectory model. This is the function that gets handed to an ODE
solver (see integrator.py) to advance the state vector forward in
time.

State vector (7 elements):
    y = [h, V, gamma, phi, lambda, psi, M]

    h      altitude, m
    V      speed, m/s
    gamma  flight-path angle, rad (below horizontal, positive = descending)
    phi    latitude, rad
    lambda longitude, rad
    psi    heading/azimuth, rad
    M      remaining mass, kg

Equations of motion follow the project brief's simplified 3D
point-mass meteor model with drag and ablation.
"""

import numpy as np
from atmosphere import density
from gravity import gravity
from wind import wind_components

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

R_E = 6371000.0  # mean Earth radius, m (kept consistent with gravity.py)


# ---------------------------------------------------------------------
# Physical/model parameters
# ---------------------------------------------------------------------
# These are NOT measured quantities — they are model assumptions/
# coefficients that must be sourced or justified separately (see
# project brief, section 3). Defaults below are placeholders; replace
# with values justified for your chosen meteor event (e.g. Cavezzo).

DEFAULT_PARAMS = {
    "Cd": 1.16,        # drag coefficient, dimensionless — Mach > 4 plateau
                        # value from the Ceplecha (1987) Cd(M) curve, as
                        # reported in Moscati et al. (2027, Icarus 461,
                        # 117303, Fig. 1). Cavezzo's speed (4-12.2 km/s)
                        # stays well above Mach 4 throughout the luminous
                        # flight, so this constant is a reasonable
                        # simplification of the full Mach-dependent curve.
    "sigma_abl": 8.0e-8,  # ablation coefficient CH/Q, kg/J — midpoint of
                        # the 7-9e-8 kg/J range empirically calibrated by
                        # Moscati et al. (2027) against several real falls,
                        # Cavezzo explicitly included among them.
    "rho_m": 3322.0,   # meteoroid bulk density, kg/m^3 — measured value
                        # for Cavezzo fragment F2 (Gardiol et al. 2021)
    "atmosphere_model": "table",  # "exp" or "table" — see atmosphere.py
    "gravity_model": "spherical",  # see gravity.py
    "wind_model": "none",  # "none" or "cavezzo" — see wind.py. Simplified
                            # horizontal-advection drift, only meaningful
                            # once the meteor is slow (dark flight).
}


def radius_from_mass(M, rho_m):
    """
    Meteoroid radius assuming a sphere of uniform density rho_m.

    r = (3M / (4 * pi * rho_m))^(1/3)

    Parameters
    ----------
    M : float
        Remaining mass, kg. Must be positive.
    rho_m : float
        Meteoroid bulk density, kg/m^3.

    Returns
    -------
    float
        Radius, m.
    """
    M = max(M, 0.0)  # guard against tiny negative mass from numerical overshoot
    return (3.0 * M / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)


def meteor_rhs(t, y, params=None):
    """
    Right-hand side of the meteoroid trajectory ODE system.

    Parameters
    ----------
    t : float
        Time, s. Unused directly (the system is autonomous / not
        explicitly time-dependent) but required by the solve_ivp
        function signature.
    y : array_like, shape (7,)
        Current state [h, V, gamma, phi, lambda, psi, M].
    params : dict, optional
        Physical/model parameters. Missing keys fall back to
        DEFAULT_PARAMS. See DEFAULT_PARAMS for expected keys.

    Returns
    -------
    ndarray, shape (7,)
        Time derivatives [h_dot, V_dot, gamma_dot, phi_dot,
        lambda_dot, psi_dot, M_dot].
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    h, V, gamma, phi, lam, psi, M = y

    # Once mass is (numerically) exhausted, freeze ablation and drag
    # area growth rather than letting radius_from_mass blow up or go
    # complex on a negative mass from solver overshoot.
    M_safe = max(M, 1e-6)

    rho = density(h, model=p["atmosphere_model"])
    g = gravity(h, model=p["gravity_model"])

    r = radius_from_mass(M_safe, p["rho_m"])
    A = np.pi * r ** 2

    # Guard against V=0 causing a divide-by-zero in gamma_dot.
    V_safe = max(V, 1e-3)

    h_dot = V * np.sin(gamma)

    V_dot = -(rho * p["Cd"] * A * V ** 2) / (2.0 * M_safe) - g * np.sin(gamma)

    gamma_dot = np.cos(gamma) * (V_safe / (R_E + h) - g / V_safe)

    phi_dot = (V * np.cos(gamma) * np.cos(psi)) / (R_E + h)

    lambda_dot = (V * np.cos(gamma) * np.sin(psi)) / ((R_E + h) * np.cos(phi))

    # Optional wind drift (simplified horizontal advection, see wind.py).
    if p["wind_model"] == "cavezzo":
        W_north, W_east = wind_components(h)
        phi_dot += W_north / (R_E + h)
        lambda_dot += W_east / ((R_E + h) * np.cos(phi))

    psi_dot = 0.0

    M_dot = -(p["sigma_abl"] * A * rho * V ** 3) / 2.0
    # Ablation stops once the meteor transitions to "dark flight" — the
    # velocity threshold below which luminous ablation becomes negligible
    # (Moilanen et al. 2021, adopted by Moscati et al. 2027 as a standard
    # operational criterion), rather than only guarding against M <= 0.
    if V < 3000.0:
        M_dot = 0.0
    elif M <= 0.0 and M_dot < 0.0:
        M_dot = 0.0

    return np.array([h_dot, V_dot, gamma_dot, phi_dot, lambda_dot, psi_dot, M_dot])


def ground_impact_event(t, y, params=None):
    """
    Event function for solve_ivp: triggers when altitude reaches zero
    (ground impact), so the solver can stop integrating there instead
    of continuing to negative altitude.

    Pass to solve_ivp as an event with terminal=True and direction=-1,
    e.g.:
        ground_impact_event.terminal = True
        ground_impact_event.direction = -1
        solve_ivp(meteor_rhs, t_span, y0, events=ground_impact_event, ...)
    """
    return y[0]  # h


ground_impact_event.terminal = True
ground_impact_event.direction = -1


if __name__ == "__main__":
    # Quick manual sanity check: one RHS evaluation at a plausible
    # mid-flight state, using default parameters.
    y_test = np.array([
        30000.0,             # h, m
        10000.0,             # V, m/s
        np.radians(-68.0),   # gamma, rad (descending steeply)
        np.radians(44.0),    # phi, rad
        np.radians(11.0),    # lambda, rad
        np.radians(238.0),   # psi, rad
        1.0,                  # M, kg
    ])
    dy = meteor_rhs(0.0, y_test)
    labels = ["h_dot", "V_dot", "gamma_dot", "phi_dot", "lambda_dot", "psi_dot", "M_dot"]
    for label, val in zip(labels, dy):
        print(f"{label:>10} = {val: .6e}")