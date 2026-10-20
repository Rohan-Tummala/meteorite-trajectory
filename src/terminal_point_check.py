"""
terminal_point_check.py

Compares the model state at the end of the observed luminous flight
(t = DURATION from each paper) with the published terminal-point
values, for both falls. Produces the numbers in Table 2 of the report.
"""

import numpy as np
from integrator import run_trajectory
from dynamics import DEFAULT_PARAMS
import cavezzo
import bunburra


def terminal_state(fall):
    """Model [h, V, M] at t = fall.DURATION, using RK45 with dense output.

    Parameters
    ----------
    fall : module
        cavezzo or bunburra (must define initial_state, DURATION, RHO_M).

    Returns
    -------
    tuple of float
        (height in m, speed in m/s, mass in kg) at the paper's terminal time.
    """
    params = {**DEFAULT_PARAMS, "rho_m": fall.RHO_M}
    sol = run_trajectory(fall.initial_state(), (0.0, 600.0), method="RK45",
                         rtol=1e-8, atol=1e-10, params=params,
                         dense_output=True)
    h, V, _, _, _, _, M = sol.sol(fall.DURATION)
    return h, V, M


def main():
    """Print model vs. paper terminal-point values for both falls."""
    for name, fall in (("Cavezzo", cavezzo), ("Bunburra", bunburra)):
        h, V, M = terminal_state(fall)
        print(f"\n{name} (t = {fall.DURATION} s)")
        print(f"  Height  : model {h/1000:7.2f} km   paper {fall.H_TERMINAL/1000:7.2f} km")
        print(f"  Velocity: model {V:7.0f} m/s  paper {fall.V_TERMINAL:7.0f} m/s")
        print(f"  Mass    : model {M:7.2f} kg   paper {fall.M_TERMINAL:7.2f} kg")


if __name__ == "__main__":
    main()