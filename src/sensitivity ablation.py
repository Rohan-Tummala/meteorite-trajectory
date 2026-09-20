"""
sensitivity_ablation.py

Sweeps the ablation coefficient sigma_abl (= CH/Q) across the range of
uncertainty found in the literature:
    - Generic CH~0.01-0.1, Q~8e6 J/kg (Chyba et al. 1993, Avramenko
      et al. 2014, Johnston et al. 2018) implies sigma_abl roughly
      1e-9 to 1e-8 kg/J.
    - Moscati et al. (2027)'s own empirically-calibrated range for
      real falls (Cavezzo included) is 7-9e-8 kg/J.

Sweeping log-spaced across ~1e-9 to ~1e-7 kg/J covers both ends of
this disagreement. For each value, the full trajectory is run to
ground impact and we record:
    - distance between predicted impact point and the real recovery
      location
    - impact velocity
    - final (impact) mass

This is the input-sensitivity-analysis component of the project: does
precise knowledge of sigma_abl actually matter for the quantity we
care about (impact location), or is the prediction robust to this
order-of-magnitude uncertainty?
"""

import numpy as np
import matplotlib.pyplot as plt

from integrator import run_trajectory, impact_state
from cavezzo import initial_state, RECOVERY_LAT, RECOVERY_LON
from plot_trajectory import great_circle_distance
from dynamics import DEFAULT_PARAMS


def run_sweep(sigma_abl_values, t_span=(0.0, 600.0)):
    """
    Run the trajectory once per sigma_abl value, holding everything
    else at DEFAULT_PARAMS.

    Returns
    -------
    dict of ndarrays, keyed by:
        'sigma_abl', 'distance_km', 'impact_velocity_ms', 'final_mass_kg'
    Runs that don't reach the ground within t_span are recorded as NaN.
    """
    y0 = initial_state()

    distances = []
    velocities = []
    masses = []

    for sigma_abl in sigma_abl_values:
        params = {**DEFAULT_PARAMS, "sigma_abl": sigma_abl}
        sol = run_trajectory(y0, t_span, method="RK45", rtol=1e-8,
                              atol=1e-10, params=params)
        y_impact = impact_state(sol)

        if y_impact is None:
            distances.append(np.nan)
            velocities.append(np.nan)
            masses.append(np.nan)
            print(f"sigma_abl={sigma_abl:.2e}: no ground impact within "
                  f"t_span — skipping.")
            continue

        h, V, gamma, phi, lam, psi, M = y_impact
        dist = great_circle_distance(phi, lam, RECOVERY_LAT, RECOVERY_LON)

        distances.append(dist / 1000.0)  # m -> km
        velocities.append(V)
        masses.append(M)

        print(f"sigma_abl={sigma_abl:.2e} kg/J  ->  "
              f"offset={dist/1000.0:6.2f} km  "
              f"V_impact={V:6.2f} m/s  M_final={M:.4f} kg")

    return {
        "sigma_abl": np.array(sigma_abl_values),
        "distance_km": np.array(distances),
        "impact_velocity_ms": np.array(velocities),
        "final_mass_kg": np.array(masses),
    }


def plot_sweep(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(results["sigma_abl"], results["distance_km"], "o-",
             color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{abl}$ (kg/J)")
    ax.set_ylabel("Distance from recovery site (km)")
    ax.set_title("Impact-location sensitivity")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(results["sigma_abl"], results["impact_velocity_ms"], "o-",
             color="tab:orange")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{abl}$ (kg/J)")
    ax.set_ylabel("Impact velocity (m/s)")
    ax.set_title("Impact-velocity sensitivity")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    ax = axes[2]
    ax.plot(results["sigma_abl"], results["final_mass_kg"], "o-",
             color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{abl}$ (kg/J)")
    ax.set_ylabel("Final mass (kg)")
    ax.set_title("Terminal-mass sensitivity")
    ax.axhline(1.5, color="gray", linestyle=":",
                label="Paper's terminal mass (1.5 kg, at h=21.5 km)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Log-spaced sweep from ~1e-9 to ~1e-7 kg/J (7 points), covering
    # both the generic-CH-derived low end and Moscati et al.'s
    # empirically-calibrated high end.
    sigma_abl_values = np.logspace(-9, -7, 7)

    results = run_sweep(sigma_abl_values)
    plot_sweep(results)