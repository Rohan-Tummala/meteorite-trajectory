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
                              atol=1e-10, params=params, max_step=5.0)
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


def plot_sweep(results, highlight_value=None):
    """
    Same as before, but optionally highlights one specific sigma_abl
    value (e.g. the project's baseline) with a distinct marker and
    vertical guide line on each panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        (axes[0], "distance_km", "tab:blue", "Distance from recovery site (km)",
         "Impact-location sensitivity"),
        (axes[1], "impact_velocity_ms", "tab:orange", "Impact velocity (m/s)",
         "Impact-velocity sensitivity"),
        (axes[2], "final_mass_kg", "tab:green", "Final mass (kg)",
         "Terminal-mass sensitivity"),
    ]

    highlight_idx = None
    if highlight_value is not None:
        highlight_idx = int(np.argmin(np.abs(results["sigma_abl"] - highlight_value)))

    for ax, key, color, ylabel, title in panels:
        ax.plot(results["sigma_abl"], results[key], "o-", color=color)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_{abl}$ (kg/J)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        if highlight_idx is not None:
            x_h = results["sigma_abl"][highlight_idx]
            y_h = results[key][highlight_idx]
            ax.plot(x_h, y_h, "*", color="red", markersize=18,
                     markeredgecolor="black", zorder=5,
                     label=f"Baseline ({x_h:.2e} kg/J)")
            ax.axvline(x_h, color="red", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8)

    axes[2].axhline(1.5, color="gray", linestyle=":",
                      label="Paper's terminal mass (1.5 kg, at h=21.5 km)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Log-spaced sweep from ~1e-9 to ~1e-7 kg/J (20 points), covering
    # both the generic-CH-derived low end and Moscati et al.'s
    # empirically-calibrated high end. Dense enough to resolve the
    # location-minimizing value precisely (see project notes).
    BASELINE_SIGMA_ABL = DEFAULT_PARAMS["sigma_abl"]  # 8.0e-8, the project's default

    sigma_abl_values = np.logspace(-9, -7, 20)
    # Insert the exact baseline value so it appears as a real,
    # labeled point on the curve rather than something interpolated.
    sigma_abl_values = np.sort(np.append(sigma_abl_values, BASELINE_SIGMA_ABL))

    results = run_sweep(sigma_abl_values)
    plot_sweep(results, highlight_value=BASELINE_SIGMA_ABL)