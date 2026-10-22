"""
sensitivity_sweep.py

One-at-a-time sensitivity sweep of Cd and sigma_abl for both falls.
Each is perturbed around its default and the shift in the predicted
impact point is reported, to check whether coefficient uncertainty
could explain the gap to the real recovery site.

Only Cd and sigma_abl are swept, since both are assumed rather than
measured for each fall. rho_m is a measured value, and M0 is already
sampled in the Monte Carlo (monte_carlo_uncertainty_*.py).

Ranges:
    - sigma_abl: 1.1e-8 to 1.5e-8 kg/J, the range reported by Ceplecha
      et al. (1998, Table VII).
    - Cd: +/-10% of the default, a generic engineering bound since no
      published range exists.
"""

import numpy as np
from integrator import run_trajectory, impact_state
from dynamics import DEFAULT_PARAMS
import cavezzo
import bunburra
from plot_trajectory_cavezzo import great_circle_distance

FALLS = {
    "Cavezzo": cavezzo,
    "Bunburra": bunburra,
}


def run_impact(y0, t_span, params):
    """Integrate to impact and return the impact (lat, lon) in radians."""
    sol = run_trajectory(y0, t_span, method="RK45", rtol=1e-8, atol=1e-10,
                          dense_output=True, params=params)
    y_impact = impact_state(sol)
    return y_impact[3], y_impact[4]


def run_sweep_for_fall(fall_module):
    """Run the one-at-a-time sensitivity sweep on Cd and sigma_abl for
    a single fall module (cavezzo or bunburra), print each parameter's
    impact-point shift relative to the baseline run, and rank them by
    maximum shift."""
    y0 = fall_module.initial_state()
    t_span = (0.0, 600.0)
    base_params = {**DEFAULT_PARAMS, "rho_m": fall_module.RHO_M}

    lat_base, lon_base = run_impact(y0, t_span, base_params)
    print(f"Baseline impact: {np.degrees(lat_base):.6f} N, "
          f"{np.degrees(lon_base):.6f} E")

    sweeps = {
        "Cd": {
            "base": base_params["Cd"],
            "low": base_params["Cd"] * 0.9,
            "high": base_params["Cd"] * 1.1,
            "note": "+/-10% (no published range; engineering bound)",
        },
        "sigma_abl": {
            "base": base_params["sigma_abl"],
            "low": 1.1e-8,
            "high": 1.5e-8,
            "note": "Ceplecha et al. (1998, Table VII) reported range",
        },
    }

    results = {}
    for name, s in sweeps.items():
        shifts = {}
        for label in ("low", "high"):
            p = {**base_params, name: s[label]}
            lat, lon = run_impact(y0, t_span, p)
            shifts[label] = great_circle_distance(
                lat_base, lon_base, lat, lon
            ) / 1000.0  # km
        results[name] = shifts
        print(f"\n{name} (base={s['base']:.4g}, {s['note']}):")
        print(f"  low  ({s['low']:.4g}): shift = {shifts['low']:.3f} km")
        print(f"  high ({s['high']:.4g}): shift = {shifts['high']:.3f} km")

    print("\n=== Ranked by max shift ===")
    ranked = sorted(results.items(), key=lambda kv: max(kv[1].values()),
                     reverse=True)
    for name, shifts in ranked:
        print(f"{name}: max shift = {max(shifts.values()):.3f} km")

    return results


def main():
    """Run the sweep for both falls in turn."""
    for fall_name, fall_module in FALLS.items():
        print(f"\n{'=' * 60}\n{fall_name}\n{'=' * 60}")
        run_sweep_for_fall(fall_module)


if __name__ == "__main__":
    main()