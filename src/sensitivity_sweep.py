"""
sensitivity_sweep.py

Parameter sensitivity sweep for both falls: perturbs Cd and sigma_abl
one at a time around their default values and measures how far the
predicted impact point shifts, to check whether coefficient
uncertainty could plausibly explain the gap between the predicted and
real recovery site (see plot_trajectory_cavezzo.py and
plot_trajectory_bunburra.py).

Only Cd and sigma_abl are swept. Both are genuinely assumed rather
than measured for either fall: Cd is a generic hypersonic blunt-body
value, and sigma_abl is fitted from a different fall entirely (Lost
City, Ceplecha et al. 1998), not Cavezzo or Bunburra. rho_m (bulk
density) and M0 (initial mass) are NOT swept here. rho_m is an actual
measured value for each fall (Cavezzo fragment F2, Gardiol et al.
2021; Bunburra's recovered fragments, Spurny et al. 2012), not an
unknown or generalized parameter. M0 is already sampled with its own
reported uncertainty as part of the entry-state Monte Carlo
(monte_carlo_uncertainty_*.py), so sweeping it here too would double
count the same uncertainty in two different checks.

Ranges used:
    - sigma_abl: the actual reported range from Ceplecha et al.
      (1998, Table VII), 1.1e-8 to 1.5e-8 kg/J.
    - Cd: no published uncertainty range exists (it is not fitted
      per fall), so it is swept +/-10% as a generic engineering
      bound rather than a sourced uncertainty. This is noted
      explicitly in the output so the two kinds of range are never
      conflated.
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