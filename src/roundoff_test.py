"""
roundoff_test.py

Checks that DOP853 has converged by the tolerance used as the "truth"
reference solution elsewhere in this project (rtol=1e-10, see
plot_numerical_comparison.py and plot_piecewise_stepping.py), rather
than assuming it without evidence.

Tightens rtol in steps past 1e-10, down to 1e-12, and measures how far
the predicted Cavezzo impact point moves at each step relative to the
previous, tighter run. If the shift has flattened out to a negligible
distance well before 1e-12, that confirms 1e-10 was already past the
point of diminishing returns, i.e. the practical accuracy floor for
this problem, and tightening further only adds cost, not accuracy.
"""

import numpy as np
from integrator import run_trajectory, impact_state
from dynamics import DEFAULT_PARAMS
from cavezzo import initial_state
from plot_trajectory_cavezzo import great_circle_distance

RTOLS = [1e-8, 1e-9, 1e-10, 1e-11, 1e-12]


def run_impact(y0, t_span, rtol):
    """Integrate with DOP853 at the given rtol (atol = rtol * 1e-2) and
    return the impact (lat, lon) in radians."""
    sol = run_trajectory(y0, t_span, method="DOP853", rtol=rtol,
                          atol=rtol * 1e-2, dense_output=True,
                          params=DEFAULT_PARAMS)
    y_impact = impact_state(sol)
    return y_impact[3], y_impact[4]


def main():
    """Run DOP853 at a sequence of tightening tolerances, print each
    run's impact point, and print the shift in impact point relative
    to the previous (tighter) run, to show where convergence stops
    producing a meaningful change."""
    y0 = initial_state()
    t_span = (0.0, 600.0)

    results = []
    for rtol in RTOLS:
        lat, lon = run_impact(y0, t_span, rtol)
        results.append((rtol, lat, lon))
        print(f"rtol={rtol:.0e}: impact at "
              f"{np.degrees(lat):.8f} N, {np.degrees(lon):.8f} E")

    print("\nShift relative to the next-tighter run:")
    for i in range(len(results) - 1):
        rtol_a, lat_a, lon_a = results[i]
        rtol_b, lat_b, lon_b = results[i + 1]
        shift_m = great_circle_distance(lat_a, lon_a, lat_b, lon_b)
        print(f"  rtol={rtol_a:.0e} -> rtol={rtol_b:.0e}: "
              f"shift = {shift_m:.3e} m")


if __name__ == "__main__":
    main()