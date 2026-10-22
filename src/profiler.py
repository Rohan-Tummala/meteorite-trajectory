"""
profiler.py

Times 20 full RK45 trajectory runs, used to measure the before/after
effect of the np.interp vs interp1d swap in atmosphere.py's
tabulated_density(). Run once before applying the swap, and once
after, and compare the printed time. Uses RK45 at the same tolerance
as the project's actual production runs (see
monte_carlo_uncertainty_cavezzo.py), rather than the tight-tolerance
DOP853 reference, so this profiling matches the method actually used.
"""

import time
from dynamics import meteor_rhs, ground_impact_event, DEFAULT_PARAMS
from cavezzo import initial_state
from scipy.integrate import solve_ivp


def run():
    """Run one Cavezzo RK45 trajectory (rtol=1e-7, atol=1e-9) to ground impact and return the solve_ivp result."""
    y0 = initial_state()
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    return solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=DEFAULT_PARAMS),
        t_span=(0.0, 1500.0), y0=y0, method="RK45",
        rtol=1e-7, atol=1e-9,
        events=ground_impact_event,
    )


if __name__ == "__main__":
    t0 = time.perf_counter()
    for _ in range(20):
        run()
    t1 = time.perf_counter()
    print(f"{t1 - t0:.3f}s for 20 runs")