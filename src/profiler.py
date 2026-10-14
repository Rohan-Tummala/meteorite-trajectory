"""
profiler.py

Times 20 full tight-tolerance DOP853 trajectory runs, used to measure
the before/after effect of the np.interp vs interp1d swap in
atmosphere.py's tabulated_density(). Run once before applying the
swap, and once after, and compare the printed time.
"""

import time
from dynamics import meteor_rhs, ground_impact_event, DEFAULT_PARAMS
from cavezzo import initial_state
from scipy.integrate import solve_ivp


def run():
    y0 = initial_state()
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    return solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=DEFAULT_PARAMS),
        t_span=(0.0, 600.0), y0=y0, method="DOP853",
        rtol=1e-12, atol=1e-14, max_step=5.0,
        events=ground_impact_event,
    )


if __name__ == "__main__":
    t0 = time.perf_counter()
    for _ in range(20):
        run()
    t1 = time.perf_counter()
    print(f"{t1 - t0:.3f}s for 20 runs")