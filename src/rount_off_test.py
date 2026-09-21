import numpy as np
import time
import warnings
from scipy.integrate import solve_ivp

from dynamics import meteor_rhs, ground_impact_event, DEFAULT_PARAMS
from cavezzo import initial_state
from plot_trajectory import great_circle_distance

y0 = initial_state()
t_span = (0.0, 600.0)
params = DEFAULT_PARAMS

def run(rtol, atol):
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    t0 = time.perf_counter()
    sol = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=t_span, y0=y0, method="DOP853",
        rtol=rtol, atol=atol, max_step=5.0,
        events=ground_impact_event,
    )
    runtime = time.perf_counter() - t0
    y_impact = sol.y_events[0][0]
    return y_impact, runtime, sol.nfev

# Sweep tolerance from loose to extremely tight (near machine epsilon)
tol_values = [1e-6, 1e-8, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 3e-15]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = []
    for rtol in tol_values:
        atol = rtol * 1e-2
        try:
            y_i, rt, nfev = run(rtol, atol)
            results.append((rtol, y_i, rt, nfev))
            print(f"rtol={rtol:.0e}  lat={np.degrees(y_i[3]):.10f}  lon={np.degrees(y_i[4]):.10f}  "
                  f"runtime={rt:.4f}s  nfev={nfev}")
        except Exception as e:
            print(f"rtol={rtol:.0e}  FAILED: {e}")

# Use the tightest successful run as the "best available" reference
ref_rtol, ref_y, _, _ = results[-1]
lat_ref, lon_ref = ref_y[3], ref_y[4]

print(f"\nUsing rtol={ref_rtol:.0e} as reference. Error of each looser run vs this reference:")
for rtol, y_i, rt, nfev in results:
    err = great_circle_distance(y_i[3], y_i[4], lat_ref, lon_ref)
    print(f"rtol={rtol:.0e}  error_vs_tightest={err:10.4f} m  runtime={rt:.4f}s  nfev={nfev}")