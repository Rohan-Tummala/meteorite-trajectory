"""
numerical_methods_comparison.py

The core numerical-methods investigation from the project brief
(section 8): compare how much the predicted impact point changes
with integration method and step size, and at what computational
cost.

Two families compared:
    - Fixed-step RK2 (Heun) and RK4 (classical), at several step
      sizes, implemented by hand in fixed_step_integrators.py since
      scipy.solve_ivp only offers adaptive methods.
    - Adaptive RK23, RK45, DOP853 via scipy.solve_ivp, at a normal
      (not ultra-tight) tolerance.

All are compared against a "truth" reference: DOP853 at very tight
tolerance (rtol=1e-12, atol=1e-14), which should be far more accurate
than any of the methods actually being compared. The distance between
each method's impact point and the truth solver's impact point is
NUMERICAL ERROR — the physical model (Cd, sigma_abl, atmosphere, etc.)
is held IDENTICAL across every run, so any difference is purely from
the integration method/step size, not from model assumptions.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dynamics import meteor_rhs, ground_impact_event, DEFAULT_PARAMS
from cavezzo import initial_state
from fixed_step_integrators import integrate_fixed_step
from plot_trajectory import great_circle_distance


def get_impact_lat_lon(y_event):
    return y_event[3], y_event[4]  # phi, lambda (rad)


def run_truth_solver(y0, t_span, params):
    """Very tight tolerance DOP853 run — the reference solution."""
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    t0 = time.perf_counter()
    sol = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=t_span, y0=y0, method="DOP853",
        rtol=1e-12, atol=1e-14, max_step=5.0,
        events=ground_impact_event,
    )
    runtime = time.perf_counter() - t0
    y_impact = sol.y_events[0][0]
    return y_impact, runtime, sol.nfev


def run_adaptive(y0, t_span, params, method, rtol=1e-8, atol=1e-10):
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    t0 = time.perf_counter()
    sol = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=t_span, y0=y0, method=method,
        rtol=rtol, atol=atol, max_step=5.0,
        events=ground_impact_event,
    )
    runtime = time.perf_counter() - t0
    y_impact = sol.y_events[0][0]
    return y_impact, runtime, sol.nfev


def run_fixed_step(y0, t_span, params, method, dt):
    f = lambda t, y: meteor_rhs(t, y, params=params)
    result = integrate_fixed_step(f, y0, t_span[0], t_span[1], dt, method=method)
    return result["event_y"], result["runtime_s"], result["n_evals"]


def main():
    y0 = initial_state()
    t_span = (0.0, 600.0)
    params = DEFAULT_PARAMS  # identical physics for every run — isolates numerical error

    print("Running truth solver (DOP853, rtol=1e-12)...")
    y_truth, t_truth, nfev_truth = run_truth_solver(y0, t_span, params)
    lat_truth, lon_truth = get_impact_lat_lon(y_truth)
    print(f"  Truth impact: {np.degrees(lat_truth):.6f} N, {np.degrees(lon_truth):.6f} E "
          f"({t_truth:.3f} s, {nfev_truth} evals)\n")

    results = []  # list of dicts: method, dt/tol label, error_m, runtime_s, n_evals

    # --- Adaptive methods at normal tolerance ---
    for method in ["RK23", "RK45", "DOP853"]:
        y_i, rt, nfev = run_adaptive(y0, t_span, params, method)
        lat_i, lon_i = get_impact_lat_lon(y_i)
        err = great_circle_distance(lat_i, lon_i, lat_truth, lon_truth)
        results.append({"family": "adaptive", "label": method,
                          "error_m": err, "runtime_s": rt, "n_evals": nfev})
        print(f"{method:>8} (adaptive, rtol=1e-8): error={err:8.2f} m  "
              f"runtime={rt:.4f} s  evals={nfev}")

    # --- Fixed-step RK2 and RK4 at several step sizes ---
    dt_values = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    for method in ["rk2", "rk4"]:
        for dt in dt_values:
            y_i, rt, nfev = run_fixed_step(y0, t_span, params, method, dt)
            if y_i is None:
                print(f"{method.upper():>8} dt={dt:>5.2f}s: never reached ground — skipping")
                continue
            lat_i, lon_i = get_impact_lat_lon(y_i)
            err = great_circle_distance(lat_i, lon_i, lat_truth, lon_truth)
            results.append({"family": method, "label": f"{method.upper()} dt={dt}",
                              "dt": dt, "error_m": err, "runtime_s": rt, "n_evals": nfev})
            print(f"{method.upper():>8} dt={dt:>5.2f}s: error={err:8.2f} m  "
                  f"runtime={rt:.5f} s  evals={nfev}")

    return results


def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel 1: convergence (error vs step size) for RK2/RK4 ---
    ax = axes[0]
    for method, color, marker in [("rk2", "tab:blue", "o"), ("rk4", "tab:orange", "s")]:
        pts = [(r["dt"], r["error_m"]) for r in results if r["family"] == method]
        pts.sort()
        dts = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        ax.plot(dts, errs, marker=marker, color=color, label=method.upper())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step size dt (s)")
    ax.set_ylabel("Impact-location error vs. truth (m)")
    ax.set_title("Convergence: error vs. step size")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # --- Panel 2: accuracy vs. computational cost, all methods ---
    ax = axes[1]
    colors = {"adaptive": "tab:green", "rk2": "tab:blue", "rk4": "tab:orange"}
    for r in results:
        ax.scatter(r["runtime_s"], r["error_m"], color=colors[r["family"]])
        ax.annotate(r["label"], (r["runtime_s"], r["error_m"]),
                     fontsize=6, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Impact-location error vs. truth (m)")
    ax.set_title("Accuracy vs. computational cost")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = main()
    plot_results(results)