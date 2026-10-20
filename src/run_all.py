"""
run_all.py

Runs the project's deliverable scripts in sequence: trajectory
validation for both falls, numerical-methods comparison, piecewise
stepping, wind comparison, wind plausibility check, both Monte Carlo
runs, and the numerical/performance validation scripts (DOP853
tolerance check, fitted convergence order, terminal-point check,
sensitivity sweep, profiling comparison). Excludes the core
physics/data modules (atmosphere.py, dynamics.py, gravity.py,
wind.py, integrator.py, fixed_step_integrators.py, bunburra.py,
cavezzo.py), which are not standalone deliverables, and profiler.py
/ profile_memory.py, which are one-off manual profiling runs (the
latter needs to be invoked as `scalene profile_memory.py`, not run
directly).

Each script runs in-process via runpy, not as a subprocess, so
plt.show() behaves the same as running the file on its own (e.g.
figures still appear in Spyder's Plots pane).
"""

import runpy

SCRIPTS = [
    ("Cavezzo trajectory validation", "plot_trajectory_cavezzo.py"),
    ("Bunburra trajectory validation", "plot_trajectory_bunburra.py"),
    ("Numerical methods comparison", "plot_numerical_comparison.py"),
    ("Piecewise stepping experiment", "plot_piecewise_stepping.py"),
    ("Wind drift comparison", "plot_wind_comparison.py"),
    ("Wind plausibility check", "wind_plausibility_check.py"),
    ("Cavezzo Monte Carlo uncertainty", "monte_carlo_uncertainty_cavezzo.py"),
    ("Bunburra Monte Carlo uncertainty", "monte_carlo_uncertainty_bunburra.py"),
    ("DOP853 tolerance convergence check", "roundoff_test.py"),
    ("Fitted RK2/RK4 convergence order", "convergence_order_check.py"),
    ("Terminal-point check", "terminal_point_check.py"),
    ("Cd/sigma_abl sensitivity sweep", "sensitivity_sweep.py"),
    ("Interpolation profiling comparison", "profile_compare.py"),
]

if __name__ == "__main__":
    for label, script in SCRIPTS:
        print(f"\n{'=' * 60}\n{label} ({script})\n{'=' * 60}")
        runpy.run_path(script, run_name="__main__")