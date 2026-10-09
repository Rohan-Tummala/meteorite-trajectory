"""
run_all.py

Runs the project's deliverable scripts in sequence (trajectory
validation for both falls, numerical-methods comparison, piecewise
stepping, wind comparison, and both Monte Carlo runs). Excludes
sensitivity_ablation.py and the core physics/data modules, which
aren't standalone deliverables.

Each script runs in-process via runpy, not as a subprocess, so
plt.show() behaves the same as running the file on its own (e.g.
figures still appear in Spyder's Plots pane).
"""

import runpy

SCRIPTS = [
    ("Cavezzo trajectory validation", "plot_trajectory.py"),
    ("Bunburra trajectory validation", "plot_trajectory_bunburra.py"),
    ("Numerical methods comparison", "plot_numerical_comparison.py"),
    ("Piecewise stepping experiment", "plot_piecewise_stepping.py"),
    ("Wind drift comparison", "plot_wind_comparison.py"),
    ("Cavezzo Monte Carlo uncertainty", "monte_carlo_uncertainty.py"),
    ("Bunburra Monte Carlo uncertainty", "monte_carlo_uncertainty_bunburra.py"),
]

if __name__ == "__main__":
    for label, script in SCRIPTS:
        print(f"\n{'=' * 60}\n{label} ({script})\n{'=' * 60}")
        runpy.run_path(script, run_name="__main__")