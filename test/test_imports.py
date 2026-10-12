"""
test_imports.py

Module-level smoke test: every source module must import cleanly on
the platform pytest is run on. A module that fails to import cannot
be exercised by any other test in the suite, so this check runs
independently of, and before, the tests that exercise each module's
actual functions.
"""

import importlib

import matplotlib
matplotlib.use("Agg")  # headless backend, so plotting modules import
                        # cleanly in a CI/terminal environment with no
                        # display

import pytest

MODULES = [
    "atmosphere",
    "bunburra",
    "cavezzo",
    "dynamics",
    "fixed_step_integrators",
    "gravity",
    "integrator",
    "monte_carlo_uncertainty_bunburra",
    "monte_carlo_uncertainty_cavezzo",
    "numerical_methods_comparison",
    "plot_numerical_comparison",
    "plot_piecewise_stepping",
    "plot_trajectory",
    "plot_trajectory_bunburra",
    "plot_wind_comparison",
    "run_all",
    "wind",
    "wind_plausibility_check",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    """Every project module should import without raising."""
    importlib.import_module(module_name)