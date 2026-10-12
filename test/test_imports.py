"""
test_imports.py

Module-level smoke test: every source module must actually import
without error on the platform pytest is run on. This exists because
a filename-case mismatch (`from cavezzo import ...` against the
actual file `Cavezzo.py`) previously broke every script that imports
the Cavezzo case on a case-sensitive filesystem (Linux), while
working silently on a case-insensitive one (macOS/Windows). None of
the project's other tests would have caught that, since a module
that fails to import can't be exercised by any test inside it.
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
    "Cavezzo",
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