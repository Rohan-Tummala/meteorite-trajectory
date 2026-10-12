"""
test_integrator.py

Unit tests for integrator.py: the scipy-based adaptive-method wrapper
and the ground-impact event extraction.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from integrator import run_trajectory, impact_state
from dynamics import meteor_rhs


def _simple_falling_state():
    """A near-vertical descent from a modest altitude, chosen so the
    trajectory reaches the ground quickly under the real meteor_rhs
    physics, keeping the test fast."""
    return np.array([
        5000.0,               # h, m
        50.0,                  # V, m/s (slow -- below the ablation
                                 # threshold, so mass stays constant)
        np.radians(-89.0),    # gamma, rad (nearly straight down)
        np.radians(0.0),      # phi, rad
        np.radians(0.0),      # lambda, rad
        np.radians(0.0),      # psi, rad
        1.0,                    # M, kg
    ])


def test_run_trajectory_stops_at_ground_impact():
    """The integration must stop at the ground-impact event, not
    continue on to negative altitude."""
    y0 = _simple_falling_state()
    result = run_trajectory(y0, (0.0, 500.0), method="RK45",
                              rtol=1e-8, atol=1e-10)
    assert result.success
    assert len(result.t_events[0]) == 1
    # The event root-finder locates the crossing to within its own
    # numerical tolerance, not exactly, so the final altitude can be
    # a tiny negative number (floating-point noise) rather than
    # exactly 0.0.
    assert result.y[0, -1] == pytest.approx(0.0, abs=1e-6)


def test_run_trajectory_respects_max_step():
    """With a small max_step, the solver should take many more
    internal steps than with a large one, over the same span."""
    y0 = _simple_falling_state()
    result_coarse = run_trajectory(y0, (0.0, 500.0), method="RK45",
                                      rtol=1e-6, atol=1e-8, max_step=50.0)
    result_fine = run_trajectory(y0, (0.0, 500.0), method="RK45",
                                   rtol=1e-6, atol=1e-8, max_step=0.5)
    assert len(result_fine.t) > len(result_coarse.t)


def test_run_trajectory_dense_output_available_when_requested():
    y0 = _simple_falling_state()
    result = run_trajectory(y0, (0.0, 500.0), dense_output=True)
    assert result.sol is not None
    # The interpolant should reproduce the initial state at t=0.
    assert np.allclose(result.sol(0.0), y0)


def test_impact_state_returns_state_when_event_fired():
    y0 = _simple_falling_state()
    result = run_trajectory(y0, (0.0, 500.0))
    y_impact = impact_state(result)
    assert y_impact is not None
    assert y_impact[0] == pytest.approx(0.0, abs=1e-6)


def test_impact_state_returns_none_when_never_reaches_ground():
    """A t_span too short to reach the ground should leave the event
    unfired, so impact_state() must return None rather than raise or
    return a mid-flight state."""
    y0 = _simple_falling_state()
    result = run_trajectory(y0, (0.0, 0.01))  # far too short to land
    assert impact_state(result) is None


# ---------------------------------------------------------------------
# Accuracy of the underlying adaptive method against a known solution
# ---------------------------------------------------------------------
#
# run_trajectory() is hardcoded to meteor_rhs, so its accuracy cannot be
# checked directly against a simple closed-form problem the way the
# hand-written fixed-step methods are in test_fixed_step_integrators.py.
# This test instead validates the same adaptive engine run_trajectory
# relies on -- scipy.integrate.solve_ivp with method="RK45" -- against
# the exact solution of dy/dt = -y, y(0) = 1, y(t) = exp(-t), closing
# the same known-solution validation gap for the adaptive method that
# the fixed-step methods already have.

def _decay_rhs(t, y):
    """dy/dt = -y, exact solution y(t) = y0 * exp(-t)."""
    return -y


def test_rk45_matches_exact_solution_on_decay_problem():
    """At tight tolerances, RK45 (the default method used throughout
    run_trajectory) should reproduce the exact solution of a simple
    decay problem to well within its own requested tolerance."""
    y0 = np.array([1.0])
    t_span = (0.0, 5.0)
    rtol, atol = 1e-10, 1e-12

    result = solve_ivp(_decay_rhs, t_span, y0, method="RK45",
                        rtol=rtol, atol=atol, dense_output=True)

    assert result.success
    t_check = np.linspace(0.0, 5.0, 50)
    y_numeric = result.sol(t_check)[0]
    y_exact = np.exp(-t_check)
    assert np.allclose(y_numeric, y_exact, rtol=1e-6, atol=1e-8)