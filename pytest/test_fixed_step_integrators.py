"""
test_fixed_step_integrators.py

Unit tests for fixed_step_integrators.py: the hand-written RK2, RK4,
implicit Euler, and implicit Heun steppers, since scipy provides none
of these as fixed-step methods.

Correctness is checked against the exact solution of a trivial linear
test ODE, dy/dt = -y, y(0) = 1, whose exact solution y(t) = exp(-t)
is known in closed form -- this isolates each method's own
discretization error from any question about the real (much more
complex) meteor right-hand side.
"""

import numpy as np
import pytest

from fixed_step_integrators import (euler_step, rk2_step, rk4_step,
                                      implicit_euler_step, implicit_heun_step,
                                      integrate_fixed_step)


def _decay_rhs(t, y):
    """dy/dt = -y, exact solution y(t) = y0 * exp(-t)."""
    return -y


def _stiff_rhs(t, y):
    """A moderately stiff linear decay, dy/dt = -50*y, used to check
    that the implicit methods remain stable where the explicit ones
    are expected to diverge at a large step size."""
    return -50.0 * y


# ---------------------------------------------------------------------
# Single-step accuracy on the simple decay problem
# ---------------------------------------------------------------------

def test_euler_step_matches_exact_to_first_order():
    y0 = np.array([1.0])
    dt = 0.001
    y1, _ = euler_step(_decay_rhs, 0.0, y0, dt)
    exact = np.exp(-dt)
    assert y1[0] == pytest.approx(exact, abs=1e-5)


def test_rk2_step_matches_exact_to_second_order():
    y0 = np.array([1.0])
    dt = 0.01
    y1, _ = rk2_step(_decay_rhs, 0.0, y0, dt)
    exact = np.exp(-dt)
    assert y1[0] == pytest.approx(exact, abs=1e-5)


def test_rk4_step_matches_exact_to_fourth_order():
    y0 = np.array([1.0])
    dt = 0.1
    y1, _ = rk4_step(_decay_rhs, 0.0, y0, dt)
    exact = np.exp(-dt)
    assert y1[0] == pytest.approx(exact, abs=1e-6)


def test_accuracy_ranking_euler_rk2_rk4_at_fixed_step():
    """At the same, deliberately coarse step size, RK4 must be more
    accurate than RK2, which must be more accurate than explicit
    Euler -- the whole reason this project implements more than one
    fixed-step method."""
    y0 = np.array([1.0])
    dt = 0.3
    exact = np.exp(-dt)

    y_euler, _ = euler_step(_decay_rhs, 0.0, y0, dt)
    y_rk2, _ = rk2_step(_decay_rhs, 0.0, y0, dt)
    y_rk4, _ = rk4_step(_decay_rhs, 0.0, y0, dt)

    err_euler = abs(y_euler[0] - exact)
    err_rk2 = abs(y_rk2[0] - exact)
    err_rk4 = abs(y_rk4[0] - exact)

    assert err_rk4 < err_rk2 < err_euler


# ---------------------------------------------------------------------
# Implicit methods: correctness and stability
# ---------------------------------------------------------------------

def test_implicit_euler_step_matches_exact_on_decay_problem():
    """Implicit Euler is only 1st-order accurate, so even a single
    step at dt=0.1 carries O(dt) error: the exact single-step update
    y1 = y0/(1+dt) = 0.909091 versus the true exp(-0.1) = 0.904837
    differ by about 0.00425. The tolerance here reflects that known
    1st-order error, not a looser correctness bar."""
    y0 = np.array([1.0])
    dt = 0.1
    y1, _ = implicit_euler_step(_decay_rhs, 0.0, y0, dt)
    exact = np.exp(-dt)
    assert y1[0] == pytest.approx(exact, abs=5e-3)


def test_implicit_heun_step_matches_exact_on_decay_problem():
    y0 = np.array([1.0])
    dt = 0.1
    y1, _ = implicit_heun_step(_decay_rhs, 0.0, y0, dt)
    exact = np.exp(-dt)
    assert y1[0] == pytest.approx(exact, abs=1e-4)


def test_implicit_euler_remains_bounded_on_stiff_problem_where_explicit_euler_diverges():
    """At dt = 0.1 and lambda = -50, explicit Euler's stability
    condition |1 + dt*lambda| <= 1 is violated (|1 - 5| = 4 > 1), so
    it must diverge, while implicit Euler is unconditionally stable
    and must stay bounded and decay toward zero."""
    y0 = np.array([1.0])
    dt = 0.1

    y_explicit, _ = euler_step(_stiff_rhs, 0.0, y0, dt)
    y_implicit, _ = implicit_euler_step(_stiff_rhs, 0.0, y0, dt)

    assert abs(y_explicit[0]) > abs(y0[0])   # explicit Euler has diverged
    assert abs(y_implicit[0]) < abs(y0[0])   # implicit Euler has decayed


# ---------------------------------------------------------------------
# integrate_fixed_step: ground-event crossing
# ---------------------------------------------------------------------

def test_integrate_fixed_step_locates_known_crossing():
    """For dh/dt = -1 (constant descent rate), starting at h=10, the
    ground (h=0) is crossed at exactly t=10. The interpolated event
    time should recover this to within one step size."""
    def constant_descent_rhs(t, y):
        return np.array([-1.0])

    y0 = np.array([10.0])
    result = integrate_fixed_step(constant_descent_rhs, y0, 0.0, 20.0,
                                     dt=0.5, method="rk4")

    assert result["event_t"] is not None
    assert result["event_t"] == pytest.approx(10.0, abs=0.5)
    assert result["event_y"][0] == pytest.approx(0.0, abs=1e-6)


def test_integrate_fixed_step_no_crossing_returns_none():
    """If the monitored state never crosses zero within the
    integration span, no event should be reported."""
    def rising_rhs(t, y):
        return np.array([1.0])

    y0 = np.array([10.0])
    result = integrate_fixed_step(rising_rhs, y0, 0.0, 5.0, dt=0.5, method="rk4")

    assert result["event_t"] is None
    assert result["event_y"] is None