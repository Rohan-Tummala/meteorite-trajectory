"""
test_convergence_order.py

Checks that the hand-written fixed-step integrators (rk2_step,
rk4_step) actually achieve their claimed global convergence order on
a problem with a known exact solution, dy/dt = -y, y(0) = 1, whose
exact solution is y(t) = exp(-t).

This is a GLOBAL convergence check: each test integrates with many
repeated fixed steps out to a fixed end time T_FINAL, not a single
step. A single-step (local truncation) error scales as O(dt^(p+1)),
one order higher than the global error O(dt^p) actually being
claimed here, so fitting against single-step error would report an
order one too high (see roundoff_test.py / convergence_order_check.py
in the project root for the standalone script this test is based on,
and the project report, Section 4, for the fitted values this test
backs: RK2 fits order 2.03, RK4 fits order 4.03).
"""

import numpy as np
import pytest

from fixed_step_integrators import rk2_step, rk4_step

T_FINAL = 2.0
DTS = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])


def _decay_rhs(t, y):
    """dy/dt = -y, exact solution y(t) = y0 * exp(-t)."""
    return -y


def _integrate_global(step_func, dt, t_final=T_FINAL):
    """Integrate from y0=1 to t_final using many repeated fixed steps
    of size dt, and return the final value."""
    y = np.array([1.0])
    t = 0.0
    n_steps = int(round(t_final / dt))
    for _ in range(n_steps):
        y, _ = step_func(_decay_rhs, t, y, dt)
        t += dt
    return y[0]


def _fit_global_order(step_func, dts=DTS, t_final=T_FINAL):
    """Fit the global convergence order by step-halving: integrate at
    each dt in dts, compare to the exact solution at t_final, and fit
    a line to log(error) vs. log(dt). The slope is the empirical
    order, since global error ~ C * dt^p implies
    log(error) ~ p * log(dt) + log(C)."""
    exact = np.exp(-t_final)
    errors = np.array([
        abs(_integrate_global(step_func, dt, t_final) - exact)
        for dt in dts
    ])
    slope, _ = np.polyfit(np.log(dts), np.log(errors), 1)
    return slope


def test_rk2_global_convergence_order_is_second_order():
    """RK2 (Heun) is a 2nd-order method, so its global error should
    scale as O(dt^2), giving a fitted slope close to 2."""
    slope = _fit_global_order(rk2_step)
    assert slope == pytest.approx(2.0, abs=0.15)


def test_rk4_global_convergence_order_is_fourth_order():
    """RK4 (classical) is a 4th-order method, so its global error
    should scale as O(dt^4), giving a fitted slope close to 4."""
    slope = _fit_global_order(rk4_step)
    assert slope == pytest.approx(4.0, abs=0.15)