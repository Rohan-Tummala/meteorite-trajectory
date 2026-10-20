"""
convergence_order_check.py

Fits the observed convergence order of rk2_step and rk4_step on the
same decay ODE used in test_fixed_step_integrators.py (dy/dt = -y,
exact solution y(t) = exp(-t)), rather than assuming they hit their
theoretical order (2nd and 4th) without checking.

This measures GLOBAL error, not single-step error: each method is run
over a fixed time span T with many fixed steps of size dt, and the
final value is compared against the exact solution at t=T. A single
step only shows the method's local truncation error, which is one
order higher (e.g. RK2's single-step error is O(dt^3), not O(dt^2)),
so that would overstate the order by one and not match what "RK2/RK4
convergence order" means elsewhere in this project (Figure 5, impact
error vs step size over the whole trajectory).

The step size is halved five times, the global error at t=T is
recorded at each dt, and a line is fit to log(error) vs log(dt). The
slope of that line is the fitted convergence order, which should come
out close to 2 for rk2_step and close to 4 for rk4_step.
"""

import numpy as np

from fixed_step_integrators import rk2_step, rk4_step

T_FINAL = 2.0  # total integration time, s


def _decay_rhs(t, y):
    """dy/dt = -y, exact solution y(t) = y0 * exp(-t)."""
    return -y


def integrate_global(step_func, dt, t_final=T_FINAL):
    """Integrate the decay ODE from t=0, y=1 to t=t_final with a fixed
    step size dt, taking as many steps of step_func as needed, and
    return the final y value."""
    y = np.array([1.0])
    t = 0.0
    n_steps = int(round(t_final / dt))
    for _ in range(n_steps):
        y, _ = step_func(_decay_rhs, t, y, dt)
        t += dt
    return y[0]


def fit_order(step_func, dts):
    """Run step_func over a range of step sizes to t=T_FINAL, compare
    each final value against the exact solution exp(-T_FINAL), and
    return (errors, fitted_order) from a log(error) vs log(dt) fit."""
    exact = np.exp(-T_FINAL)
    errors = np.array([abs(integrate_global(step_func, dt) - exact)
                        for dt in dts])
    slope, _ = np.polyfit(np.log(dts), np.log(errors), 1)
    return errors, slope


def main():
    """Fit and print the observed global convergence order of
    rk2_step and rk4_step over a 5-point step-halving sweep."""
    dts = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])

    print(f"Integrating to t={T_FINAL}, step sizes:", dts)

    for name, step_func, theory_order in [("RK2", rk2_step, 2),
                                            ("RK4", rk4_step, 4)]:
        errors, order = fit_order(step_func, dts)
        print(f"\n{name} (theoretical order {theory_order}):")
        for dt, err in zip(dts, errors):
            print(f"  dt={dt:<8} global error={err:.3e}")
        print(f"  fitted order = {order:.3f}")


if __name__ == "__main__":
    main()