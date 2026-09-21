"""
fixed_step_integrators.py

scipy.integrate.solve_ivp only offers ADAPTIVE methods (RK23, RK45,
DOP853, etc.) — there is no built-in fixed-step-size RK2 or RK4.
Since the project explicitly wants to compare fixed step size vs.
accuracy (how much does the solution change as dt decreases?), this
module implements classical fixed-step RK2 (Heun's method) and RK4
by hand.

Both integrators support a simple ground-impact event: after each
step, if altitude (state index 0) has crossed zero, the exact
crossing time/state is located by linear interpolation between the
last two accepted points, and integration stops there.
"""

import time
import numpy as np


def rk2_step(f, t, y, dt):
    """One step of Heun's method (explicit trapezoidal RK2)."""
    k1 = f(t, y)
    k2 = f(t + dt, y + dt * k1)
    y_next = y + (dt / 2.0) * (k1 + k2)
    return y_next, 2  # 2 function evaluations per step


def rk4_step(f, t, y, dt):
    """One step of the classical 4th-order Runge-Kutta method."""
    k1 = f(t, y)
    k2 = f(t + dt / 2.0, y + dt / 2.0 * k1)
    k3 = f(t + dt / 2.0, y + dt / 2.0 * k2)
    k4 = f(t + dt, y + dt * k3)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next, 4  # 4 function evaluations per step


_STEPPERS = {"rk2": rk2_step, "rk4": rk4_step}


def integrate_fixed_step(f, y0, t0, tf, dt, method="rk4",
                            event_index=0, max_steps=2_000_000):
    """
    Integrate dy/dt = f(t, y) from t0 to tf with a FIXED step size dt,
    using either 'rk2' or 'rk4'. Stops early if state[event_index]
    crosses zero (e.g. altitude reaching the ground), locating the
    exact crossing by linear interpolation between the last two
    accepted points.

    Parameters
    ----------
    f : callable(t, y) -> dy/dt
    y0 : array_like, initial state
    t0, tf : float, start/end time, s
    dt : float, fixed step size, s
    method : "rk2" or "rk4"
    event_index : int, which state component to monitor for a
        sign change (default 0, altitude)
    max_steps : int, safety cap on total steps taken

    Returns
    -------
    dict with keys:
        't', 'y'          : arrays of all accepted (t, state) points
        'event_t'         : time of the interpolated event crossing
                              (None if never crossed within [t0, tf])
        'event_y'         : state at the interpolated event crossing
        'n_evals'         : total RHS function evaluations
        'runtime_s'       : wall-clock time taken, s
        'n_steps'         : number of fixed steps actually taken
    """
    stepper = _STEPPERS[method]
    y0 = np.asarray(y0, dtype=float)

    t_start_wall = time.perf_counter()

    t = t0
    y = y0.copy()
    t_list = [t]
    y_list = [y.copy()]
    n_evals = 0
    event_t = None
    event_y = None

    n_steps = 0
    while t < tf and n_steps < max_steps:
        step_dt = min(dt, tf - t)
        y_next, evals_used = stepper(f, t, y, step_dt)
        n_evals += evals_used
        t_next = t + step_dt
        n_steps += 1

        # Check for event crossing (e.g. altitude going negative).
        if y[event_index] > 0.0 and y_next[event_index] <= 0.0:
            # Linear interpolation between (t, y) and (t_next, y_next)
            # to find where event_index crosses zero.
            frac = y[event_index] / (y[event_index] - y_next[event_index])
            event_t = t + frac * (t_next - t)
            event_y = y + frac * (y_next - y)
            t_list.append(t_next)
            y_list.append(y_next.copy())
            break

        t, y = t_next, y_next
        t_list.append(t)
        y_list.append(y.copy())

    runtime_s = time.perf_counter() - t_start_wall

    return {
        "t": np.array(t_list),
        "y": np.array(y_list).T,  # shape (n_state, n_points), matching solve_ivp's .y
        "event_t": event_t,
        "event_y": event_y,
        "n_evals": n_evals,
        "runtime_s": runtime_s,
        "n_steps": n_steps,
    }