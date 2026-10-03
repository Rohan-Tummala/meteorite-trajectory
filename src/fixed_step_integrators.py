"""
fixed_step_integrators.py

scipy.integrate.solve_ivp only offers ADAPTIVE methods (RK23, RK45,
DOP853, etc.) — there is no built-in fixed-step-size RK2 or RK4.
Since the project explicitly wants to compare fixed step size vs.
accuracy (how much does the solution change as dt decreases?), this
module implements classical fixed-step RK2 (Heun's method), RK4, and
implicit (backward) Euler by hand.

Implicit Euler is included specifically because it is unconditionally
stable (Week 9.3 notes): unlike RK2/RK4, it has no step-size ceiling
set by the local Jacobian eigenvalues, so it can take much larger
steps in the calmer dark-flight phase without going unstable -- the
trade-off being its 1st-order accuracy, which the computation-time
vs. impact-error comparison in plot_piecewise_stepping.py is meant to
quantify directly, rather than just asserting it.

Both integrators support a simple ground-impact event: after each
step, if altitude (state index 0) has crossed zero, the exact
crossing time/state is located by linear interpolation between the
last two accepted points, and integration stops there.
"""

import time
import numpy as np


def euler_step(f, t, y, dt):
    """One step of explicit (forward) Euler — 1st order."""
    k1 = f(t, y)
    y_next = y + dt * k1
    return y_next, 1  # 1 function evaluation per step


def rk2_step(f, t, y, dt):
    """
    One step of Heun's method -- the explicit trapezoidal RK2 scheme:
    predict with forward Euler (k1), correct using the average of the
    slope at the start and at the predicted end (k2). "RK2" throughout
    this project refers to this method.
    """
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


def _finite_diff_jacobian(f, t, y, eps_rel=1e-6):
    """
    Finite-difference Jacobian df/dy at (t, y), shape (n, n). Same
    scaled forward-difference approach as numerical_jacobian() in
    plot_piecewise_stepping.py -- duplicated here (rather than
    imported) so this module has no dependency on that one.
    """
    n = len(y)
    J = np.zeros((n, n))
    f0 = f(t, y)
    for j in range(n):
        dy = np.zeros(n)
        step = eps_rel * max(abs(y[j]), 1.0)
        dy[j] = step
        f1 = f(t, y + dy)
        J[:, j] = (f1 - f0) / step
    return J


def implicit_euler_step(f, t, y, dt, newton_tol=1e-8, max_newton_iter=10):
    """
    One step of backward (implicit) Euler, solved by Newton's method --
    following Week 9.3.4 of the course notes directly, extended from a
    scalar y to a state vector.

    y_{n+1} is defined implicitly as the root of
        F(y_{n+1}) = y_{n+1} - y_n - dt*f(t_{n+1}, y_{n+1}) = 0.
    Newton's method (9.3.4, Eq. 9.18) linearises F about a guess and
    solves for the correction. For a vector F, the scalar
    f'(y)-division becomes a linear solve with F's own Jacobian
    (I - dt*J), where J = df/dy:
        (I - dt*J)*delta = -F(y_guess),   y_guess <- y_guess + delta.

    Initial guess and Jacobian point: y_n itself, NOT the explicit-
    Euler predictor (y_n + dt*f(t,y_n)). This matters here specifically
    -- during peak ablation, M is falling so fast that an explicit-
    Euler predictor with a large dt can overshoot mass almost to zero
    in a single jump, landing the Jacobian evaluation right next to
    the model's M -> 0 singularity (V_dot, M_dot ~ 1/M_safe terms
    blow up there), which was observed to make Newton diverge rather
    than converge. Evaluating at y_n instead keeps the linearisation
    point physically sane.

    J is evaluated once per step (at y_n) and held fixed across the
    Newton iterations -- the standard "modified Newton" simplification,
    since re-forming and re-solving a fresh (I - dt*J) every single
    iteration would cost far more than it saves for a well-behaved step.
    """
    t_next = t + dt
    n = len(y)

    J = _finite_diff_jacobian(f, t, y)
    n_evals = n + 1  # cost of the finite-difference Jacobian itself

    A = np.eye(n) - dt * J  # Newton iteration matrix (I - dt*J)

    y_guess = y.copy()  # start the Newton iteration from y_n itself
    for _ in range(max_newton_iter):
        F = y_guess - y - dt * f(t_next, y_guess)
        n_evals += 1
        delta = np.linalg.solve(A, -F)
        y_guess = y_guess + delta
        if np.max(np.abs(delta)) < newton_tol * max(1.0, np.max(np.abs(y_guess))):
            break

    return y_guess, n_evals


def implicit_heun_step(f, t, y, dt, newton_tol=1e-8, max_newton_iter=10):
    """
    One step of the implicit (trapezoidal) Heun method -- Week 9
    Workbook, Activity 2B:
        y_{n+1} = y_n + (h/2)*[f(t_n, y_n) + f(t_{n+1}, y_{n+1})]

    2nd-order accurate (unlike implicit Euler's 1st order) and
    A-stable, but NOT L-stable: its stability function
    R(z) = (1 + z/2)/(1 - z/2) satisfies |R(z)| -> 1 (not -> 0) as
    z -> -infinity, so very stiff modes are damped much more slowly
    than with implicit Euler, even though they never grow.

    Solved by Newton's method the same way as implicit_euler_step:
        F(y_{n+1}) = y_{n+1} - y_n - (h/2)*[f_n + f(t_{n+1},y_{n+1})] = 0
        (I - (h/2)*J)*delta = -F(y_guess).
    Initial guess and (frozen) Jacobian both taken at y_n -- same
    reasoning as implicit_euler_step: a predictor-based guess can
    overshoot into the model's mass-ablation singularity during the
    stiffest part of the trajectory.
    """
    t_next = t + dt
    n = len(y)

    f_n = f(t, y)
    n_evals = 1

    J = _finite_diff_jacobian(f, t, y)
    n_evals += n + 1

    A = np.eye(n) - (dt / 2.0) * J

    y_guess = y.copy()
    for _ in range(max_newton_iter):
        F = y_guess - y - (dt / 2.0) * (f_n + f(t_next, y_guess))
        n_evals += 1
        delta = np.linalg.solve(A, -F)
        y_guess = y_guess + delta
        if np.max(np.abs(delta)) < newton_tol * max(1.0, np.max(np.abs(y_guess))):
            break

    return y_guess, n_evals


_STEPPERS = {
    "euler": euler_step,
    "rk2": rk2_step,
    "rk4": rk4_step,
    "implicit_euler": implicit_euler_step,
    "implicit_heun": implicit_heun_step,
}


def integrate_fixed_step(f, y0, t0, tf, dt, method="rk4",
                            event_index=0, max_steps=2_000_000):
    """
    Integrate dy/dt = f(t, y) from t0 to tf with a FIXED step size dt,
    using "rk2", "rk4", or "implicit_euler". Stops early if
    state[event_index] crosses zero (e.g. altitude reaching the
    ground), locating the exact crossing by linear interpolation
    between the last two accepted points.

    Parameters
    ----------
    f : callable(t, y) -> dy/dt
    y0 : array_like, initial state
    t0, tf : float, start/end time, s
    dt : float, fixed step size, s
    method : "rk2", "rk4", or "implicit_euler"
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