"""
integrator.py

Wraps scipy.integrate.solve_ivp around dynamics.meteor_rhs to propagate
a meteor trajectory forward in time from a given initial state.

The integration method and tolerances are parameters, not hardcoded,
so the SAME physics (meteor_rhs, with a fixed params dict) can be run
through different numerical methods for comparison:

    - Fixed-step-like methods: "RK23" (2nd/3rd order), "RK45" (default,
      4th/5th order Dormand-Prince), "DOP853" (8th order) — all
      adaptive in scipy, but rtol/atol can be tightened to approximate
      a "truth" reference solution, or loosened for a cheaper run.
    - Implicit methods "Radau", "BDF", "LSODA" are also available via
      scipy if needed for stiffness comparisons.

A tight-tolerance run (small rtol/atol) serves as the "truth solver"
reference for isolating numerical error from model/input error, per
the project's error-budget plan.
"""

import numpy as np
from scipy.integrate import solve_ivp

from dynamics import meteor_rhs, ground_impact_event


def run_trajectory(y0, t_span, method="RK45", rtol=1e-6, atol=1e-8,
                    params=None, max_step=np.inf, dense_output=False):
    """
    Integrate the meteor trajectory forward in time from y0.

    Parameters
    ----------
    y0 : array_like, shape (7,)
        Initial state [h, V, gamma, phi, lambda, psi, M].
    t_span : tuple (t0, tf)
        Start and end time, s. Integration stops earlier automatically
        if ground impact (h=0) is reached first, via the event function.
    method : str, optional
        Integration method passed to solve_ivp: "RK23", "RK45"
        (default), "DOP853", "Radau", "BDF", "LSODA".
    rtol, atol : float, optional
        Relative/absolute tolerance passed to solve_ivp. Tighten both
        (e.g. 1e-10/1e-12) to approximate a "truth" reference solution;
        loosen for a cheaper, less accurate run.
    params : dict, optional
        Physical/model parameters forwarded to meteor_rhs (see
        dynamics.DEFAULT_PARAMS for keys and defaults).
    max_step : float, optional
        Maximum allowed step size, s. Default: unrestricted (solver
        chooses freely). Useful to force finer sampling for plotting.
    dense_output : bool, optional
        If True, solve_ivp also returns a continuous interpolant
        (sol.sol) usable to evaluate the state at arbitrary times
        within the integration span, not just at the solver's own
        internal steps.

    Returns
    -------
    scipy.integrate.OdeResult
        The full solve_ivp result object. Key fields:
            .t       — array of times, s
            .y       — array of shape (7, N), state at each time in .t
            .t_events[0] — time of ground impact, if it occurred
            .y_events[0] — state at ground impact, if it occurred
            .success — whether the solver completed without error
    """
    y0 = np.asarray(y0, dtype=float)

    # Ground impact is a terminal event: stop integrating once h=0.
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1

    result = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=t_span,
        y0=y0,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense_output,
        events=ground_impact_event,
    )
    return result


def impact_state(result):
    """
    Extract the state at ground impact from a run_trajectory() result,
    if the ground_impact_event fired.

    Parameters
    ----------
    result : scipy.integrate.OdeResult
        Output of run_trajectory().

    Returns
    -------
    ndarray, shape (7,), or None
        State [h, V, gamma, phi, lambda, psi, M] at the moment of
        ground impact, or None if the trajectory did not reach h=0
        within t_span (e.g. it ablated to nothing, or t_span was too
        short).
    """
    if len(result.t_events[0]) == 0:
        return None
    return result.y_events[0][0]


if __name__ == "__main__":
    # Quick manual sanity check: integrate the Cavezzo trajectory with
    # default (moderate) tolerances over a generous time span, and
    # report whether/where it hit the ground.
    from cavezzo import initial_state

    y0 = initial_state()
    t_span = (0.0, 600.0)  # s — generous upper bound; dark-flight descent
                             # from ~21.5 km at near-terminal speed can
                             # take several minutes, not seconds

    sol = run_trajectory(y0, t_span, method="RK45", rtol=1e-8, atol=1e-10)

    print(f"Solver success: {sol.success}")
    print(f"Number of time points: {len(sol.t)}")

    y_impact = impact_state(sol)
    if y_impact is not None:
        h, V, gamma, phi, lam, psi, M = y_impact
        print(f"\nGround impact at t = {sol.t_events[0][0]:.3f} s")
        print(f"  V   = {V:.2f} m/s")
        print(f"  phi = {np.degrees(phi):.6f} deg N")
        print(f"  lam = {np.degrees(lam):.6f} deg E")
        print(f"  M   = {M:.4f} kg")
    else:
        h_final, V_final = sol.y[0, -1], sol.y[1, -1]
        print(f"\nNo ground impact detected within t_span — "
              f"at t={sol.t[-1]:.1f} s, h={h_final:.1f} m, "
              f"V={V_final:.2f} m/s. Increase t_span if h is still "
              f"well above zero.")