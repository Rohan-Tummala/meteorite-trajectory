"""
flops_estimate.py

Rough operation count for one call of the model's right-hand side (RHS)
and for a full RK45 run.

Method:
    1. RHS calls per run are read from solve_ivp's counter (sol.nfev).
    2. Operations per call are counted by running the real meteor_rhs
       on a number class (Counted) that tallies each +, -, *, /, power,
       sin and cos. The density lookup and gravity cannot take Counted
       values, so wrappers add their fixed operation counts.
    3. Time per call is measured with timeit to give the achieved FLOP/s.

Two totals are reported. "Plain" counts every operation as 1 FLOP.
"Weighted" counts add, multiply and divide as 1, sin and cos as 30 and
power (used for the cube root) as 40, using relative costs from
counted-float (https://counted-float.readthedocs.io/en/stable/flop_weights/).
The weighted figure is a cost estimate, not a strict FLOP count.
Comparisons and the density table search are not counted.

Run from the src/ folder:
    python flops_estimate.py
"""

import timeit
import numpy as np

import dynamics
from dynamics import DEFAULT_PARAMS, meteor_rhs
from integrator import run_trajectory
import cavezzo
import bunburra

WEIGHTS = {"add": 1.0, "mul": 1.0, "div": 1.0, "sin": 30.0, "cos": 30.0, "pow": 40.0}


class Counted:
    """A float that counts the arithmetic done on it (class-level tally)."""
    tally = {k: 0 for k in WEIGHTS}

    def __init__(self, v):
        """Wrap the value v as a counting number."""
        self.v = float(v)

    @staticmethod
    def val(x):
        """Return x as a plain float (x may be a Counted or a number)."""
        return x.v if isinstance(x, Counted) else float(x)

    def _op(self, other, f, kind):
        """Apply f to self and other, record one operation of type kind, and return the Counted result."""
        Counted.tally[kind] += 1
        return Counted(f(self.v, Counted.val(other)))

    def __add__(self, o):
        """Add, counted as one add."""
        return self._op(o, lambda a, b: a + b, "add")
    __radd__ = __add__
    def __sub__(self, o):
        """Subtract, counted as one add."""
        return self._op(o, lambda a, b: a - b, "add")
    def __rsub__(self, o):
        """Reflected subtract, counted as one add."""
        return self._op(o, lambda a, b: b - a, "add")
    def __mul__(self, o):
        """Multiply, counted as one multiply."""
        return self._op(o, lambda a, b: a * b, "mul")
    __rmul__ = __mul__
    def __truediv__(self, o):
        """Divide, counted as one divide."""
        return self._op(o, lambda a, b: a / b, "div")
    def __rtruediv__(self, o):
        """Reflected divide, counted as one divide."""
        return self._op(o, lambda a, b: b / a, "div")
    def __neg__(self):
        """Negate (a sign flip, not counted)."""
        return Counted(-self.v)  # sign flip, not counted

    def __pow__(self, p):
        """Power. Integer powers count as n-1 multiplies, other powers (the cube root) as one power."""
        if float(p) == int(p):
            Counted.tally["mul"] += max(int(p) - 1, 1)  # x**n = n-1 multiplies
        else:
            Counted.tally["pow"] += 1                   # cube root
        return Counted(self.v ** p)

    def sin(self):
        """Return sin(self), counted as one sin."""
        Counted.tally["sin"] += 1
        return Counted(np.sin(self.v))

    def cos(self):
        """Return cos(self), counted as one cos."""
        Counted.tally["cos"] += 1
        return Counted(np.cos(self.v))

    # comparisons (max, if V < 3000) are not FLOPs
    def __lt__(self, o):
        """Compare plain values (not counted)."""
        return self.v < Counted.val(o)
    def __gt__(self, o):
        """Compare plain values (not counted)."""
        return self.v > Counted.val(o)
    def __le__(self, o):
        """Compare plain values (not counted)."""
        return self.v <= Counted.val(o)
    def __ge__(self, o):
        """Compare plain values (not counted)."""
        return self.v >= Counted.val(o)

    def __float__(self):
        """Return the plain float value."""
        return self.v


def count_rhs_flops(y0, params):
    """
    Count the operations in one meteor_rhs call.

    Parameters
    ----------
    y0 : array_like, shape (7,)
        State at which to evaluate the RHS.
    params : dict
        Model parameters passed to meteor_rhs.

    Returns
    -------
    dict
        Operation counts by type: add, mul, div, sin, cos, pow.
    """
    real_density, real_gravity = dynamics.density, dynamics.gravity

    def counted_density(h, model="table"):
        """Density lookup on a plain float, adding the interpolation's fixed count (2 add, 1 divide, 1 multiply)."""
        t = Counted.tally
        t["add"] += 2; t["div"] += 1; t["mul"] += 1   # linear interpolation
        return Counted(real_density(Counted.val(h), model=model))

    def counted_gravity(h, model="spherical"):
        """Gravity on a plain float, adding the fixed count for G0*(R/(R+h))**2 (1 add, 1 divide, 2 multiply)."""
        t = Counted.tally
        t["add"] += 1; t["div"] += 1; t["mul"] += 2   # G0*(R/(R+h))**2
        return Counted(real_gravity(Counted.val(h), model=model))

    dynamics.density, dynamics.gravity = counted_density, counted_gravity
    try:
        Counted.tally = {k: 0 for k in WEIGHTS}
        y = np.array([Counted(v) for v in y0], dtype=object)
        meteor_rhs(0.0, y, params=params)
        return dict(Counted.tally)
    finally:
        dynamics.density, dynamics.gravity = real_density, real_gravity


def main():
    """Print RHS calls per run, operations per call, run totals, time per call, FLOP/s and arithmetic intensity for both falls."""
    for name, fall in (("Cavezzo", cavezzo), ("Bunburra", bunburra)):
        params = {**DEFAULT_PARAMS, "rho_m": fall.RHO_M}
        y0 = fall.initial_state()

        sol = run_trajectory(y0, (0.0, 600.0), method="RK45",
                             rtol=1e-8, atol=1e-10, params=params)
        n_calls = sol.nfev

        tally = count_rhs_flops(y0, params)
        plain = sum(tally.values())
        weighted = sum(WEIGHTS[k] * n for k, n in tally.items())

        n = 20000
        t_call = timeit.timeit(
            lambda: meteor_rhs(0.0, y0, params=params), number=n) / n

        print(f"\n{name}")
        print(f"  RHS calls in one RK45 run (sol.nfev): {n_calls}")
        print(f"  Operations per call: {tally}")
        print(f"  Per call: {plain} plain FLOPs, {weighted:.0f} weighted")
        print(f"  Per run:  {n_calls * plain:.2e} plain, "
              f"{n_calls * weighted:.2e} weighted")
        print(f"  Time per call: {t_call * 1e6:.1f} us on this machine")
        print(f"  Achieved: {plain / t_call:.2e} (plain) to "
              f"{weighted / t_call:.2e} (weighted) FLOP/s")
        # state in + derivative out = 14 doubles (state traffic only)
        print(f"  Arithmetic intensity (approx): "
              f"{plain / (14 * 8):.2f} to {weighted / (14 * 8):.2f} FLOP/byte")


if __name__ == "__main__":
    main()