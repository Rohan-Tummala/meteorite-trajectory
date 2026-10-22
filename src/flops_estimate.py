"""
flops_estimate.py

Rough arithmetic-cost estimate for the trajectory model, to back the
FLOPs / arithmetic-intensity sentence in the report.

Three separate measurements, none of them hand-counted:

1. Number of right-hand-side (RHS) calls in a real production RK45 run,
   read directly from solve_ivp's own counter (sol.nfev).
2. FLOPs per RHS call, counted by running the real meteor_rhs on a small
   counting-number class that tallies every +, -, *, / and ** it sees.
   Trig functions and the cube root are tallied separately so the total
   can be quoted either way (each as 1 FLOP, or as ~20 FLOPs each).
3. Wall-clock time per RHS call, using timeit, to get the achieved
   FLOP/s and compare it with what a CPU can actually do.

The tabulated density lookup (np.interp) and the gravity function
convert their input to a plain float internally, so a counting object
cannot pass through them. Both are swapped for wrappers that call the
real function and add their known operation counts by hand (density
interpolation: 2 add/sub, 1 divide, 1 multiply; gravity: 1 add,
1 divide, 2 multiply). Index search comparisons are not counted.

Two totals are reported. "Plain" counts every operation as 1 FLOP.
"Weighted" counts add, multiply and divide as 1 and weights the
expensive functions using the relative costs from the counted-float
project (https://counted-float.readthedocs.io/en/stable/flop_weights/),
which combines benchmarks and specs across CPUs, in units of one add:
sin 30, cos 30, power (used for the cube root) 40. The weighted figure
is a cost estimate, not a strict FLOP count.
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
        self.v = float(v)

    @staticmethod
    def val(x):
        return x.v if isinstance(x, Counted) else float(x)

    def _op(self, other, f, kind):
        Counted.tally[kind] += 1
        return Counted(f(self.v, Counted.val(other)))

    def __add__(self, o): return self._op(o, lambda a, b: a + b, "add")
    __radd__ = __add__
    def __sub__(self, o): return self._op(o, lambda a, b: a - b, "add")
    def __rsub__(self, o): return self._op(o, lambda a, b: b - a, "add")
    def __mul__(self, o): return self._op(o, lambda a, b: a * b, "mul")
    __rmul__ = __mul__
    def __truediv__(self, o): return self._op(o, lambda a, b: a / b, "div")
    def __rtruediv__(self, o): return self._op(o, lambda a, b: b / a, "div")
    def __neg__(self): return Counted(-self.v)  # sign flip, not counted

    def __pow__(self, p):
        if float(p) == int(p):
            Counted.tally["mul"] += max(int(p) - 1, 1)  # x**n = n-1 multiplies
        else:
            Counted.tally["pow"] += 1                   # cube root
        return Counted(self.v ** p)

    def sin(self):
        Counted.tally["sin"] += 1
        return Counted(np.sin(self.v))

    def cos(self):
        Counted.tally["cos"] += 1
        return Counted(np.cos(self.v))

    # comparisons (max, if V < 3000) are not FLOPs
    def __lt__(self, o): return self.v < Counted.val(o)
    def __gt__(self, o): return self.v > Counted.val(o)
    def __le__(self, o): return self.v <= Counted.val(o)
    def __ge__(self, o): return self.v >= Counted.val(o)

    def __float__(self): return self.v


def count_rhs_flops(y0, params):
    """Run one RHS call on counting numbers and return the tally dict."""
    real_density, real_gravity = dynamics.density, dynamics.gravity

    def counted_density(h, model="table"):
        t = Counted.tally
        t["add"] += 2; t["div"] += 1; t["mul"] += 1   # linear interpolation
        return Counted(real_density(Counted.val(h), model=model))

    def counted_gravity(h, model="spherical"):
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