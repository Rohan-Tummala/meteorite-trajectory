# Meteorite Atmospheric Trajectory Simulation

A 3D point-mass simulation of meteoroid atmospheric entry, validated against
two real recorded meteorite falls: the **Bunburra Rockhole** achondrite
(Australia, Spurný et al. 2012) and the **Cavezzo** ordinary chondrite
(Italy, Gardiol et al. 2021). The project compares the simulated flight
against each fall's published terminal-point data and real recovery GPS
coordinates, and separates numerical integration error from physical-model
error, i.e. the gap between this simplified model and the real, far more
complex physics of an atmospheric entry.

## Engineering problem

**Given a meteoroid's atmospheric entry conditions, predict where it lands,
and quantify how confident that prediction can be.**

- **Inputs**: entry altitude (m), speed (m/s), flight-path angle (rad),
  initial latitude/longitude (rad), heading (rad), and initial mass (kg).
  Each one has a stated measurement uncertainty, taken from the relevant
  published fall study.
- **Outputs**: a full descent trajectory (altitude, speed, position, mass
  vs. time) down to ground impact, and, from the Monte Carlo stage, a
  probability map of likely landing coordinates, built by propagating the
  input uncertainties through the full nonlinear model.
- **Scope and limitations**: the model is a 3D point-mass approximation
  (frozen heading, no fragmentation), uses a shared drag/ablation
  coefficient pair across both case studies rather than one tuned per
  meteorite. Only Cavezzo has a dark-flight wind model, a simplified
  profile fitted to the wind data Gardiol et al. (2021) report for that
  fall, not a full atmospheric reanalysis; Bunburra has no reported wind
  data to fit a profile to, so it is run without one.
  These choices and their consequences are discussed under Testing, below.

The same core physics is run two ways:

- **Deterministic trajectory validation**: comparing the simulated
  terminal point (end of the luminous trail) against each paper's reported
  values for altitude, velocity and mass (`terminal_point_check.py`), and
  against digitized published height-vs-time curves where available.
- **Monte Carlo uncertainty propagation**: sampling the stated
  measurement uncertainties on entry conditions, propagating them through
  the full nonlinear model, and building a kernel-density probability map
  of likely landing sites, checked against the real recovered coordinates.

## Computational approach and comparison

The full meteoroid trajectory system (`dynamics.py`) is a coupled nonlinear
ODE system with no closed-form solution, so there is no exact analytical
result to compare against directly. That's only possible for the simple
decay-equation test case used in `test/test_integrator.py` and
`test/test_convergence_order.py` to validate basic solver accuracy and
fitted convergence order. Instead, a tight-tolerance DOP853 run
(`rtol=1e-12`, very small `max_step`) is used as a high-accuracy reference
("truth") solution. `roundoff_test.py` confirms that tightening the
tolerance further than the `rtol=1e-10` actually used for the reference
solution elsewhere in the project moves the predicted impact point by only
about 2e-5 m, i.e. its own numerical error is negligible at the tolerance
actually used.

Three alternative integration approaches are compared against that
reference in `numerical_methods_comparison.py` and
`plot_numerical_comparison.py`: the hand-written fixed-step RK2 and RK4
methods, and scipy's adaptive RK45. Each is run across a range of step
sizes (or tolerance for RK45) and judged on accuracy (impact-location and
altitude-profile error against the reference) versus computational cost
(runtime), rather than on accuracy alone, which is why RK45 is the method
carried forward for the production trajectory and Monte Carlo runs.
`convergence_order_check.py` separately fits the global convergence order
of RK2 and RK4 by step-halving against the known exact solution of a
decay ODE (order ≈2.03 and ≈4.03, matching theory), backed by
`test/test_convergence_order.py`.

`plot_piecewise_stepping.py` then takes the best-performing methods from
that comparison and tests one specific optimisation: splitting the
integration into a fine step during luminous flight (where the dynamics
change fastest) and a coarse step during dark flight (where they do not),
instead of a single uniform fine step across the whole trajectory. This is
compared against the uniform-step baseline on the same accuracy-vs-runtime
basis, to check whether the piecewise approach actually saves computation
for the same accuracy or whether it costs accuracy to do so.

`profile_compare.py` profiles the atmosphere density lookup directly,
comparing the original `scipy.interpolate.interp1d`-based implementation
(`atmosphere_OLD_interp1d.py`) against the current `numpy.interp`-based
one, confirming the replacement is both faster and numerically identical.

Separately, comparing the deterministic trajectory output against each
paper's reported terminal-point values (see Testing, below) separates
numerical integration error (how far a method's result is from the
reference solution, with the physical model held fixed) from
physical-model error (how far the model itself is from the real fall, with
numerics held as accurate as practical). The model cannot capture every
higher-order coupling effect present in a real meteor fall, things like
fragmentation, rotation-dependent drag, ablation-drag feedback through a
changing shape and area, or a fully resolved wind field. So some residual
mismatch against the real recovered coordinates is expected even with
near-exact numerics. `sensitivity_sweep.py` checks how much of that
mismatch could plausibly come from the shared, un-fitted drag/ablation
coefficients (small, for both falls) rather than from missing physics.
The xfail tests in `test/test_monte_carlo.py` document the remaining gap
directly rather than hiding it.

## Monte Carlo uncertainty propagation

For each fall, 5000 joint samples are drawn: the seven entry-state
variables (altitude, speed, flight-path angle, latitude, longitude,
heading, mass) as independent normals centred on the paper's reported
value with its stated 1-sigma uncertainty, plus the ablation coefficient
`sigma_abl` as a uniform draw over the range Ceplecha et al. (1998, Table
VII) fit to the Lost City fall. The drag coefficient `Cd` is held fixed
instead of sampled, because unlike `sigma_abl` it has no published,
fall-specific uncertainty range to draw from; `sensitivity_sweep.py`
checks it anyway with a generic +/-10% engineering bound rather than a
sourced uncertainty, and that sweep actually shows `Cd`'s effect on the
impact point is larger than `sigma_abl`'s for both falls, so leaving it
out of the Monte Carlo is a real limitation, not a negligible-effect
assumption (see Testing, below).

Each sample is propagated through the full model to ground
impact, giving a scatter of impact coordinates. A Gaussian KDE is fitted
to that scatter and evaluated over a 200x200 grid covering the footprint,
turning the raw points into a smooth density surface.

The HPD (highest-posterior-density) threshold for a given probability mass
is found by sorting the density evaluated at every *sample*, then picking
the threshold such that that fraction of samples (by count) lies at or
above it. Since the Monte Carlo samples are already drawn proportional to
the distribution being estimated, the fraction of samples above a density
threshold is itself a direct empirical estimate of the enclosed
probability mass. An earlier version of this function instead summed the
density *values* themselves rather than counting samples, which
over-weights the highest-density points: its nominal 39.3/86.5/98.9
percent contours actually enclosed only about 22/63/90 percent of samples,
checked against a known 2D Gaussian where the true answer is known exactly
(`test/test_monte_carlo.py::test_hpd_levels_matches_known_gaussian_coverage`).
The corrected version is checked against the same Gaussian and recovers
exact coverage.

The contour bands reported as "1-sigma/2-sigma/3-sigma" are the 2D
highest-posterior-density equivalents of those probabilities (39.3%,
86.5%, 98.9% enclosed probability), not the naive 1D 68/95/99.7% figures.
This matches the convention Gardiol et al. (2021) use for their own
published Cavezzo strewn-field ellipse, so the two are directly
comparable.

Validating this probability map against the real recovered location is
the project's headline result, and it is a genuine negative finding: under
the shared default `Cd`/`sigma_abl`, the real recovery site falls outside
the predicted 3-sigma (98.9%) region for both falls. The margin is wide for
Bunburra Rockhole, and narrow (landing just outside the sample cloud) for
Cavezzo. This is consistent with Cavezzo having the smaller terminal-point
error of the two under the same default coefficients, and is interpreted
as physical-model calibration error rather than numerical error, since the
sampling, KDE and HPD-threshold logic are separately verified correct by
other tests in `test/test_monte_carlo.py`. See the two `xfail` tests there
for the full reasoning.

## Project structure

```
meteorite-trajectory/
├── src/
│   ├── dynamics.py                        # equations of motion (RHS for the ODE solver)
│   ├── atmosphere.py                      # atmospheric density models
│   ├── atmosphere_OLD_interp1d.py         # prior interp1d-based density model, kept for profile_compare.py
│   ├── gravity.py                         # altitude-dependent gravity
│   ├── wind.py                            # simplified dark-flight wind-drift model
│   ├── wind_plausibility_check.py         # wind-bearing back-calculation against published values
│   ├── integrator.py                      # adaptive (scipy) integration wrapper + impact event
│   ├── fixed_step_integrators.py          # hand-written RK2, RK4, implicit Euler, implicit Heun
│   ├── bunburra.py / cavezzo.py           # per-fall initial conditions, sourced from each paper
│   ├── data/                              # digitized reference points and cached Monte Carlo results
│   ├── plot_trajectory_bunburra.py        # Bunburra deterministic trajectory validation plot
│   ├── plot_trajectory_cavezzo.py         # Cavezzo deterministic trajectory validation plot + height RMSE
│   ├── terminal_point_check.py            # model vs. published terminal-point state, both falls
│   ├── numerical_methods_comparison.py    # accuracy/cost comparison across integrators
│   ├── plot_numerical_comparison.py       # plots for the above comparison
│   ├── plot_piecewise_stepping.py         # phase-aware vs. uniform stepping, stability analysis
│   ├── plot_wind_comparison.py            # effect of wind drift on landing position
│   ├── roundoff_test.py                   # DOP853 tolerance convergence check (truth-solver validation)
│   ├── convergence_order_check.py         # fits global convergence order of RK2/RK4 by step-halving
│   ├── sensitivity_sweep.py               # one-at-a-time Cd/sigma_abl sweep, both falls
│   ├── profile_compare.py                 # interp1d vs. np.interp atmosphere-lookup profiling
│   ├── profiler.py / profile_memory.py    # cProfile / Scalene profiling helpers
│   ├── monte_carlo_uncertainty_*.py       # Monte Carlo sampling + probability-contour maps
│   └── run_all.py                         # runs all deliverable scripts in sequence
├── test/                                  # pytest test suite (see Testing, below)
├── docs/                                  # pdoc HTML documentation of src/ (open docs/index.html)
├── conftest.py                            # adds src/ to the import path for tests
├── pytest.ini                             # pytest configuration
└── requirements.txt
```

Source documentation is generated with `pdoc` and checked into `docs/`.
Open `docs/index.html` in a browser to view it, or regenerate it after a
code change with:

```bash
pdoc -o docs src
```

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`, `pytest`, `pdoc`, `line_profiler`,
  `scalene` (see `requirements.txt` for exact pinned versions)

```bash
pip install -r requirements.txt
```

## Running the project

To reproduce every deliverable plot (both trajectory validations, the
numerical-methods comparison, the stability/piecewise-stepping experiment,
the wind-drift comparison, and both Monte Carlo uncertainty runs) in one
go:

```bash
cd src
python run_all.py
```

Individual scripts can also be run on their own, e.g.:

```bash
cd src
python plot_trajectory_cavezzo.py          # Cavezzo trajectory validation + height RMSE
python plot_trajectory_bunburra.py         # Bunburra trajectory validation
python terminal_point_check.py             # model vs. published terminal-point state, both falls
python monte_carlo_uncertainty_cavezzo.py  # Cavezzo probability map
python sensitivity_sweep.py                # Cd/sigma_abl sensitivity, both falls
python roundoff_test.py                    # DOP853 tolerance convergence check
python convergence_order_check.py          # fitted RK2/RK4 convergence order
```

The Monte Carlo scripts cache their sampled results (`mc_cache_*.npz`) so
repeated runs do not need to resample from scratch.

## Testing

The test suite lives in `test/` and is run with `pytest` from the repo
root:

```bash
pytest
```

It covers four areas, matching this project's validation requirements:

- **Known-value and ground-truth checks**: e.g. gravity at known
  altitudes, coordinate-conversion helpers against hand-computed examples,
  and the Monte Carlo HPD-threshold logic against a known 2D Gaussian
  distribution (`test_hpd_levels_matches_known_gaussian_coverage`).
- **Comparison against an established computational method**: the
  hand-written fixed-step integrators (RK2, RK4, implicit Euler, implicit
  Heun) are checked against the closed-form exact solution of a known test
  ODE, ranked by accuracy order, and their global convergence order is
  fitted directly and checked against theory (`test_convergence_order.py`).
- **Sensitivity and stability testing**: a stiff-ODE case confirming the
  implicit methods remain bounded where the explicit method diverges.
- **Physical constraints and real-world validation**: ablation only
  activates above its velocity threshold, mass stays non-negative, wind
  drift vanishes above its data range, and the Monte Carlo probability maps
  are checked against each meteorite's actual recovered GPS coordinates.

Two tests are marked `xfail` rather than hidden or deleted: the real
recovered coordinates for both falls fall outside (or at the edge of) the
simulated probability contour under the shared default drag/ablation
coefficients. This is a genuine, documented finding, not a test bug. It
isolates physical-model calibration error from numerical error, which is
one of this project's explicit goals (see the reasoning in
`test/test_monte_carlo.py` and `src/plot_numerical_comparison.py`).

Expected result: **89 passed, 2 xfailed** (91 tests total).

## References

- Spurný, P., Bland, P. A., Shrbený, L., Borovička, J., Ceplecha, Z., et
  al. (2012). *The Bunburra Rockhole meteorite fall in SW Australia:
  fireball trajectory, luminosity, dynamics, orbit, and impact position
  from photographic and photoelectric records.* Meteoritics & Planetary
  Science, 47(2), 163-185. https://doi.org/10.1111/j.1945-5100.2011.01321.x
- Gardiol, D., Barghini, D., Buzzoni, A., et al. (2021). *Cavezzo, the
  first Italian meteorite recovered by the PRISMA fireball network. Orbit,
  trajectory, and strewn-field.* Monthly Notices of the Royal Astronomical
  Society, 501(1), 1215-1227. https://doi.org/10.1093/mnras/staa3646. Wind
  speed and direction used in `wind.py` are taken directly from this
  paper's Section 3.4 and Fig. 5.
- Ceplecha, Z., Borovička, J., Elford, W. G., ReVelle, D. O., Hawkes, R. L.,
  Porubčan, V., & Šimek, M. (1998). *Meteor phenomena and bodies.* Space
  Science Reviews, 84(3-4), 327-471. Source of the drag/ablation equations
  of motion used in `dynamics.py`, and of the drag coefficient and
  ablation coefficient default values (Table VII, fitted to the Lost City
  fall).
- U.S. Committee on Extension to the Standard Atmosphere (COESA). (1976).
  *U.S. Standard Atmosphere, 1976.* U.S. Government Printing Office.
  Defining equations implemented directly in `atmosphere.py`'s tabulated
  density model.

## License

See `LICENSE`.