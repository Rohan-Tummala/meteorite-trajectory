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
  (no rotation, no fragmentation), uses a shared drag/ablation coefficient
  pair across both case studies rather than one tuned per meteorite, and
  its dark-flight wind model is a simplified profile fitted to each paper's
  reported wind data, not a full atmospheric reanalysis. These choices and
  their consequences are discussed under Testing, below.

The same core physics is run two ways:

- **Deterministic trajectory validation**: comparing the simulated
  terminal point (end of the luminous trail) against each paper's reported
  values for altitude, velocity and ground track.
- **Monte Carlo uncertainty propagation**: sampling the stated
  measurement uncertainties on entry conditions, propagating them through
  the full nonlinear model, and building a kernel-density probability map
  of likely landing sites, checked against the real recovered coordinates.

## Computational approach and comparison

The full meteoroid trajectory system (`dynamics.py`) is a coupled nonlinear
ODE system with no closed-form solution, so there is no exact analytical
result to compare against directly. That's only possible for the simple
decay-equation test case used in `test/test_integrator.py` to validate
basic solver accuracy. Instead, a tight-tolerance DOP853 run (`rtol=1e-12`,
very small `max_step`) is used as a high-accuracy reference ("truth")
solution. At that tolerance its numerical error is far smaller than any of
the candidate methods being compared, so it stands in for ground truth when
assessing them.

Four alternative integration approaches are compared against that
reference in `numerical_methods_comparison.py` and
`plot_numerical_comparison.py`: the hand-written fixed-step RK2 and RK4
methods, and scipy's adaptive RK45. Each is run across a range of step
sizes (or `max_step` for RK45) and judged on accuracy (impact-location and
altitude-profile error against the reference) versus computational cost
(runtime), rather than on accuracy alone. A method that is only marginally
more accurate but far slower is not necessarily the better choice for this
problem.

`plot_piecewise_stepping.py` then takes the best-performing methods from
that comparison and tests one specific optimisation: splitting the
integration into a fine step during luminous flight (where the dynamics
change fastest) and a coarse step during dark flight (where they do not),
instead of a single uniform fine step across the whole trajectory. This is
compared against the uniform-step baseline on the same accuracy-vs-runtime
basis, to check whether the piecewise approach actually saves computation
for the same accuracy or whether it costs accuracy to do so.

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
near-exact numerics. The xfail tests in `test/test_monte_carlo.py` document
this directly rather than hiding it.

## Monte Carlo uncertainty propagation

For each fall, 5000 joint samples are drawn: the seven entry-state
variables (altitude, speed, flight-path angle, latitude, longitude,
heading, mass) as independent normals centred on the paper's reported
value with its stated 1-sigma uncertainty, plus the ablation coefficient
`sigma_abl` as a uniform draw over the range Ceplecha et al. (1998, Table
VII) fit to the Lost City fall. The drag coefficient `Cd` is held fixed,
since its uncertainty at these entry speeds is negligible next to the
other inputs. Each sample is propagated through the full model to ground
impact, giving a scatter of impact coordinates. A Gaussian KDE is fitted
to that scatter and evaluated over a 200x200 grid covering the footprint,
turning the raw points into a smooth density surface. The density at
every sample point is then sorted and cumulatively summed to find the
threshold that encloses each target probability mass (the HPD levels),
and the map is drawn by filling the region between those thresholds, so
each shaded band is an actual region of highest density rather than a
fixed percentile of the grid.

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
│   ├── gravity.py                         # altitude-dependent gravity
│   ├── wind.py                            # simplified dark-flight wind-drift model
│   ├── integrator.py                      # adaptive (scipy) integration wrapper + impact event
│   ├── fixed_step_integrators.py          # hand-written RK2, RK4, implicit Euler, implicit Heun
│   ├── bunburra.py / cavezzo.py           # per-fall initial conditions, sourced from each paper
│   ├── data/                              # digitized reference points and cached Monte Carlo results
│   ├── plot_trajectory_bunburra.py        # Bunburra deterministic trajectory validation plot
│   ├── plot_trajectory_cavezzo.py         # Cavezzo deterministic trajectory validation plot
│   ├── numerical_methods_comparison.py    # accuracy/cost comparison across integrators
│   ├── plot_numerical_comparison.py       # plots for the above comparison
│   ├── plot_piecewise_stepping.py         # phase-aware vs. uniform stepping, stability analysis
│   ├── plot_wind_comparison.py            # effect of wind drift on landing position
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
- `numpy`, `scipy`, `matplotlib` (see `requirements.txt`)

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
python plot_trajectory_cavezzo.py          # Cavezzo trajectory validation
python plot_trajectory_bunburra.py         # Bunburra trajectory validation
python monte_carlo_uncertainty_cavezzo.py  # Cavezzo probability map
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
  altitudes, coordinate-conversion helpers against hand-computed examples.
- **Comparison against an established computational method**: the
  hand-written fixed-step integrators (RK2, RK4, implicit Euler, implicit
  Heun) are checked against the closed-form exact solution of a known test
  ODE, and ranked by accuracy order.
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

Expected result: **85 passed, 2 xfailed**.

## References

- Spurný, P., et al. (2012). *Atmospheric trajectory and heliocentric orbit
  of the Bunburra Rockhole meteorite fall.*
- Gardiol, D., et al. (2021). *The Cavezzo meteorite fall.* Wind speed and
  direction used in `wind.py` are taken directly from this paper's Section
  3.4 and Fig. 5.
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