"""
monte_carlo_uncertainty_bunburra.py

Joint Monte Carlo propagation of input uncertainty for the Bunburra
Rockhole meteorite fall (Spurny et al., 2012) -- the project's second
validation case. Mirrors monte_carlo_uncertainty.py's method (RK45,
KDE-based probability contour) so the two falls are directly
comparable.

Notes
-----
Uncertain inputs and their sampling are documented in sample_inputs().
In short: H0, V0, GAMMA0, PHI0, LAMBDA0, M0, PSI0 are independent
normals using the paper's own reported (or paper-derived) 1-sigma
values; PSI0_UNC specifically comes from the apparent radiant
uncertainty (Table 4), not from differencing the entry/terminal
coordinates -- see bunburra.py for why. sigma_abl reuses the same
Moscati et al. (2027) calibrated range as the Cavezzo run, for a
like-for-like comparison; Cd is held fixed. rho_m is overridden to
Bunburra's own measured bulk density (2700 kg/m^3) via BUNBURRA_PARAMS.

A Monte Carlo run is cached to CACHE_FILE after it completes; running
this script again offers to reuse the cached samples instead of
resimulating from scratch.
"""

import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from bunburra import (H0, H0_UNC, V0, V0_UNC, GAMMA0, GAMMA0_UNC,
                        PHI0, PHI0_UNC, LAMBDA0, LAMBDA0_UNC, M0, M0_UNC,
                        PSI0, PSI0_UNC,
                        RHO_M, RECOVERY_LAT_DEG, RECOVERY_LON_DEG,
                        initial_state)
from dynamics import DEFAULT_PARAMS
from integrator import run_trajectory, impact_state

N_SAMPLES = 5000
SIGMA_ABL_RANGE = (7e-8, 9e-8)   # Moscati et al. (2027) calibrated range -- same as Cavezzo run
RANDOM_SEED = 42

BUNBURRA_PARAMS = {**DEFAULT_PARAMS, "rho_m": RHO_M}

# Running N_SAMPLES propagations is the expensive part of this script.
# Once a run's raw impact scatter is on disk, there's no need to redo
# it just to re-plot or re-check the summary numbers.
CACHE_FILE = "mc_cache_bunburra.npz"


def _load_cache(path):
    """Load a previously-saved Monte Carlo result (see _save_cache)."""
    data = np.load(path)
    return {"lats": data["lats"], "lons": data["lons"],
            "n_failed": int(data["n_failed"])}


def _save_cache(path, result):
    """Save the raw impact scatter so a later run can skip resimulating."""
    np.savez(path, lats=result["lats"], lons=result["lons"],
              n_failed=result["n_failed"])


def sample_inputs(rng, n):
    """
    Draw n joint samples of the uncertain entry-state inputs.

    H0, V0, GAMMA0, PHI0, LAMBDA0, M0 are sampled as independent
    normal distributions centred on the paper-reported value with the
    paper-reported 1-sigma uncertainty. PSI0 (heading) is also sampled
    as an independent normal, using PSI0_UNC -- derived in bunburra.py
    from the apparent radiant uncertainty (Table 4, Spurny et al. 2012),
    NOT from differencing the entry/terminal coordinate uncertainties
    (those two points share a common trajectory fit, so their
    uncertainties are correlated; naive differencing understates the
    true heading uncertainty -- see the PSI0_UNC comment in bunburra.py).

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator to draw from (keeps sampling
        reproducible when seeded by the caller).
    n : int
        Number of joint samples to draw.

    Returns
    -------
    dict[str, ndarray]
        Keys "H0", "V0", "GAMMA0", "PHI0", "LAMBDA0", "PSI0", "M0",
        "sigma_abl", each an array of length n holding one sampled
        value per Monte Carlo draw.
    """
    return {
        "H0": rng.normal(H0, H0_UNC, n),
        "V0": rng.normal(V0, V0_UNC, n),
        "GAMMA0": rng.normal(GAMMA0, GAMMA0_UNC, n),
        "PHI0": rng.normal(PHI0, PHI0_UNC, n),
        "LAMBDA0": rng.normal(LAMBDA0, LAMBDA0_UNC, n),
        "PSI0": rng.normal(PSI0, PSI0_UNC, n),
        "M0": rng.normal(M0, M0_UNC, n),
        "sigma_abl": rng.uniform(*SIGMA_ABL_RANGE, n),
    }


def run_monte_carlo(n_samples=N_SAMPLES, seed=RANDOM_SEED, verbose_every=100):
    """
    Run the joint Monte Carlo propagation for Bunburra Rockhole.

    Draws n_samples joint input samples (sample_inputs), runs each one
    forward through the reference ODE solver (RK45) to ground impact,
    and collects the resulting impact coordinates. Samples that never
    reach the ground within the time span are counted and excluded.

    Parameters
    ----------
    n_samples : int, optional
        Number of joint Monte Carlo samples to run. Default N_SAMPLES.
    seed : int, optional
        Seed for the random number generator, for reproducibility.
        Default RANDOM_SEED.
    verbose_every : int, optional
        Print a progress line every this many samples. Set to 0 or
        None to disable progress output. Default 100.

    Returns
    -------
    dict
        "lats", "lons" : ndarray
            Impact coordinates (degrees) for every sample that
            successfully reached the ground.
        "n_failed" : int
            Number of samples that did not reach the ground within
            the time span (excluded from lats/lons).
        "samples" : dict
            The raw sampled input arrays, as returned by
            sample_inputs(), kept for reference/debugging.
    """
    rng = np.random.default_rng(seed)
    samples = sample_inputs(rng, n_samples)

    lats, lons = [], []
    n_failed = 0

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_samples):
            y0 = np.array([
                samples["H0"][i], samples["V0"][i], samples["GAMMA0"][i],
                samples["PHI0"][i], samples["LAMBDA0"][i],
                samples["PSI0"][i], samples["M0"][i],
            ])
            params = {**BUNBURRA_PARAMS, "sigma_abl": samples["sigma_abl"][i]}
            sol = run_trajectory(y0, (0.0, 1500.0), method="RK45",
                                   rtol=1e-7, atol=1e-9, params=params,
                                   max_step=5.0)
            y_impact = impact_state(sol)
            if y_impact is not None:
                lats.append(np.degrees(y_impact[3]))
                lons.append(np.degrees(y_impact[4]))
            else:
                n_failed += 1

            if verbose_every and (i + 1) % verbose_every == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i+1}/{n_samples} samples "
                      f"({elapsed:.1f}s elapsed, {n_failed} failed so far)")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {len(lats)}/{n_samples} samples reached ground "
          f"({n_failed} failed), {elapsed:.1f}s total")

    return {
        "lats": np.array(lats), "lons": np.array(lons),
        "n_failed": n_failed, "samples": samples,
    }


def hpd_levels(density_values, mass_fractions=(0.393, 0.865, 0.989)):
    """
    Find density thresholds enclosing given highest-posterior-density
    (HPD) probability masses, for contour levels on a KDE surface.

    Same logic as the Cavezzo run's hpd_levels(): sorts the density
    evaluated at every sample point, then finds the thresholds that
    enclose each requested cumulative probability mass -- NOT the
    naive 1D 68.3/95.4/99.7% percentile bands. Default mass fractions
    are the true 2D Gaussian-equivalent 1-sigma/2-sigma/3-sigma
    enclosed probabilities, matching the convention Gardiol et al.
    (2021) use for their published Cavezzo strewn-field ellipse.

    Parameters
    ----------
    density_values : ndarray
        KDE density evaluated at each sample point.
    mass_fractions : tuple of float, optional
        Cumulative probability masses to find HPD thresholds for.
        Default (0.393, 0.865, 0.989) -- 2D 1/2/3-sigma equivalents.

    Returns
    -------
    list of float
        Density threshold for each requested mass fraction, in the
        same order as mass_fractions.
    """
    sorted_density = np.sort(density_values)[::-1]
    cumulative = np.cumsum(sorted_density)
    cumulative /= cumulative[-1]
    thresholds = []
    for frac in mass_fractions:
        idx = np.searchsorted(cumulative, frac)
        idx = min(idx, len(sorted_density) - 1)
        thresholds.append(sorted_density[idx])
    return thresholds


def plot_probability_contour(result, grid_res=200):
    """
    Plot a KDE-based impact-probability contour map and print summary
    validation statistics.

    Builds a 2D kernel density estimate over the Monte Carlo impact
    scatter, draws filled 1-sigma/2-sigma/3-sigma highest-posterior-
    density contours (see hpd_levels), and marks the nominal
    (unperturbed baseline) impact point and the real recovered
    location for comparison. Same structure and HPD convention as the
    Cavezzo run's plot, so the two are directly comparable. Also
    prints great-circle distance statistics from the real recovery
    site across all samples, and the fraction of samples landing at
    least as close as the single-point baseline run.

    Parameters
    ----------
    result : dict
        Output of run_monte_carlo() -- must contain "lats", "lons",
        "n_failed".
    grid_res : int, optional
        Number of grid points per axis for the KDE evaluation grid.
        Default 200.

    Returns
    -------
    None
        Saves the figure to fig_monte_carlo_bunburra_contour.png,
        displays it, and prints summary statistics to stdout.
    """
    lats, lons = result["lats"], result["lons"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y0_nominal = initial_state()
        sol_nominal = run_trajectory(y0_nominal, (0.0, 1500.0), method="RK45",
                                       rtol=1e-8, atol=1e-10,
                                       params=BUNBURRA_PARAMS, max_step=5.0)
        y_impact_nominal = impact_state(sol_nominal)
        nominal_lat = np.degrees(y_impact_nominal[3])
        nominal_lon = np.degrees(y_impact_nominal[4])

    xy = np.vstack([lons, lats])
    kde = gaussian_kde(xy)
    density_at_samples = kde(xy)
    levels = hpd_levels(density_at_samples)

    lon_pad = (lons.max() - lons.min()) * 0.3
    lat_pad = (lats.max() - lats.min()) * 0.3
    lon_grid = np.linspace(lons.min() - lon_pad, lons.max() + lon_pad, grid_res)
    lat_grid = np.linspace(lats.min() - lat_pad, lats.max() + lat_pad, grid_res)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    grid_density = kde(np.vstack([LON.ravel(), LAT.ravel()])).reshape(LON.shape)

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(lons, lats, s=2.5, color="black", alpha=0.08,
                label="MC samples", zorder=1)

    contour_levels = sorted(levels) + [grid_density.max()]
    cs = ax.contourf(LON, LAT, grid_density, levels=contour_levels,
                       colors=["#ffeda0", "#feb24c", "#f03b20"],
                       alpha=0.62, zorder=2)
    ax.contour(LON, LAT, grid_density, levels=sorted(levels),
                colors="black", linewidths=1.0, alpha=0.7, zorder=3)

    ax.plot(nominal_lon, nominal_lat, "o", color="blue", markersize=10,
             markeredgecolor="white", label="Nominal (baseline) impact", zorder=5)
    ax.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="lime",
             markersize=18, markeredgecolor="black",
             label="Real recovery location (M1)", zorder=6)

    from matplotlib.patches import Patch
    hpd_patches = [Patch(facecolor=c, alpha=0.62, label=lbl) for c, lbl in
                    zip(["#f03b20", "#feb24c", "#ffeda0"],
                        ["1-sigma (39.3%)", "2-sigma (86.5%)", "3-sigma (98.9%)"])]

    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title(f"Bunburra Rockhole: Monte Carlo impact-probability map "
                  f"({len(lats)} samples, {result['n_failed']} failed)\n"
                  f"(axes not equal-scaled -- true footprint is far more "
                  f"elongated along the flight path than shown)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + hpd_patches, fontsize=8, loc="best")
    # Deliberately NOT ax.set_aspect("equal") here: a true 1-degree-lon
    # = 1-degree-lat scaling would stretch this footprint into a near-
    # invisible line across the figure (see module docstring / the
    # heading-uncertainty discussion -- Bunburra's real dispersion is
    # genuinely very elongated along the flight path). Auto aspect
    # makes the plot readable; the title says so explicitly so this
    # is never mistaken for the true geographic shape.
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()

    # Which HPD band does the real recovery site actually fall in? This
    # is the validation result itself, not just a description of the plot.
    density_at_recovery = kde(np.array([[RECOVERY_LON_DEG], [RECOVERY_LAT_DEG]]))[0]
    if density_at_recovery >= levels[0]:
        band = "within 1-sigma (39.3%)"
    elif density_at_recovery >= levels[1]:
        band = "within 2-sigma (86.5%)"
    elif density_at_recovery >= levels[2]:
        band = "within 3-sigma (98.9%)"
    else:
        band = "OUTSIDE 3-sigma (98.9%)"

    from plot_trajectory_bunburra import great_circle_distance
    nominal_dist = great_circle_distance(
        np.radians(nominal_lat), np.radians(nominal_lon),
        np.radians(RECOVERY_LAT_DEG), np.radians(RECOVERY_LON_DEG)) / 1000.0

    print(f"\nNominal (baseline) impact: {nominal_lat:.5f} N, {nominal_lon:.5f} E")
    print(f"Real recovery location (M1): {RECOVERY_LAT_DEG:.5f} N, {RECOVERY_LON_DEG:.5f} E")
    print(f"Distance (nominal to recovery): {nominal_dist:.3f} km")
    print(f"Real recovery point falls {band} of the Monte Carlo dispersion")
    print(f"Samples reaching ground: {len(lats)}/{len(lats) + result['n_failed']} "
          f"({result['n_failed']} failed)")


def main():
    if os.path.exists(CACHE_FILE):
        answer = input(f"Cached Monte Carlo results found ({CACHE_FILE}). "
                        f"Re-run simulation? [y/N]: ").strip().lower()
        if answer == "y":
            result = run_monte_carlo()
            _save_cache(CACHE_FILE, result)
        else:
            result = _load_cache(CACHE_FILE)
            print(f"Using cached results from {CACHE_FILE}.")
    else:
        print(f"Running Bunburra Rockhole Monte Carlo with N={N_SAMPLES} joint samples...")
        result = run_monte_carlo()
        _save_cache(CACHE_FILE, result)
    plot_probability_contour(result)


if __name__ == "__main__":
    main()