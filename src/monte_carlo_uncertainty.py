"""
monte_carlo_uncertainty.py

Joint Monte Carlo propagation of input uncertainty for the Cavezzo
meteorite fall, as distinct from the single-parameter sensitivity
sweep in sensitivity_ablation.py: every uncertain input (V0, gamma0,
phi0, lambda0, psi0, M0, sigma_abl) is sampled SIMULTANEOUSLY, many
times, and propagated to an impact point. The resulting scatter is
used to build a probability density (KDE) contour map -- the same
style of strewn-field visualization used in Moscati et al. (2027)
and comparable planetary-defense literature.

Sampled distributions:
    V0, gamma0, phi0, lambda0, psi0, M0 -- independent normal,
        centred on Gardiol et al. (2021) Table 3 values, with their
        stated 1-sigma uncertainties.
    sigma_abl -- uniform over [7e-8, 9e-8] kg/J, the calibrated range
        from Moscati et al. (2027) (see dynamics.py DEFAULT_PARAMS
        comment for the same citation).
    Cd -- held fixed. Its uncertainty at these Mach numbers is
        negligible compared to the other inputs (established via
        literature check earlier in this project).

Output: a probability contour map (1 sigma/2 sigma/3 sigma highest-density regions)
over the impact footprint, plus the raw sample scatter, the nominal
(baseline) impact point, and the real recovery location.
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from cavezzo import (H0, H0_UNC, V0, V0_UNC, GAMMA0, GAMMA0_UNC, PHI0, PHI0_UNC,
                       LAMBDA0, LAMBDA0_UNC, PSI0, PSI0_UNC, M0, M0_UNC,
                       RECOVERY_LAT_DEG, RECOVERY_LON_DEG, initial_state)
from dynamics import DEFAULT_PARAMS
from integrator import run_trajectory, impact_state

N_SAMPLES = 5000
SIGMA_ABL_RANGE = (7e-8, 9e-8)   # Moscati et al. (2027) calibrated range
RANDOM_SEED = 42


def sample_inputs(rng, n):
    """
    Draw n joint samples of the uncertain inputs. Returns a dict of
    arrays, one per parameter, all length n.
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
    Run the full joint Monte Carlo propagation.

    Returns
    -------
    dict with:
        lats, lons : ndarray -- impact coordinates (degrees) for every
            sample that successfully reached the ground
        n_failed : int -- samples that did not reach the ground within
            the time span (excluded from lats/lons)
        samples : dict -- the raw sampled input arrays (for reference)
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
            params = {**DEFAULT_PARAMS, "sigma_abl": samples["sigma_abl"][i]}
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
    Given the KDE evaluated at every sample point, find the density
    thresholds that enclose each requested probability mass fraction
    (highest-posterior-density regions, not naive percentile bands).

    Default fractions are the TRUE 2D Gaussian-equivalent 1-sigma,
    2-sigma, 3-sigma enclosed probabilities (1 - exp(-n^2/2) for n =
    1, 2, 3) -- NOT the 1D values (68.3%/95.4%/99.7%). This matches
    the convention Gardiol et al. (2021) use for their own published
    Cavezzo strewn-field ellipse, so the two can be compared directly
    on a like-for-like basis.
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
    KDE-based probability contour map of the Monte Carlo impact
    scatter, with 50/80/95% highest-density regions, the raw sample
    scatter, the nominal (baseline-parameter) impact point, and the
    real recovery location.
    """
    lats, lons = result["lats"], result["lons"]

    # Nominal (baseline, no perturbation) impact point for comparison
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y0_nominal = initial_state()
        sol_nominal = run_trajectory(y0_nominal, (0.0, 1500.0), method="RK45",
                                       rtol=1e-8, atol=1e-10,
                                       params=DEFAULT_PARAMS, max_step=5.0)
        y_impact_nominal = impact_state(sol_nominal)
        nominal_lat = np.degrees(y_impact_nominal[3])
        nominal_lon = np.degrees(y_impact_nominal[4])

    # KDE over the impact scatter
    xy = np.vstack([lons, lats])
    kde = gaussian_kde(xy)
    density_at_samples = kde(xy)
    levels = hpd_levels(density_at_samples)  # [1-sigma, 2-sigma, 3-sigma] thresholds, descending

    lon_pad = (lons.max() - lons.min()) * 0.3
    lat_pad = (lats.max() - lats.min()) * 0.3
    lon_grid = np.linspace(lons.min() - lon_pad, lons.max() + lon_pad, grid_res)
    lat_grid = np.linspace(lats.min() - lat_pad, lats.max() + lat_pad, grid_res)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    grid_density = kde(np.vstack([LON.ravel(), LAT.ravel()])).reshape(LON.shape)

    fig, ax = plt.subplots(figsize=(9, 8))

    # Raw sample scatter drawn FIRST (low zorder) and smaller/fainter,
    # so it doesn't wash out the contour fill sitting on top of it --
    # with 5000 points the dots alone were dense enough to obscure the
    # (previously too-pale) shading underneath.
    ax.scatter(lons, lats, s=2.5, color="black", alpha=0.08,
                label="MC samples", zorder=1)

    # Filled contours between HPD levels (levels must be ascending for
    # contourf). Stronger, more saturated palette + higher alpha + a
    # solid zorder above the scatter so the bands stay clearly visible
    # regardless of how many samples are plotted underneath.
    contour_levels = sorted(levels) + [grid_density.max()]
    cs = ax.contourf(LON, LAT, grid_density, levels=contour_levels,
                       colors=["#ffeda0", "#feb24c", "#f03b20"],
                       alpha=0.65, zorder=2)
    ax.contour(LON, LAT, grid_density, levels=sorted(levels),
                colors="black", linewidths=1.0, alpha=0.7, zorder=3)

    ax.plot(nominal_lon, nominal_lat, "o", color="blue", markersize=10,
             markeredgecolor="white", label="Nominal (baseline) impact", zorder=5)
    ax.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="lime",
             markersize=18, markeredgecolor="black",
             label="Real recovery location", zorder=6)

    # Manual legend entries for the HPD bands (contourf doesn't auto-label).
    # Labelled by 2D-equivalent sigma (39.3%/86.5%/98.9% enclosed
    # probability), matching Gardiol et al.'s own 1-sigma/3-sigma
    # convention for the published Cavezzo strewn-field ellipse.
    from matplotlib.patches import Patch
    hpd_patches = [Patch(facecolor=c, alpha=0.65, label=lbl) for c, lbl in
                    zip(["#f03b20", "#feb24c", "#ffeda0"],
                        ["1-sigma (39.3%)", "2-sigma (86.5%)", "3-sigma (98.9%)"])]

    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title(f"Monte Carlo impact-probability map "
                  f"({len(lats)} samples, {result['n_failed']} failed)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + hpd_patches, fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()

    # Summary stats
    from plot_trajectory import great_circle_distance
    dists_km = np.array([
        great_circle_distance(np.radians(la), np.radians(lo),
                                np.radians(RECOVERY_LAT_DEG), np.radians(RECOVERY_LON_DEG)) / 1000.0
        for la, lo in zip(lats, lons)
    ])
    print(f"\nDistance from real recovery location, across all MC samples:")
    print(f"  Median: {np.median(dists_km):.3f} km")
    print(f"  Mean:   {np.mean(dists_km):.3f} km")
    print(f"  Std:    {np.std(dists_km):.3f} km")
    print(f"  Min:    {np.min(dists_km):.3f} km")
    print(f"  Max:    {np.max(dists_km):.3f} km")
    nominal_dist = great_circle_distance(
        np.radians(nominal_lat), np.radians(nominal_lon),
        np.radians(RECOVERY_LAT_DEG), np.radians(RECOVERY_LON_DEG)) / 1000.0
    print(f"  Nominal (baseline) run: {nominal_dist:.3f} km")


if __name__ == "__main__":
    print(f"Running Monte Carlo with N={N_SAMPLES} joint samples...")
    result = run_monte_carlo()
    plot_probability_contour(result)