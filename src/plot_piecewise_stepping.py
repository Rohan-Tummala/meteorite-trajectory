"""
plot_piecewise_stepping.py

Tests a hand-crafted TWO-PHASE step size for meteor trajectory
integration.

The motivation is to investigate whether temporal resolution can be
allocated according to the physical behaviour of the trajectory:

    Phase 1: luminous / rapidly changing flight
              -> small timestep

    Phase 2: dark / slowly changing flight
              -> larger timestep

This is intentionally NOT an adaptive solver. The timestep is prescribed
manually so that the effect of numerical method and timestep size can be
studied explicitly.

Methods compared:
    - RK2
    - RK4
    - RK45 (adaptive internal timestep, with phase-specific max_step caps)

A high-accuracy DOP853 solution is used as a numerical REFERENCE solution.
It is not treated as the exact physical solution.

The experiment investigates:
    1. Effect of timestep size
    2. Difference between RK2 and RK4
    3. Stability / divergence of coarse timesteps
    4. Error in predicted impact location
    5. Error throughout the trajectory
    6. Whether manually finer stepping during luminous flight and
       coarser stepping during dark flight is effective
    7. Computation time vs. accuracy trade-off across methods and step sizes

The piecewise fixed-step experiment uses:

    fine / coarse timestep pairs:

        (0.5 s, 5.0 s)
        (0.25 s, 2.5 s)
        (0.10 s, 1.0 s)
        (0.05 s, 0.5 s)
        (0.025 s, 0.25 s)

The coarse timestep is therefore 10x the fine timestep.

For RK2/RK4, these are literal fixed timesteps.

For RK45, they are maximum-step caps only. RK45 still chooses its
own internal timestep based primarily on rtol/atol.

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dynamics import meteor_rhs, ground_impact_event, DEFAULT_PARAMS
from cavezzo import initial_state
from fixed_step_integrators import integrate_fixed_step
from plot_trajectory import (
    great_circle_distance,
    RECOVERY_LAT,
    RECOVERY_LON,
    RECOVERY_LAT_DEG,
    RECOVERY_LON_DEG,
)


# ============================================================================
# SETTINGS
# ============================================================================

params = DEFAULT_PARAMS

# Boundary between luminous and dark flight.
PHASE_SPLIT = 10.0  # seconds

# Generous upper bound for dark flight.
FULL_END = 600.0  # seconds

# Fine/coarse timestep pairs. Coarse = 10x fine.
PAIRS = [
    (0.50, 5.00),
    (0.25, 2.50),
    (0.10, 1.00),
    (0.05, 0.50),
    (0.025, 0.25),
]

# Reference solver tolerances.
REFERENCE_RTOL = 1e-10
REFERENCE_ATOL = 1e-12

# RK45 tolerances.
RK45_RTOL = 1e-8
RK45_ATOL = 1e-10

# Threshold used only to decide whether a numerical trajectory has become
# obviously non-physical / numerically unstable.
HEIGHT_LIMIT_KM = 1e5

# Colours used consistently across the summary/time plots.
METHOD_COLORS = {"RK2": "tab:orange", "RK4": "tab:green", "RK45": "tab:blue"}


# ============================================================================
# REFERENCE SOLUTION
# ============================================================================

def run_reference():
    """
    Generate a high-accuracy numerical reference solution using DOP853.

    This is a numerical reference, NOT an exact physical solution.
    """

    ground_impact_event.terminal = True
    ground_impact_event.direction = -1

    sol = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=(0.0, FULL_END),
        y0=initial_state(),
        method="DOP853",
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
        dense_output=True,
        events=ground_impact_event,
    )

    return sol


# ============================================================================
# FIXED-STEP PIECEWISE INTEGRATION
# ============================================================================

def run_piecewise_fixed(method, dt_fine, dt_coarse):
    """
    RK2/RK4 piecewise fixed-step integration.

    Phase 1:
        0 -> PHASE_SPLIT using dt_fine

    Phase 2:
        PHASE_SPLIT -> FULL_END using dt_coarse
    """

    f = lambda t, y: meteor_rhs(t, y, params=params)

    y0 = initial_state()

    phase1 = integrate_fixed_step(
        f,
        y0,
        0.0,
        PHASE_SPLIT,
        dt_fine,
        method=method,
    )

    y_mid = phase1["y"][:, -1]

    phase2 = integrate_fixed_step(
        f,
        y_mid,
        PHASE_SPLIT,
        FULL_END,
        dt_coarse,
        method=method,
    )

    t_all = np.concatenate([
        phase1["t"],
        phase2["t"][1:],
    ])

    y_all = np.concatenate([
        phase1["y"],
        phase2["y"][:, 1:],
    ], axis=1)

    event_y = phase2["event_y"]

    return (
        t_all,
        y_all,
        event_y,
        phase1["t"],
        phase1["y"],
    )


# ============================================================================
# PIECEWISE RK45
# ============================================================================

def run_piecewise_rk45(max_step_fine, max_step_coarse):
    """
    RK45 piecewise integration.

    IMPORTANT:
        max_step does NOT force RK45 to use that timestep.

    RK45 still chooses its own internal timestep using its error control.
    The supplied values therefore act only as upper bounds.
    """

    y0 = initial_state()

    sol1 = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=(0.0, PHASE_SPLIT),
        y0=y0,
        method="RK45",
        rtol=RK45_RTOL,
        atol=RK45_ATOL,
        max_step=max_step_fine,
    )

    y_mid = sol1.y[:, -1]

    ground_impact_event.terminal = True
    ground_impact_event.direction = -1

    sol2 = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=(PHASE_SPLIT, FULL_END),
        y0=y_mid,
        method="RK45",
        rtol=RK45_RTOL,
        atol=RK45_ATOL,
        max_step=max_step_coarse,
        events=ground_impact_event,
    )

    t_all = np.concatenate([
        sol1.t,
        sol2.t[1:],
    ])

    y_all = np.concatenate([
        sol1.y,
        sol2.y[:, 1:],
    ], axis=1)

    if len(sol2.t_events[0]) > 0:
        event_y = sol2.y_events[0][0]
    else:
        event_y = None

    return (
        t_all,
        y_all,
        event_y,
        sol1.t,
        sol1.y,
    )


# ============================================================================
# SANITY / STABILITY CHECKS
# ============================================================================

def path_is_sane(
    y_array,
    lat_idx=3,
    lon_idx=4,
    bound_deg=90,
):
    """
    Basic numerical sanity check.

    Returns False if:
        - NaNs/infinities occur
        - latitude exceeds physical bounds
        - longitude exceeds physical bounds
        - height becomes absurdly large
    """

    if not np.all(np.isfinite(y_array)):
        return False

    lat_deg = np.degrees(y_array[lat_idx, :])
    lon_deg = np.degrees(y_array[lon_idx, :])

    if not np.all(np.abs(lat_deg) <= bound_deg):
        return False

    if not np.all(np.abs(lon_deg) <= 180):
        return False

    height_km = np.abs(y_array[0, :]) / 1000.0

    if np.any(height_km > HEIGHT_LIMIT_KM):
        return False

    return True


# ============================================================================
# REFERENCE INTERPOLATION
# ============================================================================

def interpolate_reference(reference, t):
    """
    Evaluate the dense DOP853 reference solution at arbitrary times.
    """

    t_clipped = np.clip(
        t,
        reference.t[0],
        reference.t[-1],
    )

    return reference.sol(t_clipped)


# ============================================================================
# TRAJECTORY ERROR
# ============================================================================

def calculate_trajectory_error(t, y, reference):
    """
    Calculate trajectory/state error relative to the reference solution.

    Returns:
        errors : array with shape (n_states, n_times)
    """

    y_ref = interpolate_reference(reference, t)

    errors = np.abs(y - y_ref)

    return errors


# ============================================================================
# IMPACT ERROR
# ============================================================================

def calculate_impact_error(event_y, reference_event_y):
    """
    Great-circle distance between numerical and reference impact location.
    """

    if event_y is None:
        return None

    if reference_event_y is None:
        return None

    lat_i = event_y[3]
    lon_i = event_y[4]

    lat_ref = reference_event_y[3]
    lon_ref = reference_event_y[4]

    return great_circle_distance(
        lat_i,
        lon_i,
        lat_ref,
        lon_ref,
    )


# ============================================================================
# COMPUTE ALL RESULTS
# ============================================================================

def compute_piecewise_results():
    """
    Run the reference solution and every method/timestep combination.

    Results are retained even when a trajectory becomes unstable so that
    the divergence itself can be plotted and analysed. Wall-clock runtime
    is recorded per run for the computation-time-vs-accuracy comparison.
    """

    print("\n" + "=" * 80)
    print("REFERENCE SOLUTION")
    print("=" * 80)

    reference = run_reference()

    if len(reference.t_events[0]) > 0:
        reference_event_y = reference.y_events[0][0]
        reference_impact_time = reference.t_events[0][0]

        lat_ref = np.degrees(reference_event_y[3])
        lon_ref = np.degrees(reference_event_y[4])

        print(f"Reference impact time : {reference_impact_time:.8f} s")
        print(f"Reference latitude    : {lat_ref:.8f} deg")
        print(f"Reference longitude   : {lon_ref:.8f} deg")

    else:
        reference_event_y = None
        reference_impact_time = None
        print("WARNING: Reference solution did not reach ground.")

    t_reference_plot = np.linspace(0, reference.t[-1], 1000)
    y_reference_plot = reference.sol(t_reference_plot)
    height_reference_plot = y_reference_plot[0] / 1000.0

    methods = [
        ("RK2", "fixed", "rk2"),
        ("RK4", "fixed", "rk4"),
        ("RK45", "rk45", None),
    ]

    colors = plt.cm.plasma(np.linspace(0, 0.8, len(PAIRS)))

    results_by_method = []
    all_lons = []
    all_lats = []

    if reference_event_y is not None:
        all_lons.append(np.degrees(reference_event_y[4]))
        all_lats.append(np.degrees(reference_event_y[3]))

    all_lons.append(RECOVERY_LON_DEG)
    all_lats.append(RECOVERY_LAT_DEG)

    for label, kind, fkey in methods:

        entries = []
        phase1_entries = []
        raw_full = []
        failed = []

        print("\n" + "=" * 80)
        print(label)
        print("=" * 80)

        for (dt_fine, dt_coarse), color in zip(PAIRS, colors):

            # ------------------------------------------------------
            # Run (timed)
            # ------------------------------------------------------

            t0 = time.perf_counter()

            if kind == "fixed":
                (t_all, y_all, event_y, t_p1, y_p1) = run_piecewise_fixed(
                    fkey, dt_fine, dt_coarse,
                )
            else:
                (t_all, y_all, event_y, t_p1, y_p1) = run_piecewise_rk45(
                    dt_fine, dt_coarse,
                )

            runtime_s = time.perf_counter() - t0

            phase1_entries.append((dt_fine, dt_coarse, t_p1, y_p1, color))
            raw_full.append((dt_fine, dt_coarse, t_all, y_all, color))

            sane = path_is_sane(y_all)
            impact_error = calculate_impact_error(event_y, reference_event_y)

            if np.all(np.isfinite(y_all)):
                errors = calculate_trajectory_error(t_all, y_all, reference)
                height_error = errors[0, :]
                max_height_error = np.nanmax(height_error)
            else:
                errors = None
                max_height_error = np.inf

            n_steps = len(t_all) - 1

            if event_y is None:
                status = "NO IMPACT"
                failed.append((dt_fine, dt_coarse))
            elif not sane:
                status = "UNSTABLE"
                failed.append((dt_fine, dt_coarse))
            else:
                status = "OK"

            if impact_error is not None:
                print(
                    f"{label:>5} fine={dt_fine:<7g} coarse={dt_coarse:<7g} "
                    f"steps={n_steps:<6d} runtime={runtime_s:8.5f}s "
                    f"impact error={impact_error:10.3f} m "
                    f"max height error={max_height_error:10.3f} m [{status}]"
                )
            else:
                print(
                    f"{label:>5} fine={dt_fine:<7g} coarse={dt_coarse:<7g} "
                    f"steps={n_steps:<6d} runtime={runtime_s:8.5f}s "
                    f"impact error={'N/A':>10} "
                    f"max height error={max_height_error:>10.3g} [{status}]"
                )

            if event_y is not None and sane:
                lat_i = event_y[3]
                lon_i = event_y[4]
                all_lons.append(np.degrees(lon_i))
                all_lats.append(np.degrees(lat_i))

            entries.append({
                "dt_fine": dt_fine,
                "dt_coarse": dt_coarse,
                "t": t_all,
                "y": y_all,
                "event_y": event_y,
                "color": color,
                "sane": sane,
                "status": status,
                "impact_error": impact_error,
                "trajectory_errors": errors,
                "max_height_error": max_height_error,
                "n_steps": n_steps,
                "runtime_s": runtime_s,
            })

        results_by_method.append({
            "label": label,
            "entries": entries,
            "phase1_entries": phase1_entries,
            "raw_full": raw_full,
            "failed": failed,
        })

        if failed:
            print(f"\n{label}: {len(failed)} pair(s) failed or became unstable.")

    return {
        "reference": reference,
        "reference_event_y": reference_event_y,
        "reference_impact_time": reference_impact_time,
        "t_reference_plot": t_reference_plot,
        "height_reference_plot": height_reference_plot,
        "results_by_method": results_by_method,
        "all_lons": all_lons,
        "all_lats": all_lats,
    }


# ============================================================================
# FIGURE 1 — FULL TRAJECTORY
# ============================================================================

def figure_piecewise_full(data, outlier_threshold_km=20.0):
    """
    Full trajectory comparison.

    Successful trajectories are shown together with the reference.
    Unstable trajectories are also retained but clipped from the axis
    scaling so that they do not hide the useful results.
    """

    t_reference_plot = data["t_reference_plot"]
    height_reference_plot = data["height_reference_plot"]
    reference_event_y = data["reference_event_y"]
    results_by_method = data["results_by_method"]

    scale_lons = [RECOVERY_LON_DEG]
    scale_lats = [RECOVERY_LAT_DEG]

    if reference_event_y is not None:
        scale_lons.append(np.degrees(reference_event_y[4]))
        scale_lats.append(np.degrees(reference_event_y[3]))

    excluded_outliers = []

    for method_data in results_by_method:
        label = method_data["label"]
        for entry in method_data["entries"]:
            event_y = entry["event_y"]
            if event_y is None or not entry["sane"]:
                continue
            lat_i, lon_i = event_y[3], event_y[4]
            if reference_event_y is not None:
                dist_km = great_circle_distance(
                    lat_i, lon_i, reference_event_y[3], reference_event_y[4],
                ) / 1000.0
            else:
                dist_km = 0.0
            if dist_km <= outlier_threshold_km:
                scale_lons.append(np.degrees(lon_i))
                scale_lats.append(np.degrees(lat_i))
            else:
                excluded_outliers.append((label, entry["dt_fine"], entry["dt_coarse"], dist_km))

    if excluded_outliers:
        print("\nPoints excluded from map-axis scaling (still exist in the data):")
        for label, dt_fine, dt_coarse, dist_km in excluded_outliers:
            print(f"  {label} {dt_fine}s/{dt_coarse}s = {dist_km:.1f} km")

    lon_range = max(scale_lons) - min(scale_lons)
    lat_range = max(scale_lats) - min(scale_lats)
    lon_pad = max(lon_range * 0.15, 0.005)
    lat_pad = max(lat_range * 0.15, 0.005)
    shared_xlim = (min(scale_lons) - lon_pad, max(scale_lons) + lon_pad)
    shared_ylim = (min(scale_lats) - lat_pad, max(scale_lats) + lat_pad)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for col, method_data in enumerate(results_by_method):
        label = method_data["label"]
        ax_top, ax_bot = axes[0, col], axes[1, col]

        ax_top.plot(t_reference_plot, height_reference_plot, "-",
                     color="black", linewidth=2, label="DOP853 reference")

        for entry in method_data["entries"]:
            t = entry["t"]
            height = entry["y"][0] / 1000.0
            pair_label = f"{entry['dt_fine']:g}s/{entry['dt_coarse']:g}s"
            mask = np.isfinite(t) & np.isfinite(height) & (np.abs(height) < 1e5)
            ax_top.plot(t[mask], height[mask], "o:", color=entry["color"],
                         markersize=3, linewidth=1, label=pair_label)

        ax_top.axvline(PHASE_SPLIT, color="gray", linestyle="--", alpha=0.5,
                         label=f"phase split ({PHASE_SPLIT:g}s)")
        ax_top.set_ylim(-10, 90)
        ax_top.set_xlabel("Time (s)")
        ax_top.set_ylabel("Height (km)")
        ax_top.set_title(f"{label}: piecewise trajectory", fontsize=10)
        ax_top.legend(fontsize=6)
        ax_top.grid(True, linestyle="--", alpha=0.4)

        if reference_event_y is not None:
            ax_bot.plot(np.degrees(reference_event_y[4]), np.degrees(reference_event_y[3]),
                         "o", color="black", markersize=12, label="DOP853 reference", zorder=5)
        ax_bot.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="tab:green",
                     markersize=16, label="Real recovery", zorder=5)

        for entry in method_data["entries"]:
            event_y = entry["event_y"]
            if event_y is None:
                continue
            pair_label = f"{entry['dt_fine']:g}s/{entry['dt_coarse']:g}s"
            ax_bot.plot(np.degrees(event_y[4]), np.degrees(event_y[3]), "o",
                         color=entry["color"], markersize=8, label=pair_label)

        ax_bot.set_xlim(*shared_xlim)
        ax_bot.set_ylim(*shared_ylim)
        ax_bot.set_xlabel("Longitude (deg E)")
        ax_bot.set_ylabel("Latitude (deg N)")
        ax_bot.set_title(f"{label}: impact location", fontsize=9)
        ax_bot.legend(fontsize=6)
        ax_bot.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Piecewise stepping — full trajectory comparison", fontsize=14)
    fig.tight_layout()
    plt.show()


# ============================================================================
# FIGURE 2 — LUMINOUS PHASE
# ============================================================================

def figure_piecewise_luminous_zoom(data, zoom_end=12.0):
    """
    Detailed view of the luminous / rapidly changing region.

    Only phase-1 data is plotted, so this figure isolates the effect
    of the fine timestep.
    """

    results_by_method = data["results_by_method"]
    reference = data["reference"]

    t_reference = np.linspace(0, zoom_end, 500)
    h_reference = reference.sol(t_reference)[0] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for col, method_data in enumerate(results_by_method):
        label = method_data["label"]
        ax = axes[col]
        ax.plot(t_reference, h_reference, "-", color="black", linewidth=2,
                 label="DOP853 reference")

        for dt_fine, dt_coarse, t_p1, y_p1, color in method_data["phase1_entries"]:
            height = y_p1[0] / 1000.0
            mask = (t_p1 <= zoom_end) & np.isfinite(height)
            pair_label = f"{dt_fine:g}s/{dt_coarse:g}s"
            ax.plot(t_p1[mask], height[mask], "o:", color=color,
                     markersize=4, linewidth=1.2, label=pair_label)

        ax.axvline(PHASE_SPLIT, color="gray", linestyle="--", alpha=0.5,
                    label=f"phase split ({PHASE_SPLIT:g}s)")
        ax.set_xlim(0, zoom_end)
        ax.set_ylim(-10, 90)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (km)")
        ax.set_title(f"{label}: luminous-phase detail", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Fine-step behaviour during luminous flight", fontsize=13)
    fig.tight_layout()
    plt.show()


# ============================================================================
# FIGURE 3 — DARK PHASE
# ============================================================================

def figure_piecewise_dark_zoom(data, zoom_start=None, zoom_end=None):
    """
    Detailed view of the dark-flight phase.

    RAW trajectories are used deliberately, including unstable runs,
    so that coarse-step divergence is visible.
    """

    if zoom_start is None:
        zoom_start = PHASE_SPLIT

    reference = data["reference"]
    if zoom_end is None:
        zoom_end = reference.t[-1]

    results_by_method = data["results_by_method"]

    t_reference = np.linspace(zoom_start, zoom_end, 700)
    h_reference = reference.sol(t_reference)[0] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for col, method_data in enumerate(results_by_method):
        label = method_data["label"]
        ax = axes[col]
        ax.plot(t_reference, h_reference, "-", color="black", linewidth=2,
                 label="DOP853 reference")

        for dt_fine, dt_coarse, t_all, y_all, color in method_data["raw_full"]:
            height = y_all[0] / 1000.0
            mask = ((t_all >= zoom_start) & (t_all <= zoom_end)
                     & np.isfinite(t_all) & np.isfinite(height) & (np.abs(height) < 1e5))
            pair_label = f"{dt_fine:g}s/{dt_coarse:g}s"
            ax.plot(t_all[mask], height[mask], "o:", color=color,
                     markersize=5, linewidth=1.2, label=pair_label)

        ax.axvline(PHASE_SPLIT, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_ylim(-10, 90)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (km)")
        ax.set_title(f"{label}: dark-flight behaviour", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Coarse-step behaviour during dark flight", fontsize=13)
    fig.tight_layout()
    plt.show()


# ============================================================================
# FIGURE 4 — IMPACT ERROR VS TIMESTEP
# ============================================================================

def figure_impact_error(data):
    """
    Plot impact-location error against coarse timestep.
    """

    results_by_method = data["results_by_method"]
    fig, ax = plt.subplots(figsize=(9, 6))

    for method_data in results_by_method:
        label = method_data["label"]
        dt_values, errors = [], []

        for entry in method_data["entries"]:
            if entry["impact_error"] is None or not np.isfinite(entry["impact_error"]):
                continue
            dt_values.append(entry["dt_coarse"])
            errors.append(entry["impact_error"])

        if len(dt_values) == 0:
            continue

        order = np.argsort(dt_values)
        dt_values = np.array(dt_values)[order]
        errors = np.array(errors)[order]

        ax.loglog(dt_values, errors, "o-", linewidth=1.5, markersize=7,
                   color=METHOD_COLORS.get(label), label=label)

    ax.set_xlabel("Dark-flight timestep (s)")
    ax.set_ylabel("Impact-location error (m)")
    ax.set_title("Impact-location error vs coarse timestep")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# FIGURE 5 — MAXIMUM HEIGHT ERROR
# ============================================================================

def figure_height_error(data):
    """
    Compare maximum absolute height error against coarse timestep.
    """

    results_by_method = data["results_by_method"]
    fig, ax = plt.subplots(figsize=(9, 6))

    for method_data in results_by_method:
        label = method_data["label"]
        dt_values, errors = [], []

        for entry in method_data["entries"]:
            error = entry["max_height_error"]
            if not np.isfinite(error):
                continue
            dt_values.append(entry["dt_coarse"])
            errors.append(error)

        if len(dt_values) == 0:
            continue

        order = np.argsort(dt_values)
        dt_values = np.array(dt_values)[order]
        errors = np.array(errors)[order]

        ax.loglog(dt_values, errors, "o-", linewidth=1.5, markersize=7,
                   color=METHOD_COLORS.get(label), label=label)

    ax.set_xlabel("Dark-flight timestep (s)")
    ax.set_ylabel("Maximum height error (m)")
    ax.set_title("Maximum trajectory height error vs timestep")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# FIGURE 6 — TRAJECTORY ERROR OVER TIME
# ============================================================================

def figure_trajectory_error(data, method_name="RK2"):
    """
    Show how height error develops throughout the trajectory.
    """

    method_data = None
    for candidate in data["results_by_method"]:
        if candidate["label"] == method_name:
            method_data = candidate
            break

    if method_data is None:
        print(f"Method {method_name} not found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for entry in method_data["entries"]:
        errors = entry["trajectory_errors"]
        if errors is None:
            continue
        t = entry["t"]
        height_error = errors[0, :]
        mask = np.isfinite(t) & np.isfinite(height_error) & (height_error > 0)
        pair_label = f"{entry['dt_fine']:g}s/{entry['dt_coarse']:g}s"
        ax.semilogy(t[mask], height_error[mask], "o-", markersize=2.5,
                     linewidth=1, label=pair_label)

    ax.axvline(PHASE_SPLIT, color="gray", linestyle="--", alpha=0.5, label="phase split")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute height error (m)")
    ax.set_title(f"{method_name}: height error relative to DOP853 reference")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


# ============================================================================
# FIGURE 7 — COMPUTATION TIME VS ACCURACY
# ============================================================================

def figure_computation_time(data):
    """
    Plot computation time vs. accuracy (both impact-location error and
    max height error), log-log, one line per method, one point per
    (fine, coarse) step-size pair. This is the accuracy-vs-cost trade-off
    plot, using actual wall-clock runtime as the cost axis (as opposed
    to step count or function-evaluation count).
    """

    results_by_method = data["results_by_method"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for method_data in results_by_method:
        label = method_data["label"]
        color = METHOD_COLORS.get(label, "gray")

        rows = [
            (entry["runtime_s"], entry["impact_error"], entry["max_height_error"],
             entry["dt_fine"], entry["dt_coarse"])
            for entry in method_data["entries"]
            if entry["status"] == "OK" and entry["impact_error"] is not None
        ]
        if not rows:
            continue

        rows.sort(key=lambda r: r[0])
        times = [r[0] for r in rows]
        impact_errs = [r[1] for r in rows]
        height_errs = [r[2] for r in rows]
        dt_coarses = [r[4] for r in rows]

        axes[0].loglog(times, impact_errs, "o-", color=color, markersize=7, label=label)
        axes[1].loglog(times, height_errs, "o-", color=color, markersize=7, label=label)

        for t_val, err_val, dt_coarse in zip(times, impact_errs, dt_coarses):
            axes[0].annotate(f"{dt_coarse:g}s", (t_val, err_val), fontsize=6,
                               xytext=(4, 4), textcoords="offset points", alpha=0.7)

    axes[0].set_xlabel("Computation time (s)")
    axes[0].set_ylabel("Impact-location error vs. reference (m)")
    axes[0].set_title("Runtime vs. impact-location error")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_xlabel("Computation time (s)")
    axes[1].set_ylabel("Max height error vs. reference (m)")
    axes[1].set_title("Runtime vs. max height error")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle("Computation time vs. accuracy, by method and step size", fontsize=13)
    fig.tight_layout()
    plt.show()


# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================

def print_summary(data):
    """
    Print a compact summary suitable for comparing methods.
    """

    print("\n")
    print("=" * 125)
    print("SUMMARY")
    print("=" * 125)

    print(
        f"{'Method':<8}{'Fine':>10}{'Coarse':>10}{'Steps':>10}{'Runtime (s)':>14}"
        f"{'Impact error (m)':>20}{'Max height error (m)':>24}{'Status':>15}"
    )
    print("-" * 125)

    for method_data in data["results_by_method"]:
        label = method_data["label"]
        for entry in method_data["entries"]:
            impact_error = entry["impact_error"]
            max_height_error = entry["max_height_error"]

            impact_string = "N/A" if impact_error is None else f"{impact_error:.4f}"
            height_string = f"{max_height_error:.4f}" if np.isfinite(max_height_error) else "INF"

            print(
                f"{label:<8}{entry['dt_fine']:>10g}{entry['dt_coarse']:>10g}"
                f"{entry['n_steps']:>10d}{entry['runtime_s']:>14.5f}"
                f"{impact_string:>20}{height_string:>24}{entry['status']:>15}"
            )

    print("=" * 125)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print(
        "\n"
        "==============================================================\n"
        " PIECEWISE / MULTIRATE METEOR TRAJECTORY EXPERIMENT\n"
        "==============================================================\n"
    )

    print(f"Phase split: {PHASE_SPLIT:g} s")
    print(f"Simulation end: {FULL_END:g} s")
    print("Timestep pairs:")
    for fine, coarse in PAIRS:
        print(f"    {fine:g} s / {coarse:g} s")
    print("\nReference: DOP853 with tight tolerances")

    data = compute_piecewise_results()

    print_summary(data)

    figure_piecewise_full(data)
    figure_piecewise_luminous_zoom(data)
    figure_piecewise_dark_zoom(data)
    figure_impact_error(data)
    figure_height_error(data)
    figure_computation_time(data)

    figure_trajectory_error(data, method_name="RK2")
    figure_trajectory_error(data, method_name="RK4")