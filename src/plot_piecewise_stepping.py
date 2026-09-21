"""
plot_piecewise_stepping.py

Tests a hand-crafted TWO-PHASE step size, using physical intuition
about where the dynamics are fast (luminous flight, 0-6s) vs. slow
(dark flight, 6s-to-impact) instead of letting a solver decide
automatically. This is a real technique (multirate / composite
time-stepping), historically used before adaptive solvers were
common, and directly tests whether "manually adaptive" stepping can
approach full adaptive (RK45) performance at lower complexity.

Three (fine, coarse) pairs are tested for each method:
    (0.1s, 10s), (0.2s, 20s), (0.3s, 30s)

For RK2/RK4 (truly fixed-step), the fine/coarse values are literal
step sizes in each phase. For RK45, they're applied as max_step caps
per phase — worth remembering (per the earlier max_step-vs-rtol
finding) that this mainly tests whether a coarse cap on the ALREADY
slow dark-flight phase costs RK45 anything, not a fair "accuracy"
knob the way it is for the fixed-step methods.

Plot style: markers at every actual computed step, connected by
dotted lines, so the discrete nature of each phase's stepping is
visible (dense marks in the fast phase, widely spaced marks in the
slow phase).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dynamics import meteor_rhs, ground_impact_event, DEFAULT_PARAMS
from cavezzo import initial_state
from fixed_step_integrators import integrate_fixed_step
from plot_trajectory import great_circle_distance, RECOVERY_LAT, RECOVERY_LON, \
    RECOVERY_LAT_DEG, RECOVERY_LON_DEG

params = DEFAULT_PARAMS
PHASE_SPLIT = 10.0     # s — boundary between "luminous" and "dark flight" phases
FULL_END = 600.0       # s — generous upper bound for dark flight
# (fine, coarse) pairs, coarse = 100x fine, spanning a much wider and
# more differentiating range than the original 0.1/0.2/0.3 (which all
# behaved too similarly to be informative).
PAIRS = [(2.0, 200.0), (1.0, 100.0), (0.5, 50.0), (0.25, 25.0), (0.125, 12.5)]


def run_truth():
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    sol = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=(0.0, FULL_END), y0=initial_state(), method="DOP853",
        rtol=1e-10, atol=1e-12, dense_output=True,
        events=ground_impact_event,
    )
    return sol


def run_piecewise_fixed(method, dt_fine, dt_coarse):
    """RK2/RK4: literally fixed step size, different per phase."""
    f = lambda t, y: meteor_rhs(t, y, params=params)
    y0 = initial_state()

    # Phase 1: luminous flight, fine step. No ground-impact expected here.
    phase1 = integrate_fixed_step(f, y0, 0.0, PHASE_SPLIT, dt_fine, method=method)
    y_mid = phase1["y"][:, -1]

    # Phase 2: dark flight, coarse step, continuing from phase 1's end state.
    phase2 = integrate_fixed_step(f, y_mid, PHASE_SPLIT, FULL_END, dt_coarse, method=method)

    t_all = np.concatenate([phase1["t"], phase2["t"][1:]])
    y_all = np.concatenate([phase1["y"], phase2["y"][:, 1:]], axis=1)
    event_y = phase2["event_y"]  # None if never reached ground
    return t_all, y_all, event_y, phase1["t"], phase1["y"]


def run_piecewise_rk45(max_step_fine, max_step_coarse):
    """RK45: max_step cap per phase (see module docstring caveat)."""
    y0 = initial_state()

    sol1 = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=(0.0, PHASE_SPLIT), y0=y0, method="RK45",
        rtol=1e-8, atol=1e-10, max_step=max_step_fine,
    )
    y_mid = sol1.y[:, -1]

    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    sol2 = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=(PHASE_SPLIT, FULL_END), y0=y_mid, method="RK45",
        rtol=1e-8, atol=1e-10, max_step=max_step_coarse,
        events=ground_impact_event,
    )

    t_all = np.concatenate([sol1.t, sol2.t[1:]])
    y_all = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1)
    event_y = sol2.y_events[0][0] if len(sol2.t_events[0]) > 0 else None
    return t_all, y_all, event_y, sol1.t, sol1.y


def path_is_sane(y_array, lat_idx=3, lon_idx=4, bound_deg=90):
    lat_deg = np.degrees(y_array[lat_idx, :])
    lon_deg = np.degrees(y_array[lon_idx, :])
    return np.all(np.isfinite(y_array)) and \
        np.all(np.abs(lat_deg) <= bound_deg) and np.all(np.abs(lon_deg) <= 180)


def compute_piecewise_results():
    """Run truth + all method/pair combinations once; return everything
    needed for both the full-trajectory and zoomed-in plots."""
    truth = run_truth()
    lat_truth, lon_truth = truth.y_events[0][0][3], truth.y_events[0][0][4]
    t_truth_fine = np.linspace(0, truth.t[-1], 600)
    h_truth = truth.sol(t_truth_fine)[0] / 1000.0

    methods = [
        ("RK2", "fixed", "rk2"),
        ("RK4", "fixed", "rk4"),
        ("RK45", "rk45", None),
    ]

    colors = plt.cm.plasma(np.linspace(0, 0.8, len(PAIRS)))
    results_by_col = []
    all_lons, all_lats = [np.degrees(lon_truth), RECOVERY_LON_DEG], \
                          [np.degrees(lat_truth), RECOVERY_LAT_DEG]

    for label, kind, fkey in methods:
        entries = []       # full-trajectory entries (only if overall sane)
        zoom_entries = []  # phase-1-only entries (kept even if phase 2 fails)
        raw_full = []      # full t/y arrays, kept REGARDLESS of sanity —
                            # used for the dark-flight zoom, so a divergence
                            # is visible as "the line runs off frame" rather
                            # than being silently dropped
        failed = []
        for (dt_fine, dt_coarse), color in zip(PAIRS, colors):
            if kind == "fixed":
                t_all, y_all, event_y, t_p1, y_p1 = run_piecewise_fixed(fkey, dt_fine, dt_coarse)
            else:
                t_all, y_all, event_y, t_p1, y_p1 = run_piecewise_rk45(dt_fine, dt_coarse)

            # Phase-1 (luminous) data is kept for the zoom plot regardless
            # of what phase 2 does later — the fine step's own behavior in
            # 0-6s is a separate question from whether the coarse phase
            # that follows it happens to blow up.
            zoom_entries.append((dt_fine, dt_coarse, t_p1, y_p1[0] / 1000.0, color))
            raw_full.append((dt_fine, dt_coarse, t_all, y_all[0] / 1000.0, color))

            if event_y is None or not path_is_sane(y_all):
                failed.append((dt_fine, dt_coarse))
                continue

            lat_i, lon_i = event_y[3], event_y[4]
            err = great_circle_distance(lat_i, lon_i, lat_truth, lon_truth)
            print(f"{label:>5} fine={dt_fine}s/coarse={dt_coarse}s: "
                  f"offset from truth = {err:8.2f} m")
            entries.append((dt_fine, dt_coarse, t_all, y_all[0] / 1000.0,
                              lat_i, lon_i, color))
            all_lons.append(np.degrees(lon_i))
            all_lats.append(np.degrees(lat_i))

        results_by_col.append((label, entries, zoom_entries, raw_full, failed))
        if failed:
            print(f"{label}: pairs {failed} failed/unstable in phase 2 — "
                  f"phase-1 (luminous) data still shown in the zoom plot.")

    return {
        "t_truth_fine": t_truth_fine, "h_truth": h_truth,
        "lat_truth": lat_truth, "lon_truth": lon_truth,
        "results_by_col": results_by_col,
        "all_lons": all_lons, "all_lats": all_lats,
    }


def figure_piecewise_full(data, outlier_threshold_km=20.0):
    """Full trajectory, 0 to impact (~450s).

    outlier_threshold_km: points farther than this from the truth
    impact point are still PLOTTED (and printed/reported) but excluded
    from the shared-axis scaling calculation, so one bad outlier (e.g.
    a partially-diverged run) doesn't squash the axis and hide the
    close, informative cluster of points — same fix applied earlier
    in Figure B.
    """
    t_truth_fine, h_truth = data["t_truth_fine"], data["h_truth"]
    lat_truth, lon_truth = data["lat_truth"], data["lon_truth"]
    results_by_col = data["results_by_col"]

    # Recompute shared-axis bounds from only the "close" points (within
    # outlier_threshold_km of truth), always including truth + recovery.
    scale_lons = [np.degrees(lon_truth), RECOVERY_LON_DEG]
    scale_lats = [np.degrees(lat_truth), RECOVERY_LAT_DEG]
    excluded_outliers = []
    for label, entries, zoom_entries, raw_full, failed in results_by_col:
        for dt_fine, dt_coarse, t_all, h_all, lat_i, lon_i, color in entries:
            dist_km = great_circle_distance(lat_i, lon_i, lat_truth, lon_truth) / 1000.0
            if dist_km <= outlier_threshold_km:
                scale_lons.append(np.degrees(lon_i))
                scale_lats.append(np.degrees(lat_i))
            else:
                excluded_outliers.append((label, dt_fine, dt_coarse, dist_km))

    if excluded_outliers:
        print("\nPoints excluded from axis scaling (still plotted, just off-frame):")
        for label, dt_fine, dt_coarse, dist_km in excluded_outliers:
            print(f"  {label} {dt_fine}s/{dt_coarse}s: {dist_km:.1f} km from truth")

    lon_pad = max((max(scale_lons) - min(scale_lons)) * 0.15, 0.005)
    lat_pad = max((max(scale_lats) - min(scale_lats)) * 0.15, 0.005)
    shared_xlim = (min(scale_lons) - lon_pad, max(scale_lons) + lon_pad)
    shared_ylim = (min(scale_lats) - lat_pad, max(scale_lats) + lat_pad)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, (label, entries, zoom_entries, raw_full, failed) in enumerate(results_by_col):
        ax_top, ax_bot = axes[0, col], axes[1, col]

        ax_top.plot(t_truth_fine, h_truth, "-", color="black", linewidth=2, label="Truth")
        ax_bot.plot(np.degrees(lon_truth), np.degrees(lat_truth), "o", color="black",
                     markersize=12, label="Truth", zorder=5)
        ax_bot.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="tab:green",
                     markersize=16, label="Real recovery", zorder=5)

        for dt_fine, dt_coarse, t_all, h_all, lat_i, lon_i, color in entries:
            pair_label = f"{dt_fine}s/{dt_coarse}s"
            ax_top.plot(t_all, h_all, "o:", color=color, markersize=3,
                         linewidth=1, label=pair_label)
            ax_bot.plot(np.degrees(lon_i), np.degrees(lat_i), "o", color=color,
                         markersize=8, label=pair_label)

        ax_top.axvline(PHASE_SPLIT, color="gray", linestyle="--", alpha=0.5,
                         label=f"phase split ({PHASE_SPLIT}s)")
        ax_top.set_ylim(-10, 90)
        ax_top.set_xlabel("Time (s)")
        ax_top.set_ylabel("Height (km)")
        title = f"{label}: piecewise step (fine/coarse), full trajectory"
        if failed:
            title += f"\n({failed} failed, omitted)"
        ax_top.set_title(title, fontsize=9)
        ax_top.legend(fontsize=6)
        ax_top.grid(True, linestyle="--", alpha=0.4)

        ax_bot.set_xlim(*shared_xlim)
        ax_bot.set_ylim(*shared_ylim)
        ax_bot.set_xlabel("Longitude (deg E)")
        ax_bot.set_ylabel("Latitude (deg N)")
        this_col_outliers = [f"{f}s/{c}s ({d:.0f}km)" for lbl, f, c, d in excluded_outliers if lbl == label]
        bot_title = f"{label}: impact location"
        if this_col_outliers:
            bot_title += f"\n({this_col_outliers} off-frame)"
        ax_bot.set_title(bot_title, fontsize=8)
        ax_bot.legend(fontsize=6)
        ax_bot.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Piecewise (multirate) stepping — full trajectory view", fontsize=13)
    fig.tight_layout()
    plt.show()


def figure_piecewise_zoom(data, zoom_end=12.0):
    """Zoomed into the luminous phase (0 to zoom_end s) ONLY, so the
    individual step markers are actually visible — the full-trajectory
    view compresses this whole phase into an unreadable sliver."""
    results_by_col = data["results_by_col"]

    t_fine_zoom = np.linspace(0, zoom_end, 300)
    truth = run_truth()
    h_truth_zoom = truth.sol(t_fine_zoom)[0] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for col, (label, entries, zoom_entries, raw_full, failed) in enumerate(results_by_col):
        ax = axes[col]
        ax.plot(t_fine_zoom, h_truth_zoom, "-", color="black", linewidth=2, label="Truth")

        phase1_diverged = []
        for dt_fine, dt_coarse, t_p1, h_p1, color in zoom_entries:
            if np.any(~np.isfinite(h_p1)) or np.any(np.abs(h_p1) > 200):
                phase1_diverged.append(f"{dt_fine}/{dt_coarse}")
                continue  # would blow out the y-axis and hide everything else
            mask = t_p1 <= zoom_end
            pair_label = f"{dt_fine}s/{dt_coarse}s (phase 1 only)"
            ax.plot(t_p1[mask], h_p1[mask], "o:", color=color, markersize=5,
                     linewidth=1.2, label=pair_label)

        ax.axvline(PHASE_SPLIT, color="gray", linestyle="--", alpha=0.5,
                     label=f"phase split ({PHASE_SPLIT}s)")
        ax.set_xlim(0, zoom_end)
        ax.set_ylim(-10, 90)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (km)")
        title = f"{label}: luminous-phase detail (phase 1 only, fine step)"
        if phase1_diverged:
            title += f"\n(fine steps {phase1_diverged} diverge even in phase 1, omitted)"
        if failed:
            title += f"\n(phase 2 with {failed} failed — full trajectory not shown here)"
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Piecewise stepping — zoomed to the luminous phase (0-{zoom_end:g}s), "
                   "showing actual step markers", fontsize=13)
    fig.tight_layout()
    plt.show()


def figure_piecewise_dark_zoom(data, zoom_start=None, zoom_end=None):
    """
    Zoomed into the dark-flight phase, from the phase split all the way
    to actual ground impact (not an arbitrary cutoff) — using the RAW
    (unfiltered) trajectory data, so for RK2/RK4, where pairs diverge in
    phase 2, you actually SEE the moment the line runs off the frame
    rather than the panel just being empty.
    """
    if zoom_start is None:
        zoom_start = PHASE_SPLIT
    results_by_col = data["results_by_col"]

    truth = run_truth()
    if zoom_end is None:
        zoom_end = truth.t[-1]  # actual impact time, not a guessed cutoff
    t_fine_zoom = np.linspace(zoom_start, zoom_end, 600)
    h_truth_zoom = truth.sol(t_fine_zoom)[0] / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for col, (label, entries, zoom_entries, raw_full, failed) in enumerate(results_by_col):
        ax = axes[col]
        ax.plot(t_fine_zoom, h_truth_zoom, "-", color="black", linewidth=2, label="Truth")

        for dt_fine, dt_coarse, t_all, h_all, color in raw_full:
            mask = (t_all >= zoom_start) & (t_all <= zoom_end)
            pair_label = f"{dt_fine}s/{dt_coarse}s"
            # Values that ran off to inf/nan simply won't render / will
            # break the line naturally — that IS the point of this plot.
            ax.plot(t_all[mask], h_all[mask], "o:", color=color, markersize=5,
                     linewidth=1.2, label=pair_label)

        ax.axvline(PHASE_SPLIT, color="gray", linestyle="--", alpha=0.5,
                     label=f"phase split ({PHASE_SPLIT}s)")
        ax.set_xlim(zoom_start, zoom_end)
        ax.set_ylim(-10, 90)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (km)")
        title = f"{label}: start of dark flight (coarse step)"
        if failed:
            title += f"\n({[f'{a}/{b}' for a, b in failed]} diverge here)"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Piecewise stepping — start of dark flight ({zoom_start}-{zoom_end}s), "
                   "showing where the coarse step breaks down", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Piecewise (multirate) stepping: fine luminous-phase step, "
          "coarse dark-flight step ===")
    data = compute_piecewise_results()
    figure_piecewise_full(data)
    figure_piecewise_zoom(data)
    figure_piecewise_dark_zoom(data)