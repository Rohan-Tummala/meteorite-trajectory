"""
plot_numerical_comparison.py

Figure A ("ground truth vs step size = 5"):
    Left  - full-trajectory altitude vs time (0 to impact) for the
             truth solver and RK2/RK4/RK45 all run with step size 5s
             (for RK45, "step size 5" means max_step=5s, since it is
             adaptive internally but capped at that maximum).
    Right - ground track (lat/lon) for the same four, with the real
             recovery location marked.
    Purpose: shows, at a glance, how far a too-coarse step size (5s)
    pushes each method from the truth solution — i.e. numerical error
    at a deliberately under-resolved step size, distinguished from
    physical-model error (which is a separate investigation, e.g.
    sensitivity_ablation.py).

Figure B (6-panel step-size convergence grid):
    3 columns: RK2, RK4, RK45. Step sizes tested: 20, 10, 5, 2.5, 1,
    0.5, 0.1 (s) - for RK45 this is applied as max_step, holding rtol
    fixed at a normal value (1e-8), so all three columns are driven by
    the same nominal "step size" list for a fair visual comparison.
    Top row    - altitude vs time (0-6s, luminous flight only, where
                  step-size differences are visible) for that method
                  at each step size, plotted against the truth curve.
    Bottom row - resulting impact-location (lat/lon) for that method
                  at each step size, plotted against the truth's
                  impact point. Runs that diverge wildly are noted in
                  the console and excluded from the scatter's axis
                  scaling so the informative points stay legible.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from dynamics import meteor_rhs, ground_impact_event, DEFAULT_PARAMS
from cavezzo import initial_state, RECOVERY_LAT, RECOVERY_LON, \
    RECOVERY_LAT_DEG, RECOVERY_LON_DEG
from fixed_step_integrators import integrate_fixed_step
from plot_trajectory import great_circle_distance

params = DEFAULT_PARAMS
FULL_SPAN = (0.0, 600.0)
ZOOM_SPAN = (0.0, 6.0)
STEP_SIZES = [20.0, 10.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5, 0.3, 0.2, 0.1]

# RK45's real accuracy control is rtol, not max_step (max_step only ever
# binds during the calm dark-flight phase for this problem -- see the
# diagnostic trace showing RK45 self-selects sub-0.15s steps during the
# violent early phase regardless of any cap up to 20s). So RK45's column
# sweeps rtol instead, the fair equivalent of testing how coarse the
# method can be.
RK45_TOLERANCES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]


# ---------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------

def run_truth(t_span=FULL_SPAN):
    """DOP853 at rtol=1e-10 — the practical accuracy floor for this
    problem (see roundoff_test.py: tightening further changes nothing
    but cost)."""
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    sol = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=t_span, y0=initial_state(), method="DOP853",
        rtol=1e-10, atol=1e-12, dense_output=True,
        events=ground_impact_event,
    )
    return sol


def run_rk45_tol(rtol, t_span=FULL_SPAN):
    """RK45 with NO max_step cap — rtol is the actual accuracy knob."""
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    sol = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=t_span, y0=initial_state(), method="RK45",
        rtol=rtol, atol=rtol * 1e-2,
        dense_output=True, events=ground_impact_event,
    )
    return sol


def run_rk45_capped(max_step, t_span=FULL_SPAN, rtol=1e-8):
    """
    RK45 with a max_step CAP (not a tolerance sweep) — used only in
    Figure A, where the point is a like-for-like "same coarse step
    size as RK2/RK4" comparison, not testing RK45's own accuracy
    limits (that's what run_rk45_tol / Figure B is for). Note: because
    RK45 is adaptive, this cap only ever binds during the calm dark-
    flight phase for this problem -- see the max_step-vs-rtol finding
    from Figure B's own investigation.
    """
    ground_impact_event.terminal = True
    ground_impact_event.direction = -1
    sol = solve_ivp(
        fun=lambda t, y: meteor_rhs(t, y, params=params),
        t_span=t_span, y0=initial_state(), method="RK45",
        rtol=rtol, atol=rtol * 1e-2, max_step=max_step,
        dense_output=True, events=ground_impact_event,
    )
    return sol


def run_fixed(method, dt, t_span=FULL_SPAN):
    f = lambda t, y: meteor_rhs(t, y, params=params)
    return integrate_fixed_step(f, initial_state(), t_span[0], t_span[1], dt, method=method)


def path_is_sane(y_array, lat_idx=3, lon_idx=4, bound_deg=90):
    """
    Check the ENTIRE path, not just the final point — a run can wander
    to absurd values mid-flight and coincidentally cross back through
    a plausible-looking point right when the ground-impact event
    triggers, which would make a final-point-only check dangerously
    misleading (looks fine, is actually wildly unstable throughout).
    """
    lat_deg = np.degrees(y_array[lat_idx, :])
    lon_deg = np.degrees(y_array[lon_idx, :])
    return np.all(np.abs(lat_deg) <= bound_deg) and np.all(np.abs(lon_deg) <= 180)


def impact_latlon_fixed(result):
    if result["event_y"] is None:
        return None, None
    return result["event_y"][3], result["event_y"][4]


def impact_latlon_solveivp(sol):
    if len(sol.t_events[0]) == 0:
        return None, None
    y = sol.y_events[0][0]
    return y[3], y[4]


# ---------------------------------------------------------------------
# Figure A
# ---------------------------------------------------------------------

def figure_A():
    print("=== Figure A: truth vs. RK2/RK4/RK45 at step size 5 ===")

    truth = run_truth()
    lat_truth, lon_truth = impact_latlon_solveivp(truth)
    t_truth_fine = np.linspace(0, truth.t[-1], 600)
    h_truth = truth.sol(t_truth_fine)[0] / 1000.0

    rk2 = run_fixed("rk2", dt=5.0)
    rk4 = run_fixed("rk4", dt=5.0)
    rk45 = run_rk45_capped(max_step=5.0)

    lat_rk2, lon_rk2 = impact_latlon_fixed(rk2)
    lat_rk4, lon_rk4 = impact_latlon_fixed(rk4)
    lat_rk45, lon_rk45 = impact_latlon_solveivp(rk45)

    for name, lat, lon in [("Truth", lat_truth, lon_truth),
                             ("RK2 (dt=5s)", lat_rk2, lon_rk2),
                             ("RK4 (dt=5s)", lat_rk4, lon_rk4),
                             ("RK45 (max_step=5s)", lat_rk45, lon_rk45)]:
        if lat is None:
            print(f"{name:>20}: never reached ground")
            continue
        err = great_circle_distance(lat, lon, RECOVERY_LAT, RECOVERY_LON)
        print(f"{name:>20}: {np.degrees(lat):.6f} N, {np.degrees(lon):.6f} E  "
              f"(offset from recovery: {err/1000:.3f} km)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: full-trajectory altitude ---
    ax = axes[0]
    ax.plot(t_truth_fine, h_truth, "-", color="black", linewidth=2, label="Truth")
    ax.plot(rk2["t"], rk2["y"][0] / 1000.0, "-", color="tab:blue", label="RK2 (dt=5s)")
    ax.plot(rk4["t"], rk4["y"][0] / 1000.0, "-", color="tab:orange", label="RK4 (dt=5s)")
    t_rk45_fine = np.linspace(0, rk45.t[-1], 600)
    ax.plot(t_rk45_fine, rk45.sol(t_rk45_fine)[0] / 1000.0, "-",
             color="tab:purple", label="RK45 (max_step=5s)")
    ax.set_ylim(-10, 90)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (km)")
    ax.set_title("Full-trajectory altitude (0 to impact), step size = 5s")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Right: ground track ---
    ax = axes[1]
    ax.plot(np.degrees(truth.y[4, :]), np.degrees(truth.y[3, :]), "-",
             color="black", linewidth=2, label="Truth")

    def latlon_sane(lat_rad, lon_rad):
        """Final-point physical sanity check (kept for the impact marker)."""
        return lat_rad is not None and abs(np.degrees(lat_rad)) <= 90 and abs(np.degrees(lon_rad)) <= 180

    rk2_sane = latlon_sane(lat_rk2, lon_rk2) and path_is_sane(rk2["y"])
    rk4_sane = latlon_sane(lat_rk4, lon_rk4) and path_is_sane(rk4["y"])
    rk45_sane = latlon_sane(lat_rk45, lon_rk45) and path_is_sane(rk45.y)

    if rk2_sane:
        ax.plot(np.degrees(rk2["y"][4, :]), np.degrees(rk2["y"][3, :]), "-",
                 color="tab:blue", label="RK2 (dt=5s)")
        ax.plot(np.degrees(lon_rk2), np.degrees(lat_rk2), "o", color="tab:blue", markersize=10)
    else:
        print("RK2 (dt=5s): path wanders off physical scale at some point — omitted from right panel (unstable, not just 'a bit off').")
    if rk4_sane:
        ax.plot(np.degrees(rk4["y"][4, :]), np.degrees(rk4["y"][3, :]), "-",
                 color="tab:orange", label="RK4 (dt=5s)")
        ax.plot(np.degrees(lon_rk4), np.degrees(lat_rk4), "s", color="tab:orange", markersize=10)
    else:
        print("RK4 (dt=5s): path wanders off physical scale at some point — omitted from right panel (unstable, not just 'a bit off').")
    if rk45_sane:
        ax.plot(np.degrees(rk45.y[4, :]), np.degrees(rk45.y[3, :]), "-",
                 color="tab:purple", label="RK45 (max_step=5s)")
        ax.plot(np.degrees(lon_rk45), np.degrees(lat_rk45), "^", color="tab:purple", markersize=10)
    ax.plot(np.degrees(lon_truth), np.degrees(lat_truth), "o", color="black", markersize=10)
    ax.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="tab:green",
             markersize=18, label="Real recovery location")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Ground track and impact points, step size = 5s")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Figure A: Ground-truth solver vs. RK2/RK4/RK45 at a deliberately "
                   "coarse step size (5s) — isolating numerical error")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Figure B: 6-panel grid
# ---------------------------------------------------------------------

def figure_B():
    print("\n=== Figure B: per-method step-size convergence grid ===")

    truth_full = run_truth(t_span=FULL_SPAN)
    lat_truth, lon_truth = impact_latlon_solveivp(truth_full)
    t_truth_fine = np.linspace(0, truth_full.t[-1], 600)
    h_truth_full = truth_full.sol(t_truth_fine)[0] / 1000.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(STEP_SIZES)))

    methods = [
        ("RK2", "rk2", "fixed", STEP_SIZES, lambda v: f"dt={v}s"),
        ("RK4", "rk4", "fixed", STEP_SIZES, lambda v: f"dt={v}s"),
        ("RK45", None, "adaptive_tol", RK45_TOLERANCES, lambda v: f"rtol={v:.0e}"),
    ]

    # --- Pass 1: run everything once (full span only — reused for both
    # the top-row altitude curve and the bottom-row impact point), and
    # collect the sane impact points so shared axis limits can be set
    # across all three bottom panels before plotting anything. ---
    results_by_col = []
    all_impact_lons, all_impact_lats = [], []

    for label, fkey, kind, sweep_values, label_fn in methods:
        diverged = []
        entries = []  # (value, t_array, h_array_km, lat_i, lon_i, color, label)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(sweep_values)))

        for value, color in zip(sweep_values, colors):
            if kind == "fixed":
                res = run_fixed(fkey, dt=value, t_span=FULL_SPAN)
                t_arr, h_arr = res["t"], res["y"][0] / 1000.0
                lat_i, lon_i = impact_latlon_fixed(res)
                path_ok = path_is_sane(res["y"]) if lat_i is not None else False
            else:  # RK45, sweeping rtol, no max_step cap
                sol = run_rk45_tol(rtol=value, t_span=FULL_SPAN)
                t_arr = np.linspace(0, sol.t[-1], 400)
                h_arr = sol.sol(t_arr)[0] / 1000.0
                lat_i, lon_i = impact_latlon_solveivp(sol)
                path_ok = path_is_sane(sol.y) if lat_i is not None else False

            if lat_i is None or not path_ok:
                diverged.append(value)
                continue

            err_from_truth = great_circle_distance(lat_i, lon_i, lat_truth, lon_truth)
            print(f"{label:>5} {label_fn(value):>14}: impact offset from truth = {err_from_truth:8.2f} m")
            entries.append((value, t_arr, h_arr, lat_i, lon_i, color, label_fn(value)))
            all_impact_lons.append(np.degrees(lon_i))
            all_impact_lats.append(np.degrees(lat_i))

        results_by_col.append((label, entries, diverged, label_fn))
        if diverged:
            print(f"{label}: {[label_fn(v) for v in diverged]} produced an unstable path (or no impact) — omitted.")

    # Shared bottom-row axis limits, covering every sane impact point
    # across all three methods, plus truth and the real recovery site,
    # with a little padding so markers aren't clipped at the edge.
    all_impact_lons += [np.degrees(lon_truth), RECOVERY_LON_DEG]
    all_impact_lats += [np.degrees(lat_truth), RECOVERY_LAT_DEG]
    lon_min, lon_max = min(all_impact_lons), max(all_impact_lons)
    lat_min, lat_max = min(all_impact_lats), max(all_impact_lats)
    lon_pad = max((lon_max - lon_min) * 0.1, 0.001)
    lat_pad = max((lat_max - lat_min) * 0.1, 0.001)
    shared_xlim = (lon_min - lon_pad, lon_max + lon_pad)
    shared_ylim = (lat_min - lat_pad, lat_max + lat_pad)

    # --- Pass 2: plot, now that shared bottom-row limits are known. ---
    for col, (label, entries, diverged, label_fn) in enumerate(results_by_col):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.plot(t_truth_fine, h_truth_full, "-", color="black", linewidth=2, label="Truth")
        ax_bot.plot(np.degrees(lon_truth), np.degrees(lat_truth), "o", color="black",
                     markersize=12, label="Truth", zorder=5)
        ax_bot.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="tab:green",
                     markersize=16, label="Real recovery", zorder=5)

        for value, t_arr, h_arr, lat_i, lon_i, color, step_label in entries:
            ax_top.plot(t_arr, h_arr, "-", color=color, linewidth=1.2, label=step_label)
            ax_bot.plot(np.degrees(lon_i), np.degrees(lat_i), "o", color=color,
                         markersize=7, label=step_label)

        ax_top.set_ylim(-10, 90)
        ax_top.set_xlabel("Time (s)")
        ax_top.set_ylabel("Height (km)")
        title_top = f"{label}: convergence (full trajectory)"
        if diverged:
            title_top += f"\n({[label_fn(v) for v in diverged]} diverged, omitted)"
        ax_top.set_title(title_top, fontsize=9)
        ax_top.legend(fontsize=6)
        ax_top.grid(True, linestyle="--", alpha=0.4)

        ax_bot.set_xlim(*shared_xlim)
        ax_bot.set_ylim(*shared_ylim)
        ax_bot.set_xlabel("Longitude (deg E)")
        ax_bot.set_ylabel("Latitude (deg N)")
        title_bot = f"{label}: impact location"
        if diverged:
            title_bot += f"\n({[label_fn(v) for v in diverged]} unstable, omitted)"
        ax_bot.set_title(title_bot, fontsize=9)
        ax_bot.legend(fontsize=6)
        ax_bot.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Figure B: Convergence and resulting impact-location error, by method\n"
                   "(RK2/RK4: fixed step size  |  RK45: rtol, no max_step cap)", fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    figure_A()
    figure_B()