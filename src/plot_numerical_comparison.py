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
STEP_SIZES = [20.0, 10.0, 5.0, 2.5, 1.0, 0.5, 0.1]


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


def run_rk45_capped(max_step, t_span=FULL_SPAN, rtol=1e-8):
    """RK45, adaptive internally but capped at max_step — the closest
    equivalent of 'step size' for an adaptive method."""
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

    truth_zoom = run_truth(t_span=ZOOM_SPAN)
    t_zoom_fine = np.linspace(*ZOOM_SPAN, 400)
    h_truth_zoom = truth_zoom.sol(t_zoom_fine)[0] / 1000.0

    truth_full = run_truth(t_span=FULL_SPAN)
    lat_truth, lon_truth = impact_latlon_solveivp(truth_full)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(STEP_SIZES)))

    methods = [
        ("RK2", "rk2", "fixed"),
        ("RK4", "rk4", "fixed"),
        ("RK45", None, "adaptive_capped"),
    ]

    for col, (label, fkey, kind) in enumerate(methods):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.plot(t_zoom_fine, h_truth_zoom, "-", color="black", linewidth=2.5, label="Truth")
        ax_bot.plot(np.degrees(lon_truth), np.degrees(lat_truth), "o", color="black",
                     markersize=12, label="Truth", zorder=5)
        ax_bot.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="tab:green",
                     markersize=16, label="Real recovery", zorder=5)

        diverged_zoom = []
        diverged_full = []

        for dt, color in zip(STEP_SIZES, colors):
            # Top row: zoomed altitude curve at this step size
            if kind == "fixed":
                res_zoom = run_fixed(fkey, dt=dt, t_span=ZOOM_SPAN)
                h_vals = res_zoom["y"][0] / 1000.0
                if np.any(np.abs(h_vals) > 200):
                    diverged_zoom.append(dt)
                else:
                    ax_top.plot(res_zoom["t"], h_vals, "-", color=color,
                                 linewidth=1.2, label=f"dt={dt}s")
            else:  # RK45, capped
                sol_zoom = run_rk45_capped(max_step=dt, t_span=ZOOM_SPAN)
                h_vals = sol_zoom.sol(t_zoom_fine)[0] / 1000.0
                ax_top.plot(t_zoom_fine, h_vals, "-", color=color,
                             linewidth=1.2, label=f"max_step={dt}s")

            # Bottom row: full-trajectory impact location at this step size
            if kind == "fixed":
                res_full = run_fixed(fkey, dt=dt, t_span=FULL_SPAN)
                lat_i, lon_i = impact_latlon_fixed(res_full)
                path_ok = path_is_sane(res_full["y"]) if lat_i is not None else False
            else:
                sol_full = run_rk45_capped(max_step=dt, t_span=FULL_SPAN)
                lat_i, lon_i = impact_latlon_solveivp(sol_full)
                path_ok = path_is_sane(sol_full.y) if lat_i is not None else False

            if lat_i is None or not path_ok:
                diverged_full.append(dt)
                continue
            err_from_truth = great_circle_distance(lat_i, lon_i, lat_truth, lon_truth)
            print(f"{label:>5} dt={dt:>5}: impact offset from truth = {err_from_truth:8.2f} m")
            ax_bot.plot(np.degrees(lon_i), np.degrees(lat_i), "o", color=color,
                         markersize=7, label=f"dt={dt}s" if kind == "fixed" else f"max_step={dt}s")

        ax_top.set_xlim(*ZOOM_SPAN)
        ax_top.set_ylim(-10, 90)
        ax_top.set_xlabel("Time (s)")
        ax_top.set_ylabel("Height (km)")
        title_top = f"{label}: altitude convergence"
        if diverged_zoom:
            title_top += f"\n(dt={diverged_zoom} diverged, omitted)"
        ax_top.set_title(title_top, fontsize=9)
        ax_top.legend(fontsize=6)
        ax_top.grid(True, linestyle="--", alpha=0.4)

        ax_bot.set_xlabel("Longitude (deg E)")
        ax_bot.set_ylabel("Latitude (deg N)")
        title_bot = f"{label}: impact location vs. step size"
        if diverged_full:
            title_bot += f"\n(dt={diverged_full} unstable path, omitted)"
        ax_bot.set_title(title_bot, fontsize=9)
        ax_bot.legend(fontsize=6)
        ax_bot.grid(True, linestyle="--", alpha=0.4)

        if diverged_zoom:
            print(f"{label}: dt={diverged_zoom} diverged off-scale in the 0-6s altitude plot.")
        if diverged_full:
            print(f"{label}: dt={diverged_full} produced an unstable path (or no impact) — omitted.")

    fig.suptitle("Figure B: Step-size convergence and resulting impact-location error, "
                   "by method (RK2 / RK4 / RK45)", fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    figure_A()
    figure_B()