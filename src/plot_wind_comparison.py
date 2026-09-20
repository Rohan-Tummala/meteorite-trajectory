"""
plot_wind_comparison.py

Runs the Cavezzo trajectory twice - once with wind_model="none",
once with wind_model="cavezzo" (see wind.py) - and compares the
resulting ground tracks and impact points against each other and
against the real recovery location.
"""

import numpy as np
import matplotlib.pyplot as plt

from integrator import run_trajectory, impact_state
from cavezzo import initial_state, RECOVERY_LAT, RECOVERY_LON, \
    RECOVERY_LAT_DEG, RECOVERY_LON_DEG
from dynamics import DEFAULT_PARAMS
from plot_trajectory import great_circle_distance


def plot_wind_comparison():
    y0 = initial_state()
    t_span = (0.0, 600.0)

    params_no_wind = {**DEFAULT_PARAMS, "wind_model": "none"}
    sol_no_wind = run_trajectory(y0, t_span, method="RK45", rtol=1e-8,
                                    atol=1e-10, params=params_no_wind,
                                    max_step=5.0)

    params_wind = {**DEFAULT_PARAMS, "wind_model": "cavezzo"}
    sol_wind = run_trajectory(y0, t_span, method="RK45", rtol=1e-8,
                                 atol=1e-10, params=params_wind,
                                 max_step=5.0)

    y_impact_nw = impact_state(sol_no_wind)
    y_impact_w = impact_state(sol_wind)

    lat_nw, lon_nw = np.degrees(y_impact_nw[3]), np.degrees(y_impact_nw[4])
    lat_w, lon_w = np.degrees(y_impact_w[3]), np.degrees(y_impact_w[4])

    dist_nw = great_circle_distance(y_impact_nw[3], y_impact_nw[4],
                                       RECOVERY_LAT, RECOVERY_LON) / 1000.0
    dist_w = great_circle_distance(y_impact_w[3], y_impact_w[4],
                                      RECOVERY_LAT, RECOVERY_LON) / 1000.0
    shift = great_circle_distance(y_impact_nw[3], y_impact_nw[4],
                                     y_impact_w[3], y_impact_w[4]) / 1000.0

    print(f"No wind:   {lat_nw:.6f} N, {lon_nw:.6f} E  -> "
          f"offset from recovery: {dist_nw:.3f} km")
    print(f"With wind: {lat_w:.6f} N, {lon_w:.6f} E  -> "
          f"offset from recovery: {dist_w:.3f} km")
    print(f"Shift due to wind alone: {shift:.3f} km")

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(np.degrees(sol_no_wind.y[4, :]), np.degrees(sol_no_wind.y[3, :]),
             "-", color="tab:blue", label="No wind")
    ax.plot(np.degrees(sol_wind.y[4, :]), np.degrees(sol_wind.y[3, :]),
             "-", color="tab:purple", label="With wind (Cavezzo profile)")
    ax.plot(lon_nw, lat_nw, "o", color="tab:blue", markersize=10,
             label="Impact (no wind)")
    ax.plot(lon_w, lat_w, "o", color="tab:purple", markersize=10,
             label="Impact (with wind)")
    ax.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="tab:green",
             markersize=18, label="Real recovery location")

    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title(f"Wind effect on ground track\n"
                  f"(no-wind offset: {dist_nw:.2f} km, "
                  f"with-wind offset: {dist_w:.2f} km)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    plt.show()

    return sol_no_wind, sol_wind


if __name__ == "__main__":
    plot_wind_comparison()