"""
plot_trajectory_bunburra.py

Runs the Bunburra Rockhole trajectory and plots:
    1. Altitude vs. time
    2. Ground track (longitude vs. latitude), with the model's
       predicted impact point AND the real recovered meteorite
       location (M1) both marked, plus the distance between them
    3. Mass vs. time

Same structure as plot_trajectory.py, but for the second validation
case. The bulk density override (rho_m) is applied explicitly, since
DEFAULT_PARAMS in dynamics.py otherwise defaults to Cavezzo's value.
"""

import numpy as np
import matplotlib.pyplot as plt

from integrator import run_trajectory, impact_state
from dynamics import DEFAULT_PARAMS
from bunburra import initial_state, RECOVERY_LAT, RECOVERY_LON, \
    RECOVERY_LAT_DEG, RECOVERY_LON_DEG, RHO_M


def great_circle_distance(lat1, lon1, lat2, lon2, R=6371000.0):
    """
    Great-circle distance between two points (haversine formula).
    lat/lon in radians. Returns distance in metres.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def plot_full_trajectory():
    y0 = initial_state()
    t_span = (0.0, 600.0)
    params = {**DEFAULT_PARAMS, "rho_m": RHO_M}

    sol = run_trajectory(y0, t_span, method="RK45", rtol=1e-8, atol=1e-10,
                          dense_output=True, params=params)

    h = sol.y[0, :]
    V = sol.y[1, :]
    phi = sol.y[3, :]
    lam = sol.y[4, :]
    M = sol.y[6, :]
    t = sol.t

    y_impact = impact_state(sol)
    if y_impact is None:
        print("Warning: no ground impact detected — plots will show "
              "the trajectory up to the end of t_span instead.")
        impact_lat_deg = np.degrees(phi[-1])
        impact_lon_deg = np.degrees(lam[-1])
    else:
        impact_lat_deg = np.degrees(y_impact[3])
        impact_lon_deg = np.degrees(y_impact[4])

    distance = great_circle_distance(
        np.radians(impact_lat_deg), np.radians(impact_lon_deg),
        RECOVERY_LAT, RECOVERY_LON
    )
    print(f"Model impact point:  {impact_lat_deg:.6f} N, {impact_lon_deg:.6f} E")
    print(f"Real recovery point (M1): {RECOVERY_LAT_DEG:.6f} N, {RECOVERY_LON_DEG:.6f} E")
    print(f"Distance between them: {distance / 1000.0:.2f} km")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Altitude vs. time ---
    ax = axes[0, 0]
    ax.plot(t, h / 1000.0, "-", color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude vs. time")
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Ground track ---
    ax = axes[0, 1]
    ax.plot(np.degrees(lam), np.degrees(phi), "-", color="tab:blue",
             label="Model trajectory")
    ax.plot(impact_lon_deg, impact_lat_deg, "o", color="tab:red",
             markersize=10, label="Model impact point")
    ax.plot(RECOVERY_LON_DEG, RECOVERY_LAT_DEG, "*", color="tab:green",
             markersize=16, label="Real recovery location (M1)")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title(f"Ground track (offset: {distance/1000.0:.2f} km)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")

    # --- Mass vs. time ---
    ax = axes[1, 0]
    ax.plot(t, M, "-", color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title("Mass vs. time")
    ax.grid(True, linestyle="--", alpha=0.4)

    # No digitized Fig. 4a/4c equivalent exists for Bunburra, so the
    # 4th panel (and plot_velocity_validation, in plot_trajectory.py)
    # are intentionally omitted here.
    axes[1, 1].axis("off")

    fig.tight_layout()
    plt.show()

    return sol


if __name__ == "__main__":
    plot_full_trajectory()