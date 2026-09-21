"""
plot_trajectory.py

Runs the Cavezzo trajectory and plots:
    1. Altitude vs. time
    2. Ground track (longitude vs. latitude), with the model's
       predicted impact point AND the real recovered meteorite
       location both marked, plus the distance between them
    3. Mass vs. time

This is the first "does the whole picture make sense" check — not
yet a formal validation (that comes once Cd/sigma_abl are properly
justified/tuned), just a visual sanity check tying everything
together.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from integrator import run_trajectory, impact_state
from cavezzo import initial_state, RECOVERY_LAT, RECOVERY_LON, \
    RECOVERY_LAT_DEG, RECOVERY_LON_DEG

# Digitized reference points from Fig. 4a of Gardiol et al. (2021),
# via WebPlotDigitizer. Update this path if the filename differs.
DIGITIZED_FILE = "cavezzo_height_vs_time.csv"


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


def load_digitized_height(path):
    """
    Load digitized (time, height) points exported by WebPlotDigitizer,
    from Gardiol et al. (2021), Fig. 4a. Assumes time (s) in column 0,
    height (km) in column 1, regardless of header text. File is CSV.
    """
    df = pd.read_csv(path)
    t_data = df.iloc[:, 0].to_numpy(dtype=float)
    h_data = df.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(t_data)
    return t_data[order], h_data[order]


def plot_full_trajectory():
    y0 = initial_state()
    t_span = (0.0, 600.0)

    sol = run_trajectory(y0, t_span, method="RK45", rtol=1e-8, atol=1e-10,
                          dense_output=True)

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
    print(f"Real recovery point: {RECOVERY_LAT_DEG:.6f} N, {RECOVERY_LON_DEG:.6f} E")
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
             markersize=16, label="Real recovery location")
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

    # --- Height validation vs. digitized Fig. 4a data (0-6 s only) ---
    ax = axes[1, 1]
    t_fine = np.linspace(0.0, 5.6, 300)
    h_fine_model = sol.sol(t_fine)[0] / 1000.0  # m -> km
    ax.plot(t_fine, h_fine_model, "-", color="tab:blue", linewidth=2,
             label="Model")
    try:
        t_data, h_data = load_digitized_height(DIGITIZED_FILE)
        ax.plot(t_data, h_data, "o", color="tab:orange", markersize=5,
                 alpha=0.7, label="Digitized (Gardiol et al. 2021, Fig. 4a)")
    except FileNotFoundError:
        print(f"Note: '{DIGITIZED_FILE}' not found — skipping digitized "
              f"overlay on the validation panel.")
    ax.set_xlim(0.0, 6.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (km)")
    ax.set_title("Trajectory validation (luminous flight, 0-6 s)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    plt.show()

    return sol


# Digitized reference points from Fig. 4c of Gardiol et al. (2021),
# via WebPlotDigitizer. Update this path if the filename differs.
DIGITIZED_VELOCITY_FILE = "cavezzo_vel_vs_time.xlsx"


def load_digitized_velocity(path):
    """
    Load digitized (time, velocity) points exported by
    WebPlotDigitizer, from Gardiol et al. (2021), Fig. 4c. Assumes
    time (s) in column 0, velocity (km/s) in column 1, regardless of
    header text.
    """
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    t_data = df.iloc[:, 0].to_numpy(dtype=float)
    v_data = df.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(t_data)
    return t_data[order], v_data[order]


def plot_velocity_validation(velocity_uncertainty_kms=2.0):
    """
    Plot the model's velocity-vs-time curve over the luminous-flight
    window against digitized reference points from Fig. 4c, with a
    shaded uncertainty band of +/- velocity_uncertainty_kms around the
    digitized data.
    """
    y0 = initial_state()
    t_span = (0.0, 5.6)

    sol = run_trajectory(y0, t_span, method="RK45", rtol=1e-9, atol=1e-11,
                          dense_output=True)

    t_fine = np.linspace(t_span[0], t_span[1], 300)
    v_model = sol.sol(t_fine)[1] / 1000.0  # m/s -> km/s

    fig, ax = plt.subplots(figsize=(7, 5))

    try:
        t_data, v_data = load_digitized_velocity(DIGITIZED_VELOCITY_FILE)

        # Shaded uncertainty band around the digitized curve.
        ax.fill_between(t_data, v_data - velocity_uncertainty_kms,
                          v_data + velocity_uncertainty_kms,
                          color="tab:orange", alpha=0.2,
                          label=f"Digitized \u00b1{velocity_uncertainty_kms:g} km/s band")
        ax.plot(t_data, v_data, "o", color="tab:orange", markersize=5,
                 alpha=0.8, label="Digitized (Gardiol et al. 2021, Fig. 4c)")
    except FileNotFoundError:
        print(f"Note: '{DIGITIZED_VELOCITY_FILE}' not found — plotting "
              f"model curve only.")

    ax.plot(t_fine, v_model, "-", color="tab:blue", linewidth=2,
             label="Model")

    ax.set_xlim(0.0, 6.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (km/s)")
    ax.set_title("Trajectory validation: velocity vs. time (luminous flight)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()

    return sol


if __name__ == "__main__":
    plot_full_trajectory()
    plot_velocity_validation()