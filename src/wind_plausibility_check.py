"""
wind_plausibility_check.py

Estimates the average wind vector required to close the gap between
the no-wind nominal impact prediction and the real recovery location,
compares it against the wind speeds measured in Gardiol et al. (2021),
Section 3.4, and computes the actual displacement the wind model in
wind.py produces, for comparison against what is actually needed.
"""

import numpy as np
from integrator import run_trajectory, impact_state
from cavezzo import initial_state, RECOVERY_LAT, RECOVERY_LON, \
    RECOVERY_LAT_DEG, RECOVERY_LON_DEG
from dynamics import DEFAULT_PARAMS
from plot_trajectory import great_circle_distance
from wind import _WIND_BEARING_DEG


def bearing_deg(lat1, lon1, lat2, lon2):
    """Forward bearing (deg clockwise from North) from point 1 to point 2."""
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def main():
    y0 = initial_state()

    # No-wind nominal run.
    params_no_wind = {**DEFAULT_PARAMS, "wind_model": "none"}
    sol_no_wind = run_trajectory(y0, (0.0, 600.0), method="RK45", rtol=1e-8,
                                    atol=1e-10, params=params_no_wind, max_step=5.0)
    y_nw = impact_state(sol_no_wind)
    lat_nw, lon_nw = y_nw[3], y_nw[4]
    t_impact = sol_no_wind.t_events[0][0]

    # With-wind run (Cavezzo profile from wind.py).
    params_wind = {**DEFAULT_PARAMS, "wind_model": "cavezzo"}
    sol_wind = run_trajectory(y0, (0.0, 600.0), method="RK45", rtol=1e-8,
                                 atol=1e-10, params=params_wind, max_step=5.0)
    y_w = impact_state(sol_wind)
    lat_w, lon_w = y_w[3], y_w[4]

    # Required wind: what average wind would close the no-wind/real gap.
    dist_needed_m = great_circle_distance(lat_nw, lon_nw, RECOVERY_LAT, RECOVERY_LON)
    bearing_needed = bearing_deg(lat_nw, lon_nw, RECOVERY_LAT, RECOVERY_LON)
    luminous_duration_s = 10.0  # matches PHASE_SPLIT elsewhere in the project
    dark_duration = t_impact - luminous_duration_s
    required_speed = dist_needed_m / dark_duration

    # Actual wind: what wind.py's Cavezzo profile actually did.
    dist_induced_m = great_circle_distance(lat_nw, lon_nw, lat_w, lon_w)
    bearing_induced = bearing_deg(lat_nw, lon_nw, lat_w, lon_w)
    dist_w_to_real_m = great_circle_distance(lat_w, lon_w, RECOVERY_LAT, RECOVERY_LON)

    print(f"No-wind impact           : {np.degrees(lat_nw):.6f} N, {np.degrees(lon_nw):.6f} E")
    print(f"With-wind impact         : {np.degrees(lat_w):.6f} N, {np.degrees(lon_w):.6f} E")
    print(f"Real recovery location   : {RECOVERY_LAT_DEG:.6f} N, {RECOVERY_LON_DEG:.6f} E")
    print()
    print(f"Offset (no-wind vs real) : {dist_needed_m/1000:.3f} km, bearing {bearing_needed:.1f} deg")
    print(f"Dark-flight duration     : {dark_duration:.1f} s")
    print(f"Required average wind speed to fully explain the offset: {required_speed:.2f} m/s")
    print("Gardiol et al. (2021) Sec 3.4: ~28 m/s @22km, ~20 m/s @20km, <10 m/s below 13km.")
    print()
    print(f"wind.py's assumed bearing        : {_WIND_BEARING_DEG:.1f} deg")
    print(f"Wind ACTUALLY moved the impact by: {dist_induced_m:.1f} m at bearing {bearing_induced:.1f} deg")
    print(f"Distance from with-wind impact to real recovery: {dist_w_to_real_m/1000:.3f} km "
          f"(vs {dist_needed_m/1000:.3f} km no-wind)")


if __name__ == "__main__":
    main()