"""
wind.py

A simplified wind-drift model for the dark-flight phase, based on the
wind profile Gardiol et al. (2021) report for the Cavezzo area at
18 UTC on the day of the fall (Fig. 5 and Section 3.4).

What the paper actually states (Section 3.4):
    - ~28 m/s at ~22 km altitude (the last observed point of the
      luminous path), blowing at 45 deg clockwise relative to the
      meteoroid's own ground-track motion direction.
    - Decreasing to ~20 m/s at 20 km altitude.
    - Below 10 m/s from 13 km downward.

The paper does NOT give a precise profile below 13 km or above 22 km,
or how direction varies with altitude below 22 km. This module fills
those gaps with explicit, flagged assumptions:
    - Wind direction is held constant (45 deg clockwise of travel
      heading) at all altitudes where wind is applied, since the
      paper gives no data suggesting otherwise.
    - Speed is linearly interpolated between the paper's own points
      (13, 20, 22 km), and linearly extrapolated down to a small
      residual value (5 m/s) at the ground - this last piece is an
      assumption, not from the paper.
    - Above 22 km, wind is set to zero: no data is given, and at that
      altitude the meteoroid's own speed (multiple km/s) is so much
      larger than any plausible wind speed that the omission has
      negligible effect on the luminous-flight portion anyway.

This is a SIMPLIFIED drift model: wind is added directly as an extra
horizontal ground-velocity term (advection), not fully coupled into
the relative-velocity drag calculation. This is a common
simplification for illustrating wind's effect on drift, but is not
as rigorous as solving for drag against the true air-relative
velocity vector.
"""

import numpy as np

# Travel heading (bearing, deg clockwise from North) - matches PSI0 in
# cavezzo.py (238.1 deg reported azimuth, corrected by -180 deg for
# the classical meteor-astronomy convention).
_TRAVEL_HEADING_DEG = 238.1 - 180.0

# Wind blows 45 deg clockwise relative to the travel heading, per the
# paper. This is the bearing (deg clockwise from North) the wind is
# blowing TOWARD.
_WIND_BEARING_DEG = _TRAVEL_HEADING_DEG + 45.0
_WIND_BEARING_RAD = np.radians(_WIND_BEARING_DEG)

# Wind speed profile: altitude (m) -> speed (m/s), from Fig. 5 /
# Section 3.4. The h=0 point is NOT from the paper - it's an assumed
# taper to a small residual value, since the paper only states "below
# 10 m/s from 13 km downwards" without giving the exact ground value.
_WIND_ALT_M = np.array([0.0, 13000.0, 20000.0, 22000.0])
_WIND_SPEED_MS = np.array([5.0, 10.0, 20.0, 28.0])


def wind_speed(h):
    """
    Wind speed (m/s) at altitude h (m), per the profile above.
    Zero above 22 km (no data given, and negligible relative to the
    meteoroid's own hypersonic speed there anyway).
    """
    h = np.asarray(h, dtype=float)
    speed = np.interp(h, _WIND_ALT_M, _WIND_SPEED_MS,
                        left=_WIND_SPEED_MS[0], right=0.0)
    # np.interp's "right" only applies beyond the last x-point (22 km);
    # explicitly zero it there rather than extrapolating the trend.
    speed = np.where(h > _WIND_ALT_M[-1], 0.0, speed)
    return speed


def wind_components(h):
    """
    Wind velocity components (north, east) in m/s at altitude h,
    assuming a constant bearing (45 deg clockwise of travel heading)
    at all altitudes where wind is applied.

    Returns
    -------
    (W_north, W_east) : tuple of float or ndarray, m/s
    """
    speed = wind_speed(h)
    W_north = speed * np.cos(_WIND_BEARING_RAD)
    W_east = speed * np.sin(_WIND_BEARING_RAD)
    return W_north, W_east