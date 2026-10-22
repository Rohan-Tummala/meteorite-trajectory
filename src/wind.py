"""
wind.py

Simplified dark-flight wind-drift model for Cavezzo, based on the wind
profile in Gardiol et al. (2021, Section 3.4 and Fig. 5).

From the paper: about 28 m/s at 22 km, about 20 m/s at 20 km and below
10 m/s from 13 km down, blowing 45 deg clockwise of the meteoroid's
ground-track heading.

Assumptions (not in the paper):
    - Direction is constant at 45 deg clockwise of the travel heading.
    - Speed is interpolated linearly between the paper's points
      (13, 20 and 22 km) and falls to 5 m/s at the ground.
    - Wind is zero above 22 km, where the meteoroid's speed is far
      larger than any plausible wind speed.

Wind is added as an extra horizontal velocity term and is not coupled
into the drag calculation, so this is a simplification.
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
# Section 3.4. The h=0 point is not from the paper; it is an assumed
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