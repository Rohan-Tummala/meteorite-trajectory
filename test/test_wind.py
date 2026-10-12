"""
test_wind.py

Unit tests for wind.py: the simplified dark-flight wind-drift model.
"""

import numpy as np
import pytest

from wind import wind_speed, wind_components, _WIND_ALT_M, _WIND_SPEED_MS


def test_wind_speed_matches_table_at_node_altitudes():
    """At each altitude explicitly given in the paper-derived profile,
    wind_speed must reproduce that exact value (interpolation through
    a node point, not between two of them)."""
    for h, expected in zip(_WIND_ALT_M, _WIND_SPEED_MS):
        assert wind_speed(h) == pytest.approx(expected)


def test_wind_speed_interpolates_between_nodes():
    """Halfway between the 13 km (10 m/s) and 20 km (20 m/s) nodes,
    linear interpolation should give a value between the two, roughly
    at the midpoint."""
    h_mid = (13000.0 + 20000.0) / 2.0
    speed = wind_speed(h_mid)
    assert 10.0 < speed < 20.0


def test_wind_speed_is_zero_above_22km():
    """No data exists above 22 km; the model must return exactly
    zero there rather than extrapolating the trend."""
    assert wind_speed(25000.0) == 0.0
    assert wind_speed(50000.0) == 0.0


def test_wind_speed_handles_array_input():
    altitudes = np.array([0.0, 13000.0, 20000.0, 22000.0, 30000.0])
    speeds = wind_speed(altitudes)
    assert speeds.shape == altitudes.shape
    assert speeds[-1] == 0.0


def test_wind_components_magnitude_matches_wind_speed():
    """The (north, east) components must recombine, via Pythagoras,
    to the same magnitude as wind_speed -- i.e. the decomposition
    doesn't silently lose or add energy."""
    h = 18000.0
    W_north, W_east = wind_components(h)
    magnitude = np.sqrt(W_north ** 2 + W_east ** 2)
    assert magnitude == pytest.approx(wind_speed(h))


def test_wind_components_zero_above_22km():
    W_north, W_east = wind_components(25000.0)
    assert W_north == pytest.approx(0.0)
    assert W_east == pytest.approx(0.0)