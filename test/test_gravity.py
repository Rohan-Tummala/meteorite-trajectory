"""
test_gravity.py

Unit tests for gravity.py.
"""

import numpy as np
import pytest

from gravity import spherical_gravity, gravity, G0


def test_spherical_gravity_at_sea_level_equals_g0():
    assert spherical_gravity(0.0) == pytest.approx(G0)


def test_spherical_gravity_decreases_with_altitude():
    altitudes = np.array([0.0, 10000.0, 30000.0, 60000.0, 85000.0])
    g_values = spherical_gravity(altitudes)
    assert np.all(np.diff(g_values) < 0.0)


def test_spherical_gravity_known_value_at_high_altitude():
    """At h = R_E, gravity must be exactly g0/4, per the inverse
    square law g(h) = g0 * (R_E / (R_E + h))^2."""
    from gravity import R_E
    assert spherical_gravity(R_E) == pytest.approx(G0 / 4.0)


def test_gravity_dispatch_matches_spherical_model():
    assert gravity(10000.0, model="spherical") == spherical_gravity(10000.0)


def test_gravity_unknown_model_raises():
    with pytest.raises(ValueError):
        gravity(10000.0, model="not_a_real_model")