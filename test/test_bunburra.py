"""
test_bunburra.py

Unit tests for bunburra.py: the bearing-calculation helper and the
assembled initial-state vector.
"""

import numpy as np
import pytest

from bunburra import (_bearing_deg, initial_state,
                        H0, V0, GAMMA0, PHI0, LAMBDA0, PSI0, M0)


# ---------------------------------------------------------------------
# _bearing_deg: known cardinal-direction cases
# ---------------------------------------------------------------------

def test_bearing_due_north_is_zero():
    """Moving to higher latitude at the same longitude is due north,
    exactly 0 degrees, regardless of distance."""
    assert _bearing_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_bearing_due_south_is_180():
    assert _bearing_deg(10.0, 0.0, 0.0, 0.0) == pytest.approx(180.0, abs=1e-6)


def test_bearing_due_east_on_equator_is_90():
    """Along the equator, moving to higher longitude is due east,
    exactly 90 degrees."""
    assert _bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0, abs=1e-6)


def test_bearing_due_west_on_equator_is_270():
    assert _bearing_deg(0.0, 10.0, 0.0, 0.0) == pytest.approx(270.0, abs=1e-6)


def test_bearing_is_always_in_0_to_360_range():
    rng = np.random.default_rng(0)
    for _ in range(20):
        lat1, lon1, lat2, lon2 = rng.uniform(-80, 80, 4)
        bearing = _bearing_deg(lat1, lon1, lat2, lon2)
        assert 0.0 <= bearing < 360.0


# ---------------------------------------------------------------------
# initial_state: consistency with the module's own constants
# ---------------------------------------------------------------------

def test_initial_state_has_seven_elements():
    assert initial_state().shape == (7,)


def test_initial_state_matches_module_constants_in_order():
    """initial_state() must assemble exactly the module-level
    constants, in the [h, V, gamma, phi, lambda, psi, M] order
    dynamics.py expects -- a transcription slip here (e.g. swapping
    two entries) would silently feed the wrong physical quantity into
    the wrong slot of the ODE state vector."""
    y0 = initial_state()
    expected = np.array([H0, V0, GAMMA0, PHI0, LAMBDA0, PSI0, M0])
    assert np.array_equal(y0, expected)