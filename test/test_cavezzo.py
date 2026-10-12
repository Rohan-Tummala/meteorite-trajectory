"""
test_cavezzo.py

Unit tests for Cavezzo.py: the degrees-arcminutes-arcseconds
conversion helper and the assembled initial-state vector.
"""

import numpy as np
import pytest

from cavezzo import (_dms_to_deg, initial_state,
                       H0, V0, GAMMA0, PHI0, LAMBDA0, PSI0, M0)


# ---------------------------------------------------------------------
# _dms_to_deg: known conversions
# ---------------------------------------------------------------------

def test_dms_to_deg_whole_degrees_only():
    assert _dms_to_deg(10.0, 0.0, 0.0) == pytest.approx(10.0)


def test_dms_to_deg_half_degree_arcminutes():
    """30 arcminutes is exactly half a degree."""
    assert _dms_to_deg(1.0, 30.0, 0.0) == pytest.approx(1.5)


def test_dms_to_deg_full_degree_from_arcseconds():
    """3600 arcseconds is exactly one degree."""
    assert _dms_to_deg(0.0, 0.0, 3600.0) == pytest.approx(1.0)


def test_dms_to_deg_matches_hand_computed_example():
    """44 deg 49' 43.7" = 44 + 49/60 + 43.7/3600, computable by hand
    to several decimal places."""
    expected = 44.0 + 49.0 / 60.0 + 43.7 / 3600.0
    assert _dms_to_deg(44.0, 49.0, 43.7) == pytest.approx(expected)


# ---------------------------------------------------------------------
# initial_state: consistency with the module's own constants
# ---------------------------------------------------------------------

def test_initial_state_has_seven_elements():
    assert initial_state().shape == (7,)


def test_initial_state_matches_module_constants_in_order():
    """Same rationale as the equivalent Bunburra test: a transcription
    slip in assembling the state vector would feed the wrong physical
    quantity into the wrong slot silently."""
    y0 = initial_state()
    expected = np.array([H0, V0, GAMMA0, PHI0, LAMBDA0, PSI0, M0])
    assert np.array_equal(y0, expected)