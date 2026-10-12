"""
test_atmosphere.py

Unit tests for atmosphere.py: the exponential and tabulated
atmospheric density models.
"""

import numpy as np
import pytest

from atmosphere import (exponential_density, tabulated_density,
                          coesa_density, density, RHO0)


def test_exponential_density_at_sea_level():
    assert exponential_density(0.0) == pytest.approx(RHO0)


def test_exponential_density_decreases_with_altitude():
    altitudes = np.array([0.0, 10000.0, 30000.0, 60000.0, 85000.0])
    rho_values = exponential_density(altitudes)
    assert np.all(np.diff(rho_values) < 0.0)


def test_tabulated_density_decreases_with_altitude():
    altitudes = np.array([0.0, 10000.0, 21500.0, 50000.0, 85000.0])
    rho_values = tabulated_density(altitudes)
    assert np.all(np.diff(rho_values) < 0.0)


def test_tabulated_density_agrees_with_coesa_reference():
    """The tabulated model is built by interpolating the COESA 1976
    reference curve, so at a node altitude the two should agree
    closely (interpolation error only, not a modeling difference)."""
    for h in [0.0, 10000.0, 21500.0, 50000.0]:
        assert tabulated_density(h) == pytest.approx(coesa_density(h), rel=0.05)


def test_density_dispatch_exp_matches_exponential_density():
    assert density(20000.0, model="exp") == exponential_density(20000.0)


def test_density_dispatch_table_matches_tabulated_density():
    assert density(20000.0, model="table") == tabulated_density(20000.0)


def test_density_unknown_model_raises():
    with pytest.raises(ValueError):
        density(20000.0, model="not_a_real_model")