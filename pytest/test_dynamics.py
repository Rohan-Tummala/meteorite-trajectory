"""
test_dynamics.py

Unit tests for dynamics.py: the trajectory ODE right-hand side and
its supporting helpers.
"""

import numpy as np
import pytest

from dynamics import radius_from_mass, meteor_rhs, ground_impact_event


# ---------------------------------------------------------------------
# radius_from_mass
# ---------------------------------------------------------------------

def test_radius_from_mass_known_value():
    """A 1 kg sphere at 1000 kg/m^3 has a radius computable by hand."""
    r = radius_from_mass(1.0, 1000.0)
    expected = (3.0 / (4.0 * np.pi * 1000.0)) ** (1.0 / 3.0)
    assert np.isclose(r, expected)


def test_radius_from_mass_zero_mass():
    assert radius_from_mass(0.0, 1000.0) == 0.0


def test_radius_from_mass_negative_mass_is_clamped():
    """Numerical overshoot can hand this function a tiny negative
    mass; it must clamp to zero rather than raise or return a
    complex/NaN result."""
    assert radius_from_mass(-1.0, 1000.0) == 0.0


def test_radius_from_mass_scales_correctly_with_density():
    """Doubling density at fixed mass must shrink the radius by the
    cube root of 2, per r = (3M / (4*pi*rho))^(1/3)."""
    r_low = radius_from_mass(5.0, 1000.0)
    r_high = radius_from_mass(5.0, 2000.0)
    assert np.isclose(r_low / r_high, 2.0 ** (1.0 / 3.0))


# ---------------------------------------------------------------------
# meteor_rhs
# ---------------------------------------------------------------------

def _midflight_state():
    """A representative, physically plausible mid-flight state."""
    return np.array([
        30000.0,              # h, m
        10000.0,               # V, m/s
        np.radians(-68.0),    # gamma, rad (descending)
        np.radians(44.0),     # phi, rad
        np.radians(11.0),     # lambda, rad
        np.radians(238.0),    # psi, rad
        1.0,                   # M, kg
    ])


def test_meteor_rhs_returns_finite_values():
    """At a plausible mid-flight state, no component of the
    derivative should be NaN or infinite."""
    dy = meteor_rhs(0.0, _midflight_state())
    assert np.all(np.isfinite(dy))


def test_meteor_rhs_descending_gamma_gives_negative_h_dot():
    """h_dot = V*sin(gamma); a descending flight-path angle (negative
    gamma, per this project's sign convention) must give a negative
    altitude rate."""
    dy = meteor_rhs(0.0, _midflight_state())
    assert dy[0] < 0.0


def test_meteor_rhs_ablation_stops_below_dark_flight_threshold():
    """Below the 3000 m/s dark-flight threshold, M_dot must be
    exactly zero regardless of the other state values."""
    y = _midflight_state()
    y[1] = 2000.0  # V, below the 3000 m/s threshold
    dy = meteor_rhs(0.0, y)
    assert dy[6] == 0.0


def test_meteor_rhs_ablation_active_above_threshold():
    """Above the dark-flight threshold, with nonzero mass and
    density, mass loss should be strictly negative."""
    y = _midflight_state()
    y[1] = 10000.0  # V, above the 3000 m/s threshold
    dy = meteor_rhs(0.0, y)
    assert dy[6] < 0.0


def test_meteor_rhs_handles_near_zero_mass_without_blowing_up():
    """As M -> 0, radius and the 1/M drag/ablation terms could blow
    up; the M_safe guard should keep the derivative finite."""
    y = _midflight_state()
    y[6] = 1e-9
    dy = meteor_rhs(0.0, y)
    assert np.all(np.isfinite(dy))


def test_meteor_rhs_psi_dot_is_always_zero():
    """Heading is held constant in this model (no cross-range
    curvature term), so psi_dot must always be zero."""
    dy = meteor_rhs(0.0, _midflight_state())
    assert dy[5] == 0.0


# ---------------------------------------------------------------------
# ground_impact_event
# ---------------------------------------------------------------------

def test_ground_impact_event_returns_altitude():
    y = _midflight_state()
    assert ground_impact_event(0.0, y) == y[0]


def test_ground_impact_event_zero_at_ground():
    y = _midflight_state()
    y[0] = 0.0
    assert ground_impact_event(0.0, y) == 0.0