"""
data/cavezzo.py

Initial conditions and validation data for the Cavezzo meteorite fall
(1 January 2020, northern Italy), extracted from:

    Gardiol, D., Barghini, D., Buzzoni, A., et al. (2021).
    "Cavezzo, the first Italian meteorite recovered by the PRISMA
    fireball network. Orbit, trajectory, and strewn-field."
    Monthly Notices of the Royal Astronomical Society, 501(1), 1215-1227.
    DOI: 10.1093/mnras/staa3646

Values below are taken from Table 3 (fireball parameters at the
beginning and end of the observed bright flight) unless noted
otherwise. Latitude/longitude given in the paper as degrees-arcmin-
arcsec have been converted to decimal degrees here; everything is
converted to SI units (m, m/s, rad, kg) to match the state vector
convention used in dynamics.py: y = [h, V, gamma, phi, lambda, psi, M].

NOTE ON SIGN CONVENTION: gamma is the flight-path angle in the
dynamics model, with h_dot = V*sin(gamma). Since the meteor is
descending, gamma is taken as NEGATIVE here (matching a downward
velocity component), even though the paper reports the trajectory
"inclination" as a positive angle from horizontal (68.4 deg).
"""

import numpy as np


def _dms_to_deg(deg, arcmin, arcsec):
    """Convert degrees-arcminutes-arcseconds to decimal degrees."""
    return deg + arcmin / 60.0 + arcsec / 3600.0


# ---------------------------------------------------------------------
# Beginning of the observed bright flight (initial conditions for the model)
# ---------------------------------------------------------------------

# Height, m (Table 3: 75.9 +/- 0.2 km)
H0 = 75900.0
H0_UNC = 200.0

# Speed, m/s (Table 3, beginning: 12.2 +/- 0.2 km/s — this is the
# observed speed at the start of the LUMINOUS trail, not the
# pre-atmospheric velocity v_inf = 12.8 +/- 0.2 km/s, which is the
# speed before any atmospheric deceleration at all. Use V0 as the
# initial condition for the visible-trajectory model; v_inf is a
# separate, slightly higher reference value — see V_INF below.)
V0 = 12200.0
V0_UNC = 200.0

# Flight-path angle, rad (Table 3: trajectory inclination 68.4 +/- 0.3
# deg from horizontal; negative here since the meteor is descending —
# see sign convention note above)
GAMMA0 = -np.radians(68.4)
GAMMA0_UNC = np.radians(0.3)

# Latitude at start, rad (Table 3: 44 deg 44' 03" +/- 7" N)
PHI0_DEG = _dms_to_deg(44, 44, 3)
PHI0 = np.radians(PHI0_DEG)
PHI0_UNC = np.radians(7.0 / 3600.0)

# Longitude at start, rad (Table 3: 10 deg 43' 09" +/- 7" E)
LAMBDA0_DEG = _dms_to_deg(10, 43, 9)
LAMBDA0 = np.radians(LAMBDA0_DEG)
LAMBDA0_UNC = np.radians(7.0 / 3600.0)

# Azimuth/heading, rad. Table 3 reports the trajectory azimuth as
# 238.1 +/- 0.2 deg, but this is the classical meteor-astronomy
# convention (direction back toward the radiant), which is the
# RECIPROCAL of the direction of travel. Confirmed against the
# paper's own beginning/end coordinates: lat and lon both increase
# (bearing ~58 deg, ENE), matching 238.1 - 180 = 58.1 deg, not 238.1
# deg directly. The dynamics.py equations need the travel heading,
# so we apply the 180-degree correction here.
PSI0 = np.radians(238.1 - 180.0)
PSI0_UNC = np.radians(0.2)

# Pre-atmospheric mass, kg (Table 3, beginning: 3.5 +/- 0.8 kg)
M0 = 3.5
M0_UNC = 0.8

# Pre-atmospheric velocity (speed before ANY atmospheric deceleration,
# i.e. at the edge of the atmosphere) — a separate reference value
# from V0 above. Useful for orbital calculations, not a state-vector
# initial condition.
V_INF = 12800.0
V_INF_UNC = 200.0


# ---------------------------------------------------------------------
# Terminal point of the observed luminous trail (mid-flight validation)
# ---------------------------------------------------------------------

H_TERMINAL = 21500.0
H_TERMINAL_UNC = 100.0

V_TERMINAL = 4000.0
V_TERMINAL_UNC = 200.0

PHI_TERMINAL_DEG = _dms_to_deg(44, 50, 24)
PHI_TERMINAL = np.radians(PHI_TERMINAL_DEG)
PHI_TERMINAL_UNC = np.radians(7.0 / 3600.0)

LAMBDA_TERMINAL_DEG = _dms_to_deg(10, 57, 25)
LAMBDA_TERMINAL = np.radians(LAMBDA_TERMINAL_DEG)
LAMBDA_TERMINAL_UNC = np.radians(7.0 / 3600.0)

M_TERMINAL = 1.5   # kg (residual mass at end of LUMINOUS trail — not
M_TERMINAL_UNC = 0.4  # final ground-impact mass; paper notes this
                        # value is less certain than the initial mass)

# Duration of the observed luminous flight, s
DURATION = 5.6

# Luminous path length, m
PATH_LENGTH = 59000.0


# ---------------------------------------------------------------------
# Ground truth: recovered meteorite location (for impact-location validation)
# ---------------------------------------------------------------------
# Coordinates where fragments F1 (3.1 g) and F2 (52.2 g) were recovered.
# NOTE: this is NOT the same as the paper's computed nominal impact
# point (which accounts for dark-flight and wind effects after the
# luminous trail ends) — it is the actual find location, the strongest
# available ground truth for validating a full model's predicted
# impact point.

RECOVERY_LAT_DEG = _dms_to_deg(44, 49, 43.7)
RECOVERY_LAT = np.radians(RECOVERY_LAT_DEG)

RECOVERY_LON_DEG = _dms_to_deg(10, 58, 19.5)
RECOVERY_LON = np.radians(RECOVERY_LON_DEG)


# ---------------------------------------------------------------------
# Physical parameters relevant to model calibration/validation
# ---------------------------------------------------------------------

# Measured bulk density of recovered fragment F2, kg/m^3 (paper: 3.322
# g/cm^3, via 3D scanning) — use this instead of a generic placeholder
# for rho_m in dynamics.py's DEFAULT_PARAMS.
RHO_M = 3322.0

# Ablation coefficient, s^2/km^2 (paper's own fitted value — useful as
# an independent cross-check, NOT a direct input to CH/Q in our model,
# since it's defined differently; see Ceplecha 1987 formulation)
ABLATION_COEFF_PAPER = 0.012
ABLATION_COEFF_PAPER_UNC = 0.003

# Maximum dynamic pressure reached, and altitude at which it occurred
P_MAX = 1.0e6       # Pa (paper: 1.0 +/- 0.3 MPa)
P_MAX_UNC = 0.3e6
P_MAX_ALTITUDE = 28200.0  # m

# Estimated fragmentation strength (paper, Eq. 1)
FRAGMENTATION_STRENGTH = 0.88e6  # Pa


def initial_state():
    """
    Return the nominal initial state vector y0 = [h, V, gamma, phi,
    lambda, psi, M] for the Cavezzo fall, matching the convention in
    dynamics.py, ready to hand to an integrator.

    Returns
    -------
    ndarray, shape (7,)
    """
    return np.array([H0, V0, GAMMA0, PHI0, LAMBDA0, PSI0, M0])


if __name__ == "__main__":
    y0 = initial_state()
    labels = ["h0 (m)", "V0 (m/s)", "gamma0 (rad)", "phi0 (rad)",
              "lambda0 (rad)", "psi0 (rad)", "M0 (kg)"]
    print("Cavezzo initial state:")
    for label, val in zip(labels, y0):
        print(f"  {label:>15} = {val: .6f}")
    print(f"\nRecovery location: {RECOVERY_LAT_DEG:.6f} N, "
          f"{RECOVERY_LON_DEG:.6f} E")