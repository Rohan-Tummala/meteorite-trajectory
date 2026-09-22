"""
data/bunburra.py

Initial conditions and validation data for the Bunburra Rockhole
meteorite fall (20 July 2007, SW Australia).

Source: Spurny, P., Bland, P. A., Shrbeny, L., Borovicka, J.,
Ceplecha, Z., Singelton, A., Bevan, A. W. R., Vaughan, D., Towner,
M. C., McClafferty, T. P., Toumi, R., & Deacon, G. (2012). "The
Bunburra Rockhole meteorite fall in SW Australia: fireball
trajectory, luminosity, dynamics, orbit, and impact position from
photographic and photoelectric records." Meteoritics & Planetary
Science, 47(2), 163-185. DOI: 10.1111/j.1945-5100.2011.01321.x

Format matches cavezzo.py so it can be imported as a drop-in second
data source, with no other code changes. Bulk density differs from
Cavezzo's (achondrite vs. ordinary chondrite), so pass rho_m=RHO_M
as an override when calling run_trajectory.

PSI0 (heading) is not tabulated in the source paper and is computed
here from the beginning/end coordinates (Table 1) via the standard
bearing formula.
"""

import numpy as np


def _bearing_deg(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Forward bearing between two points, degrees clockwise from North."""
    lat1, lat2 = np.radians(lat1_deg), np.radians(lat2_deg)
    dlon = np.radians(lon2_deg - lon1_deg)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# ---------------------------------------------------------------------
# Beginning of the observed luminous trajectory (Table 1)
# ---------------------------------------------------------------------

H0 = 62830.0              # m, 62.83 +/- 0.03 km
H0_UNC = 30.0

V0 = 13310.0               # m/s, 13.31 +/- 0.02 km/s
V0_UNC = 20.0

GAMMA0 = -np.radians(31.19)   # "Slope" = 31.19 +/- 0.03 deg; negated for descent
GAMMA0_UNC = np.radians(0.03)

PHI0_DEG = -31.4496
PHI0 = np.radians(PHI0_DEG)
PHI0_UNC = np.radians(0.0003)

LAMBDA0_DEG = 129.82721
LAMBDA0 = np.radians(LAMBDA0_DEG)
LAMBDA0_UNC = np.radians(0.00018)

PSI0_DEG = _bearing_deg(-31.4496, 129.82721, -31.3710, 129.25555)  # derived, ~279.15 deg
PSI0 = np.radians(PSI0_DEG)

M0 = 22.1                  # kg, GFM dynamic mass +/- 0.3
M0_UNC = 0.3

V_INF = 13365.0             # m/s, pre-atmospheric velocity (Table 4), +/- 7
V_INF_UNC = 7.0


# ---------------------------------------------------------------------
# Terminal point of the observed luminous trail
# ---------------------------------------------------------------------

H_TERMINAL = 29590.0
H_TERMINAL_UNC = 20.0

V_TERMINAL = 5680.0
V_TERMINAL_UNC = 120.0

PHI_TERMINAL_DEG = -31.3710
PHI_TERMINAL = np.radians(PHI_TERMINAL_DEG)
PHI_TERMINAL_UNC = np.radians(0.0002)

LAMBDA_TERMINAL_DEG = 129.25555
LAMBDA_TERMINAL = np.radians(LAMBDA_TERMINAL_DEG)
LAMBDA_TERMINAL_UNC = np.radians(0.00015)

M_TERMINAL = 1.1            # kg, MFM dynamic mass, +/- ~0.3
M_TERMINAL_UNC = 0.3

DURATION = 5.68             # s
PATH_LENGTH = 64650.0       # m


# ---------------------------------------------------------------------
# Recovered meteorite locations (Table 5)
# ---------------------------------------------------------------------

RECOVERY_LON_DEG = 129.188   # M1, 150 g, found 2008-10-03
RECOVERY_LAT_DEG = -31.350
RECOVERY_LON = np.radians(RECOVERY_LON_DEG)
RECOVERY_LAT = np.radians(RECOVERY_LAT_DEG)
RECOVERY_MASS_G = 150.0

M2_LON_DEG, M2_LAT_DEG, M2_MASS_G = 129.190, -31.349, 174.0   # found 2008-10-11
M3_LON_DEG, M3_LAT_DEG, M3_MASS_G = 129.320, -31.355, 14.9    # found 2009-02-27


# ---------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------

RHO_M = 2700.0  # kg/m^3, measured bulk density of recovered fragments

# Classical ablation coefficients (paper's own units, s^2/km^2) -- not
# the same definition as this project's sigma_abl (= CH/Q, kg/J).
APPARENT_ABLATION_COEFF = 0.0331     # GFM, includes fragmentation
APPARENT_ABLATION_COEFF_UNC = 0.0007
INTRINSIC_ABLATION_COEFF = 0.002     # MFM, fragmentation-corrected
INTRINSIC_ABLATION_COEFF_UNC_RANGE = (0.001, 0.004)

FIRST_FRAGMENTATION_HEIGHT = 54900.0  # m, ~60% mass loss at this event
FIRST_FRAGMENTATION_SPEED = 13194.0   # m/s


def initial_state():
    """
    Nominal initial state vector y0 = [h, V, gamma, phi, lambda, psi, M]
    for Bunburra Rockhole, matching the dynamics.py convention.

    Returns
    -------
    ndarray, shape (7,)
    """
    return np.array([H0, V0, GAMMA0, PHI0, LAMBDA0, PSI0, M0])


if __name__ == "__main__":
    y0 = initial_state()
    labels = ["h0 (m)", "V0 (m/s)", "gamma0 (rad)", "phi0 (rad)",
              "lambda0 (rad)", "psi0 (rad)", "M0 (kg)"]
    print("Bunburra Rockhole initial state:")
    for label, val in zip(labels, y0):
        print(f"  {label:>15} = {val: .6f}")
    print(f"\nComputed travel bearing (psi0): {PSI0_DEG:.3f} deg")
    print(f"Primary recovery location (M1): {RECOVERY_LAT_DEG:.6f} S, "
          f"{RECOVERY_LON_DEG:.6f} E")