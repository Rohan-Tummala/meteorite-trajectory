"""
atmosphere.py

Atmospheric density models for the meteoroid trajectory simulation.

Two interchangeable density models are provided:
    - exponential_density(h)  : simple exponential approximation
    - tabulated_density(h)    : linear interpolation over a standard
                                 atmosphere density table

Both take altitude in metres and return density in kg/m^3, so they
can be swapped without changing any calling code (e.g. dynamics.py).

The top-level `density(h, model)` dispatcher should be used to select
which one is active, e.g.:

    from atmosphere import density
    rho = density(h, model="table")   # or model="exp"
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Exponential model
# ---------------------------------------------------------------------

RHO0 = 1.225        # sea-level density, kg/m^3
SCALE_HEIGHT = 7160.0  # atmospheric scale height, m (standard approx.)


def exponential_density(h):
    """
    Exponential atmosphere model: rho(h) = rho0 * exp(-h / H).

    Parameters
    ----------
    h : float or array_like
        Altitude above sea level, in metres. Expected domain: h >= 0.

    Returns
    -------
    float or ndarray
        Atmospheric density in kg/m^3.
    """
    h = np.asarray(h, dtype=float)
    return RHO0 * np.exp(-h / SCALE_HEIGHT)


# ---------------------------------------------------------------------
# Tabulated model (interpolated)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# COESA 1976 Standard Atmosphere (analytic, piecewise)
# ---------------------------------------------------------------------
# This is the actual defining model behind every "standard atmosphere"
# table found in textbooks — tables are just sampled points from
# these equations. Implementing it directly avoids transcription error
# and allows sampling at any resolution needed to build the
# interpolation table below, rather than relying on someone else's
# rounded, fixed-resolution table.
#
# Reference: U.S. Standard Atmosphere, 1976, U.S. Government Printing
# Office, Washington, D.C. Layer definitions per COESA (1976); see also
# https://www.pdas.com/atmos.html for a clear summary of the equations.

_G0 = 9.80665          # m/s^2
_R_AIR = 287.053        # J/(kg*K), specific gas constant for air
_R_EARTH_GEOPOTENTIAL = 6356766.0  # m, used to convert geometric -> geopotential altitude

# Each row: (base geopotential altitude H_b [m], base temperature T_b [K],
#            base pressure P_b [Pa], lapse rate L_b [K/m])
_COESA_LAYERS = np.array([
    [0.0,     288.15, 101325.0,    -0.0065],
    [11000.0, 216.65, 22632.0064,   0.0],
    [20000.0, 216.65, 5474.8890,    0.0010],
    [32000.0, 228.65, 868.0187,     0.0028],
    [47000.0, 270.65, 110.9063,     0.0],
    [51000.0, 270.65, 66.9389,     -0.0028],
    [71000.0, 214.65, 3.95642,     -0.0020],
])
_COESA_MAX_GEOMETRIC_M = 86000.0  # equations below are valid up to ~86 km geometric altitude


def _geopotential_altitude(z):
    """Convert geometric altitude z (m) to geopotential altitude H (m)."""
    return _R_EARTH_GEOPOTENTIAL * z / (_R_EARTH_GEOPOTENTIAL + z)


def _coesa_density_scalar(z):
    """COESA 1976 density at a single geometric altitude z (m)."""
    H = _geopotential_altitude(z)
    # find the highest layer whose base altitude is <= H
    idx = np.searchsorted(_COESA_LAYERS[:, 0], H, side="right") - 1
    idx = np.clip(idx, 0, len(_COESA_LAYERS) - 1)
    H_b, T_b, P_b, L_b = _COESA_LAYERS[idx]

    if L_b != 0.0:
        T = T_b + L_b * (H - H_b)
        P = P_b * (T / T_b) ** (-_G0 / (L_b * _R_AIR))
    else:
        T = T_b
        P = P_b * np.exp(-_G0 * (H - H_b) / (_R_AIR * T_b))

    return P / (_R_AIR * T)


def coesa_density(h):
    """
    COESA 1976 Standard Atmosphere density, computed directly from the
    defining piecewise equations (not interpolated from a table).

    Parameters
    ----------
    h : float or array_like
        Geometric altitude above sea level, in metres. Valid up to
        ~86,000 m; values above this are silently clipped.

    Returns
    -------
    float or ndarray
        Atmospheric density in kg/m^3.
    """
    h = np.asarray(h, dtype=float)
    h_clipped = np.clip(h, 0.0, _COESA_MAX_GEOMETRIC_M)

    if h_clipped.ndim == 0:
        return _coesa_density_scalar(float(h_clipped))
    return np.array([_coesa_density_scalar(hi) for hi in h_clipped])


# Build the interpolation table by sampling the COESA equations directly,
# rather than transcribing a textbook table by hand. Resolution here
# (1 km spacing) can be changed freely, e.g. to study how table
# resolution affects interpolation accuracy.
_TABLE_ALTITUDE_M = np.arange(0.0, _COESA_MAX_GEOMETRIC_M + 1.0, 1000.0)
_TABLE_DENSITY_KGM3 = coesa_density(_TABLE_ALTITUDE_M)


def tabulated_density(h):
    """
    Tabulated atmosphere model: linear interpolation over a standard
    atmosphere density table.

    Uses numpy.interp rather than scipy.interpolate.interp1d. Both do
    the same linear interpolation, but interp1d's general-purpose
    object overhead dominates runtime when called with a single
    scalar altitude on every RHS evaluation of the ODE solver
    (profiled, see Table 2 in the project report for the measured
    before/after split). np.interp is a
    dedicated 1D linear-interpolation routine with far less per-call
    overhead for exactly this use pattern.

    One behavioural difference from the previous interp1d-based
    version: np.interp does not extrapolate. A value outside the
    table range (0 to 86,000 m) is clamped to the nearest table
    boundary's density, rather than linearly extrapolated. This is
    arguably safer, it cannot return a negative density the way
    linear extrapolation technically could, and a model straying this
    far out of range (e.g. a diverging integration) is already
    flagged elsewhere by the caller's own stability checks regardless
    of what density value is returned here.

    Parameters
    ----------
    h : float or array_like
        Altitude above sea level, in metres. Values outside the table
        range are clamped to the nearest boundary density rather than
        extrapolated.

    Returns
    -------
    float or ndarray
        Atmospheric density in kg/m^3.
    """
    h = np.asarray(h, dtype=float)
    return np.interp(h, _TABLE_ALTITUDE_M, _TABLE_DENSITY_KGM3)


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def density(h, model="exp"):
    """
    Select and evaluate an atmospheric density model by name.

    Parameters
    ----------
    h : float or array_like
        Altitude above sea level, in metres.
    model : str, optional
        Which model to use: "exp" for the exponential approximation,
        or "table" for the tabulated/interpolated model. Default "exp".

    Returns
    -------
    float or ndarray
        Atmospheric density in kg/m^3.

    Raises
    ------
    ValueError
        If `model` is not "exp" or "table".
    """
    if model == "exp":
        return exponential_density(h)
    elif model == "table":
        return tabulated_density(h)
    else:
        raise ValueError(f"Unknown atmosphere model '{model}'. Use 'exp' or 'table'.")


def plot_atmosphere_comparison(interp_kind="linear"):
    """
    Plot the exponential and tabulated density models on the same
    axes for visual comparison.

    The exponential model is drawn as a smooth solid curve. The
    tabulated model is drawn as markers at the actual table points
    connected by a dotted line, since that dotted line literally *is*
    the interpolated result between data points (currently linear;
    pass interp_kind="cubic" once that is implemented to compare).

    Parameters
    ----------
    interp_kind : str, optional
        Interpolation method label shown in the legend. Currently the
        underlying interpolator is fixed to "linear" (see
        `tabulated_density`); this parameter only affects the
        legend text until cubic interpolation is added. Default "linear".
    """
    h_fine = np.linspace(0, _TABLE_ALTITUDE_M.max(), 500)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Exponential model: smooth curve over the full altitude range.
    ax.plot(h_fine, exponential_density(h_fine), "-", color="tab:blue",
             linewidth=2, label="Exponential model")

    # COESA 1976 model: the true underlying curve, evaluated directly
    # (no interpolation). Plotted as a dashed line so it remains visible
    # even where it nearly overlaps the interpolated table below.
    ax.plot(h_fine, coesa_density(h_fine), "--", color="tab:green",
             linewidth=2, label="COESA 1976 model (exact evaluation)")

    # Tabulated model: table points connected with a dotted line (this
    # dotted line *is* the interpolated result). Markers kept small
    # since the table is now densely sampled (1 km spacing).
    ax.plot(_TABLE_ALTITUDE_M, _TABLE_DENSITY_KGM3, "o:", color="tab:orange",
             linewidth=1.2, markersize=2,
             label=f"Tabulated model ({interp_kind} interpolation, "
                   f"COESA 1976, 1 km spacing)")

    ax.set_yscale("log")  # density spans several orders of magnitude
    ax.set_xlabel("Altitude, h (m)")
    ax.set_ylabel(r"Density, $\rho$ (kg/m$^3$)")
    ax.set_title("Atmospheric density models: exponential vs. tabulated")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()

    return fig, ax


if __name__ == "__main__":
    # Manual verification when running this file directly.
    test_altitudes = [0, 10000, 21500, 50000]
    for h in test_altitudes:
        print(f"h={h:>6} m  exp={density(h, 'exp'):.6e} kg/m^3  "
              f"table={density(h, 'table'):.6e} kg/m^3")

    plot_atmosphere_comparison(interp_kind="linear")