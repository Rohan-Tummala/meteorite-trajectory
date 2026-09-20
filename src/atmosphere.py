"""
atmosphere.py

Atmospheric density models for the meteoroid trajectory simulation.

Two interchangeable density models are provided:
    - exponential_density(h)  : simple exponential approximation
    - tabulated_density(h)    : linear interpolation over a standard
                                 atmosphere density table

Both take altitude in metres and return density in kg/m^3, so they
can be swapped without changing any calling code (e.g. dynamics.py).

Use the top-level `density(h, model)` dispatcher from your main code
to select which one is active, e.g.:

    from atmosphere import density
    rho = density(h, model="table")   # or model="exp"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

# Standard Atmosphere density table (e.g. Anderson, Introduction to
# Flight, Table 1-5). Altitude in metres (converted from the table's
# km), density in kg/m^3. The table's -1 km row is dropped since
# altitude in this simulation is never negative.
_TABLE_ALTITUDE_KM = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 25, 30, 32,
     35, 40, 45, 47, 50, 51, 60, 70, 71, 80, 84.9, 89.7, 100.4, 105, 110],
    dtype=float
)
_TABLE_ALTITUDE_M = _TABLE_ALTITUDE_KM * 1000.0

_TABLE_DENSITY_KGM3 = np.array(
    [1.2250, 1.1116, 1.0065, 0.9091, 0.8191, 0.7361, 0.6597, 0.5895,
     0.5252, 0.4664, 0.4127, 0.3639, 0.2655, 0.1937, 0.1423, 0.0880,
     0.0395, 0.0180, 0.0132, 0.0082, 0.0039, 0.0019, 0.0014, 0.0010,
     0.00086, 0.000288, 0.000074, 0.000064, 0.000015, 0.000007,
     0.000003, 0.0000005, 0.0000002, 0.0000001],
    dtype=float
)

# Build the interpolator once at import time so repeated calls are cheap.
# bounds_error=False + fill_value="extrapolate" avoids crashing on values
# slightly outside the table; a warning is raised instead (see below).
_density_interpolator = interp1d(
    _TABLE_ALTITUDE_M,
    _TABLE_DENSITY_KGM3,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate",
)


def tabulated_density(h):
    """
    Tabulated atmosphere model: linear interpolation over a standard
    atmosphere density table.

    Parameters
    ----------
    h : float or array_like
        Altitude above sea level, in metres. Values outside the table
        range (0 to 110,000 m) are linearly extrapolated; a warning is
        printed once per call if this happens, since extrapolated
        density is not physically validated.

    Returns
    -------
    float or ndarray
        Atmospheric density in kg/m^3.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < _TABLE_ALTITUDE_M.min()) or np.any(h > _TABLE_ALTITUDE_M.max()):
        print(f"Warning: altitude {h} m outside tabulated range "
              f"[{_TABLE_ALTITUDE_M.min()}, {_TABLE_ALTITUDE_M.max()}] m — extrapolating.")
    return _density_interpolator(h)


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
    pass interp_kind="cubic" once that's implemented to compare).

    Parameters
    ----------
    interp_kind : str, optional
        Interpolation method label shown in the legend. Currently the
        underlying interpolator is fixed to "linear" (see
        `_density_interpolator`); this parameter only affects the
        legend text until cubic interpolation is added. Default "linear".
    """
    h_fine = np.linspace(0, _TABLE_ALTITUDE_M.max(), 500)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Exponential model: smooth curve over the full altitude range.
    ax.plot(h_fine, exponential_density(h_fine), "-", color="tab:blue",
             linewidth=2, label="Exponential model")

    # Tabulated model: raw table points connected with a dotted line.
    ax.plot(_TABLE_ALTITUDE_M, _TABLE_DENSITY_KGM3, "o:", color="tab:orange",
             linewidth=1.5, markersize=6,
             label=f"Tabulated model ({interp_kind} interpolation)")

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
    # Quick manual sanity check when running this file directly.
    test_altitudes = [0, 10000, 21500, 50000]
    for h in test_altitudes:
        print(f"h={h:>6} m  exp={density(h, 'exp'):.6e} kg/m^3  "
              f"table={density(h, 'table'):.6e} kg/m^3")

    plot_atmosphere_comparison(interp_kind="linear")