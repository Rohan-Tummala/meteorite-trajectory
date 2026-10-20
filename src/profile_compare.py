"""
profile_compare.py

Reproduces Table 2 in the report: profiles the Cavezzo RK45 run with
the OLD atmosphere.py (scipy's interp1d) against the CURRENT
atmosphere.py (np.interp), to measure the actual speedup from the
interpolation swap rather than quoting a number nobody can check.

Profiling numbers are machine-dependent (CPU, cache, library build),
so these figures only mean something when measured on the machine
being reported on. Run this locally and use ITS numbers for the
report, not numbers copied from someone else's machine.

Needs atmosphere_OLD_interp1d.py (the pre-swap version, pulled from
git history at commit 7ffb209, the parent of 18e02d2 "Profilling
updated") sitting next to this script. It is not imported directly;
instead, both the old and current src/ are copied into separate temp
directories with only atmosphere.py swapped, so each run imports a
clean, consistent set of modules.
"""

import cProfile
import pstats
import io
import os
import shutil
import sys
import tempfile
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
N_RUNS = 20


def _build_variant(atmosphere_source):
    """Copy every src/*.py file except atmosphere.py into a fresh temp
    directory, then drop in the given atmosphere.py variant. Returns
    the temp directory path."""
    tmp_dir = tempfile.mkdtemp(prefix="profile_variant_")
    for fname in os.listdir(THIS_DIR):
        if fname.endswith(".py") and fname != "atmosphere.py" \
                and fname != "profile_compare.py" \
                and fname != "atmosphere_OLD_interp1d.py":
            shutil.copy(os.path.join(THIS_DIR, fname), tmp_dir)
    shutil.copy(atmosphere_source, os.path.join(tmp_dir, "atmosphere.py"))
    return tmp_dir


def _profile_variant(tmp_dir, label):
    """Import integrator/cavezzo/dynamics fresh from tmp_dir (so the
    swapped-in atmosphere.py is the one actually used), run RK45 on
    the Cavezzo fall N_RUNS times under cProfile, and print the
    meteor_rhs / density-lookup cumulative time and share."""
    sys.path.insert(0, tmp_dir)
    # Drop any previously imported modules so the swapped atmosphere.py
    # (and everything that imports it) is reloaded from tmp_dir.
    for mod in list(sys.modules):
        if mod in ("integrator", "cavezzo", "dynamics", "atmosphere",
                    "gravity", "wind"):
            del sys.modules[mod]

    from integrator import run_trajectory
    from cavezzo import initial_state
    from dynamics import DEFAULT_PARAMS

    y0 = initial_state()
    t_span = (0.0, 600.0)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    for _ in range(N_RUNS):
        run_trajectory(y0, t_span, method="RK45", rtol=1e-8, atol=1e-10,
                        params=DEFAULT_PARAMS)
    profiler.disable()
    wall_s = time.perf_counter() - t0

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats("cumulative")

    rhs_time = _cumtime_for(stats, "meteor_rhs")
    density_time = _cumtime_for_any(stats, ("tabulated_density",
                                              "__call__"))

    print(f"\n=== {label} ===")
    print(f"Full run, {N_RUNS}x RK45 (wall-clock): {wall_s:.3f} s")
    print(f"meteor_rhs cumulative time: {rhs_time:.3f} s")
    print(f"density lookup cumulative time: {density_time:.3f} s")
    if rhs_time > 0:
        print(f"density lookup share of meteor_rhs: "
              f"{100.0 * density_time / rhs_time:.1f}%")

    sys.path.remove(tmp_dir)
    return wall_s, rhs_time, density_time


def _cumtime_for(stats, func_name):
    """Sum cumulative time for every profiled function whose name
    matches func_name exactly."""
    total = 0.0
    for (filename, lineno, name), (cc, nc, tt, ct, callers) in \
            stats.stats.items():
        if name == func_name:
            total += ct
    return total


def _cumtime_for_any(stats, func_names):
    """Sum cumulative time for every profiled function whose name
    matches any of func_names (used for the density lookup, since its
    entry point differs between interp1d's __call__ and np.interp's
    tabulated_density)."""
    total = 0.0
    for (filename, lineno, name), (cc, nc, tt, ct, callers) in \
            stats.stats.items():
        if name in func_names and ("atmosphere" in filename):
            total += ct
    return total


def main():
    """Profile the OLD (interp1d) and CURRENT (np.interp) atmosphere.py
    on the same Cavezzo RK45 run and print both so they can be
    compared directly, as Table 2 in the report does."""
    old_path = os.path.join(THIS_DIR, "atmosphere_OLD_interp1d.py")
    if not os.path.exists(old_path):
        print("Missing atmosphere_OLD_interp1d.py next to this script.")
        return

    current_path = os.path.join(THIS_DIR, "atmosphere.py")

    old_dir = _build_variant(old_path)
    _profile_variant(old_dir, "BEFORE (interp1d)")
    shutil.rmtree(old_dir, ignore_errors=True)

    new_dir = _build_variant(current_path)
    _profile_variant(new_dir, "AFTER (np.interp)")
    shutil.rmtree(new_dir, ignore_errors=True)


if __name__ == "__main__":
    main()