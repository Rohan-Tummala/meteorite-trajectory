"""
profile_memory.py

Memory/time profiling of the Cavezzo Monte Carlo run using Scalene.
Run from the src/ folder:

    scalene profile_memory.py

Then view the results with:

    scalene view --cli
"""

from monte_carlo_uncertainty_cavezzo import run_monte_carlo

# Full 5000-sample run, same as used for the actual report results.
# Takes a few minutes on a normal machine.
result = run_monte_carlo()