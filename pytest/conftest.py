"""
conftest.py

pytest configuration root. The project's modules live in src/ and
import each other with plain top-level names (e.g. `from dynamics
import meteor_rhs`), not as a package, so src/ must be on sys.path
before any test module is collected. Adding it here, once, makes it
available to every test file under test/ without each one repeating
the same path setup.
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)