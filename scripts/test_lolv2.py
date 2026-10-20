"""Evaluate a checkpoint on LOL-v2 Real_captured, Synthetic, or both."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from scripts.evaluate_common import run_test

if __name__ == "__main__":
    run_test("lolv2")
