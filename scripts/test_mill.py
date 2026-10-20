"""Evaluate a MILL-trained SDDNet-v1 checkpoint on paired MILL images."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from scripts.evaluate_common import run_test

if __name__ == "__main__":
    run_test("mill")
