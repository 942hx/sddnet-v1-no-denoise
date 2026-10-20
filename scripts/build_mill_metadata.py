"""Generate CSV/JSON MILL metadata, or a template for unsupported filenames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from datasets.common import write_mill_metadata
from utils.runtime import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/MILLs-DSLR")
    parser.add_argument("--output", default="metadata/mill.csv")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--allow-unparsed", action="store_true", help="Leave unknown labels/GT blank for manual editing.")
    args = parser.parse_args()
    count = write_mill_metadata(project_path(args.root), project_path(args.output), args.splits, args.allow_unparsed)
    print(f"Wrote {count} rows to {project_path(args.output)}")
    if args.allow_unparsed:
        print("Check blank scene_id, illum_level, and gt_path fields before training.")


if __name__ == "__main__":
    main()
