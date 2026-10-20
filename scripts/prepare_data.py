"""Decode and hash all paired data; produce explicit manifests and a fail-closed audit."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import hashlib
from pathlib import Path

from PIL import Image
import numpy as np
from tqdm import tqdm

from datasets.common import MILL_EXCLUSIONS, image_files, parse_mill_filename, stem_index
from utils.experiment import assert_environment, experiment_config, file_hash, run_path
from utils.runtime import project_path

FIELDS = ["dataset", "filename", "split", "scene_id", "illum_level", "input_path", "gt_path",
          "width", "height", "input_sha256", "gt_sha256", "content_id", "level", "group_id", "valid",
          "input_pixel_sha256", "gt_pixel_sha256", "input_min", "input_max", "reason"]


def prepare_data(cfg):
    """Reject unknown pairs/corruption; the one official exclusion is explicit."""
    assert_environment(cfg)
    destination = run_path(cfg)
    metadata_dir = destination / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    audit = {"passed": False, "path_base": "dataset root specified in configuration",
             "splits": {}, "errors": [], "warnings": [], "official_exclusions": [], "manifest_hashes": {}}
    roots = {"mill": project_path(cfg["data"]["mill_root"]),
             "lolv1": project_path(cfg["data"]["lolv1_root"]),
             "lolv2_real": project_path(cfg["data"]["lolv2_root"]) / "Real_captured",
             "lolv2_synthetic": project_path(cfg["data"]["lolv2_root"]) / "Synthetic"}
    layouts = [("mill", s, f"{s}/input", f"{s}/gt") for s in ("train", "validation", "test")]
    layouts += [("lolv1", s, f"{folder}/low", f"{folder}/high") for s, folder in (("train", "our485"), ("test", "eval15"))]
    for dataset in ("lolv2_real", "lolv2_synthetic"):
        layouts += [(dataset, s.lower(), f"{s}/Low", f"{s}/Normal") for s in ("Train", "Test")]
    records = collections.defaultdict(list)
    image_info = {}
    group_file = cfg["data"].get("group_mapping")
    groups = json.loads(project_path(group_file).read_text(encoding="utf-8")) if group_file else {}
    audit["group_basis"] = ("Explicit user group mapping" if group_file else
        "No verified capture-series grouping. group_id blank; full content_id is used. Numeric prefixes are not physical-scene evidence.")

    def decode(path):
        key = str(path.absolute()).casefold()
        if key not in image_info:
            with path.open("rb") as stream, Image.open(stream) as image:
                image.load()  # A PNG header alone is not an integrity test.
                size = image.size
                if image.mode != "RGB":
                    raise ValueError(f"Expected exported 8-bit RGB; actual {image.mode}: {path}")
                pixels = np.asarray(image)
                pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
                low, high = float(pixels.min()) / 255, float(pixels.max()) / 255
                if not np.isfinite([low, high]).all() or low < 0 or high > 1:
                    raise ValueError(f"Invalid RGB data range: {path}")
            image_info[key] = (size, file_hash(path), pixel_hash, low, high)
        return image_info[key]

    for dataset, split, input_folder, gt_folder in layouts:
        root = roots[dataset]
        key = f"{dataset}/{split}"
        split_rows, seen, seen_ids, used_gt = [], set(), set(), set()
        try:
            lows, targets = image_files(root / input_folder), stem_index(root / gt_folder)
        except Exception as error:
            audit["errors"].append(f"{key}: {type(error).__name__}: {error}")
            continue
        if dataset == "mill" and split == "test":
            for (excluded_split, filename), reason in MILL_EXCLUSIONS.items():
                audit["official_exclusions"].append({"dataset": dataset, "split": excluded_split,
                    "filename": "Test13-9.png", "present": (root / input_folder / "Test13-9.png").exists(),
                    "reason": reason, "source": "https://github.com/CVC-Color/MILL#mill-dataset"})
        for low in tqdm(lows, desc=f"Audit {key}", leave=False):
            if dataset == "mill" and (split, low.name.casefold()) in MILL_EXCLUSIONS:
                continue
            try:
                scene, level = parse_mill_filename(low.name) if dataset == "mill" else (low.stem, "")
                sample_id = (scene.casefold(), level)
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate/ambiguous content-level ID: {sample_id}")
                seen_ids.add(sample_id)
                target = targets.get(scene.casefold())
                if target is None:
                    raise ValueError(f"No unambiguous GT for scene {scene!r}")
                low_key = str(low.absolute()).casefold()
                if low_key in seen:
                    raise ValueError("Duplicate input path")
                seen.add(low_key)
                (width, height), low_hash, low_pixels, minimum, maximum = decode(low)
                gt_size, gt_hash, gt_pixels, _, _ = decode(target)
                if (width, height) != gt_size:
                    raise ValueError(f"Input size {(width, height)} != GT size {gt_size}")
                used_gt.add(target)
                split_rows.append(dict(dataset=dataset, filename=low.name, split=split, scene_id=scene,
                                       illum_level=level, input_path=low.relative_to(root).as_posix(),
                                       gt_path=target.relative_to(root).as_posix(), width=width, height=height,
                                       input_sha256=low_hash, gt_sha256=gt_hash, content_id=scene,
                                       level=level, group_id=str(groups.get(scene, "")) if dataset == "mill" else "",
                                       valid=True, input_pixel_sha256=low_pixels, gt_pixel_sha256=gt_pixels,
                                       input_min=minimum, input_max=maximum, reason=""))
            except Exception as error:
                audit["errors"].append(f"{key}/{low.name}: {type(error).__name__}: {error}")
                # Preserve a concrete invalid row rather than silently dropping it.
                records[dataset].append(dict(dataset=dataset, filename=low.name, split=split,
                    input_path=low.relative_to(root).as_posix(), valid=False, reason=str(error)))
        unused = [p.name for p in targets.values() if p not in used_gt]
        if unused:
            audit["errors"].append(f"{key}: unused GT files: {unused}")
        audit["splits"][key] = {"input_files": len(lows), "valid_pairs": len(split_rows), "gt_files": len(targets),
                                 "scenes": len({r["scene_id"] for r in split_rows}),
                                 "level_counts": dict(sorted(collections.Counter(str(int(r["illum_level"]))
                                     for r in split_rows if r["illum_level"] != "").items(), key=lambda p: int(p[0]))),
                                 "sizes": dict(collections.Counter(f"{r['width']}x{r['height']}" for r in split_rows))}
        records[dataset].extend(split_rows)
    mill_scenes = {s: {r["scene_id"].casefold() for r in records["mill"] if r["split"] == s and r.get("valid")}
                   for s in ("train", "validation", "test")}
    audit["mill_scene_overlaps"] = {}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(mill_scenes[first] & mill_scenes[second])
        audit["mill_scene_overlaps"][f"{first}/{second}"] = overlap
        if overlap:
            audit["errors"].append(f"MILL scene leakage {first}/{second}: {overlap}")
    # Identical GT hashes under different IDs would also indicate potential leakage.
    gt_splits = collections.defaultdict(set)
    for row in records["mill"]:
        if row.get("valid"):
            gt_splits[row["gt_pixel_sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in gt_splits.values()):
        audit["errors"].append("Identical decoded MILL GT pixels occur across splits.")
    for dataset, rows in records.items():
        path = metadata_dir / f"{dataset}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        audit["manifest_hashes"][dataset] = file_hash(path)
    audit["passed"] = not audit["errors"] and len(audit["manifest_hashes"]) == 4
    (destination / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Data audit", "", f"Passed: {audit['passed']}", "", "Paths in CSV are relative to each dataset root.", "",
             "| Dataset/split | Inputs | Valid pairs | GT | Scenes |", "|---|---:|---:|---:|---:|"]
    for name, row in audit["splits"].items():
        lines.append(f"| {name} | {row['input_files']} | {row['valid_pairs']} | {row['gt_files']} | {row['scenes']} |")
    lines += ["", "Official exclusion: Test13-9.png (MILL test), whether absent or present; no source deletion.",
              "", "## Errors", *[f"- {error}" for error in audit["errors"]]]
    (destination / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Data audit passed={audit['passed']}; manifests={audit['manifest_hashes']}", flush=True)
    if not audit["passed"]:
        raise RuntimeError("Data audit failed; no training is permitted. See data_audit.json.")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sddnet_v1.yaml")
    args = parser.parse_args()
    prepare_data(experiment_config(args.config))


if __name__ == "__main__":
    main()
