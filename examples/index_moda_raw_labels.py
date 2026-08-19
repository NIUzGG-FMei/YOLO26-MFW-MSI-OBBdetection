"""Index MODA raw OBB labels without downloading the corresponding images.

The raw MODA labels use eight polygon coordinates followed by a class name and
the difficult flag, for example ``x1 y1 ... x4 y4 car 0``.  This utility scans
local label caches, excludes samples already present in the working dataset,
and writes an auditable per-file index plus class-distribution summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ("car", "bus", "van", "awning-bike", "truck", "tricycle", "bike", "pedestrian")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass
class ParsedLabel:
    """Accumulate statistics for one raw label file."""

    class_counts: Counter[int]
    class_area_sums: dict[int, float]
    object_count: int = 0
    valid_line_count: int = 0
    difficult_count: int = 0
    malformed_line_count: int = 0
    unknown_class_count: int = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-candidate-label-dir",
        type=Path,
        default=Path("/home/mofengwei/datasetObjectDetection/labels_huancun/MODA/train/labels"),
        help="Local cache containing candidate MODA train labels.",
    )
    parser.add_argument(
        "--val-candidate-label-dir",
        type=Path,
        default=Path("/home/mofengwei/datasetObjectDetection/labels_huancun/MODA/test/labels"),
        help="Local cache containing candidate MODA validation labels.",
    )
    parser.add_argument(
        "--existing-train-label-dir",
        type=Path,
        default=Path("/home/mofengwei/datasetObjectDetection/train/labels"),
        help="Labels already used by the current training split.",
    )
    parser.add_argument(
        "--existing-val-label-dir",
        type=Path,
        default=Path("/home/mofengwei/datasetObjectDetection/test/labels"),
        help="Labels already used by the current validation split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/mofengwei/datasetObjectDetection/labels_huancun/MODA/index"),
        help="Directory for the index CSV and JSON summaries.",
    )
    parser.add_argument(
        "--train-exclude-difficult",
        action="store_true",
        help="Exclude difficult objects from train statistics. Disabled by default.",
    )
    parser.add_argument(
        "--val-include-difficult",
        action="store_true",
        help="Include difficult objects in validation statistics. Disabled by default to match the existing CSV.",
    )
    return parser.parse_args()


def parse_label_file(path: Path, include_difficult: bool) -> ParsedLabel:
    """Parse one MODA raw OBB label file."""
    parsed = ParsedLabel(class_counts=Counter(), class_area_sums={})
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) < 10:
            parsed.malformed_line_count += 1
            continue

        class_name = parts[8]
        if class_name not in CLASS_TO_ID:
            parsed.unknown_class_count += 1
            continue

        try:
            difficult = int(float(parts[9]))
            points = np.asarray([float(value) for value in parts[:8]], dtype=np.float32).reshape(4, 2)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid annotation at {path}:{line_number}") from error

        if not np.isfinite(points).all():
            parsed.malformed_line_count += 1
            continue
        if difficult:
            parsed.difficult_count += 1
            if not include_difficult:
                continue

        class_id = CLASS_TO_ID[class_name]
        area = abs(float(cv2.contourArea(points)))
        parsed.class_counts[class_id] += 1
        parsed.class_area_sums[class_id] = parsed.class_area_sums.get(class_id, 0.0) + area
        parsed.object_count += 1
        parsed.valid_line_count += 1
    return parsed


def existing_stems(*label_dirs: Path) -> set[str]:
    """Return image stems already assigned to a working split."""
    return {path.stem for label_dir in label_dirs for path in label_dir.glob("*.txt")}


def scan_candidate_dir(
    label_dir: Path,
    source_split: str,
    excluded_stems: set[str],
    include_difficult: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Scan a candidate label directory and return index rows plus counters."""
    rows: list[dict[str, object]] = []
    counters = Counter(total_files=0, excluded_existing=0, malformed_files=0, unknown_files=0, empty_files=0)
    if not label_dir.exists():
        return rows, dict(counters)

    for label_path in sorted(label_dir.glob("*.txt")):
        counters["total_files"] += 1
        if label_path.stem in excluded_stems:
            counters["excluded_existing"] += 1
            continue

        parsed = parse_label_file(label_path, include_difficult)
        if parsed.malformed_line_count:
            counters["malformed_files"] += 1
        if parsed.unknown_class_count:
            counters["unknown_files"] += 1
        if parsed.object_count == 0:
            counters["empty_files"] += 1

        row: dict[str, object] = {
            "source_split": source_split,
            "stem": label_path.stem,
            "label_path": str(label_path),
            "object_count": parsed.object_count,
            "valid_line_count": parsed.valid_line_count,
            "difficult_count": parsed.difficult_count,
            "malformed_line_count": parsed.malformed_line_count,
            "unknown_class_count": parsed.unknown_class_count,
            "eligible": int(not parsed.malformed_line_count and not parsed.unknown_class_count),
        }
        for class_id in range(len(CLASS_NAMES)):
            count = parsed.class_counts.get(class_id, 0)
            row[f"class_{class_id}_count"] = count
            row[f"class_{class_id}_present"] = int(count > 0)
            row[f"class_{class_id}_area_sum"] = f"{parsed.class_area_sums.get(class_id, 0.0):.6f}"
        rows.append(row)
    return rows, dict(counters)


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate object and image counts from index rows."""
    instance_counts = [0] * len(CLASS_NAMES)
    image_counts = [0] * len(CLASS_NAMES)
    total_instances = 0
    for row in rows:
        total_instances += int(row["object_count"])
        for class_id in range(len(CLASS_NAMES)):
            count = int(row[f"class_{class_id}_count"])
            instance_counts[class_id] += count
            image_counts[class_id] += int(count > 0)

    distribution = []
    image_total = len(rows)
    for class_id, class_name in enumerate(CLASS_NAMES):
        distribution.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "instance_count": instance_counts[class_id],
                "instance_fraction": (instance_counts[class_id] / total_instances) if total_instances else 0.0,
                "image_count_with_class": image_counts[class_id],
                "image_fraction": (image_counts[class_id] / image_total) if image_total else 0.0,
            }
        )
    return {
        "image_count": image_total,
        "instance_count": total_instances,
        "distribution": distribution,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_distribution(path: Path, distribution: list[dict[str, object]]) -> None:
    """Write a class distribution table."""
    write_csv(path, distribution)


def main() -> None:
    """Build the MODA raw-label index and summary files."""
    args = parse_args()
    train_include_difficult = not args.train_exclude_difficult
    val_include_difficult = args.val_include_difficult
    excluded_stems = existing_stems(args.existing_train_label_dir, args.existing_val_label_dir)

    train_rows, train_counters = scan_candidate_dir(
        args.train_candidate_label_dir, "train", excluded_stems, train_include_difficult
    )
    val_rows, val_counters = scan_candidate_dir(
        args.val_candidate_label_dir, "val", excluded_stems, val_include_difficult
    )
    all_rows = train_rows + val_rows
    existing_train_rows, _ = scan_candidate_dir(
        args.existing_train_label_dir, "existing_train", set(), train_include_difficult
    )
    existing_val_rows, _ = scan_candidate_dir(args.existing_val_label_dir, "existing_val", set(), val_include_difficult)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0]) if all_rows else []
    write_csv(args.output_dir / "candidate_label_index.csv", all_rows)
    write_distribution(args.output_dir / "candidate_train_distribution.csv", aggregate_rows(train_rows)["distribution"])
    write_distribution(args.output_dir / "candidate_val_distribution.csv", aggregate_rows(val_rows)["distribution"])
    write_distribution(
        args.output_dir / "existing_train_distribution.csv", aggregate_rows(existing_train_rows)["distribution"]
    )
    write_distribution(
        args.output_dir / "existing_val_distribution.csv", aggregate_rows(existing_val_rows)["distribution"]
    )

    summary = {
        "class_names": list(CLASS_NAMES),
        "train_include_difficult": train_include_difficult,
        "val_include_difficult": val_include_difficult,
        "excluded_existing_stem_count": len(excluded_stems),
        "candidate_train": aggregate_rows(train_rows),
        "candidate_val": aggregate_rows(val_rows),
        "existing_train": aggregate_rows(existing_train_rows),
        "existing_val": aggregate_rows(existing_val_rows),
        "train_scan": train_counters,
        "val_scan": val_counters,
        "index_row_count": len(all_rows),
        "index_field_count": len(fieldnames),
    }
    summary_path = args.output_dir / "label_index_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
