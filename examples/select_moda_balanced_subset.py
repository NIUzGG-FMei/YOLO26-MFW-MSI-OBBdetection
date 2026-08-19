"""Select auditable MODA train/validation subsets from a raw-label index.

This script selects image names using current class-instance deficits. It never
downloads or modifies images; the resulting manifest is consumed by the later
download step. A capped per-image contribution prevents one crowded scene from
dominating the selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.index_moda_raw_labels import CLASS_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("/home/mofengwei/datasetObjectDetection/labels_huancun/MODA/index"),
        help="Directory created by index_moda_raw_labels.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/mofengwei/datasetObjectDetection/labels_huancun/MODA/index"),
        help="Directory for the selection manifests and projected distributions.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=1000,
        help="Number of new train images to select.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=200,
        help="Number of new validation images to select.",
    )
    parser.add_argument(
        "--empty-fraction",
        type=float,
        default=0.0,
        help="Optional fraction of selected images reserved for empty labels.",
    )
    parser.add_argument(
        "--per-image-class-cap",
        type=int,
        default=3,
        help="Maximum contribution of one image from one class to its score.",
    )
    parser.add_argument(
        "--weight-exponent",
        type=float,
        default=0.5,
        help="Exponent for inverse-frequency class weights; 0.5 is a moderate correction.",
    )
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=15.0,
        help="Maximum inverse-frequency weight for a class.",
    )
    parser.add_argument(
        "--train-target-counts",
        type=str,
        default=None,
        help="Optional post-selection train targets, comma-separated in the eight class order.",
    )
    parser.add_argument(
        "--val-target-counts",
        type=str,
        default=None,
        help="Optional post-selection validation targets, comma-separated in the eight class order.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def class_counts(row: dict[str, str]) -> list[int]:
    """Return per-class object counts for one index row."""
    return [int(row[f"class_{class_id}_count"]) for class_id in range(len(CLASS_NAMES))]


def distribution(rows: list[dict[str, str]], counts: list[int] | None = None) -> list[dict[str, object]]:
    """Aggregate object and image counts for index rows."""
    instance_counts = [0] * len(CLASS_NAMES) if counts is None else counts.copy()
    image_counts = [0] * len(CLASS_NAMES)
    if counts is None:
        for row in rows:
            for class_id, count in enumerate(class_counts(row)):
                instance_counts[class_id] += count
                image_counts[class_id] += int(count > 0)
    else:
        for row in rows:
            for class_id, count in enumerate(class_counts(row)):
                image_counts[class_id] += int(count > 0)

    total_instances = sum(instance_counts)
    image_count = len(rows)
    return [
        {
            "class_id": class_id,
            "class_name": class_name,
            "instance_count": instance_counts[class_id],
            "instance_fraction": (instance_counts[class_id] / total_instances) if total_instances else 0.0,
            "image_count_with_class": image_counts[class_id],
            "image_fraction": (image_counts[class_id] / image_count) if image_count else 0.0,
        }
        for class_id, class_name in enumerate(CLASS_NAMES)
    ]


def load_initial_counts(path: Path) -> list[int]:
    """Read existing per-class instance counts from an index distribution CSV."""
    rows = read_csv(path)
    counts = [0] * len(CLASS_NAMES)
    for row in rows:
        counts[int(row["class_id"])] = int(row["instance_count"])
    return counts


def parse_target_counts(text: str | None) -> list[int] | None:
    """Parse an optional comma-separated target vector."""
    if text is None:
        return None
    values = [int(value.strip()) for value in text.split(",")]
    if len(values) != len(CLASS_NAMES) or any(value < 0 for value in values):
        raise ValueError(f"Target counts must contain {len(CLASS_NAMES)} non-negative integers.")
    return values


def score_row(
    row: dict[str, str],
    current_counts: list[int],
    args: argparse.Namespace,
    target_counts: list[int] | None,
) -> float:
    """Score one candidate using inverse frequency or explicit target deficits."""
    reference = max(current_counts, default=1)
    score = 0.0
    for class_id, count in enumerate(class_counts(row)):
        if count <= 0:
            continue
        if target_counts is None:
            weight = (reference / max(current_counts[class_id], 1)) ** args.weight_exponent
        else:
            deficit = max(target_counts[class_id] - current_counts[class_id], 0)
            if deficit <= 0:
                continue
            weight = (deficit / max(current_counts[class_id], 1)) ** args.weight_exponent
        weight = min(weight, args.max_class_weight)
        score += weight * min(count, args.per_image_class_cap)
    return score


def select_positive_rows(
    rows: list[dict[str, str]],
    limit: int,
    initial_counts: list[int],
    args: argparse.Namespace,
    target_counts: list[int] | None,
) -> tuple[list[dict[str, object]], list[int]]:
    """Greedily select positive candidates and update instance counts."""
    remaining = [row for row in rows if int(row["eligible"]) and int(row["object_count"]) > 0]
    selected: list[dict[str, object]] = []
    current_counts = initial_counts.copy()

    for order in range(1, min(limit, len(remaining)) + 1):
        best = max(
            remaining,
            key=lambda row: (
                score_row(row, current_counts, args, target_counts),
                len([count for count in class_counts(row) if count > 0]),
                str(row["stem"]),
            ),
        )
        score = score_row(best, current_counts, args, target_counts)
        counts = class_counts(best)
        selected.append({**best, "selection_order": order, "selection_score": f"{score:.6f}"})
        current_counts = [current + added for current, added in zip(current_counts, counts)]
        remaining.remove(best)
    return selected, current_counts


def select_empty_rows(rows: list[dict[str, str]], limit: int, start_order: int) -> list[dict[str, object]]:
    """Select empty-label candidates in source order for optional hard negatives."""
    selected = []
    for offset, row in enumerate(
        (row for row in rows if int(row["eligible"]) and int(row["object_count"]) == 0), start=0
    ):
        if offset >= limit:
            break
        selected.append({**row, "selection_order": start_order + offset, "selection_score": "0.000000"})
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_split(
    rows: list[dict[str, str]],
    split: str,
    limit: int,
    initial_counts: list[int],
    args: argparse.Namespace,
    target_counts: list[int] | None,
) -> tuple[list[dict[str, object]], list[int]]:
    """Select one split and attach its destination/remote paths."""
    empty_limit = min(limit, round(limit * args.empty_fraction))
    positive_limit = limit - empty_limit
    positive_rows, final_counts = select_positive_rows(rows, positive_limit, initial_counts, args, target_counts)
    empty_rows = select_empty_rows(rows, empty_limit, len(positive_rows) + 1)
    selected = positive_rows + empty_rows
    for row in selected:
        source_dir = "train" if split == "train" else "test"
        destination_dir = "train_1" if split == "train" else "test_1"
        row.update(
            {
                "destination_split": destination_dir,
                "remote_image_path": f"MODA/{source_dir}/images/{row['stem']}.npy",
                "remote_label_path": f"MODA/{source_dir}/labels/{row['stem']}.txt",
            }
        )
    return selected, final_counts


def main() -> None:
    """Create train/test selection manifests and projected distributions."""
    args = parse_args()
    if args.train_limit < 0 or args.val_limit < 0:
        raise ValueError("Selection limits must be non-negative.")
    if not 0.0 <= args.empty_fraction < 1.0:
        raise ValueError("empty_fraction must be in [0, 1).")
    if args.per_image_class_cap < 1:
        raise ValueError("per_image_class_cap must be positive.")

    index_rows = read_csv(args.index_dir / "candidate_label_index.csv")
    train_rows = [row for row in index_rows if row["source_split"] == "train"]
    val_rows = [row for row in index_rows if row["source_split"] == "val"]
    train_counts = load_initial_counts(args.index_dir / "existing_train_distribution.csv")
    val_counts = load_initial_counts(args.index_dir / "existing_val_distribution.csv")
    train_target_counts = parse_target_counts(args.train_target_counts)
    val_target_counts = parse_target_counts(args.val_target_counts)

    selected_train, projected_train_counts = select_split(
        train_rows, "train", args.train_limit, train_counts, args, train_target_counts
    )
    selected_val, projected_val_counts = select_split(
        val_rows, "val", args.val_limit, val_counts, args, val_target_counts
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "train_selection_manifest.csv", selected_train)
    write_csv(args.output_dir / "val_selection_manifest.csv", selected_val)
    write_csv(
        args.output_dir / "projected_train_distribution.csv", distribution(selected_train, projected_train_counts)
    )
    write_csv(args.output_dir / "projected_val_distribution.csv", distribution(selected_val, projected_val_counts))

    summary = {
        "train_requested": args.train_limit,
        "train_selected": len(selected_train),
        "val_requested": args.val_limit,
        "val_selected": len(selected_val),
        "empty_fraction": args.empty_fraction,
        "per_image_class_cap": args.per_image_class_cap,
        "weight_exponent": args.weight_exponent,
        "max_class_weight": args.max_class_weight,
        "train_initial_counts": train_counts,
        "train_projected_counts": projected_train_counts,
        "val_initial_counts": val_counts,
        "val_projected_counts": projected_val_counts,
        "train_target_counts": train_target_counts,
        "val_target_counts": val_target_counts,
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
