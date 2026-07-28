"""Shared utilities for deterministic multi-view OBB dataset preparation.

This module is intentionally independent from the training entry point.  Both
``custom_obb_prepare_and_train.py`` and
``build_target_class_augmented_obb_dataset.py`` use it in the
``balanced_multiscale`` profile, while the legacy profile keeps its original
code path.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

VIEW_TYPES = ("patch256", "patch512", "full_scaled")
IDENTITY_TRANSFORM = "identity"


@dataclass(frozen=True)
class ViewRatios:
    """Target sample ratios for the three input views."""

    patch256: float = 0.70
    patch512: float = 0.20
    full_scaled: float = 0.10

    def as_dict(self) -> dict[str, float]:
        """Return ratios keyed by view name."""
        return asdict(self)


@dataclass(frozen=True)
class ViewCandidate:
    """A cheap description of one materializable dataset sample."""

    candidate_id: str
    source_path: str
    split: str
    view_type: str
    size: int
    x0: int = 0
    y0: int = 0
    transform: str = IDENTITY_TRANSFORM
    has_object: bool = False
    has_target: bool = False
    object_count: int = 0
    anchor_class_id: int = -1
    anchor_gt_index: int = -1


def validate_view_ratios(ratios: ViewRatios, tolerance: float = 1e-6) -> ViewRatios:
    """Validate non-negative view ratios that sum to one."""
    values = ratios.as_dict()
    if any(value < 0 for value in values.values()):
        raise ValueError(f"View ratios must be non-negative, got {values}.")
    total = sum(values.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"View ratios must sum to 1.0, got {total:.8f} ({values}).")
    if any(value == 0 for value in values.values()):
        raise ValueError(f"Each balanced view ratio must be greater than zero, got {values}.")
    return ratios


def allocate_view_counts(total: int, ratios: ViewRatios) -> dict[str, int]:
    """Allocate an integer total with the largest-remainder method."""
    validate_view_ratios(ratios)
    if total < 3:
        raise ValueError(f"At least 3 samples are required for three view types, got {total}.")
    names = list(VIEW_TYPES)
    raw = [total * ratios.as_dict()[name] for name in names]
    # First round down the ideal quotas, while reserving one sample for each
    # view.  The old implementation applied the minimum after subtracting
    # one from every quota and could return four samples for ``total=3``.
    counts = [max(1, math.floor(value)) for value in raw]
    if sum(counts) > total:
        # This only occurs for very small totals.  Exact 70/20/10 is
        # impossible with three non-empty integer buckets, so the safest
        # deterministic fallback is one sample per view.
        counts = [1, 1, 1]

    remaining = total - sum(counts)
    fractional = [value - math.floor(value) for value in raw]
    order = sorted(range(len(names)), key=lambda i: (-fractional[i], i))
    for index in order[:remaining]:
        counts[index] += 1
    if sum(counts) != total:
        raise RuntimeError(f"Internal view quota error: counts={counts}, total={total}, ratios={ratios}.")
    return dict(zip(names, counts))


def maximum_feasible_total(candidate_counts: dict[str, int], ratios: ViewRatios) -> int:
    """Return the largest total whose requested quotas fit all candidate pools."""
    validate_view_ratios(ratios)
    if any(name not in candidate_counts for name in VIEW_TYPES):
        raise ValueError(f"candidate_counts must contain {VIEW_TYPES}, got {candidate_counts}.")
    limits = [candidate_counts[name] / ratios.as_dict()[name] for name in VIEW_TYPES]
    total = min(sum(candidate_counts.values()), math.floor(min(limits)))
    # Integer rounding can make the quota for one bucket exceed its pool by
    # one.  Walk down only as far as needed and validate the actual allocator.
    while total >= 3:
        counts = allocate_view_counts(total, ratios)
        if all(counts[name] <= candidate_counts[name] for name in VIEW_TYPES):
            return total
        total -= 1
    return 0


def stable_rank(key: str, seed: int = 0) -> str:
    """Return a deterministic ranking key for reproducible candidate sampling."""
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def normalize_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an HWC array to the uint8 convention used by training."""
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    image = image.astype(np.float32)
    max_value = float(image.max())
    min_value = float(image.min())
    if max_value <= 1.0 and min_value >= 0.0:
        image *= 255.0
    elif max_value > 255.0 or min_value < 0.0:
        image = (image - min_value) * (255.0 / max(max_value - min_value, 1e-6))
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def to_hwc(image: np.ndarray, layout: str) -> np.ndarray:
    """Convert a 2D/CHW/CWH/HWC array to HWC."""
    if image.ndim == 2:
        image = image[..., None]
    elif image.ndim == 3:
        if layout == "CHW":
            image = np.transpose(image, (1, 2, 0))
        elif layout == "CWH":
            image = np.transpose(image, (2, 1, 0))
        elif layout != "HWC":
            raise ValueError(f"Unsupported NPY layout {layout!r}; expected CHW, CWH or HWC.")
    else:
        raise ValueError(f"Expected a 2D or 3D image, got shape={image.shape}.")
    return normalize_hwc_uint8(image)


def load_npy_hwc(path: Path, layout: str) -> np.ndarray:
    """Load an NPY image and normalize it to contiguous HWC uint8."""
    return to_hwc(np.load(path, allow_pickle=False), layout)


def load_multichannel_tiff(path: Path) -> np.ndarray:
    """Load a single- or multi-page TIFF as contiguous HWC data.

    OpenCV's normal ``imread`` returns only the first page of a multi-page
    TIFF.  The prepared dataset stores one channel per page, so validation
    and offline inspection must use ``imdecodemulti`` and stack all pages.
    """
    file_bytes = np.fromfile(str(path), dtype=np.uint8)
    success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
    if not success or not frames:
        raise RuntimeError(f"Failed to read TIFF image: {path}")
    if len(frames) == 1 and frames[0].ndim == 3:
        image = frames[0]
    else:
        image = np.stack([frame if frame.ndim == 2 else frame[..., 0] for frame in frames], axis=2)
    return np.ascontiguousarray(image)


def save_multichannel_tiff(path: Path, image: np.ndarray) -> None:
    """Save an HWC uint8 image as a multi-page TIFF."""
    image = normalize_hwc_uint8(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwritemulti(str(path), np.ascontiguousarray(image.transpose(2, 0, 1))):
        raise RuntimeError(f"Failed to write multi-page TIFF: {path}")


def polygon_area(points: np.ndarray) -> float:
    """Return the absolute area of a polygon."""
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def clip_polygon_to_rect(points: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray | None:
    """Clip a convex polygon to an axis-aligned rectangle."""
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    area, clipped = cv2.intersectConvexConvex(points.astype(np.float32), rect)
    if area <= 1e-6 or clipped is None or len(clipped) < 3:
        return None
    return clipped.reshape(-1, 2).astype(np.float32)


def rebuild_obb_from_polygon(points: np.ndarray) -> np.ndarray:
    """Fit a four-corner OBB to a polygon."""
    return cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32))).astype(np.float32)


def clip_points_to_bounds(points: np.ndarray, width: float, height: float) -> np.ndarray:
    """Clip polygon coordinates to image bounds."""
    result = points.astype(np.float32, copy=True)
    result[:, 0] = np.clip(result[:, 0], 0.0, width)
    result[:, 1] = np.clip(result[:, 1], 0.0, height)
    return result


def project_annotation_to_patch(
    points: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_iof: float,
) -> tuple[np.ndarray | None, float]:
    """Clip and rebuild one OBB if its intersection-over-foreground is sufficient."""
    original_area = polygon_area(points)
    clipped = clip_polygon_to_rect(points, float(x0), float(y0), float(x1), float(y1))
    if clipped is None or original_area <= 1e-6:
        return None, 0.0
    iof = polygon_area(clipped) / original_area
    if iof < min_iof:
        return None, iof
    shifted = clipped - np.array([x0, y0], dtype=np.float32)
    rebuilt = clip_points_to_bounds(rebuild_obb_from_polygon(shifted), float(x1 - x0), float(y1 - y0))
    return (rebuilt if polygon_area(rebuilt) > 1e-6 else None), iof


def get_starts(length: int, window: int, overlap: bool) -> list[int]:
    """Return deterministic sliding-window starts, including the final edge."""
    if length <= window:
        return [0]
    step = max(window // 2, 1) if overlap else window
    starts = list(range(0, length - window + 1, step))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def collect_patch_labels(
    annotations: Sequence[tuple[int, np.ndarray]], x0: int, y0: int, x1: int, y1: int, min_iof: float
) -> list[tuple[int, np.ndarray]]:
    """Project all annotations into one patch."""
    labels: list[tuple[int, np.ndarray]] = []
    for class_id, points in annotations:
        projected, _ = project_annotation_to_patch(points, x0, y0, x1, y1, min_iof)
        if projected is not None:
            labels.append((class_id, projected))
    return labels


def apply_transform(
    image: np.ndarray, labels: Sequence[tuple[int, np.ndarray]], transform: str
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Apply a geometric transform and the corresponding OBB point transform."""
    image_h, image_w = image.shape[:2]
    if transform == IDENTITY_TRANSFORM:
        return np.ascontiguousarray(image), [(int(c), p.copy()) for c, p in labels]
    if transform == "rot90":
        transformed_image = np.ascontiguousarray(np.rot90(image, 1))

        def transform_points(points: np.ndarray) -> np.ndarray:
            return np.stack((points[:, 1], image_w - points[:, 0]), axis=1)

    elif transform == "rot180":
        transformed_image = np.ascontiguousarray(np.rot90(image, 2))

        def transform_points(points: np.ndarray) -> np.ndarray:
            return np.stack((image_w - points[:, 0], image_h - points[:, 1]), axis=1)

    elif transform == "rot270":
        transformed_image = np.ascontiguousarray(np.rot90(image, 3))

        def transform_points(points: np.ndarray) -> np.ndarray:
            return np.stack((image_h - points[:, 1], points[:, 0]), axis=1)

    elif transform == "flip_h":
        transformed_image = np.ascontiguousarray(np.flip(image, axis=1))

        def transform_points(points: np.ndarray) -> np.ndarray:
            return np.stack((image_w - points[:, 0], points[:, 1]), axis=1)

    elif transform == "flip_v":
        transformed_image = np.ascontiguousarray(np.flip(image, axis=0))

        def transform_points(points: np.ndarray) -> np.ndarray:
            return np.stack((points[:, 0], image_h - points[:, 1]), axis=1)

    else:
        raise ValueError(f"Unsupported geometric transform {transform!r}.")

    new_h, new_w = transformed_image.shape[:2]
    transformed_labels = []
    for class_id, points in labels:
        rebuilt = clip_points_to_bounds(rebuild_obb_from_polygon(transform_points(points)), new_w, new_h)
        if polygon_area(rebuilt) > 1e-6:
            transformed_labels.append((int(class_id), rebuilt))
    return transformed_image, transformed_labels


def letterbox_full_image(
    image: np.ndarray, annotations: Sequence[tuple[int, np.ndarray]], target_size: int, padding_value: int = 114
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], dict[str, float | int]]:
    """Resize a full image to a square and transform all OBB corners consistently."""
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}.")
    image = normalize_hwc_uint8(image)
    image_h, image_w = image.shape[:2]
    scale = min(target_size / image_w, target_size / image_h)
    new_w = max(1, round(image_w * scale))
    new_h = max(1, round(image_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    canvas = np.full((target_size, target_size, image.shape[2]), padding_value, dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    transformed_labels = []
    for class_id, points in annotations:
        transformed = points.astype(np.float32) * scale
        transformed += np.array([pad_left, pad_top], dtype=np.float32)
        transformed = clip_points_to_bounds(rebuild_obb_from_polygon(transformed), target_size, target_size)
        if polygon_area(transformed) > 1e-6:
            transformed_labels.append((int(class_id), transformed))
    metadata = {
        "scale": float(scale),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "source_width": int(image_w),
        "source_height": int(image_h),
        "output_size": int(target_size),
    }
    return canvas, transformed_labels, metadata


def contains_target(labels: Sequence[tuple[int, np.ndarray]], target_class_ids: set[int] | None) -> bool:
    """Return whether labels contain at least one requested class."""
    return target_class_ids is None or any(class_id in target_class_ids for class_id, _ in labels)


def generate_candidates(
    image_path: Path,
    split: str,
    image: np.ndarray,
    annotations: Sequence[tuple[int, np.ndarray]],
    patch_sizes: tuple[int, int],
    full_view_size: int,
    overlap: bool,
    keep_empty_patches: bool,
    min_iof: float,
    target_class_ids: set[int] | None = None,
    require_target: bool = False,
    transforms: Sequence[str] = (IDENTITY_TRANSFORM,),
) -> list[ViewCandidate]:
    """Generate cheap candidate records for all requested views."""
    image_h, image_w = image.shape[:2]
    candidates: list[ViewCandidate] = []
    source_key = str(image_path.resolve())

    def add_candidate(
        view_type: str,
        size: int,
        x0: int,
        y0: int,
        labels: Sequence[tuple[int, np.ndarray]],
        transform: str,
    ) -> None:
        has_target = contains_target(labels, target_class_ids)
        if require_target and not has_target:
            return
        if not labels and not keep_empty_patches:
            return
        anchor_class = next((int(c) for c, _ in labels if target_class_ids is None or c in target_class_ids), -1)
        candidate_key = f"{source_key}|{split}|{view_type}|{size}|{x0}|{y0}|{transform}"
        candidates.append(
            ViewCandidate(
                candidate_id=hashlib.sha1(candidate_key.encode("utf-8")).hexdigest()[:20],
                source_path=source_key,
                split=split,
                view_type=view_type,
                size=size,
                x0=x0,
                y0=y0,
                transform=transform,
                has_object=bool(labels),
                has_target=bool(has_target and target_class_ids is not None),
                object_count=len(labels),
                anchor_class_id=anchor_class,
            )
        )

    for size, view_type in zip(patch_sizes, ("patch256", "patch512")):
        for y0 in get_starts(image_h, size, overlap):
            for x0 in get_starts(image_w, size, overlap):
                x1, y1 = min(x0 + size, image_w), min(y0 + size, image_h)
                labels = collect_patch_labels(annotations, x0, y0, x1, y1, min_iof)
                for transform in transforms:
                    add_candidate(view_type, size, x0, y0, labels, transform)

    full_labels = list(annotations)
    for transform in transforms:
        add_candidate("full_scaled", full_view_size, 0, 0, full_labels, transform)
    return candidates


def materialize_candidate(
    candidate: ViewCandidate,
    image: np.ndarray,
    annotations: Sequence[tuple[int, np.ndarray]],
    min_iof: float,
    padding_value: int = 114,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], dict[str, float | int]]:
    """Materialize one candidate image and its pixel-coordinate OBB labels."""
    if candidate.view_type == "full_scaled":
        result, labels, metadata = letterbox_full_image(image, annotations, candidate.size, padding_value)
    else:
        image_h, image_w = image.shape[:2]
        x1, y1 = min(candidate.x0 + candidate.size, image_w), min(candidate.y0 + candidate.size, image_h)
        labels = collect_patch_labels(annotations, candidate.x0, candidate.y0, x1, y1, min_iof)
        result = image[candidate.y0 : y1, candidate.x0 : x1]
        padded = np.full((candidate.size, candidate.size, image.shape[2]), padding_value, dtype=np.uint8)
        padded[: result.shape[0], : result.shape[1]] = result
        result = padded
        metadata = {"scale": 1.0, "pad_left": 0, "pad_top": 0, "source_width": image_w, "source_height": image_h}
    result, labels = apply_transform(result, labels, candidate.transform)
    metadata["output_size"] = int(result.shape[0])
    return normalize_hwc_uint8(result), labels, metadata


def select_candidates(
    candidates: Sequence[ViewCandidate],
    ratios: ViewRatios,
    seed: int = 0,
    total_samples: int = 0,
) -> tuple[list[ViewCandidate], dict[str, int], dict[str, int]]:
    """Select a deterministic 70/20/10-style subset and return capacity data."""
    validate_view_ratios(ratios)
    pools = {
        view: sorted(
            (c for c in candidates if c.view_type == view),
            key=lambda c: stable_rank(c.candidate_id, seed),
        )
        for view in VIEW_TYPES
    }
    capacities = {view: len(pool) for view, pool in pools.items()}
    feasible = maximum_feasible_total(capacities, ratios)
    if total_samples > 0:
        if total_samples > feasible:
            raise ValueError(
                f"Requested {total_samples} samples but only {feasible} are feasible; capacities={capacities}."
            )
        total = total_samples
    else:
        total = feasible
    if total < 3:
        raise ValueError(f"Not enough candidates for balanced views: capacities={capacities}.")
    counts = allocate_view_counts(total, ratios)
    selected = [candidate for view in VIEW_TYPES for candidate in pools[view][: counts[view]]]
    return selected, counts, capacities


def write_obb_labels(path: Path, labels: Sequence[tuple[int, np.ndarray]], image_size: int) -> None:
    """Write normalized YOLO OBB labels."""
    lines = []
    for class_id, points in labels:
        normalized = points.astype(np.float32).copy()
        normalized[:, 0] /= image_size
        normalized[:, 1] /= image_size
        if np.any(normalized < -1e-6) or np.any(normalized > 1.000001):
            raise ValueError(f"Normalized OBB is out of range for {path}: {normalized.tolist()}")
        coords = " ".join(f"{float(value):.6f}" for value in normalized.reshape(-1))
        lines.append(f"{int(class_id)} {coords}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_candidates(
    selected: Sequence[ViewCandidate],
    output_root: Path,
    source_loader,
    min_iof: float,
    padding_value: int = 114,
    seed: int = 0,
) -> tuple[list[Path], list[dict[str, object]]]:
    """Materialize selected candidates and write a manifest."""
    image_paths: list[Path] = []
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(selected):
        # Do not retain all full-resolution 8-channel source images: the source
        # cache can otherwise become a second, hidden RAM dataset.
        image, annotations = source_loader(Path(candidate.source_path))
        materialized, labels, metadata = materialize_candidate(candidate, image, annotations, min_iof, padding_value)
        stem = f"{index:07d}__{candidate.view_type}__{candidate.candidate_id}"
        image_path = output_root / "images" / candidate.split / f"{stem}.tiff"
        label_path = output_root / "labels" / candidate.split / f"{stem}.txt"
        save_multichannel_tiff(image_path, materialized)
        write_obb_labels(label_path, labels, materialized.shape[0])
        image_paths.append(image_path)
        rows.append(
            {
                **asdict(candidate),
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "seed": int(seed),
                **metadata,
                "label_count": len(labels),
            }
        )
        del materialized, labels
    return image_paths, rows


def write_manifest(rows: Sequence[dict[str, object]], path: Path) -> Path:
    """Write a deterministic CSV manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty manifest: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def manifest_view_counts(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    """Count view types in manifest rows."""
    counts = {view: 0 for view in VIEW_TYPES}
    for row in rows:
        view = str(row["view_type"])
        if view not in counts:
            raise ValueError(f"Unknown view type in manifest: {view}")
        counts[view] += 1
    return counts


def validate_manifest_rows(
    rows: Sequence[dict[str, object]], ratios: ViewRatios, tolerance: float = 0.02
) -> dict[str, int]:
    """Validate non-empty manifest counts and approximate requested ratios."""
    counts = manifest_view_counts(rows)
    total = sum(counts.values())
    if total < 3 or any(counts[view] == 0 for view in VIEW_TYPES):
        raise ValueError(f"Manifest must contain all view types, got {counts}.")
    for view, ratio in ratios.as_dict().items():
        actual = counts[view] / total
        if abs(actual - ratio) > tolerance:
            raise ValueError(f"View ratio for {view} is {actual:.4f}, expected {ratio:.4f} ± {tolerance}.")
    return counts
