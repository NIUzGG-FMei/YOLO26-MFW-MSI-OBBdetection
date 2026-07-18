from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.custom_obb_prepare_and_train import (
    DEFAULT_CONFIG,
    DatasetSplitConfig,
    draw_obb_labels,
    get_obb_size_metrics,
    load_npy_image,
    load_yolo_obb_label_file,
    parse_raw_label_file,
    sanitize_annotation,
)
from examples.multichannel_preview_utils import (
    build_preview_bgr,
    parse_display_channels,
    validate_stretch_percentiles,
)
from ultralytics.utils.patches import imread

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "dataset_analysis"

# 预览图显示模式
# - "first3": 沿用旧逻辑，直接显示前 3 个通道
# - "manual": 使用 ANALYZE_DISPLAY_CHANNELS 手工指定 3 个通道，按 (R, G, B) 顺序解释
# - "rgb_like": 根据 ANALYZE_CHANNEL_WAVELENGTHS_NM 自动选取最接近可见光 RGB 的 3 个通道
# - "false_color": 根据 ANALYZE_CHANNEL_WAVELENGTHS_NM 自动选取近红外伪彩组合
ANALYZE_PREVIEW_MODE = "rgb_like"
ANALYZE_DISPLAY_CHANNELS = (4, 2, 1)
ANALYZE_CHANNEL_WAVELENGTHS_NM = (395.0, 474.285714, 553.571429, 632.857143, 712.142857, 791.428571, 870.714286, 950.0)
ANALYZE_PERCENTILE_STRETCH = (2.0, 98.0)


@dataclass(frozen=True)
class PreviewConfig:
    preview_mode: str
    display_channels: tuple[int, int, int]
    channel_wavelengths_nm: tuple[float, ...]
    stretch_low: float
    stretch_high: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one OBB split, export per-class size/count CSVs, and save annotated example images. "
            "Supports both raw NPY+raw-label datasets and prepared/augmented TIFF+YOLO-OBB datasets."
        )
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="prepared",
        choices=("raw", "prepared"),
        help="Dataset format. raw=npy+raw labels, prepared=tiff+yolo obb labels. 中文：数据集格式。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Dataset split alias. 中文：数据划分别名。",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/mnt/d/Vscode work_place/datasetObjectDetection/augmented_target_patch_dataset/images/train",
        # 默认None，搭配--split使用，默认分析raw数据集（原始的数据集）
        # 增强数据集路径 "/mnt/d/Vscode work_place/datasetObjectDetection/augmented_target_patch_dataset/images/train"
        help="Override image directory. 中文：覆盖默认图像目录。",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default="/mnt/d/Vscode work_place/datasetObjectDetection/augmented_target_patch_dataset/labels/train",
        # 默认None，搭配--split使用，默认分析raw数据集（原始的数据集）
        # 增强数据集路径 "/mnt/d/Vscode work_place/datasetObjectDetection/augmented_target_patch_dataset/labels/train"
        help="Override label directory. 中文：覆盖默认标签目录。",
    )
    parser.add_argument(
        "--include-difficult",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to keep difficult objects. Only used for raw datasets. 中文：是否保留 difficult 目标。",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="car,bus,van,awning-bike,truck,tricycle,bike,pedestrian",
        help="Comma-separated class names in dataset order. 中文：按数据集顺序排列的类别名称。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/d/Vscode work_place/datasetObjectDetection/keshihua",
        help="Output directory. Defaults to runs/dataset_analysis/<dataset_format>/<split>/. 中文：输出目录。",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=6,
        help="Maximum number of annotated example images to save. 中文：最多保存多少张标注预览图。",
    )
    parser.add_argument(
        "--preview-mode",
        type=str,
        default=ANALYZE_PREVIEW_MODE,
        choices=("first3", "manual", "rgb_like", "false_color"),
        help="Preview mode for multi-channel images. 中文：多通道图像的可视化模式。",
    )
    parser.add_argument(
        "--display-channels",
        type=str,
        default=",".join(str(idx) for idx in ANALYZE_DISPLAY_CHANNELS),
        help="Three channel indices in RGB order, e.g. 4,2,1. 中文：按 RGB 顺序指定三个显示通道。",
    )
    parser.add_argument(
        "--stretch-low",
        type=float,
        default=ANALYZE_PERCENTILE_STRETCH[0],
        help="Lower percentile for per-channel stretch. 中文：按通道拉伸的低百分位。",
    )
    parser.add_argument(
        "--stretch-high",
        type=float,
        default=ANALYZE_PERCENTILE_STRETCH[1],
        help="Upper percentile for per-channel stretch. 中文：按通道拉伸的高百分位。",
    )
    return parser.parse_args()


def resolve_split_config(split_name: str) -> DatasetSplitConfig:
    return DEFAULT_CONFIG.train if split_name == "train" else DEFAULT_CONFIG.val


def parse_class_names(text: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    if not names:
        raise ValueError("At least one class name is required.")
    return names


def build_preview_config(args: argparse.Namespace) -> PreviewConfig:
    display_channels = parse_display_channels(args.display_channels)
    stretch_low, stretch_high = validate_stretch_percentiles(args.stretch_low, args.stretch_high)
    return PreviewConfig(
        preview_mode=args.preview_mode,
        display_channels=display_channels,
        channel_wavelengths_nm=tuple(float(value) for value in ANALYZE_CHANNEL_WAVELENGTHS_NM),
        stretch_low=stretch_low,
        stretch_high=stretch_high,
    )


def resolve_default_dirs(dataset_format: str, split_name: str, split_cfg: DatasetSplitConfig) -> tuple[Path, Path]:
    if dataset_format == "raw":
        return split_cfg.image_dir, split_cfg.label_dir
    if dataset_format == "prepared":
        dataset_root = DEFAULT_CONFIG.prepared_dataset_dir
        return dataset_root / "images" / split_name, dataset_root / "labels" / split_name
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def resolve_dataset_args(
    args: argparse.Namespace,
) -> tuple[str, str, Path, Path, bool, tuple[str, ...], Path, PreviewConfig]:
    split_cfg = resolve_split_config(args.split)
    split_name = args.split
    dataset_format = args.dataset_format
    default_image_dir, default_label_dir = resolve_default_dirs(dataset_format, split_name, split_cfg)
    image_dir = Path(args.image_dir) if args.image_dir else default_image_dir
    label_dir = Path(args.label_dir) if args.label_dir else default_label_dir
    include_difficult = split_cfg.include_difficult if args.include_difficult is None else bool(args.include_difficult)
    class_names = parse_class_names(args.class_names)
    output_dir = Path(args.output_dir) if args.output_dir else (DEFAULT_OUTPUT_ROOT / dataset_format / split_name)
    preview_cfg = build_preview_config(args)
    return split_name, dataset_format, image_dir, label_dir, include_difficult, class_names, output_dir, preview_cfg


def normalize_image_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        max_value = float(image.max())
        min_value = float(image.min())
        if max_value <= 1.0 and min_value >= 0.0:
            image *= 255.0
        elif max_value > 255.0 or min_value < 0.0:
            denom = max(max_value - min_value, 1e-6)
            image = (image - min_value) * (255.0 / denom)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def load_tiff_image(path: Path) -> np.ndarray:
    ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
    if ok and pages:
        if pages[0].ndim == 2:
            return normalize_image_uint8(np.stack(pages, axis=2))
        return normalize_image_uint8(np.ascontiguousarray(pages[0]))
    image = imread(str(path), flags=cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read TIFF image: {path}")
    if image.ndim == 2:
        image = image[..., None]
    return normalize_image_uint8(image)


def get_image_paths(image_dir: Path, dataset_format: str) -> list[Path]:
    if dataset_format == "raw":
        return sorted(image_dir.glob("*.npy"))
    if dataset_format == "prepared":
        return sorted(list(image_dir.glob("*.tiff")) + list(image_dir.glob("*.tif")))
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def load_dataset_image(image_path: Path, dataset_format: str) -> np.ndarray:
    if dataset_format == "raw":
        return load_npy_image(image_path)
    if dataset_format == "prepared":
        return load_tiff_image(image_path)
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def load_dataset_labels(
    image_path: Path,
    label_dir: Path,
    class_names: tuple[str, ...],
    include_difficult: bool,
    dataset_format: str,
    image_h: int,
    image_w: int,
) -> list[tuple[int, np.ndarray]]:
    label_path = label_dir / f"{image_path.stem}.txt"
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    labels: list[tuple[int, np.ndarray]] = []
    if dataset_format == "raw":
        raw_annotations = parse_raw_label_file(label_path, class_to_id, include_difficult)
        for class_id, points in raw_annotations:
            sanitized = sanitize_annotation(points, image_w, image_h)
            if sanitized is not None:
                labels.append((class_id, sanitized))
        return labels
    if dataset_format == "prepared":
        yolo_labels = load_yolo_obb_label_file(label_path, image_h, image_w)
        for class_id, points in yolo_labels:
            if not 0 <= class_id < len(class_names):
                continue
            sanitized = sanitize_annotation(points, image_w, image_h)
            if sanitized is not None:
                labels.append((class_id, sanitized))
        return labels
    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def build_preview_image(image: np.ndarray, preview_cfg: PreviewConfig) -> np.ndarray:
    preview_bgr, _ = build_preview_bgr(
        image=image,
        preview_mode=preview_cfg.preview_mode,
        display_channels=preview_cfg.display_channels,
        stretch_low=preview_cfg.stretch_low,
        stretch_high=preview_cfg.stretch_high,
        channel_wavelengths_nm=preview_cfg.channel_wavelengths_nm,
    )
    return preview_bgr


def collect_split_statistics(
    image_dir: Path,
    label_dir: Path,
    class_names: tuple[str, ...],
    include_difficult: bool,
    dataset_format: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    per_class_sizes = [{"long_side": [], "short_side": [], "area": []} for _ in class_names]
    per_class_counts = [0 for _ in class_names]
    per_image_records: list[dict[str, object]] = []
    image_paths = get_image_paths(image_dir, dataset_format)

    for image_path in image_paths:
        image = load_dataset_image(image_path, dataset_format)
        image_h, image_w = image.shape[:2]
        labels = load_dataset_labels(
            image_path=image_path,
            label_dir=label_dir,
            class_names=class_names,
            include_difficult=include_difficult,
            dataset_format=dataset_format,
            image_h=image_h,
            image_w=image_w,
        )
        present_class_ids: set[int] = set()
        for class_id, points in labels:
            present_class_ids.add(class_id)
            long_side, short_side, area = get_obb_size_metrics(points)
            per_class_counts[class_id] += 1
            per_class_sizes[class_id]["long_side"].append(float(long_side))
            per_class_sizes[class_id]["short_side"].append(float(short_side))
            per_class_sizes[class_id]["area"].append(float(area))

        per_image_records.append(
            {
                "image_path": image_path,
                "label_path": label_dir / f"{image_path.stem}.txt",
                "image_h": int(image_h),
                "image_w": int(image_w),
                "channels": int(image.shape[2]) if image.ndim == 3 else 1,
                "labels": labels,
                "present_class_ids": present_class_ids,
            }
        )

    size_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    total_instances = sum(per_class_counts)
    for class_id, class_name in enumerate(class_names):
        long_values = per_class_sizes[class_id]["long_side"]
        short_values = per_class_sizes[class_id]["short_side"]
        area_values = per_class_sizes[class_id]["area"]
        instance_count = per_class_counts[class_id]
        image_count_with_class = sum(1 for record in per_image_records if class_id in record["present_class_ids"])
        size_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "instance_count": instance_count,
                "avg_long_side_px": statistics_or_zero(long_values, "mean"),
                "median_long_side_px": statistics_or_zero(long_values, "median"),
                "min_long_side_px": min(long_values) if long_values else 0.0,
                "max_long_side_px": max(long_values) if long_values else 0.0,
                "avg_short_side_px": statistics_or_zero(short_values, "mean"),
                "median_short_side_px": statistics_or_zero(short_values, "median"),
                "min_short_side_px": min(short_values) if short_values else 0.0,
                "max_short_side_px": max(short_values) if short_values else 0.0,
                "avg_area_px2": statistics_or_zero(area_values, "mean"),
                "median_area_px2": statistics_or_zero(area_values, "median"),
                "min_area_px2": min(area_values) if area_values else 0.0,
                "max_area_px2": max(area_values) if area_values else 0.0,
            }
        )
        count_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "instance_count": instance_count,
                "instance_fraction": (instance_count / total_instances) if total_instances else 0.0,
                "image_count_with_class": image_count_with_class,
                "image_fraction": (image_count_with_class / len(per_image_records)) if per_image_records else 0.0,
            }
        )

    return size_rows, count_rows, per_image_records


def statistics_or_zero(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "mean":
        return float(sum(values) / len(values))
    if mode == "median":
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2 == 1:
            return float(sorted_values[mid])
        return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)
    raise ValueError(f"Unsupported mode: {mode}")


def write_size_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "instance_count",
                "avg_long_side_px",
                "median_long_side_px",
                "min_long_side_px",
                "max_long_side_px",
                "avg_short_side_px",
                "median_short_side_px",
                "min_short_side_px",
                "max_short_side_px",
                "avg_area_px2",
                "median_area_px2",
                "min_area_px2",
                "max_area_px2",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(format_numeric_row(row))
    return output_path


def write_count_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "instance_count",
                "instance_fraction",
                "image_count_with_class",
                "image_fraction",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(format_numeric_row(row))
    return output_path


def format_numeric_row(row: dict[str, object]) -> dict[str, object]:
    formatted: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.6f}"
        else:
            formatted[key] = value
    return formatted


def select_visualization_records(
    records: list[dict[str, object]], class_names: tuple[str, ...], max_visualizations: int
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    uncovered = {
        class_id for class_id in range(len(class_names)) if any(class_id in r["present_class_ids"] for r in records)
    }
    candidates = [record for record in records if record["labels"]]

    while uncovered and candidates and len(selected) < max_visualizations:
        best = max(
            candidates,
            key=lambda record: (
                len(record["present_class_ids"] & uncovered),
                len(record["present_class_ids"]),
                len(record["labels"]),
            ),
        )
        if not (best["present_class_ids"] & uncovered):
            break
        selected.append(best)
        uncovered -= best["present_class_ids"]
        candidates.remove(best)

    for record in candidates:
        if len(selected) >= max_visualizations:
            break
        if record not in selected:
            selected.append(record)

    return selected


def save_visualizations(
    records: list[dict[str, object]],
    class_names: tuple[str, ...],
    output_dir: Path,
    dataset_format: str,
    preview_cfg: PreviewConfig,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for index, record in enumerate(records, start=1):
        image = load_dataset_image(Path(record["image_path"]), dataset_format)
        preview = build_preview_image(image, preview_cfg)
        annotated = draw_obb_labels(preview, record["labels"], class_names)
        cv2.putText(
            annotated,
            record["image_path"].stem,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        save_path = output_dir / f"{index:02d}_{record['image_path'].stem}.jpg"
        ok = cv2.imwrite(str(save_path), annotated)
        if not ok:
            raise RuntimeError(f"Failed to write visualization: {save_path}")
        saved_paths.append(str(save_path))
    return saved_paths


def build_summary(
    split_name: str,
    dataset_format: str,
    image_dir: Path,
    label_dir: Path,
    include_difficult: bool,
    class_names: tuple[str, ...],
    size_csv_path: Path,
    count_csv_path: Path,
    visualization_paths: list[str],
    per_image_records: list[dict[str, object]],
    preview_cfg: PreviewConfig,
) -> dict[str, object]:
    present_class_ids = sorted({class_id for record in per_image_records for class_id in record["present_class_ids"]})
    return {
        "split": split_name,
        "dataset_format": dataset_format,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "include_difficult": include_difficult,
        "class_names": list(class_names),
        "source_image_count": len(per_image_records),
        "labeled_image_count": sum(1 for record in per_image_records if record["labels"]),
        "covered_class_ids": present_class_ids,
        "covered_class_names": [class_names[class_id] for class_id in present_class_ids],
        "preview_mode": preview_cfg.preview_mode,
        "preview_display_channels": list(preview_cfg.display_channels),
        "preview_channel_wavelengths_nm": list(preview_cfg.channel_wavelengths_nm),
        "preview_percentile_stretch": [preview_cfg.stretch_low, preview_cfg.stretch_high],
        "size_csv": str(size_csv_path),
        "count_csv": str(count_csv_path),
        "visualization_paths": visualization_paths,
    }


def main() -> None:
    args = parse_args()
    split_name, dataset_format, image_dir, label_dir, include_difficult, class_names, output_dir, preview_cfg = (
        resolve_dataset_args(args)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    size_rows, count_rows, per_image_records = collect_split_statistics(
        image_dir=image_dir,
        label_dir=label_dir,
        class_names=class_names,
        include_difficult=include_difficult,
        dataset_format=dataset_format,
    )

    size_csv_path = write_size_csv(size_rows, output_dir / f"{split_name}_class_size_stats.csv")
    count_csv_path = write_count_csv(count_rows, output_dir / f"{split_name}_class_instance_counts.csv")
    selected_records = select_visualization_records(per_image_records, class_names, args.max_visualizations)
    visualization_paths = save_visualizations(
        selected_records,
        class_names,
        output_dir / "visualizations",
        dataset_format,
        preview_cfg,
    )

    summary = build_summary(
        split_name=split_name,
        dataset_format=dataset_format,
        image_dir=image_dir,
        label_dir=label_dir,
        include_difficult=include_difficult,
        class_names=class_names,
        size_csv_path=size_csv_path,
        count_csv_path=count_csv_path,
        visualization_paths=visualization_paths,
        per_image_records=per_image_records,
        preview_cfg=preview_cfg,
    )
    summary_path = output_dir / f"{split_name}_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
