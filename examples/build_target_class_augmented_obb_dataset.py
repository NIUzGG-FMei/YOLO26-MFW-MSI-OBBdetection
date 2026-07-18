"""Build the optional target-class OBB dataset.

使用手册
========

本脚本保留 ``legacy`` 和 ``balanced_multiscale`` 两条互相隔离的路径。

``legacy``（默认）保持原来的目标中心 patch、平移、旋转/翻转增强方式，适合
复现实验和回退；``balanced_multiscale`` 则从同一批原始 NPY/TXT 数据生成
只包含目标类别的可审计训练视图：70% ``patch256``、20% ``patch512``、10%
``full_scaled``。两种 patch 使用重叠滑窗，候选池保留空 patch 选项，最终
由 manifest 配额决定实际写出的样本。balanced 模式的候选要求至少包含一个
目标类别，因此额外数据集不会用无关背景淹没定向增强；``keep_empty_patches``
仍然开启，确保处理策略与主训练数据一致。

推荐先生成，再检查 ``train_manifest.csv``、``dataset_validation.json`` 和
``augmentation_summary.json``：

    python examples/build_target_class_augmented_obb_dataset.py \
        --preprocess-profile balanced_multiscale \
        --output-dir /path/to/augmented_target_patch_dataset \
        --target-classes bus \
        --output-mode reset

确认三类视图计数、8 通道 uint8 TIFF、正方形输出和 OBB 坐标均通过后，再在
``custom_obb_prepare_and_train.py`` 中使用 ``--use-augmented-dataset``。主训练
脚本会再次读取并验证这个 manifest；验证失败时不会开始训练。原始目录、主
prepared 数据集和额外数据集始终分开，``--output-mode reset`` 可重建额外目录。

直接点击 IDE 运行时，legacy 默认写入原有增强目录；balanced_multiscale 会自动
切换到带有 ``_balanced_multiscale`` 后缀的独立目录，并在未指定输出模式时自动
使用 ``reset``。命令行显式传入 ``--output-dir`` 时，以显式路径为准。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.custom_obb_prepare_and_train import (  # noqa: E402
    DEFAULT_CONFIG,
    PATCH_IOF_THRESHOLD,
    DatasetSplitConfig,
    clip_points_to_bounds,
    load_npy_image,
    parse_raw_label_file,
    polygon_area,
    project_annotation_to_patch,
    rebuild_obb_from_polygon,
    sanitize_annotation,
    save_multichannel_tiff,
    validate_balanced_dataset_files,
    write_label_file,
)
from examples.multiscale_dataset_utils import (  # noqa: E402
    IDENTITY_TRANSFORM,
    ViewRatios,
    generate_candidates,
    maximum_feasible_total,
    select_candidates,
    validate_manifest_rows,
    validate_view_ratios,
    write_manifest,
    write_selected_candidates,
)

# =========================
# IDE Quick Config
# 顶部快速配置区
# 直接修改这里的几个值，然后点击右上角运行即可。
# =========================

# IDE_SOURCE_SPLITS:
# - ("train",): 仅从训练原始集生成补充 patch
# - ("train", "val"): 同时从 train/val 生成到独立补充数据集目录
# 中文：通常建议只处理 train，避免后续误把补充样本接到验证集里。
IDE_SOURCE_SPLITS = ("train",)

# IDE_TARGET_CLASSES:
# 中文：需要定向补充的少样本类别，名称必须来自 DEFAULT_CONFIG.class_names。
IDE_TARGET_CLASSES = ("bus", "van", "truck", "tricycle", "awning-bike")

# IDE_LEGACY_AUGMENTED_DATASET_DIR:
# 中文：legacy 模式使用的原有额外增强数据集目录。
IDE_LEGACY_AUGMENTED_DATASET_DIR = Path(
    "/mnt/d/Vscode work_place/datasetObjectDetection/augmented_target_patch_dataset"
)

# IDE_BALANCED_AUGMENTED_DATASET_DIR:
# 中文：balanced_multiscale 模式专用额外增强数据集目录。
IDE_BALANCED_AUGMENTED_DATASET_DIR = Path(
    "/mnt/d/Vscode work_place/datasetObjectDetection/augmented_target_patch_dataset_balanced_multiscale"
)

# IDE_AUGMENTED_DATASET_DIR:
# 中文：legacy 兼容别名；legacy 模式仍使用它作为额外增强目录。
IDE_AUGMENTED_DATASET_DIR = IDE_LEGACY_AUGMENTED_DATASET_DIR

# IDE_RANDOM_SEED:
# 中文：控制平移候选与几何增强候选的采样顺序，便于复现。
IDE_RANDOM_SEED = 0

# IDE_OUTPUT_MODE:
# - "reset": 每次运行前清空输出目录，再生成新的增强结果
# - "append": 保留已有 images/labels，仅对新样本增量写入；manifest/summary 仍按本次运行覆盖
# 中文：legacy 默认 append 以保持旧行为；balanced_multiscale 未显式指定时自动使用 reset。
IDE_OUTPUT_MODE = "append"

# IDE_INCLUDE_CENTER_CROP:
# 中文：是否保留每个目标的基础“目标中心 patch”。
IDE_INCLUDE_CENTER_CROP = True

# IDE_TRANSLATION_VARIANTS_PER_GT:
# 中文：每个目标额外生成多少个“小范围平移”版本。
IDE_TRANSLATION_VARIANTS_PER_GT = 2

# IDE_TRANSLATION_MAX_OFFSET:
# 中文：平移增强允许的最大偏移，格式为 (dy, dx)。
IDE_TRANSLATION_MAX_OFFSET = (32, 32)

# IDE_GEOMETRIC_VARIANTS_PER_GT:
# 中文：每个目标额外生成多少个“翻转/旋转”版本。
IDE_GEOMETRIC_VARIANTS_PER_GT = 2

# IDE_TRANSLATION_GEOMETRIC_VARIANTS_PER_GT:
# 中文：每个目标额外生成多少个“合法平移后再旋转/翻转”的组合版本。
IDE_TRANSLATION_GEOMETRIC_VARIANTS_PER_GT = 2

# IDE_GEOMETRIC_TRANSFORMS:
# 可选值：rot90, rot180, rot270, flip_h, flip_v
# 中文：几何增强候选集合。脚本会从中采样不重复的若干种变换。
IDE_GEOMETRIC_TRANSFORMS = ("rot90", "rot180", "rot270", "flip_h", "flip_v")

# IDE_PREPROCESS_PROFILE:
# - "legacy": 保留当前目标中心/平移/几何增强流程，保证历史实验可复现。
# - "balanced_multiscale": 生成 70% patch256、20% patch512、10% full_scaled。
# 中文：默认跟随训练脚本的 profile；如需独立实验，也可以在本文件单独覆盖。
IDE_PREPROCESS_PROFILE = DEFAULT_CONFIG.preprocess_profile


def resolve_profile_augmented_dataset_dir(profile: str) -> Path:
    """Return the IDE-default augmented directory for one preprocessing profile."""
    if profile == "balanced_multiscale":
        return IDE_BALANCED_AUGMENTED_DATASET_DIR
    return IDE_AUGMENTED_DATASET_DIR


# IDE_VIEW_RATIOS:
# 中文：balanced_multiscale 最终增强 train manifest 的视图比例，顺序固定为 256、512、整图。
IDE_VIEW_RATIOS = (0.70, 0.20, 0.10)

# IDE_MULTISCALE_PATCH_SIZES:
# 中文：balanced_multiscale 的两类重叠滑窗尺寸，必须是 256 和 512。
IDE_MULTISCALE_PATCH_SIZES = (256, 512)

# IDE_FULL_VIEW_SIZE:
# 中文：整图 letterbox 输出的方形尺寸；主训练默认将其设为 256。
IDE_FULL_VIEW_SIZE = 256

# IDE_TOTAL_SAMPLES:
# 中文：balanced 增强数据集样本总数上限；0 表示按各视图候选容量自动取最大可行值。
IDE_TOTAL_SAMPLES = 0

# IDE_STRICT_VIEW_RATIO:
# 中文：是否严格检查最终 manifest 的 70/20/10 比例。
IDE_STRICT_VIEW_RATIO = True

# IDE_BALANCED_TRANSFORMS:
# 中文：balanced 候选使用的几何变换；identity 始终自动加入。
IDE_BALANCED_TRANSFORMS = IDE_GEOMETRIC_TRANSFORMS


@dataclass(frozen=True)
class TargetPatchAugmentConfig:
    """Collect settings for the extra target-class patch dataset.

    中文：集中管理定向补充 patch 数据集的生成参数。
    """

    source_splits: tuple[str, ...]
    output_dir: Path
    target_class_names: tuple[str, ...]
    patch_size: tuple[int, int]
    random_seed: int
    output_mode: str
    include_center_crop: bool
    translation_variants_per_gt: int
    translation_max_offset: tuple[int, int]
    geometric_variants_per_gt: int
    translation_geometric_variants_per_gt: int
    geometric_transforms: tuple[str, ...]
    # preprocess_profile: legacy preserves the old target-centered variants; balanced adds 70/20/10 views.
    # 中文：数据处理模式。
    preprocess_profile: str = "legacy"
    # view_ratios: final balanced manifest ratios in patch256, patch512, full_scaled order.
    # 中文：三类视图比例。
    view_ratios: tuple[float, float, float] = IDE_VIEW_RATIOS
    # multiscale_patch_sizes: candidate crop sizes for the two patch views.
    # 中文：两类 patch 候选尺寸。
    multiscale_patch_sizes: tuple[int, int] = IDE_MULTISCALE_PATCH_SIZES
    # full_view_size: square output size after full-image letterbox.
    # 中文：整图缩放输出尺寸。
    full_view_size: int = IDE_FULL_VIEW_SIZE
    # total_samples: balanced sample cap; zero selects the largest feasible total.
    # 中文：balanced 样本总数上限。
    total_samples: int = IDE_TOTAL_SAMPLES
    # strict_view_ratio: reject a manifest whose integer counts miss the target ratio.
    # 中文：是否严格校验比例。
    strict_view_ratio: bool = IDE_STRICT_VIEW_RATIO
    # overlap: use half-window sliding stride in balanced mode.
    # 中文：是否使用重叠滑窗。
    overlap: bool = True
    # keep_empty_patches: retain empty candidates before target filtering and quota selection.
    # 中文：是否保留空 patch 候选。
    keep_empty_patches: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an extra target-class OBB patch dataset from the raw source images. "
            "The output stays isolated from the original dataset and prepared_Dataset."
        )
    )
    parser.add_argument(
        "--preprocess-profile",
        type=str,
        default=IDE_PREPROCESS_PROFILE,
        choices=("legacy", "balanced_multiscale"),
        help="Dataset pipeline. 中文：legacy 保持旧增强流程，balanced_multiscale 启用 70/20/10 视图。",
    )
    parser.add_argument(
        "--view-ratios",
        type=str,
        default=",".join(f"{value:.6f}" for value in IDE_VIEW_RATIOS),
        help="Ratios for patch256,patch512,full_scaled. 中文：balanced 模式三类视图比例，例如 0.7,0.2,0.1。",
    )
    parser.add_argument(
        "--multiscale-patch-sizes",
        type=str,
        default=",".join(str(value) for value in IDE_MULTISCALE_PATCH_SIZES),
        help="Two crop sizes. 中文：balanced 模式 patch 尺寸，必须为 256,512。",
    )
    parser.add_argument(
        "--full-view-size",
        type=int,
        default=IDE_FULL_VIEW_SIZE,
        help="Square full_scaled output size. 中文：整图 letterbox 后的方形尺寸。",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=IDE_TOTAL_SAMPLES,
        help="Balanced sample cap; 0 means maximum feasible. 中文：balanced 总样本数上限，0 表示自动。",
    )
    parser.add_argument(
        "--strict-view-ratio",
        action=argparse.BooleanOptionalAction,
        default=IDE_STRICT_VIEW_RATIO,
        help="Fail on invalid view ratios. 中文：是否严格校验最终 70/20/10 比例。",
    )
    parser.add_argument(
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use overlapping windows. 中文：是否使用重叠滑窗；balanced 模式强制为 True。",
    )
    parser.add_argument(
        "--keep-empty-patches",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep empty candidates. 中文：是否保留空 patch 候选；balanced 模式强制为 True。",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=",".join(IDE_SOURCE_SPLITS),
        help="Comma-separated source splits. Choices: train,val. 中文：需要处理的源数据划分。",
    )
    parser.add_argument(
        "--target-classes",
        type=str,
        default=",".join(IDE_TARGET_CLASSES),
        help="Comma-separated target class names. 中文：需要定向补充的类别名称。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory of the extra augmented dataset. If omitted, the IDE-default directory for "
            "--preprocess-profile is selected automatically. 中文：不指定时按模式自动选择输出目录。"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=IDE_RANDOM_SEED,
        help="Random seed for deterministic candidate sampling. 中文：随机种子。",
    )
    parser.add_argument(
        "--include-center-crop",
        action=argparse.BooleanOptionalAction,
        default=IDE_INCLUDE_CENTER_CROP,
        help="Whether to save the base center crop for each GT. 中文：是否保留基础中心切片。",
    )
    parser.add_argument(
        "--translation-variants-per-gt",
        type=int,
        default=IDE_TRANSLATION_VARIANTS_PER_GT,
        help="Number of translated variants generated for each target GT. 中文：每个GT的平移增强数量。",
    )
    parser.add_argument(
        "--translation-max-offset-y",
        type=int,
        default=IDE_TRANSLATION_MAX_OFFSET[0],
        help="Max absolute Y offset for translation variants. 中文：平移增强最大纵向偏移。",
    )
    parser.add_argument(
        "--translation-max-offset-x",
        type=int,
        default=IDE_TRANSLATION_MAX_OFFSET[1],
        help="Max absolute X offset for translation variants. 中文：平移增强最大横向偏移。",
    )
    parser.add_argument(
        "--geometric-variants-per-gt",
        type=int,
        default=IDE_GEOMETRIC_VARIANTS_PER_GT,
        help="Number of rotation/flip variants generated for each target GT. 中文：每个GT的几何增强数量。",
    )
    parser.add_argument(
        "--translation-geometric-variants-per-gt",
        type=int,
        default=IDE_TRANSLATION_GEOMETRIC_VARIANTS_PER_GT,
        help="Number of translated-plus-geometric variants generated for each target GT. 中文：每个GT的平移+几何增强数量。",
    )
    parser.add_argument(
        "--geometric-transforms",
        type=str,
        default=",".join(IDE_GEOMETRIC_TRANSFORMS),
        help=("Comma-separated transforms chosen from rot90,rot180,rot270,flip_h,flip_v. 中文：几何增强候选集合。"),
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default=None,
        choices=("reset", "append"),
        help=(
            "Output handling mode. reset=recreate output dir; append=keep existing images/labels and only add new ones. "
            "If omitted, legacy uses append and balanced uses reset. "
            "中文：输出处理模式。"
        ),
    )
    return parser.parse_args()


def parse_name_tuple(text: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in text.split(",") if item.strip())
    if not names:
        raise ValueError("At least one name must be provided.")
    return names


def validate_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    valid = {"train", "val"}
    if any(split not in valid for split in splits):
        raise ValueError(f"Unsupported split list: {splits}. Expected values from {sorted(valid)}.")
    return splits


def validate_target_classes(target_class_names: tuple[str, ...]) -> tuple[str, ...]:
    valid = set(DEFAULT_CONFIG.class_names)
    invalid = [name for name in target_class_names if name not in valid]
    if invalid:
        raise ValueError(
            f"Unsupported target classes: {invalid}. They must come from "
            f"DEFAULT_CONFIG.class_names={DEFAULT_CONFIG.class_names}."
        )
    return target_class_names


def validate_geometric_transforms(transforms: tuple[str, ...]) -> tuple[str, ...]:
    valid = {"rot90", "rot180", "rot270", "flip_h", "flip_v"}
    invalid = [name for name in transforms if name not in valid]
    if invalid:
        raise ValueError(f"Unsupported geometric transforms: {invalid}. Expected values from {sorted(valid)}.")
    return transforms


def validate_output_mode(output_mode: str) -> str:
    valid = {"reset", "append"}
    if output_mode not in valid:
        raise ValueError(f"Unsupported output mode: {output_mode}. Expected one of {sorted(valid)}.")
    return output_mode


def parse_numeric_tuple(text: str, length: int, cast, name: str) -> tuple:
    """Parse a comma-separated numeric tuple with an exact length."""
    try:
        values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric comma-separated values, got {text!r}.") from exc
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values, got {values}.")
    return values


def build_config(args: argparse.Namespace) -> TargetPatchAugmentConfig:
    profile = str(args.preprocess_profile)
    output_dir = Path(args.output_dir) if args.output_dir else resolve_profile_augmented_dataset_dir(profile)
    requested_output_mode = args.output_mode or ("reset" if profile == "balanced_multiscale" else IDE_OUTPUT_MODE)
    ratios = parse_numeric_tuple(args.view_ratios, 3, float, "view_ratios")
    validate_view_ratios(ViewRatios(*ratios))
    multiscale_sizes = parse_numeric_tuple(args.multiscale_patch_sizes, 2, int, "multiscale_patch_sizes")
    if profile == "balanced_multiscale" and multiscale_sizes != (256, 512):
        raise ValueError(f"balanced_multiscale requires patch sizes 256,512, got {multiscale_sizes}.")
    if any(value <= 0 for value in multiscale_sizes) or int(args.full_view_size) <= 0:
        raise ValueError("multiscale patch sizes and full_view_size must be positive.")
    if int(args.total_samples) < 0:
        raise ValueError(f"total_samples must be >= 0, got {args.total_samples}.")
    output_mode = validate_output_mode(str(requested_output_mode))
    if profile == "balanced_multiscale":
        # ``None`` selects the balanced profile defaults: overlapping windows
        # and an enabled empty-candidate pool.
        overlap = True if args.overlap is None else bool(args.overlap)
        keep_empty = True if args.keep_empty_patches is None else bool(args.keep_empty_patches)
        if not overlap:
            raise ValueError("balanced_multiscale requires overlap=True; remove --no-overlap.")
        if not keep_empty:
            raise ValueError("balanced_multiscale requires keep_empty_patches=True; remove --no-keep-empty-patches.")
        if output_mode != "reset":
            raise ValueError(
                "balanced_multiscale requires --output-mode reset so stale legacy files cannot change "
                "the 70/20/10 ratio."
            )
    else:
        overlap = False if args.overlap is None else bool(args.overlap)
        keep_empty = True if args.keep_empty_patches is None else bool(args.keep_empty_patches)
        overlap = True
        keep_empty = True
    return TargetPatchAugmentConfig(
        source_splits=validate_splits(parse_name_tuple(args.splits)),
        output_dir=output_dir,
        target_class_names=validate_target_classes(parse_name_tuple(args.target_classes)),
        patch_size=DEFAULT_CONFIG.patch_size,
        random_seed=int(args.seed),
        output_mode=output_mode,
        include_center_crop=bool(args.include_center_crop),
        translation_variants_per_gt=max(int(args.translation_variants_per_gt), 0),
        translation_max_offset=(
            max(int(args.translation_max_offset_y), 0),
            max(int(args.translation_max_offset_x), 0),
        ),
        geometric_variants_per_gt=max(int(args.geometric_variants_per_gt), 0),
        translation_geometric_variants_per_gt=max(int(args.translation_geometric_variants_per_gt), 0),
        geometric_transforms=validate_geometric_transforms(parse_name_tuple(args.geometric_transforms)),
        preprocess_profile=profile,
        view_ratios=ratios,
        multiscale_patch_sizes=multiscale_sizes,
        full_view_size=int(args.full_view_size),
        total_samples=int(args.total_samples),
        strict_view_ratio=bool(args.strict_view_ratio),
        overlap=overlap,
        keep_empty_patches=keep_empty,
    )


def resolve_split_config(split_name: str) -> DatasetSplitConfig:
    return DEFAULT_CONFIG.train if split_name == "train" else DEFAULT_CONFIG.val


def sanitize_raw_annotations(
    image_path: Path, split_cfg: DatasetSplitConfig, class_to_id: dict[str, int]
) -> list[tuple[int, np.ndarray]]:
    image = load_npy_image(image_path)
    image_h, image_w = image.shape[:2]
    raw_annotations = parse_raw_label_file(
        split_cfg.label_dir / f"{image_path.stem}.txt",
        class_to_id=class_to_id,
        include_difficult=split_cfg.include_difficult,
    )
    annotations: list[tuple[int, np.ndarray]] = []
    for class_id, points in raw_annotations:
        sanitized = sanitize_annotation(points, image_w, image_h)
        if sanitized is not None:
            annotations.append((class_id, sanitized))
    return annotations


def compute_patch_extent(image_h: int, image_w: int, patch_size: tuple[int, int]) -> tuple[int, int]:
    patch_h, patch_w = patch_size
    return min(int(patch_h), int(image_h)), min(int(patch_w), int(image_w))


def compute_containment_interval(length: int, patch: int, coord_min: float, coord_max: float) -> tuple[int, int] | None:
    if patch <= 0 or length <= 0:
        return None
    if coord_max - coord_min > patch + 1e-6:
        return None
    max_start = max(length - patch, 0)
    low = max(0, math.ceil(coord_max - patch))
    high = min(max_start, math.floor(coord_min))
    return (low, high) if low <= high else None


def clamp_int(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def compute_center_crop_window(
    points: np.ndarray, image_h: int, image_w: int, patch_size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    patch_h, patch_w = compute_patch_extent(image_h, image_w, patch_size)
    min_x = float(points[:, 0].min())
    max_x = float(points[:, 0].max())
    min_y = float(points[:, 1].min())
    max_y = float(points[:, 1].max())

    x_interval = compute_containment_interval(image_w, patch_w, min_x, max_x)
    y_interval = compute_containment_interval(image_h, patch_h, min_y, max_y)
    if x_interval is None or y_interval is None:
        return None

    center_x = float(points[:, 0].mean())
    center_y = float(points[:, 1].mean())
    centered_x0 = round(center_x - patch_w / 2.0)
    centered_y0 = round(center_y - patch_h / 2.0)
    x0 = clamp_int(centered_x0, x_interval[0], x_interval[1])
    y0 = clamp_int(centered_y0, y_interval[0], y_interval[1])
    return x0, y0, x0 + patch_w, y0 + patch_h


def sample_translation_windows(
    points: np.ndarray,
    image_h: int,
    image_w: int,
    patch_size: tuple[int, int],
    center_x0: int,
    center_y0: int,
    variant_count: int,
    max_offset: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    if variant_count <= 0:
        return []
    candidates = build_translation_window_candidates(
        points=points,
        image_h=image_h,
        image_w=image_w,
        patch_size=patch_size,
        center_x0=center_x0,
        center_y0=center_y0,
        max_offset=max_offset,
    )
    rng.shuffle(candidates)
    return candidates[:variant_count]


def build_translation_window_candidates(
    points: np.ndarray,
    image_h: int,
    image_w: int,
    patch_size: tuple[int, int],
    center_x0: int,
    center_y0: int,
    max_offset: tuple[int, int],
) -> list[tuple[int, int]]:
    patch_h, patch_w = compute_patch_extent(image_h, image_w, patch_size)
    min_x = float(points[:, 0].min())
    max_x = float(points[:, 0].max())
    min_y = float(points[:, 1].min())
    max_y = float(points[:, 1].max())

    x_interval = compute_containment_interval(image_w, patch_w, min_x, max_x)
    y_interval = compute_containment_interval(image_h, patch_h, min_y, max_y)
    if x_interval is None or y_interval is None:
        return []

    max_offset_y, max_offset_x = max_offset
    x_low = max(x_interval[0], center_x0 - max_offset_x)
    x_high = min(x_interval[1], center_x0 + max_offset_x)
    y_low = max(y_interval[0], center_y0 - max_offset_y)
    y_high = min(y_interval[1], center_y0 + max_offset_y)
    if x_low > x_high or y_low > y_high:
        return []

    return [
        (x0, y0)
        for y0 in range(y_low, y_high + 1)
        for x0 in range(x_low, x_high + 1)
        if not (x0 == center_x0 and y0 == center_y0)
    ]


def collect_patch_labels(
    annotations: list[tuple[int, np.ndarray]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> list[tuple[int, np.ndarray]]:
    kept_labels: list[tuple[int, np.ndarray]] = []
    for class_id, points in annotations:
        projected, _ = project_annotation_to_patch(points, x0, y0, x1, y1)
        if projected is None:
            continue
        kept_labels.append((class_id, projected))
    return kept_labels


def contains_target_class(labels: list[tuple[int, np.ndarray]], target_class_ids: set[int]) -> bool:
    return any(class_id in target_class_ids for class_id, _ in labels)


def select_geometric_ops(candidates: tuple[str, ...], variant_count: int, rng: random.Random) -> list[str]:
    if variant_count <= 0 or not candidates:
        return []
    pool = list(dict.fromkeys(candidates))
    rng.shuffle(pool)
    return pool[:variant_count]


def select_translation_geometric_pairs(
    translation_windows: list[tuple[int, int]],
    geometric_transforms: tuple[str, ...],
    variant_count: int,
    rng: random.Random,
) -> list[tuple[int, int, str]]:
    if variant_count <= 0 or not translation_windows or not geometric_transforms:
        return []
    unique_transforms = list(dict.fromkeys(geometric_transforms))
    candidates = [(x0, y0, op_name) for (x0, y0) in translation_windows for op_name in unique_transforms]
    rng.shuffle(candidates)
    return candidates[:variant_count]


def transform_points(points: np.ndarray, image_h: int, image_w: int, op_name: str) -> tuple[np.ndarray, int, int]:
    if op_name == "rot90":
        transformed = np.stack((points[:, 1], image_w - points[:, 0]), axis=1)
        return transformed.astype(np.float32), image_w, image_h
    if op_name == "rot180":
        transformed = np.stack((image_w - points[:, 0], image_h - points[:, 1]), axis=1)
        return transformed.astype(np.float32), image_h, image_w
    if op_name == "rot270":
        transformed = np.stack((image_h - points[:, 1], points[:, 0]), axis=1)
        return transformed.astype(np.float32), image_w, image_h
    if op_name == "flip_h":
        transformed = np.stack((image_w - points[:, 0], points[:, 1]), axis=1)
        return transformed.astype(np.float32), image_h, image_w
    if op_name == "flip_v":
        transformed = np.stack((points[:, 0], image_h - points[:, 1]), axis=1)
        return transformed.astype(np.float32), image_h, image_w
    raise ValueError(f"Unsupported transform op: {op_name}")


def apply_geometric_transform(
    image: np.ndarray, labels: list[tuple[int, np.ndarray]], op_name: str
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    image_h, image_w = image.shape[:2]
    if op_name == "rot90":
        transformed_image = np.ascontiguousarray(np.rot90(image, k=1))
    elif op_name == "rot180":
        transformed_image = np.ascontiguousarray(np.rot90(image, k=2))
    elif op_name == "rot270":
        transformed_image = np.ascontiguousarray(np.rot90(image, k=3))
    elif op_name == "flip_h":
        transformed_image = np.ascontiguousarray(np.flip(image, axis=1))
    elif op_name == "flip_v":
        transformed_image = np.ascontiguousarray(np.flip(image, axis=0))
    else:
        raise ValueError(f"Unsupported transform op: {op_name}")

    transformed_labels: list[tuple[int, np.ndarray]] = []
    new_h, new_w = transformed_image.shape[:2]
    for class_id, points in labels:
        transformed_points, _, _ = transform_points(points, image_h, image_w, op_name)
        rebuilt = rebuild_obb_from_polygon(transformed_points)
        rebuilt = clip_points_to_bounds(rebuilt, float(new_w), float(new_h))
        if polygon_area(rebuilt) <= 1e-6:
            continue
        transformed_labels.append((class_id, rebuilt))
    return transformed_image, transformed_labels


def format_patch_stem(
    image_stem: str,
    class_name: str,
    target_index: int,
    x0: int,
    y0: int,
    variant_name: str,
) -> str:
    safe_class_name = class_name.replace(" ", "_")
    safe_variant_name = variant_name.replace(" ", "_")
    return f"{image_stem}__cls{safe_class_name}__gt{target_index:05d}__x{x0}_y{y0}__{safe_variant_name}"


def save_patch_variant(
    output_root: Path,
    split_name: str,
    image_path: Path,
    class_name: str,
    target_index: int,
    x0: int,
    y0: int,
    variant_kind: str,
    variant_name: str,
    patch_image: np.ndarray,
    patch_labels: list[tuple[int, np.ndarray]],
    target_class_ids: set[int],
    manifest_rows: list[dict[str, object]],
    seen_keys: set[str],
    stats: dict[str, int],
    output_mode: str,
) -> None:
    if not patch_labels:
        stats["skipped_empty_variants"] += 1
        return
    if not contains_target_class(patch_labels, target_class_ids):
        stats["skipped_variants_without_target"] += 1
        return

    unique_key = f"{split_name}:{image_path.stem}:{x0}:{y0}:{variant_name}"
    if unique_key in seen_keys:
        stats["deduplicated_variants"] += 1
        return
    seen_keys.add(unique_key)

    patch_h, patch_w = patch_image.shape[:2]
    patch_stem = format_patch_stem(image_path.stem, class_name, target_index, x0, y0, variant_name)
    image_out_dir = output_root / "images" / split_name
    label_out_dir = output_root / "labels" / split_name
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    patch_image_path = image_out_dir / f"{patch_stem}.tiff"
    patch_label_path = label_out_dir / f"{patch_stem}.txt"
    if output_mode == "append" and (patch_image_path.exists() or patch_label_path.exists()):
        stats["skipped_existing_variants"] += 1
        return
    save_multichannel_tiff(patch_image_path, patch_image)
    write_label_file(patch_label_path, patch_labels, patch_h, patch_w)

    target_count_in_patch = sum(1 for class_id, _ in patch_labels if class_id in target_class_ids)
    manifest_rows.append(
        {
            "split": split_name,
            "source_image_path": str(image_path),
            "patch_image_path": str(patch_image_path),
            "patch_label_path": str(patch_label_path),
            "anchor_class_name": class_name,
            "anchor_gt_index": int(target_index),
            "variant_kind": variant_kind,
            "variant_name": variant_name,
            "crop_x0": int(x0),
            "crop_y0": int(y0),
            "crop_width": int(patch_w),
            "crop_height": int(patch_h),
            "object_count_in_patch": len(patch_labels),
            "target_object_count_in_patch": int(target_count_in_patch),
        }
    )
    stats["saved_patches"] += 1
    stats["saved_labels"] += 1
    stats[f"saved_{variant_kind}_variants"] += 1


def process_split(
    split_name: str,
    split_cfg: DatasetSplitConfig,
    cfg: TargetPatchAugmentConfig,
    class_to_id: dict[str, int],
    target_class_ids: set[int],
    manifest_rows: list[dict[str, object]],
) -> dict[str, int]:
    rng = random.Random(cfg.random_seed + (0 if split_name == "train" else 1))
    seen_keys: set[str] = set()
    image_paths = sorted(split_cfg.image_dir.glob("*.npy"))
    stats = {
        "source_images": 0,
        "images_with_target_gt": 0,
        "source_target_gt_count": 0,
        "skipped_targets_larger_than_patch": 0,
        "skipped_empty_variants": 0,
        "skipped_variants_without_target": 0,
        "skipped_existing_variants": 0,
        "deduplicated_variants": 0,
        "saved_patches": 0,
        "saved_labels": 0,
        "saved_center_variants": 0,
        "saved_translation_variants": 0,
        "saved_geometric_variants": 0,
        "saved_translation_geometric_variants": 0,
    }

    for image_path in image_paths:
        stats["source_images"] += 1
        image = load_npy_image(image_path)
        image_h, image_w = image.shape[:2]
        annotations = sanitize_raw_annotations(image_path, split_cfg, class_to_id)
        target_annotations = [
            (target_index, class_id, points)
            for target_index, (class_id, points) in enumerate(annotations)
            if class_id in target_class_ids
        ]
        if not target_annotations:
            continue
        stats["images_with_target_gt"] += 1
        stats["source_target_gt_count"] += len(target_annotations)

        for target_index, class_id, points in target_annotations:
            crop_window = compute_center_crop_window(points, image_h, image_w, cfg.patch_size)
            if crop_window is None:
                stats["skipped_targets_larger_than_patch"] += 1
                continue

            x0, y0, x1, y1 = crop_window
            center_patch = image[y0:y1, x0:x1]
            center_labels = collect_patch_labels(annotations, x0, y0, x1, y1)
            class_name = DEFAULT_CONFIG.class_names[class_id]

            if cfg.include_center_crop:
                save_patch_variant(
                    output_root=cfg.output_dir,
                    split_name=split_name,
                    image_path=image_path,
                    class_name=class_name,
                    target_index=target_index,
                    x0=x0,
                    y0=y0,
                    variant_kind="center",
                    variant_name="center",
                    patch_image=center_patch,
                    patch_labels=center_labels,
                    target_class_ids=target_class_ids,
                    manifest_rows=manifest_rows,
                    seen_keys=seen_keys,
                    stats=stats,
                    output_mode=cfg.output_mode,
                )

            translation_windows = sample_translation_windows(
                points=points,
                image_h=image_h,
                image_w=image_w,
                patch_size=cfg.patch_size,
                center_x0=x0,
                center_y0=y0,
                variant_count=cfg.translation_variants_per_gt,
                max_offset=cfg.translation_max_offset,
                rng=rng,
            )
            for variant_index, (shift_x0, shift_y0) in enumerate(translation_windows, start=1):
                shift_x1 = shift_x0 + center_patch.shape[1]
                shift_y1 = shift_y0 + center_patch.shape[0]
                shifted_patch = image[shift_y0:shift_y1, shift_x0:shift_x1]
                shifted_labels = collect_patch_labels(annotations, shift_x0, shift_y0, shift_x1, shift_y1)
                delta_x = shift_x0 - x0
                delta_y = shift_y0 - y0
                save_patch_variant(
                    output_root=cfg.output_dir,
                    split_name=split_name,
                    image_path=image_path,
                    class_name=class_name,
                    target_index=target_index,
                    x0=shift_x0,
                    y0=shift_y0,
                    variant_kind="translation",
                    variant_name=f"shift{variant_index:02d}_dx{delta_x:+d}_dy{delta_y:+d}",
                    patch_image=shifted_patch,
                    patch_labels=shifted_labels,
                    target_class_ids=target_class_ids,
                    manifest_rows=manifest_rows,
                    seen_keys=seen_keys,
                    stats=stats,
                    output_mode=cfg.output_mode,
                )

            for op_name in select_geometric_ops(cfg.geometric_transforms, cfg.geometric_variants_per_gt, rng):
                transformed_image, transformed_labels = apply_geometric_transform(center_patch, center_labels, op_name)
                save_patch_variant(
                    output_root=cfg.output_dir,
                    split_name=split_name,
                    image_path=image_path,
                    class_name=class_name,
                    target_index=target_index,
                    x0=x0,
                    y0=y0,
                    variant_kind="geometric",
                    variant_name=f"geom_{op_name}",
                    patch_image=transformed_image,
                    patch_labels=transformed_labels,
                    target_class_ids=target_class_ids,
                    manifest_rows=manifest_rows,
                    seen_keys=seen_keys,
                    stats=stats,
                    output_mode=cfg.output_mode,
                )

            translation_candidates_for_combo = build_translation_window_candidates(
                points=points,
                image_h=image_h,
                image_w=image_w,
                patch_size=cfg.patch_size,
                center_x0=x0,
                center_y0=y0,
                max_offset=cfg.translation_max_offset,
            )
            for combo_index, (shift_x0, shift_y0, op_name) in enumerate(
                select_translation_geometric_pairs(
                    translation_windows=translation_candidates_for_combo,
                    geometric_transforms=cfg.geometric_transforms,
                    variant_count=cfg.translation_geometric_variants_per_gt,
                    rng=rng,
                ),
                start=1,
            ):
                shift_x1 = shift_x0 + center_patch.shape[1]
                shift_y1 = shift_y0 + center_patch.shape[0]
                shifted_patch = image[shift_y0:shift_y1, shift_x0:shift_x1]
                shifted_labels = collect_patch_labels(annotations, shift_x0, shift_y0, shift_x1, shift_y1)
                transformed_image, transformed_labels = apply_geometric_transform(
                    shifted_patch, shifted_labels, op_name
                )
                delta_x = shift_x0 - x0
                delta_y = shift_y0 - y0
                save_patch_variant(
                    output_root=cfg.output_dir,
                    split_name=split_name,
                    image_path=image_path,
                    class_name=class_name,
                    target_index=target_index,
                    x0=shift_x0,
                    y0=shift_y0,
                    variant_kind="translation_geometric",
                    variant_name=f"shiftgeom{combo_index:02d}_dx{delta_x:+d}_dy{delta_y:+d}_{op_name}",
                    patch_image=transformed_image,
                    patch_labels=transformed_labels,
                    target_class_ids=target_class_ids,
                    manifest_rows=manifest_rows,
                    seen_keys=seen_keys,
                    stats=stats,
                    output_mode=cfg.output_mode,
                )

    return stats


def make_balanced_source_loader(split_cfg: DatasetSplitConfig, class_to_id: dict[str, int]):
    """Return a bounded-memory loader for one balanced source split."""

    def load_source(image_path: Path) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
        image = load_npy_image(image_path)
        image_h, image_w = image.shape[:2]
        raw = parse_raw_label_file(
            split_cfg.label_dir / f"{image_path.stem}.txt",
            class_to_id=class_to_id,
            include_difficult=split_cfg.include_difficult,
        )
        annotations = [
            (class_id, sanitized)
            for class_id, points in raw
            if (sanitized := sanitize_annotation(points, image_w, image_h)) is not None
        ]
        return image, annotations

    return load_source


def prepare_balanced_multiscale_split(
    split_name: str,
    split_cfg: DatasetSplitConfig,
    cfg: TargetPatchAugmentConfig,
    class_to_id: dict[str, int],
    target_class_ids: set[int],
) -> tuple[dict[str, object], Path]:
    """Build, materialize and validate the target-only balanced manifest."""
    if cfg.multiscale_patch_sizes != (256, 512):
        raise ValueError(
            f"balanced_multiscale requires multiscale_patch_sizes=(256, 512), got {cfg.multiscale_patch_sizes}."
        )
    if not cfg.overlap or not cfg.keep_empty_patches:
        raise ValueError("balanced_multiscale requires overlap=True and keep_empty_patches=True.")
    if cfg.output_mode != "reset":
        raise ValueError("balanced_multiscale requires output_mode='reset' to prevent stale files changing the ratio.")
    ratios = ViewRatios(*cfg.view_ratios)
    validate_view_ratios(ratios)
    image_paths = sorted(split_cfg.image_dir.glob("*.npy"))
    if not image_paths:
        raise FileNotFoundError(f"No source NPY images found in {split_cfg.image_dir}.")

    source_loader = make_balanced_source_loader(split_cfg, class_to_id)
    candidates = []
    for image_path in image_paths:
        image, annotations = source_loader(image_path)
        candidates.extend(
            generate_candidates(
                image_path=image_path,
                split=split_name,
                image=image,
                annotations=annotations,
                patch_sizes=cfg.multiscale_patch_sizes,
                full_view_size=cfg.full_view_size,
                overlap=cfg.overlap,
                keep_empty_patches=cfg.keep_empty_patches,
                min_iof=PATCH_IOF_THRESHOLD,
                target_class_ids=target_class_ids,
                require_target=True,
                transforms=(IDENTITY_TRANSFORM, *cfg.geometric_transforms),
            )
        )
        del image, annotations

    selected, selected_counts, capacities = select_candidates(
        candidates,
        ratios=ratios,
        seed=cfg.random_seed,
        total_samples=cfg.total_samples,
    )
    image_paths_out, rows = write_selected_candidates(
        selected=selected,
        output_root=cfg.output_dir,
        source_loader=source_loader,
        min_iof=PATCH_IOF_THRESHOLD,
        padding_value=114,
        seed=cfg.random_seed,
    )
    actual_counts = (
        validate_manifest_rows(rows, ratios)
        if cfg.strict_view_ratio
        else {
            view: sum(1 for row in rows if row["view_type"] == view) for view in ("patch256", "patch512", "full_scaled")
        }
    )
    manifest_path = write_manifest(rows, cfg.output_dir / f"{split_name}_manifest.csv")
    sample_image, _ = source_loader(image_paths[0])
    validation = validate_balanced_dataset_files(
        manifest_path=manifest_path,
        expected_channels=int(sample_image.shape[2]),
        ratios=ratios,
        strict_ratio=cfg.strict_view_ratio,
        expected_num_classes=len(class_to_id),
    )
    stats: dict[str, object] = {
        "source_images": len(image_paths),
        "candidate_counts": capacities,
        "selected_counts": selected_counts,
        "actual_counts": actual_counts,
        "selected_total": len(image_paths_out),
        "view_ratios": ratios.as_dict(),
        "overlap": cfg.overlap,
        "keep_empty_patches": cfg.keep_empty_patches,
        "require_target": True,
        "target_class_ids": sorted(target_class_ids),
        "manifest": str(manifest_path),
        "dataset_validation": validation,
        "max_feasible_total": maximum_feasible_total(capacities, ratios),
    }
    return stats, manifest_path


def write_manifest_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    import csv

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "source_image_path",
                "patch_image_path",
                "patch_label_path",
                "anchor_class_name",
                "anchor_gt_index",
                "variant_kind",
                "variant_name",
                "crop_x0",
                "crop_y0",
                "crop_width",
                "crop_height",
                "object_count_in_patch",
                "target_object_count_in_patch",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def build_summary(
    cfg: TargetPatchAugmentConfig,
    split_stats: dict[str, dict[str, object]],
    manifest_path: Path,
) -> dict[str, object]:
    total_saved_patches = sum(
        int(stats.get("saved_patches", stats.get("selected_total", 0))) for stats in split_stats.values()
    )
    total_target_gt = sum(int(stats.get("source_target_gt_count", 0)) for stats in split_stats.values())
    return {
        "output_dir": str(cfg.output_dir),
        "preprocess_profile": cfg.preprocess_profile,
        "source_splits": list(cfg.source_splits),
        "target_class_names": list(cfg.target_class_names),
        "class_names": list(DEFAULT_CONFIG.class_names),
        "patch_size": list(cfg.patch_size),
        "view_ratios": dict(zip(("patch256", "patch512", "full_scaled"), cfg.view_ratios)),
        "multiscale_patch_sizes": list(cfg.multiscale_patch_sizes),
        "full_view_size": cfg.full_view_size,
        "total_samples": cfg.total_samples,
        "strict_view_ratio": cfg.strict_view_ratio,
        "overlap": cfg.overlap,
        "keep_empty_patches": cfg.keep_empty_patches,
        "patch_iof_threshold": PATCH_IOF_THRESHOLD,
        "output_mode": cfg.output_mode,
        "translation_variants_per_gt": cfg.translation_variants_per_gt,
        "translation_max_offset": list(cfg.translation_max_offset),
        "geometric_variants_per_gt": cfg.geometric_variants_per_gt,
        "translation_geometric_variants_per_gt": cfg.translation_geometric_variants_per_gt,
        "geometric_transforms": list(cfg.geometric_transforms),
        "random_seed": cfg.random_seed,
        "include_center_crop": cfg.include_center_crop,
        "manifest_csv": str(manifest_path),
        "total_source_target_gt_count": total_target_gt,
        "total_saved_patches": total_saved_patches,
        "split_stats": split_stats,
    }


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    class_to_id = {name: idx for idx, name in enumerate(DEFAULT_CONFIG.class_names)}
    target_class_ids = {class_to_id[name] for name in cfg.target_class_names}

    if cfg.output_mode == "reset" and cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    split_stats: dict[str, dict[str, object]] = {}
    if cfg.preprocess_profile == "balanced_multiscale":
        manifest_paths: list[Path] = []
        for split_name in cfg.source_splits:
            stats, split_manifest_path = prepare_balanced_multiscale_split(
                split_name=split_name,
                split_cfg=resolve_split_config(split_name),
                cfg=cfg,
                class_to_id=class_to_id,
                target_class_ids=target_class_ids,
            )
            split_stats[split_name] = stats
            manifest_paths.append(split_manifest_path)
        # The train manifest is the one consumed by the main training script;
        # with train absent, expose the first requested split for inspection.
        manifest_path = next((path for path in manifest_paths if path.name == "train_manifest.csv"), manifest_paths[0])
    else:
        for split_name in cfg.source_splits:
            split_stats[split_name] = process_split(
                split_name=split_name,
                split_cfg=resolve_split_config(split_name),
                cfg=cfg,
                class_to_id=class_to_id,
                target_class_ids=target_class_ids,
                manifest_rows=manifest_rows,
            )
        manifest_path = write_manifest_csv(manifest_rows, cfg.output_dir / "augmentation_manifest.csv")
    summary = build_summary(cfg, split_stats, manifest_path)
    summary_path = cfg.output_dir / "augmentation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
