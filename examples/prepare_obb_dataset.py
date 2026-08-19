"""Strict 8-channel OBB dataset preparation: group-level stratified split, dual difficult labels.

本文件把原始 ``8 x W x H`` 的 CWH uint8 NPY 影像与
``x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult`` 原始标签，转成 Ultralytics
官方 OBB 数据集，并严格实现以下科学设计：

1. 合流与分层划分(group-level stratified split)
   - 多个原始目录(train/test)合流，按数据来源(日期前缀，如 20040301)分层；
   - 划分粒度是"场景组"而非单帧：文件名秒级时间戳(``20231220093922443_00-00589``),
     同一分钟级前缀内的连续帧高度相关，帧级随机划分会把近重复帧泄漏进 train/test。
     默认以时间戳前 12 位（分钟）作为组 ID,组内整组划入同一子集;
   - 每个来源内部按 ``test_fraction`` / ``val_fraction`` 在组级别分层抽取，
     保证各子集的来源比例与总体一致；小来源（组数 < 阈值）合并为 ``misc``;
   - 可选 holdout 模式：指定若干来源整体作为 test(终评泛化),其余分层出 train/val。

2. difficult 双口径
   - ``labels/`` 保留 difficult 目标（训练与标准评估统一口径，避免把难例变成隐式负样本）；
   - ``labels_clean/`` 剔除 difficult 目标（clean 指标专用）；
   - ``<output>_clean/`` 独立 clean 数据集根目录：图像为硬链接（零额外磁盘），
     标签为剔除 difficult 的真实副本。不能使用 ``images_clean -> images`` 符号链接：
     官方 ``check_det_dataset`` 用 ``Path.resolve()`` 解引用符号链接，会把路径解析回
     ``images/``，导致标签推导静默回落到标准 ``labels/``（clean 口径失效）。

3. 夜间批次过采样(train 侧专用)
   - 过采样仅作用于 train 的图片列表文件 ``train_oversampled.txt``（重复行），
     val/test 不参与；磁盘不复制图像。

4. 标签清洗与审计
   - 越界目标默认丢弃（不重建几何）；凸性校验；点序规范化；最小面积过滤；
   - 图-标配对缺失即报错；全程输出逐目标审计报告与划分清单（可复现）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.multichannel_preview_utils import build_preview_bgr  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402


# =========================
# IDE Quick Config
# 直接修改这里的值，然后点击 IDE 的运行按钮即可。
# 所有项也可用命令行参数覆盖（--help 查看）。
# =========================

# =========================
# 1. 原始数据来源（合流）
# =========================

# IDE_RAW_SOURCES:
# - 参与合流的原始目录列表，每个元组 = (来源名, 根目录)。
# - 根目录下必须含 images/（NPY）与 labels/（原始 txt）两个子目录，
#   原始标签格式: "x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult"。
# - 所有来源合流后统一按数据来源（日期前缀）分层划分；
#   来源名仅用于审计，不影响划分逻辑。
# - 注意：重名帧（相同时间戳）默认直接报错——同一帧同时进入两个子集
#   就是数据泄漏；若实验数据确实重叠，用 --dedupe 跳过（会告警）。
# 中文：原始数据目录（合流）；每个元组 (来源名, 根目录)。
IDE_RAW_SOURCES = (
    ("train", Path("/home/mofengwei/datasetObjectDetection/train")),
    ("test", Path("/home/mofengwei/datasetObjectDetection/test")),
)

# IDE_OUTPUT_DIR:
# - 生成数据集的输出目录。脚本每次运行都会先删除重建（rmtree）！
#   不要把它指向任何有其它用途的目录。
# - 产物结构:
#     images/         多页 TIFF（8 通道，HWC，官方 loader 可读）
#     labels/          YOLO-OBB 归一化标签（difficult 保留，训练与标准评估用）
#     labels_clean/    YOLO-OBB 归一化标签（difficult 剔除，clean 指标用）
#     <out>_clean/     独立 clean 数据集根目录（图像硬链接 + clean 标签副本 + data.yaml）
#     data.yaml        训练主数据文件（images/ + labels/）
#     split_manifest.json / audit_report.json / dataset_stats.json
#     previews/        GT 叠加伪彩预览图（人工目检标签质量）
# 中文：生成数据集输出目录（每次运行会重建！）。
IDE_OUTPUT_DIR = Path("/home/mofengwei/datasetObjectDetection/prepared_obb_dataset")

# IDE_CLASS_NAMES:
# - 类别名列表，顺序即 YOLO 类别 id（0~N-1）。
# - 必须与原始标签中的 class_name 完全一致；未知类名不会静默跳过（会记录为坏行）。
# - 修改它 = 改变类别映射，已生成的数据集需要重新 prepare。
# 中文：类别名顺序，即 YOLO 类别 id 映射。
IDE_CLASS_NAMES = ("car", "bus", "van", "awning-bike", "truck", "tricycle", "bike", "pedestrian")

# =========================
# 2. 数据划分（公平性核心）
# =========================

# IDE_SPLIT_MODE:
# - "stratified": 所有来源合流后，每个来源内部按场景组随机分层到
#   train/val/test（组级分层，保证各子集的来源比例与总体一致）。
# - "holdout"   : IDE_HOLDOUT_SOURCES 指定的来源整体作为 test（终评泛化），
#   其余来源分层出 train/val。用于"评估模型对未见来源的泛化能力"。
# 中文：划分模式；stratified=按来源分层，holdout=整来源留出作 test。
IDE_SPLIT_MODE = "stratified"

# IDE_GROUP_KEY_LEN:
# - 场景组 ID = 文件名时间戳前 N 位。
# - 文件名形如 20231220093922443_00-00589，同一分钟内的连续帧是同一场景
#   的高相关帧（实测帧间隔 10-15 秒）。帧级随机划分会把近重复帧漏进
#   train/test，指标虚高，因此必须按"场景组"整组划分。
# - 12 = 分钟级（推荐）；调小（如 10）组更细、比例更精确，但组间相关性上升。
# 中文：场景组 ID 前缀位数（12=分钟级，防帧级泄漏）。
IDE_GROUP_KEY_LEN = 12

# IDE_SMALL_SOURCE_MIN_GROUPS:
# - 组数少于该值的来源合并为 "misc" 来源再参与分层，
#   避免小来源按比例抽出 0-1 组导致子集缺失或噪声过大。
# 中文：组数少于该值的来源并入 misc。
IDE_SMALL_SOURCE_MIN_GROUPS = 3

# IDE_TEST_FRACTION / IDE_VAL_FRACTION:
# - 每个来源内部：先按 test_fraction 抽 test 组，再从剩余中按
#   val_fraction 抽 val 组，其余为 train。
# - 组数很小的来源（如 3-5 组）会出现某子集为空的情况，脚本会告警；
#   小实验数据请调大比例（如 0.25 / 0.5）或减少来源数。
# 中文：每个来源的 test / val 组占比。
IDE_TEST_FRACTION = 0.15
IDE_VAL_FRACTION = 0.15

# IDE_HOLDOUT_SOURCES:
# - 仅 IDE_SPLIT_MODE="holdout" 时生效：整体划入 test 的来源（日期前缀），
#   例如 ("20040301",) 把夜间批次整体留作终评 test。
# - 这些来源的帧完全不会出现在 train/val 中。
# 中文：holdout 模式整体作 test 的来源（日期前缀）。
IDE_HOLDOUT_SOURCES: tuple[str, ...] = ()

# IDE_SEED:
# - 划分随机种子；固定后可复现划分（split_manifest.json 记录归属）。
# 中文：划分随机种子（可复现）。
IDE_SEED = 0

# =========================
# 3. difficult 目标策略（双口径）
# =========================

# IDE_INCLUDE_DIFFICULT:
# - True : difficult 目标保留进 labels/（参与训练与标准评估），
#         同时 labels_clean/ 剔除它们（clean 指标用）。
# - False: difficult 目标完全丢弃（labels/ 与 labels_clean/ 都不含）。
# - 建议保持 True：difficult 目标是真实目标，删除会把它们变成隐式负样本，
#   弱类（bike/tricycle/truck）的召回会受损。
# 中文：difficult 目标是否保留进 labels/（labels_clean/ 始终剔除）。
IDE_INCLUDE_DIFFICULT = True

# =========================
# 4. 训练侧过采样（仅 train）
# =========================

# IDE_OVERSAMPLE_SOURCES:
# - 需要过采样的来源（日期前缀），如夜间批次 ("20040301",)。
# - 仅影响 train：生成 train_oversampled.txt 列表文件（重复行），
#   val/test 不参与，磁盘不复制图像；data.yaml 的 train 指向该列表。
# - 空元组 = 不过采样。
# 中文：train 侧过采样的来源（如夜间批次）。
IDE_OVERSAMPLE_SOURCES: tuple[str, ...] = ()

# IDE_OVERSAMPLE_REPEATS:
# - 每个过采样 train 帧额外重复的次数（1 = 翻倍）。
# 中文：过采样重复倍数。
IDE_OVERSAMPLE_REPEATS = 1

# =========================
# 5. 标签清洗与预览
# =========================

# IDE_MIN_AREA_PX:
# - 目标多边形面积小于该值（像素²）视为噪声丢弃。
# - 本数据集最小有效目标约几十像素²，4 是安全下限。
# 中文：最小目标面积（像素²）。
IDE_MIN_AREA_PX = 4.0

# IDE_PREVIEW_SAMPLES:
# - 每个子集抽样多少帧画 GT 叠加预览图（previews/gt_grid_*.jpg）。
# 中文：每个子集的预览抽样帧数。
IDE_PREVIEW_SAMPLES = 6

# IDE_DISPLAY_CHANNELS / IDE_CHANNEL_WAVELENGTHS_NM / IDE_STRETCH:
# - 仅影响预览伪彩可视化，不影响数据本身：
#   (R,G,B) 三个显示通道、各通道中心波长、百分位拉伸范围。
# 中文：预览伪彩的显示通道、波长与拉伸百分位。
IDE_DISPLAY_CHANNELS = (4, 2, 1)
IDE_CHANNEL_WAVELENGTHS_NM = (395.0, 474.285714, 553.571429, 632.857143, 712.142857, 791.428571, 870.714286, 950.0)
IDE_STRETCH = (2.0, 98.0)


# =========================
# 审计数据结构
# =========================


@dataclass
class AuditEntry:
    """One object-level audit record with the keep/drop reason."""

    image: str
    class_name: str
    difficult: bool
    action: str  # kept / kept_clean_excluded / dropped_oob / dropped_tiny / dropped_nonconvex / dropped_bad_line
    oob: bool
    area_px: float = 0.0


@dataclass
class AuditReport:
    """Aggregated label-quality audit report."""

    entries: list[AuditEntry] = field(default_factory=list)
    kept_classes: Counter = field(default_factory=Counter)
    clean_excluded_classes: Counter = field(default_factory=Counter)
    dropped_reasons: Counter = field(default_factory=Counter)

    def add(self, entry: AuditEntry) -> None:
        """Record one audit entry and update aggregate counters."""
        self.entries.append(entry)
        if entry.action == "kept":
            self.kept_classes[entry.class_name] += 1
        elif entry.action == "kept_clean_excluded":
            self.clean_excluded_classes[entry.class_name] += 1
        else:
            self.dropped_reasons[entry.action] += 1

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable summary of the audit."""
        return {
            "total_objects": len(self.entries),
            "kept_in_labels": int(sum(self.kept_classes.values())),
            "kept_in_labels_but_excluded_from_clean": int(sum(self.clean_excluded_classes.values())),
            "dropped_by_reason": dict(self.dropped_reasons),
            "kept_by_class": dict(self.kept_classes),
            "clean_excluded_by_class": dict(self.clean_excluded_classes),
        }


# =========================
# 图像读写
# =========================


def load_npy_cwh_to_hwc(path: Path) -> np.ndarray:
    """Load one CWH (channel, width, height) uint8 NPY image into contiguous HWC layout.

    中文：读取 CWH 格式 NPY（8, W, H），转成 HWC（H, W, 8）连续数组。
    """
    image = np.load(path, allow_pickle=False)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D CWH array, got shape={image.shape} from {path}")
    image = np.transpose(image, (2, 1, 0))  # (C, W, H) -> (H, W, C)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype} from {path}")
    return np.ascontiguousarray(image)


def save_multichannel_tiff(path: Path, image: np.ndarray) -> None:
    """Save an HWC multi-channel image as a multi-page TIFF (one page per channel)."""
    pages = np.ascontiguousarray(image.transpose(2, 0, 1))  # (C, H, W)
    ok = cv2.imwritemulti(str(path), pages)
    if not ok:
        raise RuntimeError(f"Failed to write multi-page TIFF: {path}")


# =========================
# 标签解析与几何校验
# =========================


def parse_raw_label_line(line: str, image_w: int, image_h: int) -> tuple[bool, str | None, dict[str, object]]:
    """Parse one raw label line into audit metadata, returning (ok, error, fields)."""
    parts = line.split()
    if len(parts) != 10:
        return False, "bad_line", {}
    try:
        coords = np.array([float(v) for v in parts[:8]], dtype=np.float32)
        difficult = int(float(parts[9]))
    except ValueError:
        return False, "bad_line", {}
    if not np.isfinite(coords).all():
        return False, "bad_line", {}
    points = coords.reshape(4, 2)
    xs, ys = points[:, 0], points[:, 1]
    oob = bool((xs < 0).any() or (ys < 0).any() or (xs > image_w).any() or (ys > image_h).any())
    return (
        True,
        None,
        {
            "class_name": parts[8],
            "difficult": bool(difficult),
            "oob": oob,
            "points": points,
        },
    )


def signed_area(points: np.ndarray) -> float:
    """Return the shoelace signed area (positive = CCW in x-right/y-up convention)."""
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_convex(points: np.ndarray) -> bool:
    """Return whether the 4 points form a convex polygon (all cross products share sign)."""
    crosses = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        v1, v2 = b - a, c - b
        crosses.append(float(v1[0] * v2[1] - v1[1] * v2[0]))
    return all(c > 1e-9 for c in crosses) or all(c < -1e-9 for c in crosses)


def normalize_polygon_order(points: np.ndarray) -> np.ndarray:
    """Normalize point order to counter-clockwise starting at the top-left-most vertex."""
    if signed_area(points) < 0:
        points = points[::-1]
    start = int(np.lexsort((points[:, 0], points[:, 1]))[0])  # min y, then min x
    return np.roll(points, -start, axis=0)


def polygon_area_abs(points: np.ndarray) -> float:
    """Return the absolute polygon area in pixels."""
    return abs(signed_area(points))


def clip_polygon_to_bounds(points: np.ndarray, image_w: float, image_h: float) -> np.ndarray:
    """Clamp vertices into the image canvas (only used when ``--keep-oob``)."""
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, image_w)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, image_h)
    return clipped


# =========================
# 合流、分组与分层划分
# =========================


def collect_frames(
    raw_sources: tuple[tuple[str, Path], ...],
    group_key_len: int,
    dedupe: bool = False,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Collect all frames from merged raw sources with (date, group) keys.

    中文：合流所有来源的帧，计算日期前缀与场景组 ID，并做重名校验。
    默认严格报错；``dedupe=True`` 时保留首次出现的帧并告警（仅用于实验数据自查）。
    """
    frames: list[dict[str, object]] = []
    seen_stems: dict[str, str] = {}
    source_frames: Counter = Counter()
    for source_name, root in raw_sources:
        image_dir = root / "images"
        label_dir = root / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Source {source_name} must contain images/ and labels/: {root}")
        for image_path in sorted(image_dir.glob("*.npy")):
            stem = image_path.stem
            if stem in seen_stems:
                if not dedupe:
                    raise ValueError(
                        f"Duplicate frame stem {stem!r} across sources {seen_stems[stem]!r} and {source_name!r}: "
                        "合流数据不允许重名，请先排查 train/test 是否包含相同时间戳（同帧同时进两个子集=泄漏）。"
                    )
                print(
                    f"[WARN] dedupe: skipping duplicate stem {stem!r} from source {source_name!r} "
                    f"(first seen in {seen_stems[stem]!r})."
                )
                continue
            seen_stems[stem] = source_name
            label_path = label_dir / f"{stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Label file missing for {image_path}: {label_path}")
            frames.append(
                {
                    "stem": stem,
                    "image_path": image_path,
                    "label_path": label_path,
                    "date": stem[:8],
                    "group": stem[:group_key_len],
                    "source": source_name,
                }
            )
            source_frames[source_name] += 1
    return frames, dict(source_frames)


def merge_small_sources(frames: list[dict[str, object]], min_groups: int) -> tuple[dict[str, str], dict[str, int]]:
    """Merge date sources with too few groups into 'misc'; returns (date->effective_source, per_source group counts)."""
    date_groups: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        date_groups[frame["date"]].add(frame["group"])
    date_to_source: dict[str, str] = {}
    for date, groups in sorted(date_groups.items()):
        date_to_source[date] = date if len(groups) >= min_groups else "misc"
    source_groups: Counter = Counter(date_to_source.values())
    return date_to_source, dict(source_groups)


def split_groups_stratified(
    frames: list[dict[str, object]],
    date_to_source: dict[str, str],
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, int]]]:
    """Per-source, group-level stratified split into train/val/test.

    中文：每个来源内部按场景组随机分层到 train/val/test，保证来源比例一致。
    """
    if not 0.0 <= test_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("test_fraction and val_fraction must be in [0, 1)")
    rng = random.Random(seed)
    frames_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for frame in frames:
        frames_by_source[date_to_source[frame["date"]]].append(frame)

    assigned: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    per_source_counts: dict[str, dict[str, int]] = {}
    for source in sorted(frames_by_source):
        source_frames = frames_by_source[source]
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for frame in source_frames:
            groups[frame["group"]].append(frame)
        group_keys = sorted(groups)
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        n_test = int(round(n_groups * test_fraction))
        n_val = int(round((n_groups - n_test) * val_fraction))
        n_val = min(n_val, n_groups - n_test)

        per_source_counts[source] = {
            "groups": n_groups,
            "train_groups": n_groups - n_test - n_val,
            "val_groups": n_val,
            "test_groups": n_test,
        }
        if n_test == 0 or n_val == 0:
            print(
                f"[WARN] source {source!r}: n_groups={n_groups}, test_groups={n_test}, val_groups={n_val} "
                "-> 该来源在部分子集中缺失（组数过少或比例过小）。"
            )
        for key in group_keys[:n_test]:
            assigned["test"].extend(groups[key])
        for key in group_keys[n_test : n_test + n_val]:
            assigned["val"].extend(groups[key])
        for key in group_keys[n_test + n_val :]:
            assigned["train"].extend(groups[key])
    return assigned, per_source_counts


def split_groups_holdout(
    frames: list[dict[str, object]],
    date_to_source: dict[str, str],
    holdout_sources: set[str],
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, int]]]:
    """Hold out entire sources as test; stratify the rest into train/val."""
    test_frames = [f for f in frames if date_to_source[f["date"]] in holdout_sources]
    dev_frames = [f for f in frames if date_to_source[f["date"]] not in holdout_sources]
    assigned, per_source_counts = split_groups_stratified(
        dev_frames, date_to_source, test_fraction=0.0, val_fraction=val_fraction, seed=seed
    )
    assigned["test"] = test_frames
    for frame in test_frames:
        source = date_to_source[frame["date"]]
        per_source_counts.setdefault(source, {"groups": 0, "train_groups": 0, "val_groups": 0, "test_groups": 0})
        per_source_counts[source]["test_groups"] += 1
    return assigned, per_source_counts


# =========================
# 数据集写出
# =========================


def process_split(
    split_frames: list[dict[str, object]],
    split_name: str,
    output_root: Path,
    class_to_id: dict[str, int],
    image_w: int,
    image_h: int,
    include_difficult: bool,
    min_area_px: float,
    audit: AuditReport,
) -> dict[str, int]:
    """Convert one split: NPY -> multi-page TIFF; write labels/ and labels_clean/.

    中文：转换一个划分：图像转多页 TIFF；写 labels/（含 difficult）与 labels_clean/（剔除 difficult）。
    """
    out_img_dir = output_root / "images" / split_name
    out_lab_dir = output_root / "labels" / split_name
    out_clean_dir = output_root / "labels_clean" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)
    out_clean_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "objects": 0, "objects_clean": 0}
    for frame in split_frames:
        label_lines: list[str] = []
        clean_lines: list[str] = []
        for line in frame["label_path"].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ok, error, fields = parse_raw_label_line(line, image_w, image_h)
            if not ok:
                audit.add(AuditEntry(frame["stem"], "", False, error, False))
                continue
            class_name = fields["class_name"]
            difficult = fields["difficult"]
            points = fields["points"]
            area = polygon_area_abs(points)

            if difficult and not include_difficult:
                audit.add(AuditEntry(frame["stem"], class_name, True, "dropped_difficult", fields["oob"], area))
                continue
            if fields["oob"]:
                audit.add(AuditEntry(frame["stem"], class_name, difficult, "dropped_oob", True, area))
                continue
            if area < min_area_px:
                audit.add(AuditEntry(frame["stem"], class_name, difficult, "dropped_tiny", False, area))
                continue
            if not is_convex(points):
                audit.add(AuditEntry(frame["stem"], class_name, difficult, "dropped_nonconvex", False, area))
                continue

            points = normalize_polygon_order(points)
            cls_id = class_to_id[class_name]
            normalized = points.copy()
            normalized[:, 0] /= image_w
            normalized[:, 1] /= image_h
            coords = " ".join(f"{v:.6f}" for v in normalized.reshape(-1))
            label_lines.append(f"{cls_id} {coords}")
            audit.add(AuditEntry(frame["stem"], class_name, difficult, "kept", False, area))
            if difficult:
                audit.add(AuditEntry(frame["stem"], class_name, True, "kept_clean_excluded", False, area))
            else:
                clean_lines.append(f"{cls_id} {coords}")

        image = load_npy_cwh_to_hwc(frame["image_path"])
        save_multichannel_tiff(out_img_dir / f"{frame['stem']}.tiff", image)
        out_lab_dir.joinpath(f"{frame['stem']}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
        )
        out_clean_dir.joinpath(f"{frame['stem']}.txt").write_text(
            "\n".join(clean_lines) + ("\n" if clean_lines else ""), encoding="utf-8"
        )
        stats["images"] += 1
        stats["objects"] += len(label_lines)
        stats["objects_clean"] += len(clean_lines)
    return stats


def write_data_yaml(output_root: Path, class_names: tuple[str, ...], channels: int) -> Path:
    """Write the standard data.yaml (labels/ with difficult)."""
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        f"path: {output_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"channels: {channels}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def build_clean_dataset_root(output_root: Path, class_names: tuple[str, ...], channels: int) -> Path:
    """Build a standalone clean-metric dataset root with hard-linked images and clean labels.

    IMPORTANT: a symlink-based clean dataset does NOT work. ``check_det_dataset`` resolves
    the split paths with ``Path.resolve()``, which dereferences symlinks, so a
    ``images_clean -> images`` link silently falls back to the standard ``labels/``.
    Instead this builds a sibling root ``<output_root>_clean/`` in which:

    - ``images/{split}/*.tiff`` are HARD LINKS to the standard images (zero extra disk,
      ``resolve()`` keeps the clean-root path since hard links are regular files);
    - ``labels/{split}/*.txt`` are real copies of the clean labels (difficult excluded).

    The official loader then derives labels from ``<clean_root>/images/...`` to
    ``<clean_root>/labels/...``, i.e. the clean GT, with a separate ``*.cache``.

    中文：构建独立的 clean 口径数据集根目录。图像用硬链接（不占额外磁盘），
    标签为剔除 difficult 的真实副本；官方 loader 的标签推导会正确指向 clean 标签。
    """
    clean_root = output_root.with_name(f"{output_root.name}_clean")
    if clean_root.exists():
        shutil.rmtree(clean_root)
    for split_name in ("train", "val", "test"):
        src_img_dir = output_root / "images" / split_name
        src_lab_dir = output_root / "labels_clean" / split_name
        dst_img_dir = clean_root / "images" / split_name
        dst_lab_dir = clean_root / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lab_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(src_img_dir.glob("*.tiff")):
            try:
                os.link(image_path, dst_img_dir / image_path.name)
            except OSError:
                shutil.copy2(image_path, dst_img_dir / image_path.name)
        for label_path in sorted(src_lab_dir.glob("*.txt")):
            shutil.copy2(label_path, dst_lab_dir / label_path.name)

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        f"path: {clean_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"channels: {channels}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path = clean_root / "data.yaml"
    yaml_path.write_text(content, encoding="utf-8")

    # 自检：官方标签推导必须落在 clean 根目录内的真实标签文件上
    from ultralytics.data.utils import img2label_paths

    sample = sorted((clean_root / "images" / "test").glob("*.tiff"))[0]
    derived = img2label_paths([str(sample)])[0]
    if not Path(derived).exists():
        raise RuntimeError(
            f"Clean dataset self-check failed: label derivation produced {derived}, "
            "which does not exist. Clean root construction is broken."
        )
    print(f"[INFO] Clean dataset root: {clean_root} (images hard-linked, zero extra image disk)")
    return clean_root


def write_oversampled_train_list(
    train_frames: list[dict[str, object]],
    oversample_sources: set[str],
    repeats: int,
    output_root: Path,
    date_to_source: dict[str, str],
) -> Path | None:
    """Write train_oversampled.txt repeating train frames of selected sources (train-only).

    中文：为过采样来源的 train 帧生成重复行列表文件，val/test 不受影响。
    """
    if not oversample_sources or repeats < 1:
        return None
    base = [str(output_root / "images" / "train" / f"{f['stem']}.tiff") for f in train_frames]
    extra = []
    for f in train_frames:
        if date_to_source[f["date"]] in oversample_sources:
            extra.extend([base[train_frames.index(f)]] * repeats)
    list_path = output_root / "train_oversampled.txt"
    list_path.write_text("\n".join(base + extra) + "\n", encoding="utf-8")
    print(
        f"[INFO] Oversampled train list: {len(base)} base + {len(extra)} repeated entries "
        f"for sources {sorted(oversample_sources)} -> {list_path.name}"
    )
    return list_path


# =========================
# 校验与统计
# =========================


def verify_prepared_dataset(output_root: Path, class_names: tuple[str, ...], label_dir_name: str) -> dict[str, object]:
    """Run OBB-specific verification over one label variant (labels or labels_clean)."""
    report: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        img_dir = output_root / "images" / split_name
        lab_dir = output_root / label_dir_name / split_name
        images = sorted(img_dir.glob("*.tiff"))
        labels = sorted(lab_dir.glob("*.txt"))
        problems: list[str] = []
        objects = 0
        for lab in labels:
            for line in lab.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 9:
                    problems.append(f"{lab.name}: {len(values)} columns")
                    continue
                cls_id = int(values[0])
                coords = np.array([float(v) for v in values[1:]], dtype=np.float32)
                if not 0 <= cls_id < len(class_names):
                    problems.append(f"{lab.name}: class id {cls_id} out of range")
                if coords.min() < 0 or coords.max() > 1.000001:
                    problems.append(f"{lab.name}: coords out of [0,1]")
                objects += 1
        report[split_name] = {
            "images": len(images),
            "labels": len(labels),
            "objects": objects,
            "problems": problems[:20],
            "problem_count": len(problems),
        }
    return report


def build_gt_preview_grid(
    split_name: str,
    output_root: Path,
    sample_count: int,
    display_channels: tuple[int, int, int],
    wavelengths: tuple[float, ...],
    stretch: tuple[float, float],
) -> Path:
    """Render a grid of GT-overlaid previews from one prepared split for visual inspection."""
    image_paths = sorted((output_root / "images" / split_name).glob("*.tiff"))
    if not image_paths:
        raise RuntimeError(f"No prepared images under {output_root / 'images' / split_name}")
    rng = random.Random(0)
    sampled = rng.sample(image_paths, min(sample_count, len(image_paths)))

    tiles: list[np.ndarray] = []
    for image_path in sampled:
        ok, pages = cv2.imreadmulti(str(image_path), flags=cv2.IMREAD_UNCHANGED)
        if not ok or not pages:
            raise RuntimeError(f"Failed to read prepared TIFF: {image_path}")
        hwc = np.stack(pages, axis=2)
        canvas, _ = build_preview_bgr(
            image=hwc,
            preview_mode="rgb_like",
            display_channels=display_channels,
            stretch_low=stretch[0],
            stretch_high=stretch[1],
            channel_wavelengths_nm=wavelengths,
        )
        label_path = output_root / "labels" / split_name / f"{image_path.stem}.txt"
        lines = [ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        canvas_h, canvas_w = canvas.shape[:2]
        for line in lines:
            values = line.split()
            coords = np.array([float(v) for v in values[1:9]], dtype=np.float32).reshape(4, 2)
            coords[:, 0] *= canvas_w
            coords[:, 1] *= canvas_h
            polygon = coords.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [polygon], True, (0, 0, 255), 2)
        cv2.putText(canvas, image_path.stem, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)

    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    tile_h, tile_w = tiles[0].shape[:2]
    grid = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    out_dir = output_root / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gt_grid_{split_name}.jpg"
    if not cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Failed to write preview grid: {out_path}")
    return out_path


# =========================
# CLI 与主流程
# =========================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the strict dataset preparation script."""
    parser = argparse.ArgumentParser(
        description="Strict 8-channel OBB dataset preparation: merged sources, group-level stratified split, "
        "dual difficult labels (labels/ + labels_clean/), optional train-only oversampling."
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Semicolon-separated name:root pairs, e.g. 'train:/path/a;test:/path/b'. "
        "Overrides the IDE_RAW_SOURCES default.",
    )
    parser.add_argument("--out-dir", type=Path, default=IDE_OUTPUT_DIR, help="Prepared dataset output directory.")
    parser.add_argument("--split-mode", type=str, default=IDE_SPLIT_MODE, choices=("stratified", "holdout"))
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=IDE_TEST_FRACTION,
        help="Per-source test group fraction (stratified mode).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=IDE_VAL_FRACTION,
        help="Per-source val group fraction of the post-test remainder.",
    )
    parser.add_argument(
        "--holdout-sources",
        type=str,
        default=",".join(IDE_HOLDOUT_SOURCES),
        help="Comma-separated date prefixes used as entire test (holdout mode).",
    )
    parser.add_argument(
        "--small-source-min-groups",
        type=int,
        default=IDE_SMALL_SOURCE_MIN_GROUPS,
        help="Sources with fewer groups than this are merged into 'misc'.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        default=False,
        help="Skip duplicate stems across sources with a warning instead of failing (experiment data only).",
    )
    parser.add_argument(
        "--group-key-len",
        type=int,
        default=IDE_GROUP_KEY_LEN,
        help="Scene group id = first N chars of the timestamp filename prefix.",
    )
    parser.add_argument("--seed", type=int, default=IDE_SEED, help="Split random seed.")
    parser.add_argument(
        "--include-difficult",
        action="store_true",
        default=IDE_INCLUDE_DIFFICULT,
        help="Keep difficult objects in labels/ (labels_clean/ is always generated).",
    )
    parser.add_argument("--no-include-difficult", action="store_false", dest="include_difficult")
    parser.add_argument(
        "--oversample-sources",
        type=str,
        default=",".join(IDE_OVERSAMPLE_SOURCES),
        help="Comma-separated sources whose TRAIN frames are repeated (train-only).",
    )
    parser.add_argument("--oversample-repeats", type=int, default=IDE_OVERSAMPLE_REPEATS)
    parser.add_argument("--min-area-px", type=float, default=IDE_MIN_AREA_PX)
    parser.add_argument("--preview-samples", type=int, default=IDE_PREVIEW_SAMPLES)
    return parser.parse_args()


def parse_sources(text: str | None) -> tuple[tuple[str, Path], ...]:
    """Parse 'name:root;name:root' into source tuples."""
    if not text:
        return IDE_RAW_SOURCES
    sources = []
    for item in text.split(";"):
        name, _, root = item.strip().partition(":")
        if not name or not root:
            raise ValueError(f"Invalid source spec {item!r}; expected 'name:root'.")
        sources.append((name, Path(root)))
    return tuple(sources)


def main() -> None:
    """Run the full strict dataset preparation pipeline."""
    args = parse_args()
    raw_sources = parse_sources(args.sources)
    output_root = args.out_dir

    frames, source_frames = collect_frames(raw_sources, args.group_key_len, dedupe=args.dedupe)
    if len(frames) == 0:
        raise RuntimeError("No frames collected from the given sources.")
    date_to_source, source_group_counts = merge_small_sources(frames, args.small_source_min_groups)
    print(
        f"[INFO] Merged {len(frames)} frames, {len(set(f['date'] for f in frames))} raw dates, "
        f"{len(set(date_to_source.values()))} effective sources."
    )
    groups_by_source: Counter = Counter()
    for source in set(date_to_source.values()):
        groups_by_source[source] = len({f["group"] for f in frames if date_to_source[f["date"]] == source})
    print(f"[INFO] Effective source group counts: {dict(groups_by_source)}")

    holdout_sources = set(s for s in args.holdout_sources.split(",") if s)
    if args.split_mode == "holdout":
        if not holdout_sources:
            raise ValueError("holdout mode requires --holdout-sources (e.g. '20040301,20231220').")
        missing = holdout_sources - set(date_to_source.values())
        if missing:
            raise ValueError(f"Holdout sources not present in data: {sorted(missing)}")
        assigned, per_source_counts = split_groups_holdout(
            frames, date_to_source, holdout_sources, args.val_fraction, args.seed
        )
    else:
        if holdout_sources:
            print("[WARN] --holdout-sources given but split-mode=stratified; ignored.")
        assigned, per_source_counts = split_groups_stratified(
            frames, date_to_source, args.test_fraction, args.val_fraction, args.seed
        )

    sample = load_npy_cwh_to_hwc(frames[0]["image_path"])
    image_h, image_w, channels = sample.shape
    if channels != 8:
        print(f"[WARN] Expected 8-channel data, got {channels} channels: {frames[0]['image_path'].name}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    class_to_id = {name: idx for idx, name in enumerate(IDE_CLASS_NAMES)}
    audit = AuditReport()
    split_stats: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_stats[split_name] = process_split(
            split_frames=assigned[split_name],
            split_name=split_name,
            output_root=output_root,
            class_to_id=class_to_id,
            image_w=image_w,
            image_h=image_h,
            include_difficult=args.include_difficult,
            min_area_px=args.min_area_px,
            audit=audit,
        )

    data_yaml = write_data_yaml(output_root, IDE_CLASS_NAMES, channels)

    clean_root = build_clean_dataset_root(output_root, IDE_CLASS_NAMES, channels)
    data_clean_yaml = clean_root / "data.yaml"

    oversample_sources = set(s for s in args.oversample_sources.split(",") if s)
    unknown = oversample_sources - set(date_to_source.values())
    if unknown:
        raise ValueError(f"Oversample sources not present: {sorted(unknown)}")
    oversampled_list = write_oversampled_train_list(
        train_frames=assigned["train"],
        oversample_sources=oversample_sources,
        repeats=args.oversample_repeats,
        output_root=output_root,
        date_to_source=date_to_source,
    )
    if oversampled_list is not None:
        payload = data_yaml.read_text(encoding="utf-8")
        payload = payload.replace("train: images/train", f"train: {oversampled_list.name}")
        data_yaml.write_text(payload, encoding="utf-8")

    verify_report = {
        "labels": verify_prepared_dataset(output_root, IDE_CLASS_NAMES, "labels"),
        "labels_clean": verify_prepared_dataset(output_root, IDE_CLASS_NAMES, "labels_clean"),
    }

    manifest = {
        "seed": args.seed,
        "split_mode": args.split_mode,
        "test_fraction": args.test_fraction,
        "val_fraction": args.val_fraction,
        "group_key_len": args.group_key_len,
        "effective_sources": sorted(set(date_to_source.values())),
        "date_to_source": date_to_source,
        "per_source_split": per_source_counts,
        "frames": {
            split_name: sorted(f["stem"] for f in assigned[split_name]) for split_name in ("train", "val", "test")
        },
        "oversample": {
            "sources": sorted(oversample_sources),
            "repeats": args.oversample_repeats,
            "train_list": str(oversampled_list) if oversampled_list else None,
        },
        "source_frame_counts": source_frames,
    }
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    audit_payload = {
        "config": {
            "raw_sources": [(name, str(root)) for name, root in raw_sources],
            "include_difficult": args.include_difficult,
            "min_area_px": args.min_area_px,
            "image_size": [image_w, image_h],
            "channels": channels,
            "class_names": list(IDE_CLASS_NAMES),
        },
        "summary": audit.summary(),
        "drop_details": [
            {
                "image": e.image,
                "class_name": e.class_name,
                "difficult": e.difficult,
                "action": e.action,
                "oob": e.oob,
                "area_px": round(e.area_px, 2),
            }
            for e in audit.entries
        ],
    }
    (output_root / "audit_report.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    stats_payload = {
        "splits": split_stats,
        "verify": verify_report,
        "difficult_policy": {
            "labels": "difficult included (training + standard eval)",
            "labels_clean": "difficult excluded (clean metric eval)",
        },
    }
    (output_root / "dataset_stats.json").write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    preview_paths: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        if not assigned[split_name]:
            print(f"[WARN] split {split_name!r} is empty; skipping its preview and stats.")
            continue
        preview_paths[split_name] = build_gt_preview_grid(
            split_name=split_name,
            output_root=output_root,
            sample_count=args.preview_samples,
            display_channels=IDE_DISPLAY_CHANNELS,
            wavelengths=IDE_CHANNEL_WAVELENGTHS_NM,
            stretch=IDE_STRETCH,
        )

    print("=" * 70)
    print("Audit summary:", json.dumps(audit_payload["summary"], ensure_ascii=False))
    print("Split stats:", json.dumps(split_stats, ensure_ascii=False))
    print("Per-source split:", json.dumps(per_source_counts, ensure_ascii=False))
    print(f"data.yaml:        {data_yaml}")
    print(f"data_clean.yaml:  {data_clean_yaml}")
    print("Previews:", {k: str(v) for k, v in preview_paths.items()})

    for yaml_name, yaml_path in (("data.yaml", data_yaml), ("data_clean.yaml", data_clean_yaml)):
        empty_splits = [s for s in ("train", "val", "test") if not assigned[s]]
        if empty_splits:
            print(
                f"[WARN] {yaml_name}: skipping official check_det_dataset because splits are empty: "
                f"{empty_splits}（小实验数据建议调大 --val-fraction / --test-fraction 或减小比例）"
            )
            continue
        try:
            check_det_dataset(str(yaml_path))
            print(f"Official check_det_dataset {yaml_name}: PASS")
        except Exception as exc:
            print(f"Official check_det_dataset {yaml_name}: FAILED ({exc})")

    print("=" * 70)
    print(
        f"Prepared dataset ready: {output_root}\n"
        "  labels/        : difficult 保留（训练与标准评估）\n"
        f"  {clean_root.name}/ : 独立 clean 口径数据集（difficult 剔除，data.yaml 为其根）\n"
        f"  split_manifest.json : 组级划分清单（种子 {args.seed}，可复现）\n"
        "  audit_report.json   : 逐目标清洗审计"
    )


if __name__ == "__main__":
    main()
