"""Visualize the OBB patch pipeline: raw 256x256 patch -> minAreaRect rebuild -> vertex clamp.

本脚本从原始 NPY 数据中挑选一张含"越界目标"的图，切出 256x256 patch，并严格复刻
``custom_obb_prepare_and_train.py`` 中 patch 标签投影的几何链条：

1. ``clip_polygon_to_rect``  : 用 ``cv2.intersectConvexConvex`` 把目标裁剪进 patch；
2. ``rebuild_obb_from_polygon``: 对裁剪后的可见残片做 ``cv2.minAreaRect`` 拟合，
   重建为标准旋转矩形（注意：拟合出的矩形可能伸出 patch 边界，因为目标在
   patch 外还有不可见部分）；
3. ``clip_points_to_bounds`` : 把重建矩形的 4 个顶点独立钳制回 patch 画布，
   破坏矩形形状，产生"不规则四边形"。

输出 3 张教学图片到 ``--out-dir``：

- ``1_patch_256x256.jpg``         : 256x256 伪彩切片（无标注）；
- ``2_patch_minarect_rebuilt.jpg``: 切片 + 重建标准矩形（绿色，普通 / 亮黄加粗=越界）；
- ``3_patch_clamped_quad.jpg``    : 切片 + 顶点钳制后的不规则四边形
  （红色，普通 / 橙色加粗=被钳制变形）。

另有 ``selection_info.txt`` 记录所选窗口、每目标 IoF、面积与顶点位移量。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.multichannel_preview_utils import build_preview_bgr  # noqa: E402

# =========================
# IDE Quick Config
# 直接修改这里的值后点击 IDE 运行按钮即可。
# =========================

# IDE_SOURCE_DIR:
# 中文：原始数据根目录，下含 images/（NPY）与 labels/（原始 txt）两个子目录，
#      与 prepare_obb_dataset.py 的 IDE_RAW_SOURCES 目录结构一致。
IDE_SOURCE_DIR = Path("/home/mofengwei/datasetObjectDetection/train")

# IDE_OUTPUT_DIR:
# 中文：三张教学图片的输出目录（会自动创建）。
IDE_OUTPUT_DIR = Path("/home/mofengwei/datasetObjectDetection/keshihua/patch-bad")

# IDE_PATCH_SIZE:
# 中文：切片边长（与训练 patch_size 一致，256）。
IDE_PATCH_SIZE = 256

# IDE_WINDOW_STEP:
# 中文：挑选窗口时的滑窗步长（128=重叠滑窗，仅影响挑选，不影响几何链条）。
IDE_WINDOW_STEP = 128

# IDE_MIN_IOF:
# 中文：与 PATCH_IOF_THRESHOLD 一致的目标保留阈值。
IDE_MIN_IOF = 0.6

# IDE_MAX_SCAN_IMAGES:
# 中文：最多扫描多少张原图寻找教学窗口；找到足够好的越界目标会提前停止。
IDE_MAX_SCAN_IMAGES = 300

# IDE_CLASS_NAMES:
# 中文：类别名，仅用于 selection_info.txt 输出。
IDE_CLASS_NAMES = ("car", "bus", "van", "awning-bike", "truck", "tricycle", "bike", "pedestrian")

# IDE_DISPLAY_CHANNELS / IDE_CHANNEL_WAVELENGTHS_NM / IDE_STRETCH:
# 中文：伪彩显示通道、波长与百分位拉伸（与 prepare_obb_dataset.py 相同）。
IDE_DISPLAY_CHANNELS = (4, 2, 1)
IDE_CHANNEL_WAVELENGTHS_NM = (395.0, 474.285714, 553.571429, 632.857143, 712.142857, 791.428571, 870.714286, 950.0)
IDE_STRETCH = (2.0, 98.0)


# =========================
# 几何链条（与训练脚本保持一致）
# =========================


def polygon_area(points: np.ndarray) -> float:
    """Return the absolute polygon area."""
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def clip_polygon_to_rect(points: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray | None:
    """Clip one convex polygon against an axis-aligned rectangle."""
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    area, clipped = cv2.intersectConvexConvex(points.astype(np.float32), rect)
    if area <= 1e-6 or clipped is None or len(clipped) < 3:
        return None
    return clipped.reshape(-1, 2).astype(np.float32)


def rebuild_obb_from_polygon(points: np.ndarray) -> np.ndarray:
    """Rebuild an oriented box from a clipped polygon using the minimum-area rectangle."""
    rect = cv2.minAreaRect(points.astype(np.float32))
    return cv2.boxPoints(rect).astype(np.float32)


def clip_points_to_bounds(points: np.ndarray, width: float, height: float) -> np.ndarray:
    """Clamp polygon coordinates into patch bounds (the step that breaks the rectangle)."""
    clipped = points.astype(np.float32, copy=True)
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, height)
    return clipped


def sanitize_annotation(points: np.ndarray, image_w: int, image_h: int) -> np.ndarray | None:
    """Image-level sanitize: clip OOB annotations to the canvas, then rebuild + clamp."""
    if not np.isfinite(points).all():
        return None
    clipped = clip_polygon_to_rect(points, 0.0, 0.0, float(image_w), float(image_h))
    if clipped is None or polygon_area(clipped) <= 1e-6:
        return None
    rebuilt = clip_points_to_bounds(rebuild_obb_from_polygon(clipped), float(image_w), float(image_h))
    return rebuilt if polygon_area(rebuilt) > 1e-6 else None


@dataclass
class ProjectedObject:
    """One object projected into a patch with rebuild/clamp diagnostics."""

    class_id: int
    iof: float
    rebuilt: np.ndarray  # minAreaRect fitted rect (may exceed patch bounds)
    clamped: np.ndarray  # vertex-clamped irregular quad
    deformed: bool  # rebuilt rect sticks out of the patch, clamp moved vertices
    displacement: float  # max vertex movement caused by clamping (px)


def project_annotation_to_patch(
    points: np.ndarray, class_id: int, x0: int, y0: int, x1: int, y1: int, min_iof: float
) -> ProjectedObject | None:
    """Mirror of the training pipeline: clip -> IoF filter -> rebuild -> clamp."""
    object_area = polygon_area(points)
    if object_area <= 1e-6:
        return None
    clipped = clip_polygon_to_rect(points, float(x0), float(y0), float(x1), float(y1))
    if clipped is None:
        return None
    iof = polygon_area(clipped) / object_area
    if iof < min_iof:
        return None
    shifted = clipped - np.array([x0, y0], dtype=np.float32)
    patch_w, patch_h = x1 - x0, y1 - y0
    rebuilt = rebuild_obb_from_polygon(shifted)
    clamped = clip_points_to_bounds(rebuilt, float(patch_w), float(patch_h))
    displacement = float(np.max(np.linalg.norm(clamped - rebuilt, axis=1)))
    overflow = float(
        max(
            0.0,
            -float(rebuilt[:, 0].min()),
            float(rebuilt[:, 0].max()) - patch_w,
            -float(rebuilt[:, 1].min()),
            float(rebuilt[:, 1].max()) - patch_h,
        )
    )
    deformed = overflow > 1.0
    return ProjectedObject(class_id, iof, rebuilt, clamped, deformed, displacement)


# =========================
# 图像与标签读取
# =========================


def load_npy_cwh_to_hwc(path: Path) -> np.ndarray:
    """Load one CWH uint8 NPY image into contiguous HWC layout."""
    image = np.load(path, allow_pickle=False)
    image = np.transpose(image, (2, 1, 0))  # (C, W, H) -> (H, W, C)
    return np.ascontiguousarray(image)


def parse_raw_label_file(label_path: Path, class_to_id: dict[str, int]) -> list[tuple[int, np.ndarray]]:
    """Parse raw 'x1 y1 ... x8 y8 class_name difficult' labels into (class_id, points)."""
    annotations: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 10 or parts[8] not in class_to_id:
            continue
        points = np.array([float(v) for v in parts[:8]], dtype=np.float32).reshape(4, 2)
        annotations.append((class_to_id[parts[8]], points))
    return annotations


# =========================
# 教学窗口挑选
# =========================


def score_window(objects: list[ProjectedObject]) -> float:
    """Prefer windows with large, clearly deformed boundary-crossing objects.

    A small bonus per extra object makes windows that contrast a deformed
    boundary object with undisturbed inside objects rank higher (better teaching).
    """
    deformed = [obj for obj in objects if obj.deformed]
    if not deformed:
        return -1.0
    score = float(max(obj.displacement + polygon_area(obj.clamped) / 500.0 for obj in deformed))
    return score + 0.5 * min(len(objects), 3)


def find_teaching_window(
    source_dir: Path, class_to_id: dict[str, int], patch_size: int, step: int, min_iof: float, max_scan: int
) -> tuple[Path, int, int, list[ProjectedObject], list[tuple[int, np.ndarray]]] | None:
    """Scan raw frames and return the best-scoring (image, x0, y0, projected, sanitized) window."""
    image_dir = source_dir / "images"
    label_dir = source_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"{source_dir} must contain images/ and labels/ subdirectories.")
    image_paths = sorted(image_dir.glob("*.npy"))
    best: tuple[float, Path, int, int, list[ProjectedObject]] | None = None
    for image_index, image_path in enumerate(image_paths[:max_scan]):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = load_npy_cwh_to_hwc(image_path)
        image_h, image_w = image.shape[:2]
        sanitized: list[tuple[int, np.ndarray]] = []
        for class_id, points in parse_raw_label_file(label_path, class_to_id):
            fixed = sanitize_annotation(points, image_w, image_h)
            if fixed is not None:
                sanitized.append((class_id, fixed))

        y_starts = list(range(0, max(image_h - patch_size + 1, 1), step))
        if y_starts and y_starts[-1] != image_h - patch_size:
            y_starts.append(image_h - patch_size)
        x_starts = list(range(0, max(image_w - patch_size + 1, 1), step))
        if x_starts and x_starts[-1] != image_w - patch_size:
            x_starts.append(image_w - patch_size)

        for y0 in y_starts:
            for x0 in x_starts:
                x1, y1 = min(x0 + patch_size, image_w), min(y0 + patch_size, image_h)
                projected = [
                    obj
                    for class_id, points in sanitized
                    if (obj := project_annotation_to_patch(points, class_id, x0, y0, x1, y1, min_iof)) is not None
                ]
                score = score_window(projected)
                if score < 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, image_path, x0, y0, projected)
        if (image_index + 1) % 50 == 0:
            print(f"[INFO] scanned {image_index + 1} images...")
    if best is None:
        return None
    score, image_path, x0, y0, projected = best
    print(
        f"[INFO] selected {image_path.name} window x={x0} y={y0} "
        f"({len(projected)} objects, score={score:.1f})"
    )
    return image_path, x0, y0, projected, []


# =========================
# 可视化
# =========================


def add_caption(canvas: np.ndarray, text: str) -> np.ndarray:
    """Add a white-on-black caption strip at the top of a BGR canvas."""
    strip_h = 34
    padded = np.zeros((canvas.shape[0] + strip_h, canvas.shape[1], 3), dtype=np.uint8)
    padded[strip_h:] = canvas
    cv2.putText(
        padded,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return padded


def draw_polygon(canvas: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int, tag: str = "") -> None:
    """Draw one polygon and an optional tag near its first vertex."""
    polygon = points.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [polygon], True, color, thickness, cv2.LINE_AA)
    if tag:
        x, y = polygon[0, 0].tolist()
        cv2.putText(canvas, tag, (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def build_patch_canvas(patch: np.ndarray) -> np.ndarray:
    """Convert an HWC 8-channel patch into a pseudo-color BGR canvas."""
    canvas, _ = build_preview_bgr(
        image=patch,
        preview_mode="rgb_like",
        display_channels=IDE_DISPLAY_CHANNELS,
        stretch_low=IDE_STRETCH[0],
        stretch_high=IDE_STRETCH[1],
        channel_wavelengths_nm=IDE_CHANNEL_WAVELENGTHS_NM,
    )
    return canvas


def visualize_window(
    image_path: Path,
    x0: int,
    y0: int,
    patch_size: int,
    projected: list[ProjectedObject],
    out_dir: Path,
) -> dict[str, Path]:
    """Render and save the three teaching images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_npy_cwh_to_hwc(image_path)
    patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
    base = build_patch_canvas(patch)

    canvas_patch = base.copy()
    canvas_rebuilt = base.copy()
    canvas_clamped = base.copy()

    for obj in projected:
        if obj.deformed:
            draw_polygon(canvas_rebuilt, obj.rebuilt, (0, 255, 255), 3, "OOB")
            draw_polygon(canvas_clamped, obj.clamped, (0, 165, 255), 3, "clamped")
        else:
            draw_polygon(canvas_rebuilt, obj.rebuilt, (0, 255, 0), 1)
            draw_polygon(canvas_clamped, obj.clamped, (0, 0, 255), 1)

    outputs = {
        "1_patch": add_caption(canvas_patch, f"Step 1: raw {patch_size}x{patch_size} patch (no labels)"),
        "2_rebuilt": add_caption(canvas_rebuilt, "Step 2: minAreaRect rebuild (regular rectangles)"),
        "3_clamped": add_caption(canvas_clamped, "Step 3: clip_points_to_bounds (irregular quads)"),
    }
    saved: dict[str, Path] = {}
    for key, canvas in outputs.items():
        path = out_dir / f"{key.replace('1_patch', '1_patch_256x256').replace('2_rebuilt', '2_patch_minarect_rebuilt').replace('3_clamped', '3_patch_clamped_quad')}.jpg"
        if not cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"Failed to write {path}")
        saved[key] = path
    return saved


def write_selection_info(
    out_dir: Path,
    image_path: Path,
    x0: int,
    y0: int,
    patch_size: int,
    projected: list[ProjectedObject],
    class_names: tuple[str, ...],
) -> Path:
    """Write a human-readable sidecar describing the selected window and its objects."""
    lines = [
        f"source_image: {image_path.name}",
        f"window: x0={x0} y0={y0} size={patch_size}",
        f"patch_size_px: {patch_size}x{patch_size}",
        f"min_iof: {IDE_MIN_IOF}",
        "",
    ]
    for index, obj in enumerate(projected):
        class_name = class_names[obj.class_id] if 0 <= obj.class_id < len(class_names) else str(obj.class_id)
        state = "DEFORMED_BY_CLAMP" if obj.deformed else "inside_only"
        lines.append(
            f"object {index}: class={class_name} iof={obj.iof:.3f} "
            f"rebuilt_area_px={polygon_area(obj.rebuilt):.1f} "
            f"clamped_area_px={polygon_area(obj.clamped):.1f} "
            f"max_vertex_shift_px={obj.displacement:.2f} state={state}"
        )
    info_path = out_dir / "selection_info.txt"
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info_path


# =========================
# CLI 与主流程
# =========================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize the OBB patch pipeline: raw patch -> minAreaRect rebuild -> vertex-clamped quad."
    )
    parser.add_argument(
        "--source-dir", type=Path, default=IDE_SOURCE_DIR, help="Raw data root containing images/ and labels/."
    )
    parser.add_argument("--out-dir", type=Path, default=IDE_OUTPUT_DIR, help="Output directory for the 3 JPEGs.")
    parser.add_argument("--patch-size", type=int, default=IDE_PATCH_SIZE)
    parser.add_argument("--window-step", type=int, default=IDE_WINDOW_STEP)
    parser.add_argument("--min-iof", type=float, default=IDE_MIN_IOF)
    parser.add_argument("--max-scan-images", type=int, default=IDE_MAX_SCAN_IMAGES)
    return parser.parse_args()


def main() -> None:
    """Run the full visualization pipeline."""
    args = parse_args()
    class_to_id = {name: idx for idx, name in enumerate(IDE_CLASS_NAMES)}
    result = find_teaching_window(
        args.source_dir, class_to_id, args.patch_size, args.window_step, args.min_iof, args.max_scan_images
    )
    if result is None:
        raise RuntimeError(
            "No teaching-worthy window found: no boundary-crossing object whose rebuilt rect "
            "sticks out of the patch. Try --max-scan-images 0 (scan all) or a smaller --window-step."
        )
    image_path, x0, y0, projected, _ = result
    saved = visualize_window(image_path, x0, y0, args.patch_size, projected, args.out_dir)
    info_path = write_selection_info(args.out_dir, image_path, x0, y0, args.patch_size, projected, IDE_CLASS_NAMES)
    deformed = [obj for obj in projected if obj.deformed]
    print("=" * 70)
    print(f"source image : {image_path.name}")
    print(f"window       : x={x0} y={y0}")
    print(f"objects kept : {len(projected)} ({len(deformed)} deformed by vertex clamp)")
    for key, path in saved.items():
        print(f"{key:10s}: {path}")
    print(f"selection info: {info_path}")


if __name__ == "__main__":
    main()
