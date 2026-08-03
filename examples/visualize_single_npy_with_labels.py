from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.utils.patches import imread
from examples.multichannel_preview_utils import (
    build_preview_bgr,
    parse_display_channels,
    validate_stretch_percentiles,
)


# =========================
# Quick Config
# 顶部快速配置区
# 直接修改这里的路径后运行即可。
# =========================

# 输入单张图片路径，支持 .npy / .tif / .tiff
# INPUT_NPY_PATH = Path("/mnt/d/Vscode work_place/datasetObjectDetection/augmented_target_patch_dataset_balanced_multiscale/images/train/0027833__full_scaled__5803755e39dc997829e6.tiff")
INPUT_NPY_PATH = Path("/mnt/d/Vscode work_place/datasetObjectDetection/test/images/20040301000900643_00-02107.npy")

# 输入与该图片对应的标签文件路径，通常是 .txt
# INPUT_LABEL_PATH = Path("/mnt/d/Vscode work_place/datasetObjectDetection/augmented_target_patch_dataset_balanced_multiscale/labels/train/0027833__full_scaled__5803755e39dc997829e6.txt")
INPUT_LABEL_PATH = Path("/mnt/d/Vscode work_place/datasetObjectDetection/test/labels/20040301000900643_00-02107.txt")
# 输出可视化图片路径
OUTPUT_VIS_PATH = Path("/mnt/d/Vscode work_place/datasetObjectDetection/labels_huancun/20040301000900643_00-02107.jpg")

# NPY 原始轴顺序
# - "CHW": (channel, height, width)
# - "CWH": (channel, width, height)
# - "HWC": (height, width, channel)
NPY_LAYOUT = "CWH"

# 预览图显示模式
# - "first3": 沿用旧逻辑，直接显示前 3 个通道
# - "manual": 使用 DISPLAY_CHANNELS 手工指定 3 个通道，按 (R, G, B) 顺序解释
# - "rgb_like": 根据 CHANNEL_WAVELENGTHS_NM 自动选取最接近可见光 RGB 的 3 个通道
# - "false_color": 根据 CHANNEL_WAVELENGTHS_NM 自动选取近红外伪彩组合
PREVIEW_MODE = "rgb_like"

# 手工指定 3 个显示通道，按 (R, G, B) 顺序书写，仅在 PREVIEW_MODE="manual" 时使用
#  (4, 2, 1)最接近可见光 RGB
DISPLAY_CHANNELS = (4, 2, 1)

# 8 通道对应的中心波长，默认按 395nm 到 950nm 近似均匀采样
# 切换数据集时需要修改这个
CHANNEL_WAVELENGTHS_NM = (395.0, 474.285714, 553.571429, 632.857143, 712.142857, 791.428571, 870.714286, 950.0)

# 每个显示通道分别执行百分位拉伸，提高伪彩显示对比度
PERCENTILE_STRETCH = (2.0, 98.0)

# 标签格式
# - "auto": 自动识别
# - "raw": 原始标签格式，形如 x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
# - "yolo_obb": YOLO OBB 标签格式，形如 class_id x1 y1 x2 y2 x3 y3 x4 y4，坐标为归一化值
LABEL_FORMAT = "auto"

# 类别名称，仅在 YOLO OBB 标签可视化时使用
CLASS_NAMES = (
    "car",
    "bus",
    "van",
    "awning-bike",
    "truck",
    "tricycle",
    "bike",
    "pedestrian",
)

# 是否显示 difficult=1 的原始标签目标
INCLUDE_DIFFICULT = True


# =========================
# Feature-map Visualization Config
# 特征图可视化配置区
# =========================

# 特征图可视化功能开关
# - False: 只执行默认的单图 + 标签可视化
# - True: 在原有功能之上，额外通过模型推理输出 FeatureProbe 位置的特征热力图
ENABLE_FEATURE_VISUALIZATION = True

# 目标模型 YAML（应在若干位置手工插入 FeatureProbe，例如：
#   - [-1, 1, FeatureProbe, ["p3_after_c3k2"]]
# 训练时 YAML 不需要包含 FeatureProbe）
FEATURE_VIS_MODEL_YAML = Path(
    "/home/mofengwei/ultralytics/ultralytics/cfg/models/26/yolo26-obb-7-F.yaml"
)

# 训练好的权重（.pt），训练所用 YAML 不包含 FeatureProbe
# 原模型yolo26_obb_car_bike_pedestrian_8ch-6
FEATURE_VIS_CHECKPOINT = Path(
    "/mnt/d/Vscode work_place/datasetObjectDetection/checkpoints/yolo26n_obb_7_8ch/weights/best.pt"
)

# 特征图输出目录，脚本会为每个 FeatureProbe 生成一张热力图（可选生成通道网格图）
FEATURE_VIS_OUTPUT_DIR = Path(
    "/mnt/d/Vscode work_place/datasetObjectDetection/labels_huancun/model_heat_maps_obb26n_7_20040301000900643_00-02107"
)

# 推理设备："cuda:0" / "cpu"，指定的 CUDA 不可用时自动回退到 CPU
FEATURE_VIS_DEVICE = "cuda:0"

# 模型最大步长。原图不会缩放；仅当宽高不是该步长的整数倍时，才在右侧和底部做最小补边。
FEATURE_VIS_STRIDE = 32

# 补边像素值，与 Ultralytics 常用的 LetterBox 填充值一致；它只出现在原图区域之外。
FEATURE_VIS_PAD_VALUE = 114

# 模型 head 使用的类别数；默认取 CLASS_NAMES 的长度
FEATURE_VIS_NUM_CLASSES = len(CLASS_NAMES)

# 通道维度聚合方式："mean" | "max" | "l2"
FEATURE_VIS_REDUCTION = "l2"

# 是否额外输出通道网格图（挑选激活强度 top-K 的通道单独渲染）
FEATURE_VIS_CHANNEL_GRID = True

# 通道网格图最多显示的通道数
FEATURE_VIS_CHANNEL_GRID_TOPK = 16

# 覆盖到预览图上的透明度：0.0 = 纯热力图，1.0 = 纯预览图
FEATURE_VIS_OVERLAY_ALPHA = 0.35

# 热力图使用的 OpenCV colormap
FEATURE_VIS_COLORMAP = cv2.COLORMAP_JET

# 是否在热力图上叠加真实标签框
FEATURE_VIS_DRAW_LABELS_ON_HEATMAP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one image (.npy/.tif/.tiff) together with its OBB label file.")
    parser.add_argument("--image", type=str, default=str(INPUT_NPY_PATH), help="Path to one input .npy/.tif/.tiff image.")
    parser.add_argument("--label", type=str, default=str(INPUT_LABEL_PATH), help="Path to one input label .txt file.")
    parser.add_argument("--output", type=str, default=str(OUTPUT_VIS_PATH), help="Path to save visualization image.")
    parser.add_argument("--layout", type=str, default=NPY_LAYOUT, choices=("CHW", "CWH", "HWC"))
    parser.add_argument(
        "--preview-mode",
        type=str,
        default=PREVIEW_MODE,
        choices=("first3", "manual", "rgb_like", "false_color"),
        help="Preview mode for multi-channel images. 中文：多通道图像的可视化模式。",
    )
    parser.add_argument(
        "--display-channels",
        type=str,
        default=",".join(str(idx) for idx in DISPLAY_CHANNELS),
        help="Three channel indices in RGB order, e.g. 4,2,1. 中文：按 RGB 顺序指定三个显示通道。",
    )
    parser.add_argument(
        "--stretch-low",
        type=float,
        default=PERCENTILE_STRETCH[0],
        help="Lower percentile for per-channel stretch. 中文：按通道拉伸的低百分位。",
    )
    parser.add_argument(
        "--stretch-high",
        type=float,
        default=PERCENTILE_STRETCH[1],
        help="Upper percentile for per-channel stretch. 中文：按通道拉伸的高百分位。",
    )
    parser.add_argument("--label-format", type=str, default=LABEL_FORMAT, choices=("auto", "raw", "yolo_obb"))
    parser.add_argument("--include-difficult", action=argparse.BooleanOptionalAction, default=INCLUDE_DIFFICULT)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    image_path = Path(args.image)
    label_path = Path(args.label)
    output_path = Path(args.output)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if image_path.suffix.lower() not in {".npy", ".tif", ".tiff"}:
        raise ValueError(f"Input image must be a .npy/.tif/.tiff file, but got: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Input label file not found: {label_path}")
    if output_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        raise ValueError(f"Unsupported output image suffix: {output_path.suffix}")
    return image_path, label_path, output_path


def load_npy_image(path: Path, layout: str) -> np.ndarray:
    image = np.load(path, allow_pickle=False)
    if image.ndim == 2:
        image = image[..., None]
    elif image.ndim == 3:
        if layout == "CHW":
            image = np.transpose(image, (1, 2, 0))
        elif layout == "CWH":
            image = np.transpose(image, (2, 1, 0))
        elif layout == "HWC":
            pass
        else:
            raise ValueError(f"Unsupported NPY layout: {layout}")
    else:
        raise ValueError(f"Unsupported array rank for {path}: shape={image.shape}")
    return normalize_image_uint8(image)


def load_tiff_image(path: Path) -> np.ndarray:
    image = imread(str(path), flags=cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read TIFF image: {path}")
    if image.ndim == 2:
        image = image[..., None]
    return normalize_image_uint8(image)


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


def load_input_image(path: Path, layout: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return load_npy_image(path, layout)
    if suffix in {".tif", ".tiff"}:
        return load_tiff_image(path)
    raise ValueError(f"Unsupported input image suffix: {suffix}")


def to_preview_bgr(
    image: np.ndarray,
    preview_mode: str,
    display_channels: tuple[int, int, int],
    stretch_low: float,
    stretch_high: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    return build_preview_bgr(
        image=image,
        preview_mode=preview_mode,
        display_channels=display_channels,
        stretch_low=stretch_low,
        stretch_high=stretch_high,
        channel_wavelengths_nm=CHANNEL_WAVELENGTHS_NM,
    )


def parse_raw_label_file(
    label_path: Path, class_names: tuple[str, ...], include_difficult: bool
) -> list[tuple[int, np.ndarray]]:
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    annotations: list[tuple[int, np.ndarray]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"{label_path}:{line_number} has {len(parts)} columns, expected at least 10")
        class_name = parts[8]
        if class_name not in class_to_id:
            continue
        difficult = int(float(parts[9]))
        if difficult and not include_difficult:
            continue
        points = np.array([float(v) for v in parts[:8]], dtype=np.float32).reshape(4, 2)
        annotations.append((class_to_id[class_name], points))
    return annotations


def load_yolo_obb_label_file(label_path: Path, image_h: int, image_w: int) -> list[tuple[int, np.ndarray]]:
    labels: list[tuple[int, np.ndarray]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"{label_path}:{line_number} has {len(parts)} columns, expected at least 9")
        class_id = int(float(parts[0]))
        coords = np.array([float(v) for v in parts[1:9]], dtype=np.float32).reshape(4, 2)
        coords[:, 0] *= image_w
        coords[:, 1] *= image_h
        labels.append((class_id, coords))
    return labels


def detect_label_format(label_path: Path) -> str:
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 10:
            try:
                float(parts[9])
                float(parts[0])
                float(parts[7])
                return "raw"
            except ValueError:
                pass
        if len(parts) >= 9:
            try:
                float(parts[0])
                float(parts[8])
                return "yolo_obb"
            except ValueError:
                pass
    raise ValueError(f"Failed to auto-detect label format for file: {label_path}")


def draw_obb_labels(image: np.ndarray, labels: list[tuple[int, np.ndarray]], class_names: tuple[str, ...]) -> np.ndarray:
    canvas = image.copy()
    for class_id, points in labels:
        polygon = points.astype(np.int32).reshape(-1, 1, 2)
        color = (
            int((37 * (class_id + 1)) % 255),
            int((97 * (class_id + 1)) % 255),
            int((157 * (class_id + 1)) % 255),
        )
        cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=2)
        x, y = polygon[0, 0].tolist()
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        cv2.putText(canvas, class_name, (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


# =========================
# Feature-map Visualization
# =========================


def _round_to_multiple(value: int, multiple: int = 32) -> int:
    return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


def prepare_model_tensor(
    image: np.ndarray, stride: int = 32, pad_value: int = 114
) -> tuple["torch.Tensor", tuple[int, int]]:
    """Convert an image without resizing and minimally pad its bottom/right edges for model inference."""
    import torch
    import torch.nn.functional as F

    if image.ndim == 2:
        image = image[..., None]
    image_h, image_w = image.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    padded_h = _round_to_multiple(image_h, stride)
    padded_w = _round_to_multiple(image_w, stride)
    pad_h = padded_h - image_h
    pad_w = padded_w - image_w
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=float(pad_value) / 255.0)
    return tensor, (image_h, image_w)


def _reduce_feature_map(feature: "torch.Tensor", reduction: str) -> np.ndarray:
    """Reduce a (1, C, H, W) tensor to a (H, W) heatmap in float32."""
    import torch

    if feature.ndim != 4:
        raise ValueError(f"FeatureProbe captured tensor must be 4D, got shape {tuple(feature.shape)}")
    fmap = feature[0].float()
    if reduction == "mean":
        heat = fmap.mean(dim=0)
    elif reduction == "max":
        heat = fmap.amax(dim=0)
    elif reduction == "l2":
        heat = torch.sqrt((fmap ** 2).sum(dim=0) + 1e-12)
    else:
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    return heat.cpu().numpy().astype(np.float32)


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi - lo < 1e-12:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def resize_heatmap_to_image(
    heat_map: np.ndarray, model_input_hw: tuple[int, int], image_hw: tuple[int, int]
) -> np.ndarray:
    """Smoothly upsample a heatmap to model-input space, then remove bottom/right inference padding."""
    model_h, model_w = model_input_hw
    image_h, image_w = image_hw
    if image_h > model_h or image_w > model_w:
        raise ValueError(f"Image size {image_hw} cannot exceed model input size {model_input_hw}")
    heat_uint8 = _normalize_to_uint8(heat_map)
    heat_model = cv2.resize(heat_uint8, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    return heat_model[:image_h, :image_w]


def build_probe_state_dict(
    target_model: "torch.nn.Module", ckpt_state_dict: dict
) -> dict:
    """Remap checkpoint keys onto a model that has extra FeatureProbe layers inserted.

    The checkpoint was trained without FeatureProbe modules, so target indices `i`
    (skipping over probes) map to consecutive checkpoint indices `j`.
    """
    from ultralytics.nn.modules import FeatureProbe

    layers = list(target_model.model)
    # target layer index -> checkpoint layer index (for layers that are not probes)
    idx_map: dict[int, int] = {}
    ckpt_counter = 0
    for target_i, layer in enumerate(layers):
        if isinstance(layer, FeatureProbe):
            continue
        idx_map[target_i] = ckpt_counter
        ckpt_counter += 1

    remapped: dict = {}
    prefix = "model.model."  # keys inside checkpoint typically start like this
    plain_prefix = "model."
    for key, value in ckpt_state_dict.items():
        # normalize away an optional outer "model." wrapper
        if key.startswith(prefix):
            body = key[len(prefix):]
            outer = prefix
        elif key.startswith(plain_prefix):
            body = key[len(plain_prefix):]
            outer = plain_prefix
        else:
            remapped[key] = value
            continue
        # body looks like "<layer_idx>.<rest...>"
        head, _, tail = body.partition(".")
        try:
            ckpt_i = int(head)
        except ValueError:
            remapped[key] = value
            continue
        # find target_i whose ckpt index equals ckpt_i
        target_i = next((ti for ti, ci in idx_map.items() if ci == ckpt_i), None)
        if target_i is None:
            remapped[key] = value
            continue
        remapped[f"{outer}{target_i}.{tail}"] = value
    return remapped


def _extract_state_dict_from_ckpt(ckpt, source_desc: str) -> dict:
    """Return a ``{param_name: tensor}`` mapping from any Ultralytics-style checkpoint.

    Ultralytics saves several variants:
      - training-time: ``{"model": nn.Module, "ema": nn.Module | None, "optimizer": ..., ...}``
      - stripped:     ``{"model": None, "ema": nn.Module, ...}``  (after ``strip_optimizer``)
      - state-only:   plain ``{param_name: tensor, ...}`` (rare, but supported for portability)
    """
    import torch  # noqa: F401 — only for isinstance checks via duck-typing

    def _from_module_or_dict(payload):
        if payload is None:
            return None
        if hasattr(payload, "state_dict"):
            module = payload
            if hasattr(module, "float"):
                module = module.float()
            return module.state_dict()
        if isinstance(payload, dict):
            # heuristics: treat as state_dict if all values look tensor-ish
            if payload and all(hasattr(v, "shape") for v in payload.values()):
                return payload
        return None

    if not isinstance(ckpt, dict):
        result = _from_module_or_dict(ckpt)
        if result is not None:
            return result
        raise TypeError(f"Unsupported checkpoint payload type ({type(ckpt).__name__}) in {source_desc}")

    for key in ("ema", "model"):
        result = _from_module_or_dict(ckpt.get(key))
        if result:
            return result

    # last resort: the outer dict itself might already be a plain state_dict
    result = _from_module_or_dict(ckpt)
    if result is not None:
        return result

    keys_preview = list(ckpt.keys())[:8]
    raise TypeError(
        f"Could not locate a state_dict in checkpoint {source_desc}. "
        f"Top-level keys observed: {keys_preview}"
    )


def load_feature_vis_model(
    yaml_path: Path, checkpoint_path: Path, num_classes: int, num_channels: int, device: "torch.device"
):
    """Build the OBB model from YAML (with FeatureProbes) and load remapped weights."""
    import torch

    from ultralytics.nn.tasks import OBBModel
    from ultralytics.utils.patches import torch_load

    if not yaml_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {yaml_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = OBBModel(cfg=str(yaml_path), ch=num_channels, nc=num_classes, verbose=False)

    ckpt = torch_load(str(checkpoint_path), map_location="cpu")
    ckpt_state = _extract_state_dict_from_ckpt(ckpt, str(checkpoint_path))

    remapped = build_probe_state_dict(model, ckpt_state)
    own_state = model.state_dict()
    filtered = {k: v for k, v in remapped.items() if k in own_state and own_state[k].shape == v.shape}
    missing_after = set(own_state) - set(filtered)
    model.load_state_dict(filtered, strict=False)

    # informative summary
    loaded_from_ckpt = len(filtered)
    print(f"[feature-vis] Loaded {loaded_from_ckpt}/{len(own_state)} tensors from checkpoint into model")
    if missing_after:
        # ignore FeatureProbe modules — they have no params — and running stats we accept as defaults
        non_trivial_missing = [k for k in missing_after if not any(tag in k for tag in ("num_batches_tracked",))]
        if non_trivial_missing:
            print(f"[feature-vis] {len(non_trivial_missing)} parameters kept at model init (likely probe / newly added layers)")

    model.eval().to(device)
    return model


def find_feature_probes(model: "torch.nn.Module") -> list[tuple[int, str, "torch.nn.Module"]]:
    """Return the ordered list of (layer_index, probe_name, module) for every FeatureProbe."""
    from ultralytics.nn.modules import FeatureProbe

    probes: list[tuple[int, str, torch.nn.Module]] = []
    for layer_i, layer in enumerate(model.model):
        if isinstance(layer, FeatureProbe):
            name = layer.probe_name or f"probe_{layer_i}"
            probes.append((layer_i, name, layer))
    return probes


def render_heatmap_overlay(
    heat_map: np.ndarray,
    preview_bgr: np.ndarray,
    model_input_hw: tuple[int, int],
    labels: list[tuple[int, np.ndarray]],
    class_names: tuple[str, ...],
    colormap: int,
    alpha: float,
    draw_labels: bool,
) -> np.ndarray:
    heat_resized = resize_heatmap_to_image(heat_map, model_input_hw, preview_bgr.shape[:2])
    heat_color = cv2.applyColorMap(heat_resized, colormap)
    overlay = cv2.addWeighted(heat_color, 1.0 - alpha, preview_bgr, alpha, 0.0)
    if draw_labels:
        overlay = draw_obb_labels(overlay, labels, class_names)
    return overlay


def render_channel_grid(feature: "torch.Tensor", top_k: int, colormap: int, tile_hw: tuple[int, int]) -> np.ndarray:
    """Render the top-K most active channels as a grid of colored heatmaps."""
    fmap = feature[0].float().cpu().numpy()  # (C, H, W)
    activations = np.abs(fmap).mean(axis=(1, 2))
    order = np.argsort(-activations)
    picked = order[: min(top_k, fmap.shape[0])]

    cols = int(np.ceil(np.sqrt(len(picked))))
    rows = int(np.ceil(len(picked) / cols))
    th, tw = tile_hw
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for tile_idx, ch_idx in enumerate(picked):
        r, c = divmod(tile_idx, cols)
        tile_gray = _normalize_to_uint8(fmap[ch_idx])
        tile_resized = cv2.resize(tile_gray, (tw, th), interpolation=cv2.INTER_LINEAR)
        tile_color = cv2.applyColorMap(tile_resized, colormap)
        cv2.putText(
            tile_color,
            f"c{ch_idx}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        canvas[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile_color
    return canvas


def run_feature_visualization(
    image: np.ndarray,
    preview_bgr: np.ndarray,
    labels: list[tuple[int, np.ndarray]],
    output_stem: str,
) -> None:
    """Run inference through the probe-instrumented model and save one heatmap per probe."""
    import torch

    from ultralytics.nn.modules import FeatureProbe

    device_str = FEATURE_VIS_DEVICE
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[feature-vis] CUDA requested but unavailable, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)

    if image.ndim == 2:
        num_channels = 1
    else:
        num_channels = image.shape[2]

    model = load_feature_vis_model(
        yaml_path=FEATURE_VIS_MODEL_YAML,
        checkpoint_path=FEATURE_VIS_CHECKPOINT,
        num_classes=FEATURE_VIS_NUM_CLASSES,
        num_channels=num_channels,
        device=device,
    )

    probes = find_feature_probes(model)
    if not probes:
        raise RuntimeError(
            f"No FeatureProbe modules found in model YAML: {FEATURE_VIS_MODEL_YAML}. "
            f"Insert lines like `[-1, 1, FeatureProbe, [\"my_probe\"]]` at positions you want to visualize."
        )

    for _, _, probe in probes:
        probe.enable_capture = True
        probe.clear()

    tensor, original_hw = prepare_model_tensor(
        image, stride=FEATURE_VIS_STRIDE, pad_value=FEATURE_VIS_PAD_VALUE
    )
    model_input_hw = tuple(tensor.shape[-2:])
    if model_input_hw == original_hw:
        print(f"[feature-vis] Inference size: {model_input_hw} (original size, no resize or padding)")
    else:
        print(
            f"[feature-vis] Inference size: {model_input_hw}; original content: {original_hw} "
            f"(no resize, bottom/right padding only)"
        )
    tensor = tensor.to(device)
    with torch.inference_mode():
        _ = model(tensor)

    FEATURE_VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for layer_i, probe_name, probe in probes:
        feat = probe.last_feature
        if feat is None:
            print(f"[feature-vis] probe #{layer_i} ({probe_name}) captured nothing — skipped")
            continue
        heat = _reduce_feature_map(feat, FEATURE_VIS_REDUCTION)

        overlay = render_heatmap_overlay(
            heat_map=heat,
            preview_bgr=preview_bgr,
            model_input_hw=model_input_hw,
            labels=labels,
            class_names=CLASS_NAMES,
            colormap=FEATURE_VIS_COLORMAP,
            alpha=FEATURE_VIS_OVERLAY_ALPHA,
            draw_labels=FEATURE_VIS_DRAW_LABELS_ON_HEATMAP,
        )
        overlay_path = FEATURE_VIS_OUTPUT_DIR / f"{output_stem}__probe{layer_i:02d}_{probe_name}_overlay.jpg"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"Failed to save overlay: {overlay_path}")

        heat_resized = resize_heatmap_to_image(heat, model_input_hw, preview_bgr.shape[:2])
        heat_color = cv2.applyColorMap(heat_resized, FEATURE_VIS_COLORMAP)
        heat_path = FEATURE_VIS_OUTPUT_DIR / f"{output_stem}__probe{layer_i:02d}_{probe_name}_heatmap.jpg"
        if not cv2.imwrite(str(heat_path), heat_color):
            raise RuntimeError(f"Failed to save heatmap: {heat_path}")

        if FEATURE_VIS_CHANNEL_GRID:
            grid = render_channel_grid(
                feature=feat,
                top_k=FEATURE_VIS_CHANNEL_GRID_TOPK,
                colormap=FEATURE_VIS_COLORMAP,
                tile_hw=(128, 128),
            )
            grid_path = FEATURE_VIS_OUTPUT_DIR / f"{output_stem}__probe{layer_i:02d}_{probe_name}_channels.jpg"
            if not cv2.imwrite(str(grid_path), grid):
                raise RuntimeError(f"Failed to save channel grid: {grid_path}")

        print(f"[feature-vis] probe #{layer_i} ({probe_name}) shape={tuple(feat.shape)} -> {overlay_path.name}")

    # release cached tensors
    for _, _, probe in probes:
        probe.enable_capture = False
        probe.clear()


def main() -> None:
    args = parse_args()
    image_path, label_path, output_path = validate_args(args)
    display_channels = parse_display_channels(args.display_channels)
    stretch_low, stretch_high = validate_stretch_percentiles(args.stretch_low, args.stretch_high)

    image = load_input_image(image_path, args.layout)
    preview, rgb_channels = to_preview_bgr(
        image=image,
        preview_mode=args.preview_mode,
        display_channels=display_channels,
        stretch_low=stretch_low,
        stretch_high=stretch_high,
    )
    image_h, image_w = preview.shape[:2]

    label_format = args.label_format
    if label_format == "auto":
        label_format = detect_label_format(label_path)

    if label_format == "raw":
        labels = parse_raw_label_file(label_path, CLASS_NAMES, args.include_difficult)
    elif label_format == "yolo_obb":
        labels = load_yolo_obb_label_file(label_path, image_h, image_w)
    else:
        raise ValueError(f"Unsupported label format: {label_format}")

    visualized = draw_obb_labels(preview, labels, CLASS_NAMES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), visualized)
    if not ok:
        raise RuntimeError(f"Failed to save visualization image: {output_path}")

    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    print(f"Preview mode: {args.preview_mode}")
    print(f"Preview RGB channels: R={rgb_channels[0]}, G={rgb_channels[1]}, B={rgb_channels[2]}")
    print(f"Percentile stretch: low={stretch_low:.2f}, high={stretch_high:.2f}")
    print(f"Label format: {label_format}")
    print(f"Objects visualized: {len(labels)}")
    print(f"Saved to: {output_path}")

    if ENABLE_FEATURE_VISUALIZATION:
        print("")
        print(f"[feature-vis] Running model inference to visualize feature maps at FeatureProbe positions")
        print(f"[feature-vis] Model YAML: {FEATURE_VIS_MODEL_YAML}")
        print(f"[feature-vis] Checkpoint: {FEATURE_VIS_CHECKPOINT}")
        run_feature_visualization(
            image=image,
            preview_bgr=preview,
            labels=labels,
            output_stem=image_path.stem,
        )
        print(f"[feature-vis] Feature maps saved under: {FEATURE_VIS_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
