from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.custom_obb_prepare_and_train import (
    DEFAULT_CONFIG,
    load_npy_image,
    parse_raw_label_file,
    sanitize_annotation,
    save_multichannel_tiff,
    write_label_file,
)
from examples.multichannel_preview_utils import (
    build_preview_bgr,
    parse_display_channels,
    validate_stretch_percentiles,
)
from ultralytics import YOLO
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import nms, ops
from ultralytics.utils.metrics import batch_probiou

# =========================
# Validation Config
# 验证参数配置区
# 直接修改这里即可切换权重、数据集路径与验证超参数。
# 默认值参考 custom_obb_prepare_and_train.py 与 Ultralytics OBB 官方验证默认配置。
# =========================

# VAL_MODEL_WEIGHTS = DEFAULT_CONFIG.save_dir / DEFAULT_CONFIG.run_name / "weights" / "best.pt"
# VAL_MODEL_WEIGHTS = "/mnt/d/Vscode work_place/datasetObjectDetection/checkpoints/yolo26_obb_car_bike_pedestrian_8ch-7/weights/best.pt"
VAL_MODEL_WEIGHTS = "/mnt/d/Vscode work_place/datasetObjectDetection/checkpoints/yolo26n_obb_5_8ch/weights/best.pt"
VAL_DATA_YAML = DEFAULT_CONFIG.data_yaml

# VAL_CONF = DEFAULT_CONFIG.metric_eval.conf
# 图上的“all classes 0.61 at 0.304”意指：部署时置信阈值最佳为0.304
VAL_CONF = 0.01
VAL_NMS_IOU = DEFAULT_CONFIG.metric_eval.iou
VAL_BATCH = 1
VAL_DEVICE = str(DEFAULT_CONFIG.device) if DEFAULT_CONFIG.device is not None else "cpu"
VAL_RESULTS_ROOT = REPO_ROOT / "runs" / "trained_obb_validation"

# 额外验证与归档参数
VAL_MAX_DET = DEFAULT_CONFIG.metric_eval.max_det
VAL_AGNOSTIC_NMS = DEFAULT_CONFIG.metric_eval.agnostic_nms
VAL_SPLIT = "val"
VAL_WORKERS = DEFAULT_CONFIG.workers
VAL_HALF = False
VAL_PLOTS = True
VAL_SAVE_JSON = False
VAL_ERROR_MATCH_IOU = 0.50
VAL_MAX_ERROR_SAMPLES = 30

# 数据集来源模式
# - "prepared": 直接验证 prepared_Dataset/data.yaml
# - "raw_full_image": 把原始完整图 NPY + raw txt 临时转换成整图版 YOLO OBB 数据集后再验证
VAL_DATASET_MODE = "raw_full_image"
# VAL_IMGSZ = max(DEFAULT_CONFIG.patch_size)  # "prepared"
VAL_IMGSZ = 1200  # "raw_full_image"


# 原始完整图验证模式下使用的输入目录
VAL_RAW_IMAGE_DIR = DEFAULT_CONFIG.val.image_dir
VAL_RAW_LABEL_DIR = DEFAULT_CONFIG.val.label_dir
VAL_RAW_INCLUDE_DIFFICULT = DEFAULT_CONFIG.val.include_difficult

# 多通道图像可视化配置
# - "first3": 直接显示前 3 个通道
# - "manual": 使用 VAL_DISPLAY_CHANNELS 手工指定 3 个通道，按 (R, G, B) 顺序解释
# - "rgb_like": 根据 VAL_CHANNEL_WAVELENGTHS_NM 自动选取最接近可见光 RGB 的 3 个通道
# - "false_color": 根据 VAL_CHANNEL_WAVELTHS_NM 自动选取近红外伪彩组合
VAL_PREVIEW_MODE = "rgb_like"
# 仅在 VAL_PREVIEW_MODE="manual" 时使用，按 (R, G, B) 顺序书写
VAL_DISPLAY_CHANNELS = (4, 2, 1)
# 8 通道对应的中心波长，默认按 395nm 到 950nm 近似均匀采样
# 切换数据集时需要修改这个
VAL_CHANNEL_WAVELENGTHS_NM = (395.0, 474.285714, 553.571429, 632.857143, 712.142857, 791.428571, 870.714286, 950.0)
# 按显示通道分别执行百分位拉伸，提高伪彩显示对比度
VAL_PERCENTILE_STRETCH = (2.0, 98.0)


# =========================
# 分析的数据集完全由上面的原配置控制
# Error Analysis Export Config
# 错误分析导出配置区
# 说明：
# - 当 ENABLE_ERROR_ANALYSIS_EXPORT=False 时，保留原有验证逻辑不变。
# - 当 ENABLE_ERROR_ANALYSIS_EXPORT=True 时，仅执行按类漏检/误检图片导出与分析功能。
# =========================
# 是否启用错误分析导出模式。False=正常跑原验证逻辑；True=仅运行新增的错误分析导出功能。
ENABLE_ERROR_ANALYSIS_EXPORT = False
# 需要分析的目标类别列表，支持类别名、类别ID，或二者混用；例如 ("bike", "pedestrian")、(6, 7)。
#  单个类别时，输入("truck",)，而不是("truck")
TARGET_CLASSES: tuple[str | int, ...] = ("truck",)
# 最多导出多少张符合条件的图片。None=不限制；正整数=仅导出前 N 张。
MAX_EXPORT_IMAGES: int | None = None
# 需要导出的错误类型，可选："false_negative"/"漏检"、"false_positive"/"误检"，或同时配置两者。
ERROR_TYPES_TO_EXPORT: tuple[str, ...] = ("false_negative", "false_positive")
# 错误分析输出根目录；程序会在该目录下自动创建本次运行的时间戳子目录及其子目录。
ERROR_ANALYSIS_OUTPUT_ROOT = REPO_ROOT / "runs" / "trained_obb_error_analysis"

ERROR_TYPE_FALSE_NEGATIVE = "false_negative"
ERROR_TYPE_FALSE_POSITIVE = "false_positive"
ERROR_TYPE_ALIASES = {
    ERROR_TYPE_FALSE_NEGATIVE: ERROR_TYPE_FALSE_NEGATIVE,
    "fn": ERROR_TYPE_FALSE_NEGATIVE,
    "漏检": ERROR_TYPE_FALSE_NEGATIVE,
    ERROR_TYPE_FALSE_POSITIVE: ERROR_TYPE_FALSE_POSITIVE,
    "fp": ERROR_TYPE_FALSE_POSITIVE,
    "误检": ERROR_TYPE_FALSE_POSITIVE,
}


# =========================
# VAL_CONF Sweep Config
# VAL_CONF 扫描绘图配置区
# 说明：
# - 当 ENABLE_VAL_CONF_SWEEP_EXPORT=False 时，不启用该链路。
# - 当 ENABLE_VAL_CONF_SWEEP_EXPORT=True 时，仅执行按 VAL_CONF 扫描并输出8张总体指标关系图。
# - `VAL_CONF_TARGET_CLASSES` 可额外指定单类/多类/ALL，并在子文件夹中为每个目标各自输出4张官方指标关系图。
# - `ALL` 表示自动展开数据集中的全部类别，等价于把所有类别名按多类形式全部输入。
# - `VAL_CONF_TARGET_CLASSES=None` 或 `VAL_CONF_TARGET_CLASSES=()` 时，不额外导出按目标拆分的图。
# =========================
ENABLE_VAL_CONF_SWEEP_EXPORT = False
VAL_CONF_SWEEP_START = 0.01
VAL_CONF_SWEEP_END = 0.30
VAL_CONF_SWEEP_STEP = 0.02
VAL_CONF_SWEEP_OUTPUT_ROOT = REPO_ROOT / "runs" / "trained_obb_conf_sweep"
VAL_CONF_TARGET_CLASSES: tuple[str | int, ...] | None = ("ALL",)


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    enabled: bool
    target_classes: tuple[str | int, ...]
    max_export_images: int | None
    error_types_to_export: tuple[str, ...]
    output_root: Path


@dataclass(frozen=True)
class ConfSweepConfig:
    enabled: bool
    start: float
    end: float
    step: float
    output_root: Path
    target_classes: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class ConfSweepTargetSelection:
    key: str
    kind: str
    display_name: str
    folder_name: str
    class_id: int | None = None


@dataclass(frozen=True)
class ValidationConfig:
    weights: Path
    data: Path
    imgsz: int
    conf: float
    iou: float
    batch: int
    device: str
    results_root: Path
    max_det: int
    agnostic_nms: bool
    split: str
    workers: int
    half: bool
    plots: bool
    save_json: bool
    error_match_iou: float
    max_error_samples: int
    run_name: str
    dataset_mode: str
    raw_image_dir: Path
    raw_label_dir: Path
    raw_include_difficult: bool
    preview_mode: str
    display_channels: tuple[int, int, int]
    channel_wavelengths_nm: tuple[float, ...]
    stretch_low: float
    stretch_high: float
    error_analysis: ErrorAnalysisConfig
    conf_sweep: ConfSweepConfig


@dataclass(frozen=True)
class FalseAlarmImageStats:
    gt_count: int
    candidate_prediction_count: int
    negative_candidate_count: int
    passed_conf_count: int
    true_positive_count: int
    false_positive_count: int
    true_negative_count: int
    false_alarm_rate: float | None


@dataclass(frozen=True)
class FalseAlarmDatasetStats:
    image_count: int
    image_count_with_valid_denominator: int
    candidate_prediction_total: int
    negative_candidate_total: int
    passed_conf_total: int
    true_positive_total: int
    false_positive_total: int
    true_negative_total: int
    false_alarm_rate: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained YOLO OBB model, report official metrics and custom false-detection metrics."
    )
    parser.add_argument("--weights", type=str, default=str(VAL_MODEL_WEIGHTS), help="Path to trained .pt weights.")
    parser.add_argument("--data", type=str, default=str(VAL_DATA_YAML), help="Path to dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=VAL_IMGSZ, help="Validation image size.")
    parser.add_argument("--conf", type=float, default=VAL_CONF, help="Confidence threshold for validation.")
    parser.add_argument("--iou", type=float, default=VAL_NMS_IOU, help="NMS IoU threshold for validation.")
    parser.add_argument("--batch", type=int, default=VAL_BATCH, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=VAL_DEVICE, help="Validation device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--results-root", type=str, default=str(VAL_RESULTS_ROOT), help="Directory to save results.")
    parser.add_argument("--max-det", type=int, default=VAL_MAX_DET, help="Maximum detections per image.")
    parser.add_argument(
        "--agnostic-nms",
        action=argparse.BooleanOptionalAction,
        default=VAL_AGNOSTIC_NMS,
        help="Whether to use class-agnostic NMS.",
    )
    parser.add_argument("--split", type=str, default=VAL_SPLIT, choices=("val", "test", "train"))
    parser.add_argument("--workers", type=int, default=VAL_WORKERS)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=VAL_HALF)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=VAL_PLOTS)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=VAL_SAVE_JSON)
    parser.add_argument("--error-match-iou", type=float, default=VAL_ERROR_MATCH_IOU)
    parser.add_argument("--max-error-samples", type=int, default=VAL_MAX_ERROR_SAMPLES)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--dataset-mode", type=str, default=VAL_DATASET_MODE, choices=("prepared", "raw_full_image"))
    parser.add_argument("--raw-image-dir", type=str, default=str(VAL_RAW_IMAGE_DIR))
    parser.add_argument("--raw-label-dir", type=str, default=str(VAL_RAW_LABEL_DIR))
    parser.add_argument(
        "--raw-include-difficult",
        action=argparse.BooleanOptionalAction,
        default=VAL_RAW_INCLUDE_DIFFICULT,
        help="Whether to keep difficult=1 objects when validating the raw full-image dataset.",
    )
    parser.add_argument(
        "--preview-mode",
        type=str,
        default=VAL_PREVIEW_MODE,
        choices=("first3", "manual", "rgb_like", "false_color"),
        help="Preview mode for multi-channel validation visualizations. 中文：验证可视化的多通道预览模式。",
    )
    parser.add_argument(
        "--display-channels",
        type=str,
        default=",".join(str(idx) for idx in VAL_DISPLAY_CHANNELS),
        help="Three channel indices in RGB order, e.g. 4,2,1. 中文：按 RGB 顺序指定三个显示通道。",
    )
    parser.add_argument(
        "--stretch-low",
        type=float,
        default=VAL_PERCENTILE_STRETCH[0],
        help="Lower percentile for per-channel stretch. 中文：按通道拉伸的低百分位。",
    )
    parser.add_argument(
        "--stretch-high",
        type=float,
        default=VAL_PERCENTILE_STRETCH[1],
        help="Upper percentile for per-channel stretch. 中文：按通道拉伸的高百分位。",
    )
    parser.add_argument(
        "--enable-error-analysis-export",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_ERROR_ANALYSIS_EXPORT,
        help="Whether to run the target-class false-positive/false-negative export mode.",
    )
    parser.add_argument(
        "--target-classes",
        nargs="*",
        default=list(_normalize_target_class_specs(TARGET_CLASSES)),
        help="Target classes for error analysis. Supports class names, ids, or a mix of both.",
    )
    parser.add_argument(
        "--max-export-images",
        type=int,
        default=MAX_EXPORT_IMAGES,
        help="Maximum number of exported images for error analysis. Use None in config area for no limit.",
    )
    parser.add_argument(
        "--error-types-to-export",
        nargs="*",
        default=list(ERROR_TYPES_TO_EXPORT),
        help="Error types to export for error analysis. Supports false_negative/漏检 and false_positive/误检.",
    )
    parser.add_argument(
        "--error-analysis-output-root",
        type=str,
        default=str(ERROR_ANALYSIS_OUTPUT_ROOT),
        help="Root directory for the new error-analysis export outputs.",
    )
    parser.add_argument(
        "--enable-val-conf-sweep-export",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_VAL_CONF_SWEEP_EXPORT,
        help="Whether to run the VAL_CONF sweep mode that only saves four metric-vs-conf plots.",
    )
    parser.add_argument("--val-conf-sweep-start", type=float, default=VAL_CONF_SWEEP_START)
    parser.add_argument("--val-conf-sweep-end", type=float, default=VAL_CONF_SWEEP_END)
    parser.add_argument("--val-conf-sweep-step", type=float, default=VAL_CONF_SWEEP_STEP)
    parser.add_argument(
        "--val-conf-sweep-output-root",
        type=str,
        default=str(VAL_CONF_SWEEP_OUTPUT_ROOT),
        help="Root directory for the VAL_CONF sweep plots.",
    )
    parser.add_argument(
        "--val-conf-target-classes",
        nargs="*",
        default=list(_normalize_conf_sweep_target_class_specs(VAL_CONF_TARGET_CLASSES)),
        help=(
            "Extra target selections for VAL_CONF sweep official plots. "
            "Supports class names, ids, or ALL. ALL expands to every dataset class. Leave empty to disable extra target plots."
        ),
    )
    return parser.parse_args()


def _normalize_target_class_spec(value: Any) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("target class spec cannot be empty")
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _normalize_target_class_specs(values: Any) -> tuple[str | int, ...]:
    if values is None:
        return tuple()
    if isinstance(values, (str, int)):
        values = (values,)
    normalized: list[str | int] = []
    for value in values:
        normalized.append(_normalize_target_class_spec(value))
    return tuple(normalized)


def _normalize_conf_sweep_target_class_specs(values: Any) -> tuple[str | int, ...]:
    normalized_specs = _normalize_target_class_specs(values)
    normalized: list[str | int] = []
    for spec in normalized_specs:
        if isinstance(spec, str):
            lowered = spec.strip().lower()
            if lowered == "all":
                canonical: str | int = "ALL"
            else:
                canonical = spec.strip()
        else:
            canonical = spec
        if canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def _normalize_error_types(values: list[Any] | tuple[Any, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in values:
        key = str(raw).strip().lower()
        if not key:
            continue
        canonical = ERROR_TYPE_ALIASES.get(key)
        if canonical is None:
            raise ValueError(
                f"Unsupported error type '{raw}'. Supported values: false_negative/漏检, false_positive/误检."
            )
        if canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def build_config(args: argparse.Namespace) -> ValidationConfig:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"val_{timestamp}"
    display_channels = parse_display_channels(args.display_channels)
    stretch_low, stretch_high = validate_stretch_percentiles(args.stretch_low, args.stretch_high)
    return ValidationConfig(
        weights=Path(args.weights),
        data=Path(args.data),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        results_root=Path(args.results_root),
        max_det=args.max_det,
        agnostic_nms=bool(args.agnostic_nms),
        split=args.split,
        workers=args.workers,
        half=bool(args.half),
        plots=bool(args.plots),
        save_json=bool(args.save_json),
        error_match_iou=args.error_match_iou,
        max_error_samples=args.max_error_samples,
        run_name=run_name,
        dataset_mode=args.dataset_mode,
        raw_image_dir=Path(args.raw_image_dir),
        raw_label_dir=Path(args.raw_label_dir),
        raw_include_difficult=bool(args.raw_include_difficult),
        preview_mode=args.preview_mode,
        display_channels=display_channels,
        channel_wavelengths_nm=tuple(float(value) for value in VAL_CHANNEL_WAVELENGTHS_NM),
        stretch_low=stretch_low,
        stretch_high=stretch_high,
        error_analysis=ErrorAnalysisConfig(
            enabled=bool(args.enable_error_analysis_export),
            target_classes=_normalize_target_class_specs(args.target_classes),
            max_export_images=args.max_export_images,
            error_types_to_export=_normalize_error_types(args.error_types_to_export),
            output_root=Path(args.error_analysis_output_root),
        ),
        conf_sweep=ConfSweepConfig(
            enabled=bool(args.enable_val_conf_sweep_export),
            start=float(args.val_conf_sweep_start),
            end=float(args.val_conf_sweep_end),
            step=float(args.val_conf_sweep_step),
            output_root=Path(args.val_conf_sweep_output_root),
            target_classes=_normalize_conf_sweep_target_class_specs(args.val_conf_target_classes),
        ),
    )


def validate_config(cfg: ValidationConfig) -> None:
    if not cfg.weights.exists():
        raise FileNotFoundError(f"Weight file not found: {cfg.weights}")
    if cfg.dataset_mode not in {"prepared", "raw_full_image"}:
        raise ValueError(f"Unsupported dataset_mode: {cfg.dataset_mode}")
    if cfg.dataset_mode == "prepared" and not cfg.data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {cfg.data}")
    if cfg.dataset_mode == "raw_full_image":
        if not cfg.raw_image_dir.exists():
            raise FileNotFoundError(f"Raw image directory not found: {cfg.raw_image_dir}")
        if not cfg.raw_label_dir.exists():
            raise FileNotFoundError(f"Raw label directory not found: {cfg.raw_label_dir}")
    if cfg.imgsz < 32:
        raise ValueError(f"imgsz must be >= 32, but got {cfg.imgsz}")
    if not 0.0 <= cfg.conf <= 1.0:
        raise ValueError(f"conf must be within [0, 1], but got {cfg.conf}")
    if not 0.0 <= cfg.iou <= 1.0:
        raise ValueError(f"iou must be within [0, 1], but got {cfg.iou}")
    if cfg.batch < 1:
        raise ValueError(f"batch must be >= 1, but got {cfg.batch}")
    if cfg.max_det < 1:
        raise ValueError(f"max_det must be >= 1, but got {cfg.max_det}")
    if cfg.workers < 0:
        raise ValueError(f"workers must be >= 0, but got {cfg.workers}")
    if not 0.0 <= cfg.error_match_iou <= 1.0:
        raise ValueError(f"error_match_iou must be within [0, 1], but got {cfg.error_match_iou}")
    if cfg.max_error_samples < 0:
        raise ValueError(f"max_error_samples must be >= 0, but got {cfg.max_error_samples}")
    if cfg.error_analysis.max_export_images is not None and cfg.error_analysis.max_export_images < 1:
        raise ValueError(
            f"error_analysis.max_export_images must be >= 1 or None, but got {cfg.error_analysis.max_export_images}"
        )
    if cfg.error_analysis.enabled and cfg.conf_sweep.enabled:
        raise ValueError("enable_error_analysis_export and enable_val_conf_sweep_export cannot both be True.")
    if cfg.error_analysis.enabled:
        if not cfg.error_analysis.target_classes:
            raise ValueError("At least one target class must be configured when error-analysis export is enabled.")
        if not cfg.error_analysis.error_types_to_export:
            raise ValueError("At least one error type must be configured when error-analysis export is enabled.")
    if cfg.conf_sweep.enabled:
        if not 0.0 <= cfg.conf_sweep.start <= 1.0:
            raise ValueError(f"conf_sweep.start must be within [0, 1], but got {cfg.conf_sweep.start}")
        if not 0.0 <= cfg.conf_sweep.end <= 1.0:
            raise ValueError(f"conf_sweep.end must be within [0, 1], but got {cfg.conf_sweep.end}")
        if cfg.conf_sweep.end < cfg.conf_sweep.start:
            raise ValueError(
                f"conf_sweep.end must be >= conf_sweep.start, but got start={cfg.conf_sweep.start}, end={cfg.conf_sweep.end}"
            )
        if cfg.conf_sweep.step <= 0:
            raise ValueError(f"conf_sweep.step must be > 0, but got {cfg.conf_sweep.step}")
    device = cfg.device.strip().lower()
    if device != "cpu":
        wants_cuda = device.isdigit() or device.startswith("cuda") or "," in device
        if wants_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device '{cfg.device}', but torch.cuda.is_available() is False. "
                "Please switch to '--device cpu' or run on a CUDA-enabled environment."
            )


def setup_logger(run_dir: Path, log_filename: str | None = "validation.log") -> logging.Logger:
    logger_name = f"trained_obb_validation.{run_dir.name}.{log_filename or 'console_only'}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_filename:
        file_handler = logging.FileHandler(run_dir / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def load_dataset_names(data_yaml_path: Path) -> dict[int, str]:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    names = payload.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError(f"Unsupported dataset names format in {data_yaml_path}: {type(names)!r}")


def write_config_json(cfg: ValidationConfig, run_dir: Path) -> Path:
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        return value

    payload = _to_jsonable(asdict(cfg))
    path = run_dir / "validation_config.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def load_data_yaml_payload(data_yaml_path: Path) -> dict[str, Any]:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_split_label_dir(data_yaml_path: Path, split: str) -> Path:
    payload = load_data_yaml_payload(data_yaml_path)
    split_entry = payload.get(split)
    if not split_entry:
        raise ValueError(f"Dataset yaml does not define split '{split}': {data_yaml_path}")
    dataset_root = Path(payload.get("path", data_yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()
    split_path = Path(split_entry)
    split_path = split_path if split_path.is_absolute() else (dataset_root / split_path)
    split_path_str = str(split_path)
    if f"{Path('/') if split_path.is_absolute() else ''}images{Path('/').as_posix()}" in split_path_str:
        label_path_str = split_path_str.replace("/images/", "/labels/")
    else:
        label_path_str = str(dataset_root / "labels" / split)
    return Path(label_path_str)


def count_dataset_instances_by_class(data_yaml_path: Path, split: str, class_names: dict[int, str]) -> dict[int, int]:
    label_dir = resolve_split_label_dir(data_yaml_path, split)
    if not label_dir.exists():
        raise FileNotFoundError(f"Resolved label directory does not exist: {label_dir}")
    counts = {class_id: 0 for class_id in sorted(class_names)}
    for label_path in sorted(label_dir.glob("*.txt")):
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to read label file {label_path}: {exc}") from exc
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            try:
                class_id = int(float(parts[0]))
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid class id in {label_path}:{line_no}: '{line}' ({exc})") from exc
            counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def generate_class_distribution_chart(
    run_dir: Path, class_names: dict[int, str], counts: dict[int, int], logger: logging.Logger
) -> Path:
    output_path = run_dir / "class_instance_distribution.jpg"
    width, height = 1800, 900
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    palette = [
        (66, 133, 244),
        (219, 68, 55),
        (244, 180, 0),
        (15, 157, 88),
        (171, 71, 188),
        (0, 172, 193),
        (255, 112, 67),
        (124, 179, 66),
        (92, 107, 192),
        (141, 110, 99),
    ]

    cv2.putText(
        canvas, "Class Instance Distribution", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA
    )

    items = [(class_id, class_names[class_id], counts.get(class_id, 0)) for class_id in sorted(class_names)]
    total_instances = sum(value for _, _, value in items)

    # Left: bar chart
    left_x0, left_y0, left_x1, left_y1 = 60, 110, 980, 820
    cv2.rectangle(canvas, (left_x0, left_y0), (left_x1, left_y1), (230, 230, 230), 1)
    cv2.putText(canvas, "Bar Chart", (left_x0, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2, cv2.LINE_AA)
    chart_h = left_y1 - left_y0 - 60
    chart_w = left_x1 - left_x0 - 50
    origin_x, origin_y = left_x0 + 25, left_y1 - 35
    cv2.line(canvas, (origin_x, left_y0 + 10), (origin_x, origin_y), (120, 120, 120), 2)
    cv2.line(canvas, (origin_x, origin_y), (origin_x + chart_w, origin_y), (120, 120, 120), 2)
    max_count = max([count for _, _, count in items] + [1])
    bar_gap = 16
    bar_w = max(20, int((chart_w - bar_gap * (len(items) + 1)) / max(len(items), 1)))
    for idx, (_, class_name, count) in enumerate(items):
        x0 = origin_x + bar_gap + idx * (bar_w + bar_gap)
        bar_h = int((count / max_count) * (chart_h - 35))
        y0 = origin_y - bar_h
        color = palette[idx % len(palette)]
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, origin_y), color, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, origin_y), (90, 90, 90), 1)
        cv2.putText(
            canvas,
            str(count),
            (x0 - 4, max(y0 - 8, left_y0 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        label = class_name if len(class_name) <= 12 else class_name[:11] + "."
        cv2.putText(
            canvas, label, (x0 - 8, origin_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA
        )

    # Right: pie chart + legend
    right_x0, _right_y0 = 1060, 120
    cv2.putText(canvas, "Pie Chart", (right_x0, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2, cv2.LINE_AA)
    center = (1315, 430)
    radius = 240
    if total_instances > 0:
        start_angle = 0.0
        for idx, (_, _, count) in enumerate(items):
            if count <= 0:
                continue
            angle = 360.0 * count / total_instances
            color = palette[idx % len(palette)]
            cv2.ellipse(canvas, center, (radius, radius), 0, start_angle, start_angle + angle, color, -1)
            start_angle += angle
        cv2.circle(canvas, center, radius, (120, 120, 120), 2)
    else:
        cv2.circle(canvas, center, radius, (220, 220, 220), -1)
        cv2.circle(canvas, center, radius, (120, 120, 120), 2)
        cv2.putText(
            canvas,
            "No Instances",
            (center[0] - 80, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )

    legend_x, legend_y = 1060, 700
    for idx, (_, class_name, count) in enumerate(items):
        color = palette[idx % len(palette)]
        y = legend_y + idx * 24
        ratio = (count / total_instances * 100.0) if total_instances > 0 else 0.0
        cv2.rectangle(canvas, (legend_x, y - 12), (legend_x + 18, y + 6), color, -1)
        cv2.rectangle(canvas, (legend_x, y - 12), (legend_x + 18, y + 6), (90, 90, 90), 1)
        legend_text = f"{class_name}: {count} ({ratio:.1f}%)"
        cv2.putText(
            canvas, legend_text, (legend_x + 28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA
        )

    footer = f"Total instances: {total_instances}"
    cv2.putText(canvas, footer, (1060, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2, cv2.LINE_AA)

    ok = cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if not ok:
        raise RuntimeError(f"Failed to save class distribution chart: {output_path}")
    logger.info(f"Saved class instance distribution chart: {output_path}")
    return output_path


def write_runtime_data_yaml(
    dataset_root: Path, split: str, class_names: tuple[str, ...], channels: int, dataset_mode: str
) -> Path:
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        f"path: {dataset_root}\n"
        f"train: images/{split}\n"
        f"val: images/{split}\n"
        f"test: images/{split}\n"
        f"channels: {channels}\n"
        f"names:\n{names_block}\n"
        f"dataset_mode: {dataset_mode}\n"
    )
    yaml_path = dataset_root / "data.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def build_raw_full_image_dataset(cfg: ValidationConfig, run_dir: Path) -> Path:
    dataset_root = run_dir / "raw_full_image_dataset"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    image_out_dir = dataset_root / "images" / cfg.split
    label_out_dir = dataset_root / "labels" / cfg.split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    class_names = tuple(DEFAULT_CONFIG.class_names)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    saved_images = 0
    saved_labels = 0
    channel_count: int | None = None

    for image_path in sorted(cfg.raw_image_dir.glob("*.npy")):
        image = load_npy_image(image_path)
        image_h, image_w = image.shape[:2]
        if channel_count is None:
            channel_count = int(image.shape[2]) if image.ndim == 3 else 1

        raw_label_path = cfg.raw_label_dir / f"{image_path.stem}.txt"
        raw_annotations = parse_raw_label_file(raw_label_path, class_to_id, cfg.raw_include_difficult)
        kept_labels: list[tuple[int, np.ndarray]] = []
        for class_id, points in raw_annotations:
            sanitized = sanitize_annotation(points, image_w, image_h)
            if sanitized is not None:
                kept_labels.append((class_id, sanitized))

        save_multichannel_tiff(image_out_dir / f"{image_path.stem}.tiff", image)
        saved_images += 1
        if kept_labels:
            write_label_file(label_out_dir / f"{image_path.stem}.txt", kept_labels, image_h, image_w)
            saved_labels += 1

    if saved_images == 0:
        raise RuntimeError(f"No raw .npy images found under: {cfg.raw_image_dir}")
    if channel_count is None:
        raise RuntimeError(f"Failed to infer channel count from raw images under: {cfg.raw_image_dir}")

    data_yaml = write_runtime_data_yaml(
        dataset_root=dataset_root,
        split=cfg.split,
        class_names=class_names,
        channels=channel_count,
        dataset_mode="raw_full_image",
    )
    meta_path = dataset_root / "conversion_summary.json"
    meta_path.write_text(
        json.dumps(
            {
                "source_image_dir": str(cfg.raw_image_dir),
                "source_label_dir": str(cfg.raw_label_dir),
                "split": cfg.split,
                "include_difficult": cfg.raw_include_difficult,
                "saved_images": saved_images,
                "saved_label_files": saved_labels,
                "channels": channel_count,
                "data_yaml": str(data_yaml),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return data_yaml


def resolve_validation_data_yaml(cfg: ValidationConfig, run_dir: Path, logger: logging.Logger) -> Path:
    if cfg.dataset_mode == "prepared":
        logger.info(f"Using prepared dataset yaml directly: {cfg.data}")
        return cfg.data
    data_yaml = build_raw_full_image_dataset(cfg, run_dir)
    logger.info(
        "Built temporary raw full-image validation dataset: "
        f"raw_image_dir={cfg.raw_image_dir}, raw_label_dir={cfg.raw_label_dir}, data_yaml={data_yaml}"
    )
    return data_yaml


def draw_obb_polygon(
    image: np.ndarray, box_xywhr: np.ndarray, color: tuple[int, int, int], label: str | None = None
) -> None:
    polygon = ops.xywhr2xyxyxyxy(torch.from_numpy(box_xywhr[None]).float()).view(4, 2).cpu().numpy().astype(np.int32)
    cv2.polylines(image, [polygon.reshape(-1, 1, 2)], True, color, 2)
    if label:
        text_origin = _resolve_label_origin(polygon, image.shape)
        _draw_box_label(image, label, text_origin, color)


def _resolve_label_origin(polygon: np.ndarray, image_shape: tuple[int, ...]) -> tuple[int, int]:
    x_min = int(np.min(polygon[:, 0]))
    y_min = int(np.min(polygon[:, 1]))
    x_max = int(np.max(polygon[:, 0]))
    h, w = image_shape[:2]
    anchor_x = max(2, min(x_min, w - 2))
    if y_min >= 18:
        anchor_y = y_min - 6
    else:
        anchor_y = min(h - 4, int(np.max(polygon[:, 1])) + 16)
    if anchor_x >= x_max:
        anchor_x = max(2, x_min - 4)
    return anchor_x, anchor_y


def _draw_box_label(
    image: np.ndarray, label: str, origin: tuple[int, int], color: tuple[int, int, int], font_scale: float = 0.45
) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x, y = origin
    x = max(2, min(x, image.shape[1] - text_w - 4))
    y = max(text_h + 4, min(y, image.shape[0] - baseline - 2))
    top_left = (x - 2, y - text_h - 2)
    bottom_right = (x + text_w + 2, y + baseline + 2)
    cv2.rectangle(image, top_left, bottom_right, color, -1)
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def _resolve_class_name(names: dict[int, str] | list[str] | tuple[str, ...], class_id: int) -> str:
    if hasattr(names, "get"):
        return str(names.get(class_id, str(class_id)))
    return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)


def resize_preview_tile(
    image: np.ndarray, target_size: tuple[int, int] = (640, 640)
) -> tuple[np.ndarray, float, int, int]:
    target_h, target_w = target_size
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid preview image shape: {image.shape}")
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, round(src_w * scale))
    resized_h = max(1, round(src_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return canvas, float(scale), int(pad_x), int(pad_y)


def remap_boxes_to_resized_tile(boxes_xywhr: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    if boxes_xywhr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    remapped = boxes_xywhr.astype(np.float32, copy=True)
    remapped[:, 0] = remapped[:, 0] * scale + pad_x
    remapped[:, 1] = remapped[:, 1] * scale + pad_y
    remapped[:, 2] *= scale
    remapped[:, 3] *= scale
    return remapped


def load_plot_preview_canvas_and_boxes(
    batch_img: torch.Tensor,
    pbatch: dict[str, Any],
    boxes_xywhr: np.ndarray,
    cfg: ValidationConfig,
    logger: logging.Logger,
    warning_context: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        canvas = load_original_preview_image(Path(pbatch["im_file"]), cfg)
        scaled_boxes = scale_obb_boxes_to_original(
            boxes_xywhr, pbatch["imgsz"], pbatch["ori_shape"], pbatch["ratio_pad"]
        )
        return canvas, scaled_boxes
    except Exception as exc:
        logger.warning(f"读取原图失败，{warning_context} 回退到验证输入可视化：image={pbatch['im_file']}, reason={exc}")
        fallback_canvas = tensor_to_preview_bgr(batch_img, cfg)
        return fallback_canvas, boxes_xywhr.astype(np.float32, copy=False)


def build_validation_batch_mosaic(tiles: list[np.ndarray]) -> np.ndarray:
    if not tiles:
        raise ValueError("Expected at least one tile to build validation mosaic.")
    tile_h, tile_w = tiles[0].shape[:2]
    grid_cols = max(1, math.ceil(math.sqrt(len(tiles))))
    grid_rows = max(1, math.ceil(len(tiles) / grid_cols))
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    padded_tiles = tiles + [blank.copy() for _ in range(grid_rows * grid_cols - len(tiles))]
    row_images = []
    for row_idx in range(grid_rows):
        row_tiles = padded_tiles[row_idx * grid_cols : (row_idx + 1) * grid_cols]
        row_images.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(row_images, axis=0)


def save_validation_batch_labels_plot(
    batch: dict[str, Any],
    ni: int,
    cfg: ValidationConfig,
    names: dict[int, str] | list[str] | tuple[str, ...],
    save_dir: Path,
    logger: logging.Logger,
    prepare_batch_fn,
) -> None:
    tiles: list[np.ndarray] = []
    batch_size = int(batch["img"].shape[0])
    for si in range(batch_size):
        pbatch = prepare_batch_fn(si, batch)
        canvas, gt_boxes = load_plot_preview_canvas_and_boxes(
            batch_img=batch["img"][si],
            pbatch=pbatch,
            boxes_xywhr=pbatch["bboxes"].detach().cpu().numpy(),
            cfg=cfg,
            logger=logger,
            warning_context="验证标签预览",
        )
        gt_classes = pbatch["cls"].detach().cpu().numpy().astype(int)
        tile, scale, pad_x, pad_y = resize_preview_tile(canvas)
        tile_boxes = remap_boxes_to_resized_tile(gt_boxes, scale, pad_x, pad_y)
        for class_id, box in zip(gt_classes.tolist(), tile_boxes):
            draw_obb_polygon(tile, box, (255, 0, 0), _resolve_class_name(names, int(class_id)))
        _draw_text_banner(tile, [Path(pbatch["im_file"]).name], text_color=(255, 255, 255))
        tiles.append(tile)

    mosaic = build_validation_batch_mosaic(tiles)
    output_path = save_dir / f"val_batch{ni}_labels.jpg"
    ok = cv2.imwrite(str(output_path), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if not ok:
        raise RuntimeError(f"Failed to write validation label preview mosaic: {output_path}")


def save_validation_batch_predictions_plot(
    batch: dict[str, Any],
    preds: list[dict[str, torch.Tensor]],
    ni: int,
    cfg: ValidationConfig,
    names: dict[int, str] | list[str] | tuple[str, ...],
    save_dir: Path,
    logger: logging.Logger,
    prepare_batch_fn,
    prepare_pred_fn,
) -> None:
    if not preds:
        return
    tiles: list[np.ndarray] = []
    batch_size = min(int(batch["img"].shape[0]), len(preds))
    for si in range(batch_size):
        pbatch = prepare_batch_fn(si, batch)
        predn = prepare_pred_fn(preds[si])
        pred_boxes = (
            predn["bboxes"].detach().cpu().numpy() if predn["bboxes"].numel() else np.zeros((0, 5), dtype=np.float32)
        )
        canvas, scaled_pred_boxes = load_plot_preview_canvas_and_boxes(
            batch_img=batch["img"][si],
            pbatch=pbatch,
            boxes_xywhr=pred_boxes,
            cfg=cfg,
            logger=logger,
            warning_context="验证预测预览",
        )
        pred_classes = (
            predn["cls"].detach().cpu().numpy().astype(int) if predn["cls"].numel() else np.zeros(0, dtype=int)
        )
        pred_confs = predn["conf"].detach().cpu().numpy() if predn["conf"].numel() else np.zeros(0, dtype=np.float32)
        tile, scale, pad_x, pad_y = resize_preview_tile(canvas)
        tile_boxes = remap_boxes_to_resized_tile(scaled_pred_boxes, scale, pad_x, pad_y)
        for class_id, conf, box in zip(pred_classes.tolist(), pred_confs.tolist(), tile_boxes):
            label = f"{_resolve_class_name(names, int(class_id))} {float(conf):.2f}"
            draw_obb_polygon(tile, box, (0, 255, 0), label)
        _draw_text_banner(tile, [Path(pbatch["im_file"]).name], text_color=(255, 255, 255))
        tiles.append(tile)

    mosaic = build_validation_batch_mosaic(tiles)
    output_path = save_dir / f"val_batch{ni}_pred.jpg"
    ok = cv2.imwrite(str(output_path), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if not ok:
        raise RuntimeError(f"Failed to write validation prediction preview mosaic: {output_path}")


def compute_false_alarm_rate(false_positive_count: int, true_negative_count: int) -> float | None:
    """Compute false alarm rate as FP / (FP + TN)."""
    if false_positive_count < 0 or true_negative_count < 0:
        raise ValueError(
            f"false alarm counts must be non-negative, but got FP={false_positive_count}, TN={true_negative_count}"
        )
    denominator = false_positive_count + true_negative_count
    if denominator <= 0:
        return None
    return false_positive_count / denominator


def compute_image_false_alarm_stats(
    gt_count: int,
    candidate_prediction_count: int,
    passed_conf_count: int,
    true_positive_count: int,
) -> FalseAlarmImageStats:
    """Compute single-image false alarm statistics from candidate and thresholded predictions.

    Metric definition requested by the user:
    - M = gt_count
    - N = candidate_prediction_count
    - K = N - M
    - L = passed_conf_count
    - J = true_positive_count
    - FP' = L - J
    - TN' = K - FP'

    Defensive clamping is applied for degenerate test scenarios such as N < M or N = 0 to avoid negative counts.
    """
    values = {
        "gt_count": gt_count,
        "candidate_prediction_count": candidate_prediction_count,
        "passed_conf_count": passed_conf_count,
        "true_positive_count": true_positive_count,
    }
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0, but got {value}")
    if passed_conf_count > candidate_prediction_count:
        raise ValueError(
            "passed_conf_count cannot exceed candidate_prediction_count, "
            f"but got {passed_conf_count} > {candidate_prediction_count}"
        )
    if true_positive_count > passed_conf_count:
        raise ValueError(
            f"true_positive_count cannot exceed passed_conf_count, but got {true_positive_count} > {passed_conf_count}"
        )
    if true_positive_count > gt_count:
        raise ValueError(f"true_positive_count cannot exceed gt_count, but got {true_positive_count} > {gt_count}")

    negative_candidate_count = max(candidate_prediction_count - gt_count, 0)
    false_positive_count = passed_conf_count - true_positive_count
    true_negative_count = max(negative_candidate_count - false_positive_count, 0)
    false_alarm_rate = compute_false_alarm_rate(false_positive_count, true_negative_count)
    return FalseAlarmImageStats(
        gt_count=gt_count,
        candidate_prediction_count=candidate_prediction_count,
        negative_candidate_count=negative_candidate_count,
        passed_conf_count=passed_conf_count,
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        true_negative_count=true_negative_count,
        false_alarm_rate=false_alarm_rate,
    )


def summarize_false_alarm_dataset(image_stats: list[FalseAlarmImageStats]) -> FalseAlarmDatasetStats:
    """Aggregate single-image false alarm counts into dataset-level totals."""
    false_positive_total = sum(item.false_positive_count for item in image_stats)
    true_negative_total = sum(item.true_negative_count for item in image_stats)
    return FalseAlarmDatasetStats(
        image_count=len(image_stats),
        image_count_with_valid_denominator=sum(item.false_alarm_rate is not None for item in image_stats),
        candidate_prediction_total=sum(item.candidate_prediction_count for item in image_stats),
        negative_candidate_total=sum(item.negative_candidate_count for item in image_stats),
        passed_conf_total=sum(item.passed_conf_count for item in image_stats),
        true_positive_total=sum(item.true_positive_count for item in image_stats),
        false_positive_total=false_positive_total,
        true_negative_total=true_negative_total,
        false_alarm_rate=compute_false_alarm_rate(false_positive_total, true_negative_total),
    )


def format_false_alarm_rate(false_alarm_rate: float | None) -> str:
    return "N/A (FP+TN=0，当前虚警率不适用)" if false_alarm_rate is None else f"{float(false_alarm_rate):.6f}"


def log_validation_summary(
    logger: logging.Logger, official_stats: dict[str, float], custom_metrics: dict[str, float | int | None]
) -> None:
    logger.info(
        "Official metrics summary: P=%.6f, R=%.6f, mAP50=%.6f, mAP50-95=%.6f",
        float(official_stats.get("metrics/precision(B)", 0.0)),
        float(official_stats.get("metrics/recall(B)", 0.0)),
        float(official_stats.get("metrics/mAP50(B)", 0.0)),
        float(official_stats.get("metrics/mAP50-95(B)", 0.0)),
    )
    logger.info(
        "Fixed-threshold error summary: TP=%s, FP=%s, FN=%s, false_detection_rate=%s, missed_detection_rate=%s, "
        "alarm_image_ratio=%s, avg_false_positive_boxes_per_image=%s",
        custom_metrics["tp_iou50"],
        custom_metrics["fp_iou50"],
        custom_metrics["fn_iou50"],
        f"{float(custom_metrics['false_detection_rate']):.6f}",
        f"{float(custom_metrics['missed_detection_rate']):.6f}",
        f"{float(custom_metrics['alarm_image_ratio']):.6f}",
        f"{float(custom_metrics['avg_false_positive_boxes_per_image']):.6f}",
    )
    logger.info(
        "Candidate-level false alarm summary: candidate_total=%s, negative_total=%s, above_conf_total=%s, "
        "tp_after_conf_total=%s, fp_total=%s, tn_total=%s, false_alarm_rate=%s",
        custom_metrics["candidate_predictions_total"],
        custom_metrics["negative_candidates_total"],
        custom_metrics["predictions_above_conf_total"],
        custom_metrics["tp_after_conf_total"],
        custom_metrics["false_alarm_fp_total"],
        custom_metrics["false_alarm_tn_total"],
        format_false_alarm_rate(custom_metrics["false_alarm_rate"]),
    )


class ExtendedOBBValidator(OBBValidator):
    """OBB validator with extra false-detection metrics and error-sample archiving."""

    VIS_COLOR_GT_MISSED = (0, 255, 255)
    VIS_COLOR_PRED_CORRECT = (0, 255, 0)
    VIS_COLOR_PRED_LOC_WRONG = (0, 0, 255)
    VIS_COLOR_PRED_CLASS_WRONG = (255, 0, 255)
    VIS_COLOR_PRED_BOTH_WRONG = (128, 128, 128)

    def __init__(
        self,
        runtime_cfg: ValidationConfig,
        run_dir: Path,
        logger: logging.Logger,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks: dict | None = None,
    ) -> None:
        self.runtime_cfg = runtime_cfg
        self.run_dir = run_dir
        self.logger = logger
        self.error_dir = run_dir / "error_samples"
        self.error_dir.mkdir(parents=True, exist_ok=True)
        self.error_records: list[dict[str, Any]] = []
        self.error_counter = 0
        self.false_alarm_image_stats: list[FalseAlarmImageStats] = []
        self.extra_counts = {
            "tp_iou50": 0,
            "fp_iou50": 0,
            "fn_iou50": 0,
            "images_with_fp": 0,
            "images_with_fn": 0,
            "candidate_prediction_total": 0,
            "negative_candidate_total": 0,
            "predictions_above_conf_total": 0,
            "tp_after_conf_total": 0,
            "false_alarm_fp_total": 0,
            "false_alarm_tn_total": 0,
            "false_alarm_defined_image_count": 0,
        }
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Retain post-NMS candidate detections before the runtime confidence threshold.

        The default validator applies `self.args.conf` inside NMS, which loses the pre-threshold candidate count `N`
        needed by the custom false alarm metric. Here we keep all post-NMS candidates with `conf > 0` and apply the
        runtime threshold later inside `update_metrics()`.
        """
        outputs = nms.non_max_suppression(
            preds,
            0.0,
            self.args.iou,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=True,
        )
        return [{"bboxes": torch.cat([x[:, :4], x[:, 6:]], dim=-1), "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]

    @staticmethod
    def _filter_predictions_by_conf(predn: dict[str, torch.Tensor], conf_threshold: float) -> dict[str, torch.Tensor]:
        conf_mask = predn["conf"] >= conf_threshold
        return {k: v[conf_mask] for k, v in predn.items()}

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn_all = self._prepare_pred(pred)
            predn = self._filter_predictions_by_conf(predn_all, self.runtime_cfg.conf)
            preds[si] = predn  # keep downstream plotting/saving behavior aligned with the runtime confidence threshold

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            processed = self._process_batch(predn, pbatch)
            self.metrics.update_stats(
                {
                    **processed,
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "im_name": Path(pbatch["im_file"]).name,
                }
            )
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            custom = self._match_for_error_metrics(predn, pbatch)
            image_false_alarm_stats = compute_image_false_alarm_stats(
                gt_count=int(pbatch["cls"].shape[0]),
                candidate_prediction_count=int(predn_all["cls"].shape[0]),
                passed_conf_count=int(predn["cls"].shape[0]),
                true_positive_count=custom["tp_count"],
            )
            self.false_alarm_image_stats.append(image_false_alarm_stats)
            self.extra_counts["tp_iou50"] += custom["tp_count"]
            self.extra_counts["fp_iou50"] += custom["fp_count"]
            self.extra_counts["fn_iou50"] += custom["fn_count"]
            self.extra_counts["candidate_prediction_total"] += image_false_alarm_stats.candidate_prediction_count
            self.extra_counts["negative_candidate_total"] += image_false_alarm_stats.negative_candidate_count
            self.extra_counts["predictions_above_conf_total"] += image_false_alarm_stats.passed_conf_count
            self.extra_counts["tp_after_conf_total"] += image_false_alarm_stats.true_positive_count
            self.extra_counts["false_alarm_fp_total"] += image_false_alarm_stats.false_positive_count
            self.extra_counts["false_alarm_tn_total"] += image_false_alarm_stats.true_negative_count
            if image_false_alarm_stats.false_alarm_rate is not None:
                self.extra_counts["false_alarm_defined_image_count"] += 1
            if custom["fp_count"] > 0:
                self.extra_counts["images_with_fp"] += 1
            if custom["fn_count"] > 0:
                self.extra_counts["images_with_fn"] += 1

            if (
                custom["fp_count"] > 0 or custom["fn_count"] > 0
            ) and self.error_counter < self.runtime_cfg.max_error_samples:
                self._save_error_sample(batch["img"][si], pbatch, custom)

            if no_pred:
                continue
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        save_validation_batch_labels_plot(
            batch=batch,
            ni=ni,
            cfg=self.runtime_cfg,
            names=self.names,
            save_dir=Path(self.save_dir),
            logger=self.logger,
            prepare_batch_fn=self._prepare_batch,
        )

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        save_validation_batch_predictions_plot(
            batch=batch,
            preds=preds,
            ni=ni,
            cfg=self.runtime_cfg,
            names=self.names,
            save_dir=Path(self.save_dir),
            logger=self.logger,
            prepare_batch_fn=self._prepare_batch,
            prepare_pred_fn=self._prepare_pred,
        )

    def _match_for_error_metrics(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, Any]:
        gt_cls = pbatch["cls"]
        gt_boxes = pbatch["bboxes"]

        gt_count = int(gt_cls.shape[0])
        pred_count = int(predn["cls"].shape[0])
        if gt_count == 0 and pred_count == 0:
            return {
                "filtered": predn,
                "matched_gt_idx": np.zeros(0, dtype=int),
                "matched_pred_idx": np.zeros(0, dtype=int),
                "missed_gt_idx": np.zeros(0, dtype=int),
                "loc_wrong_pred_idx": np.zeros(0, dtype=int),
                "cls_wrong_pred_idx": np.zeros(0, dtype=int),
                "both_wrong_pred_idx": np.zeros(0, dtype=int),
                "duplicate_pred_idx": np.zeros(0, dtype=int),
                "tp_count": 0,
                "fp_count": 0,
                "fn_count": 0,
            }
        if gt_count == 0:
            return {
                "filtered": predn,
                "matched_gt_idx": np.zeros(0, dtype=int),
                "matched_pred_idx": np.zeros(0, dtype=int),
                "missed_gt_idx": np.zeros(0, dtype=int),
                "loc_wrong_pred_idx": np.zeros(0, dtype=int),
                "cls_wrong_pred_idx": np.zeros(0, dtype=int),
                "both_wrong_pred_idx": np.arange(pred_count, dtype=int),
                "duplicate_pred_idx": np.zeros(0, dtype=int),
                "tp_count": 0,
                "fp_count": pred_count,
                "fn_count": 0,
            }
        if pred_count == 0:
            return {
                "filtered": predn,
                "matched_gt_idx": np.zeros(0, dtype=int),
                "matched_pred_idx": np.zeros(0, dtype=int),
                "missed_gt_idx": np.arange(gt_count, dtype=int),
                "loc_wrong_pred_idx": np.zeros(0, dtype=int),
                "cls_wrong_pred_idx": np.zeros(0, dtype=int),
                "both_wrong_pred_idx": np.zeros(0, dtype=int),
                "duplicate_pred_idx": np.zeros(0, dtype=int),
                "tp_count": 0,
                "fp_count": 0,
                "fn_count": gt_count,
            }

        iou_all = batch_probiou(gt_boxes, predn["bboxes"]).cpu().numpy()
        correct_class = (gt_cls[:, None] == predn["cls"]).cpu().numpy()
        iou_same_class = iou_all * correct_class
        matches = self._greedy_match(iou_same_class, self.runtime_cfg.error_match_iou)
        matched_gt_idx = matches[:, 0].astype(int) if matches.shape[0] else np.zeros(0, dtype=int)
        matched_pred_idx = matches[:, 1].astype(int) if matches.shape[0] else np.zeros(0, dtype=int)
        tp_count = int(matched_pred_idx.size)
        fp_count = int(pred_count - tp_count)
        fn_count = int(gt_count - matched_gt_idx.size)
        extra_pred_classes = self._classify_prediction_errors(
            iou_all=iou_all,
            iou_same_class=iou_same_class,
            gt_cls=gt_cls.detach().cpu().numpy().astype(int),
            pred_cls=predn["cls"].detach().cpu().numpy().astype(int),
            matched_pred_idx=matched_pred_idx,
        )
        return {
            "filtered": predn,
            "matched_gt_idx": matched_gt_idx,
            "matched_pred_idx": matched_pred_idx,
            "missed_gt_idx": np.setdiff1d(np.arange(gt_count, dtype=int), matched_gt_idx, assume_unique=False),
            "loc_wrong_pred_idx": extra_pred_classes["loc_wrong_pred_idx"],
            "cls_wrong_pred_idx": extra_pred_classes["cls_wrong_pred_idx"],
            "both_wrong_pred_idx": extra_pred_classes["both_wrong_pred_idx"],
            "duplicate_pred_idx": extra_pred_classes["duplicate_pred_idx"],
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
        }

    def _save_error_sample(self, batch_img: torch.Tensor, pbatch: dict[str, Any], custom: dict[str, Any]) -> None:
        gt_boxes = pbatch["bboxes"].detach().cpu().numpy()
        gt_classes = pbatch["cls"].detach().cpu().numpy().astype(int)
        pred_boxes = custom["filtered"]["bboxes"].detach().cpu().numpy()
        pred_classes = custom["filtered"]["cls"].detach().cpu().numpy().astype(int)
        pred_confs = custom["filtered"]["conf"].detach().cpu().numpy()
        try:
            canvas = load_original_preview_image(Path(pbatch["im_file"]), self.runtime_cfg)
            gt_boxes = scale_obb_boxes_to_original(gt_boxes, pbatch["imgsz"], pbatch["ori_shape"], pbatch["ratio_pad"])
            pred_boxes = scale_obb_boxes_to_original(
                pred_boxes, pbatch["imgsz"], pbatch["ori_shape"], pbatch["ratio_pad"]
            )
        except Exception as exc:
            self.logger.warning(
                f"读取原图失败，默认错误样例回退到验证输入可视化：image={pbatch['im_file']}, reason={exc}"
            )
            canvas = tensor_to_preview_bgr(batch_img, self.runtime_cfg)
            gt_boxes = gt_boxes.astype(np.float32, copy=False)
            pred_boxes = pred_boxes.astype(np.float32, copy=False)

        for idx in custom["missed_gt_idx"].tolist():
            gt_label = self.names.get(int(gt_classes[idx]), str(int(gt_classes[idx])))
            draw_obb_polygon(canvas, gt_boxes[idx], self.VIS_COLOR_GT_MISSED, gt_label)

        for idx in custom["matched_pred_idx"].tolist():
            pred_label = f"{self.names.get(int(pred_classes[idx]), str(int(pred_classes[idx])))} {pred_confs[idx]:.2f}"
            draw_obb_polygon(canvas, pred_boxes[idx], self.VIS_COLOR_PRED_CORRECT, pred_label)

        for idx in custom["loc_wrong_pred_idx"].tolist():
            pred_label = f"{self.names.get(int(pred_classes[idx]), str(int(pred_classes[idx])))} {pred_confs[idx]:.2f}"
            draw_obb_polygon(canvas, pred_boxes[idx], self.VIS_COLOR_PRED_LOC_WRONG, pred_label)

        for idx in custom["cls_wrong_pred_idx"].tolist():
            pred_label = f"{self.names.get(int(pred_classes[idx]), str(int(pred_classes[idx])))} {pred_confs[idx]:.2f}"
            draw_obb_polygon(canvas, pred_boxes[idx], self.VIS_COLOR_PRED_CLASS_WRONG, pred_label)

        for idx in custom["both_wrong_pred_idx"].tolist():
            pred_label = f"{self.names.get(int(pred_classes[idx]), str(int(pred_classes[idx])))} {pred_confs[idx]:.2f}"
            draw_obb_polygon(canvas, pred_boxes[idx], self.VIS_COLOR_PRED_BOTH_WRONG, pred_label)

        for idx in custom["duplicate_pred_idx"].tolist():
            draw_obb_polygon(canvas, pred_boxes[idx], self.VIS_COLOR_PRED_BOTH_WRONG)

        save_path = self.error_dir / f"{self.error_counter + 1:03d}_{Path(pbatch['im_file']).stem}.jpg"
        ok = cv2.imwrite(str(save_path), canvas)
        if ok:
            self.error_counter += 1
            self.error_records.append(
                {
                    "image_path": str(pbatch["im_file"]),
                    "error_sample_path": str(save_path),
                    "fp_count": custom["fp_count"],
                    "fn_count": custom["fn_count"],
                }
            )

    @staticmethod
    def _greedy_match(iou_matrix: np.ndarray, threshold: float) -> np.ndarray:
        matches = np.argwhere(iou_matrix >= threshold)
        if matches.shape[0] == 0:
            return np.zeros((0, 2), dtype=int)
        pair_iou = iou_matrix[matches[:, 0], matches[:, 1]]
        order = pair_iou.argsort()[::-1]
        matches = matches[order]
        pair_iou = pair_iou[order]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        pair_iou = iou_matrix[matches[:, 0], matches[:, 1]]
        order = pair_iou.argsort()[::-1]
        matches = matches[order]
        return matches[np.unique(matches[:, 0], return_index=True)[1]]

    def _classify_prediction_errors(
        self,
        iou_all: np.ndarray,
        iou_same_class: np.ndarray,
        gt_cls: np.ndarray,
        pred_cls: np.ndarray,
        matched_pred_idx: np.ndarray,
    ) -> dict[str, np.ndarray]:
        loc_wrong: list[int] = []
        cls_wrong: list[int] = []
        both_wrong: list[int] = []
        duplicate: list[int] = []
        matched_pred_set = set(matched_pred_idx.tolist())
        threshold = self.runtime_cfg.error_match_iou

        for pred_idx in range(pred_cls.shape[0]):
            if pred_idx in matched_pred_set:
                continue
            best_same_iou = float(iou_same_class[:, pred_idx].max()) if iou_same_class.shape[0] else 0.0
            best_all_gt_idx = int(iou_all[:, pred_idx].argmax()) if iou_all.shape[0] else -1
            best_all_iou = float(iou_all[best_all_gt_idx, pred_idx]) if best_all_gt_idx >= 0 else 0.0
            has_same_class_candidate = best_same_iou > 0.0
            has_diff_class_match = best_all_iou >= threshold and gt_cls[best_all_gt_idx] != pred_cls[pred_idx]
            has_duplicate_tp_overlap = best_same_iou >= threshold

            if has_diff_class_match:
                cls_wrong.append(pred_idx)
            elif has_same_class_candidate and best_same_iou < threshold:
                loc_wrong.append(pred_idx)
            elif has_duplicate_tp_overlap:
                duplicate.append(pred_idx)
            else:
                both_wrong.append(pred_idx)

        return {
            "loc_wrong_pred_idx": np.array(loc_wrong, dtype=int),
            "cls_wrong_pred_idx": np.array(cls_wrong, dtype=int),
            "both_wrong_pred_idx": np.array(both_wrong, dtype=int),
            "duplicate_pred_idx": np.array(duplicate, dtype=int),
        }

    def get_custom_metrics(self) -> dict[str, float | int | None]:
        tp = self.extra_counts["tp_iou50"]
        fp = self.extra_counts["fp_iou50"]
        fn = self.extra_counts["fn_iou50"]
        total_images = max(self.seen, 1)
        false_alarm_dataset_stats = summarize_false_alarm_dataset(self.false_alarm_image_stats)
        false_detection_rate = fp / max(tp + fp, 1)
        alarm_image_ratio = self.extra_counts["images_with_fp"] / total_images
        avg_false_positive_boxes_per_image = fp / total_images
        missed_detection_rate = fn / max(tp + fn, 1)
        return {
            "tp_iou50": tp,
            "fp_iou50": fp,
            "fn_iou50": fn,
            "false_detection_rate": false_detection_rate,
            "false_alarm_rate": false_alarm_dataset_stats.false_alarm_rate,
            "alarm_image_ratio": alarm_image_ratio,
            "avg_false_positive_boxes_per_image": avg_false_positive_boxes_per_image,
            "missed_detection_rate": missed_detection_rate,
            "images_with_false_alarm": self.extra_counts["images_with_fp"],
            "images_with_missed_detection": self.extra_counts["images_with_fn"],
            "candidate_predictions_total": false_alarm_dataset_stats.candidate_prediction_total,
            "negative_candidates_total": false_alarm_dataset_stats.negative_candidate_total,
            "predictions_above_conf_total": false_alarm_dataset_stats.passed_conf_total,
            "tp_after_conf_total": false_alarm_dataset_stats.true_positive_total,
            "false_alarm_fp_total": false_alarm_dataset_stats.false_positive_total,
            "false_alarm_tn_total": false_alarm_dataset_stats.true_negative_total,
            "false_alarm_valid_image_count": false_alarm_dataset_stats.image_count_with_valid_denominator,
            "archived_error_samples": self.error_counter,
        }


def build_validator_args(cfg: ValidationConfig) -> dict[str, Any]:
    return {
        "task": "obb",
        "mode": "val",
        "model": str(cfg.weights),
        "data": str(cfg.data),
        "imgsz": cfg.imgsz,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "max_det": cfg.max_det,
        "batch": cfg.batch,
        "device": cfg.device,
        "split": cfg.split,
        "workers": cfg.workers,
        "half": cfg.half,
        "plots": cfg.plots,
        "save_json": cfg.save_json,
        "agnostic_nms": cfg.agnostic_nms,
        "rect": True,
    }


def generate_conf_sweep_values(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    epsilon = step * 1e-6 + 1e-12
    while current <= end + epsilon:
        values.append(round(min(max(current, 0.0), 1.0), 10))
        current += step
    if not values:
        values.append(round(start, 10))
    if abs(values[-1] - end) > epsilon:
        values.append(round(end, 10))
    deduped: list[float] = []
    for value in values:
        if not deduped or abs(deduped[-1] - value) > 1e-10:
            deduped.append(value)
    return deduped


def get_metric_plot_display_name(metric_name: str) -> str:
    metric_aliases = {
        "误检率": "False Detection Rate",
        "漏检率": "Missed Detection Rate",
        "告警图占比": "Alarm Image Ratio",
        "候选级虚警率(自定义)": "Candidate False Alarm Rate (Custom)",
    }
    return metric_aliases.get(metric_name, metric_name)


def _draw_metric_curve_plot(
    metric_name: str, conf_values: list[float], metric_values: list[float], output_path: Path
) -> Path:
    display_name = get_metric_plot_display_name(metric_name)
    width, height = 1600, 900
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 120, 70, 110, 110
    plot_x0, plot_y0 = left, top
    plot_x1, plot_y1 = width - right, height - bottom
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0

    cv2.putText(
        canvas,
        f"{display_name} vs VAL_CONF",
        (left, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (210, 210, 210), 1)

    for tick in range(6):
        ratio = tick / 5.0
        y = int(plot_y1 - ratio * plot_h)
        cv2.line(canvas, (plot_x0, y), (plot_x1, y), (235, 235, 235), 1)
        value_text = f"{ratio:.1f}"
        cv2.putText(
            canvas,
            value_text,
            (45, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    x_min = min(conf_values)
    x_max = max(conf_values)
    x_span = max(x_max - x_min, 1e-9)
    max_x_labels = min(8, len(conf_values))
    for tick in range(max_x_labels):
        ratio = 0.0 if max_x_labels == 1 else tick / (max_x_labels - 1)
        x = int(plot_x0 + ratio * plot_w)
        cv2.line(canvas, (x, plot_y0), (x, plot_y1), (240, 240, 240), 1)
        conf_value = x_min + ratio * x_span
        cv2.putText(
            canvas,
            f"{conf_value:.3f}",
            (x - 28, plot_y1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (120, 120, 120), 2)
    cv2.line(canvas, (plot_x0, plot_y0), (plot_x0, plot_y1), (120, 120, 120), 2)
    cv2.putText(
        canvas, "VAL_CONF", (plot_x1 - 120, plot_y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA
    )
    cv2.putText(canvas, display_name, (20, plot_y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    points: list[tuple[int, int]] = []
    for conf_value, metric_value in zip(conf_values, metric_values):
        x_ratio = 0.5 if x_span <= 1e-9 else (conf_value - x_min) / x_span
        x = int(plot_x0 + x_ratio * plot_w)
        y = int(plot_y1 - float(np.clip(metric_value, 0.0, 1.0)) * plot_h)
        points.append((x, y))

    if len(points) >= 2:
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, (66, 133, 244), 3, cv2.LINE_AA)
    for idx, (point, conf_value, metric_value) in enumerate(zip(points, conf_values, metric_values)):
        cv2.circle(canvas, point, 6, (219, 68, 55), -1, cv2.LINE_AA)
        label_y = point[1] - 10 if idx % 2 == 0 else point[1] + 24
        cv2.putText(
            canvas,
            f"{metric_value:.4f}",
            (point[0] - 34, max(plot_y0 + 16, min(label_y, plot_y1 - 6))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{conf_value:.3f}",
            (point[0] - 28, plot_y1 + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    ok = cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError(f"Failed to save metric curve plot: {output_path}")
    return output_path


def create_official_metric_curve_buffer() -> dict[str, list[float]]:
    return {
        "P": [],
        "R": [],
        "mAP50": [],
        "mAP50-95": [],
    }


def save_conf_sweep_plots(
    run_dir: Path, conf_values: list[float], metrics_by_name: dict[str, list[float]], logger: logging.Logger
) -> list[Path]:
    metric_to_filename = {
        "P": "precision_vs_val_conf.png",
        "R": "recall_vs_val_conf.png",
        "mAP50": "map50_vs_val_conf.png",
        "mAP50-95": "map50_95_vs_val_conf.png",
        "误检率": "false_detection_rate_vs_val_conf.png",
        "漏检率": "missed_detection_rate_vs_val_conf.png",
        "告警图占比": "alarm_image_ratio_vs_val_conf.png",
        "候选级虚警率(自定义)": "candidate_false_alarm_rate_vs_val_conf.png",
    }
    output_paths: list[Path] = []
    for metric_name, filename in metric_to_filename.items():
        output_path = _draw_metric_curve_plot(
            metric_name, conf_values, metrics_by_name[metric_name], run_dir / filename
        )
        output_paths.append(output_path)
        logger.info("Saved VAL_CONF sweep plot: %s", output_path)
    return output_paths


def draw_conf_sweep_target_plot(
    target_label: str,
    metric_name: str,
    conf_values: list[float],
    metric_values: list[float],
    output_path: Path,
) -> Path:
    plot_path = _draw_metric_curve_plot(metric_name, conf_values, metric_values, output_path)
    if target_label:
        image = cv2.imread(str(plot_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to reload target plot for annotation: {plot_path}")
        cv2.putText(
            image,
            f"Target: {target_label}",
            (120, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (70, 70, 70),
            2,
            cv2.LINE_AA,
        )
        ok = cv2.imwrite(str(plot_path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if not ok:
            raise RuntimeError(f"Failed to write target-annotated plot: {plot_path}")
    return plot_path


def save_conf_sweep_target_plots(
    run_dir: Path,
    conf_values: list[float],
    target_selection: ConfSweepTargetSelection,
    metrics_by_name: dict[str, list[float]],
    logger: logging.Logger,
) -> list[Path]:
    metric_to_filename = {
        "P": "precision_vs_val_conf.png",
        "R": "recall_vs_val_conf.png",
        "mAP50": "map50_vs_val_conf.png",
        "mAP50-95": "map50_95_vs_val_conf.png",
    }
    target_dir = run_dir / "target_class_plots" / target_selection.folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for metric_name, filename in metric_to_filename.items():
        output_path = draw_conf_sweep_target_plot(
            target_label=target_selection.display_name,
            metric_name=metric_name,
            conf_values=conf_values,
            metric_values=metrics_by_name[metric_name],
            output_path=target_dir / filename,
        )
        output_paths.append(output_path)
        logger.info("Saved VAL_CONF sweep target plot [%s]: %s", target_selection.display_name, output_path)
    return output_paths


def extract_official_metrics_by_class(
    validator: ExtendedOBBValidator, class_names: dict[int, str]
) -> dict[int, dict[str, float]]:
    per_class_metrics = {int(class_id): {"P": 0.0, "R": 0.0, "mAP50": 0.0, "mAP50-95": 0.0} for class_id in class_names}
    for metric_index, class_id in enumerate(getattr(validator.metrics, "ap_class_index", [])):
        precision, recall, map50, map50_95 = validator.metrics.class_result(metric_index)
        per_class_metrics[int(class_id)] = {
            "P": float(precision),
            "R": float(recall),
            "mAP50": float(map50),
            "mAP50-95": float(map50_95),
        }
    return per_class_metrics


def resolve_conf_sweep_target_selections(
    target_specs: tuple[str | int, ...], class_names: dict[int, str]
) -> tuple[ConfSweepTargetSelection, ...]:
    if not target_specs:
        return tuple()
    name_to_id = {name: class_id for class_id, name in class_names.items()}
    resolved: list[ConfSweepTargetSelection] = []
    seen_keys: set[str] = set()
    for spec in _normalize_conf_sweep_target_class_specs(target_specs):
        if spec == "ALL":
            class_ids_to_add = sorted(class_names)
        elif isinstance(spec, int):
            class_ids_to_add = [spec]
        else:
            if spec not in name_to_id:
                raise ValueError(f"Unknown VAL_CONF target class: {spec!r}. Available dataset classes: {class_names}")
            class_ids_to_add = [name_to_id[spec]]
        for class_id in class_ids_to_add:
            if class_id not in class_names:
                raise ValueError(
                    f"Unknown VAL_CONF target class id: {class_id}. Available dataset classes: {class_names}"
                )
            class_name = class_names[class_id]
            selection = ConfSweepTargetSelection(
                key=f"class:{class_id}",
                kind="class",
                display_name=class_name,
                folder_name=f"class_{class_id}_{sanitize_filename_component(class_name)}",
                class_id=class_id,
            )
            if selection.key not in seen_keys:
                resolved.append(selection)
                seen_keys.add(selection.key)
    return tuple(resolved)


def get_conf_sweep_target_metric_snapshot(
    target_selection: ConfSweepTargetSelection,
    official_stats: dict[str, float],
    class_metrics_by_id: dict[int, dict[str, float]],
) -> dict[str, float]:
    if target_selection.class_id is None:
        raise ValueError(f"Missing class_id for target selection: {target_selection}")
    return class_metrics_by_id.get(target_selection.class_id, {"P": 0.0, "R": 0.0, "mAP50": 0.0, "mAP50-95": 0.0})


def build_metrics_summary_payload(
    cfg: ValidationConfig,
    official_stats: dict[str, float],
    custom_metrics: dict[str, float | int | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    json_payload = {
        "config": {
            "weights": str(cfg.weights),
            "data": str(cfg.data),
            "imgsz": cfg.imgsz,
            "conf": cfg.conf,
            "nms_iou": cfg.iou,
            "batch": cfg.batch,
            "device": cfg.device,
            "max_det": cfg.max_det,
            "agnostic_nms": cfg.agnostic_nms,
            "split": cfg.split,
        },
        "official_metrics": {
            "description": "Ultralytics official PR-curve metrics averaged over classes.",
            "precision": official_stats.get("metrics/precision(B)", 0.0),
            "recall": official_stats.get("metrics/recall(B)", 0.0),
            "map50": official_stats.get("metrics/mAP50(B)", 0.0),
            "map50_95": official_stats.get("metrics/mAP50-95(B)", 0.0),
        },
        "fixed_threshold_error_metrics": {
            "description": "Business-point metrics under fixed VAL_CONF and error_match_iou.",
            "tp_iou50": custom_metrics["tp_iou50"],
            "fp_iou50": custom_metrics["fp_iou50"],
            "fn_iou50": custom_metrics["fn_iou50"],
            "false_detection_rate": custom_metrics["false_detection_rate"],
            "missed_detection_rate": custom_metrics["missed_detection_rate"],
            "alarm_image_ratio": custom_metrics["alarm_image_ratio"],
            "avg_false_positive_boxes_per_image": custom_metrics["avg_false_positive_boxes_per_image"],
        },
        "candidate_level_false_alarm_metrics": {
            "description": "Custom candidate-level false alarm metrics using pre-threshold candidate predictions.",
            "candidate_predictions_total": custom_metrics["candidate_predictions_total"],
            "negative_candidates_total": custom_metrics["negative_candidates_total"],
            "predictions_above_conf_total": custom_metrics["predictions_above_conf_total"],
            "tp_after_conf_total": custom_metrics["tp_after_conf_total"],
            "false_alarm_fp_total": custom_metrics["false_alarm_fp_total"],
            "false_alarm_tn_total": custom_metrics["false_alarm_tn_total"],
            "false_alarm_valid_image_count": custom_metrics["false_alarm_valid_image_count"],
            "false_alarm_rate": custom_metrics["false_alarm_rate"],
        },
        "artifacts": {
            "images_with_false_alarm": custom_metrics["images_with_false_alarm"],
            "images_with_missed_detection": custom_metrics["images_with_missed_detection"],
            "archived_error_samples": custom_metrics["archived_error_samples"],
        },
    }
    csv_row = {
        "config.weights": str(cfg.weights),
        "config.data": str(cfg.data),
        "config.imgsz": cfg.imgsz,
        "config.conf": cfg.conf,
        "config.nms_iou": cfg.iou,
        "config.batch": cfg.batch,
        "config.device": cfg.device,
        "config.max_det": cfg.max_det,
        "config.agnostic_nms": cfg.agnostic_nms,
        "config.split": cfg.split,
        "official.precision": official_stats.get("metrics/precision(B)", 0.0),
        "official.recall": official_stats.get("metrics/recall(B)", 0.0),
        "official.map50": official_stats.get("metrics/mAP50(B)", 0.0),
        "official.map50_95": official_stats.get("metrics/mAP50-95(B)", 0.0),
        "fixed_threshold.tp_iou50": custom_metrics["tp_iou50"],
        "fixed_threshold.fp_iou50": custom_metrics["fp_iou50"],
        "fixed_threshold.fn_iou50": custom_metrics["fn_iou50"],
        "fixed_threshold.false_detection_rate": custom_metrics["false_detection_rate"],
        "fixed_threshold.missed_detection_rate": custom_metrics["missed_detection_rate"],
        "fixed_threshold.alarm_image_ratio": custom_metrics["alarm_image_ratio"],
        "fixed_threshold.avg_false_positive_boxes_per_image": custom_metrics["avg_false_positive_boxes_per_image"],
        "candidate_false_alarm.candidate_predictions_total": custom_metrics["candidate_predictions_total"],
        "candidate_false_alarm.negative_candidates_total": custom_metrics["negative_candidates_total"],
        "candidate_false_alarm.predictions_above_conf_total": custom_metrics["predictions_above_conf_total"],
        "candidate_false_alarm.tp_after_conf_total": custom_metrics["tp_after_conf_total"],
        "candidate_false_alarm.false_alarm_fp_total": custom_metrics["false_alarm_fp_total"],
        "candidate_false_alarm.false_alarm_tn_total": custom_metrics["false_alarm_tn_total"],
        "candidate_false_alarm.false_alarm_valid_image_count": custom_metrics["false_alarm_valid_image_count"],
        "candidate_false_alarm.false_alarm_rate": custom_metrics["false_alarm_rate"],
        "artifacts.images_with_false_alarm": custom_metrics["images_with_false_alarm"],
        "artifacts.images_with_missed_detection": custom_metrics["images_with_missed_detection"],
        "artifacts.archived_error_samples": custom_metrics["archived_error_samples"],
    }
    return json_payload, csv_row


def save_metrics_files(
    cfg: ValidationConfig,
    run_dir: Path,
    logger: logging.Logger,
    official_stats: dict[str, float],
    custom_metrics: dict[str, float | int | None],
) -> tuple[Path, Path]:
    json_payload, csv_row = build_metrics_summary_payload(cfg, official_stats, custom_metrics)
    csv_path = run_dir / "metrics_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in csv_row.items()})

    json_path = run_dir / "metrics_summary.json"
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info(f"Saved metric summary CSV: {csv_path}")
    logger.info(f"Saved metric summary JSON: {json_path}")
    return csv_path, json_path


def save_error_records_csv(run_dir: Path, error_records: list[dict[str, Any]]) -> Path:
    csv_path = run_dir / "error_samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "error_sample_path", "fp_count", "fn_count"])
        writer.writeheader()
        for row in error_records:
            writer.writerow(row)
    return csv_path


def extract_original_stem_from_error_sample(error_sample_path: Path) -> str:
    stem = error_sample_path.stem
    if stem.endswith("_GT"):
        stem = stem[:-3]
    prefix, sep, remainder = stem.partition("_")
    source_stem = remainder if sep and prefix.isdigit() else stem
    if "__x" in source_stem:
        source_stem = source_stem.split("__x", 1)[0]
    return source_stem


def find_original_image_path(raw_image_dir: Path, source_stem: str) -> Path:
    for suffix in (".npy", ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        candidate = raw_image_dir / f"{source_stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find original image for stem '{source_stem}' under '{raw_image_dir}'. "
        "Expected one of: .npy/.tiff/.tif/.png/.jpg/.jpeg/.bmp/.webp"
    )


def build_validation_preview_bgr(image: np.ndarray, cfg: ValidationConfig) -> np.ndarray:
    preview_bgr, _ = build_preview_bgr(
        image=image,
        preview_mode=cfg.preview_mode,
        display_channels=cfg.display_channels,
        stretch_low=cfg.stretch_low,
        stretch_high=cfg.stretch_high,
        channel_wavelengths_nm=cfg.channel_wavelengths_nm,
    )
    return preview_bgr


def tensor_to_preview_bgr(image_tensor: torch.Tensor, cfg: ValidationConfig) -> np.ndarray:
    image = image_tensor.detach().float().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor for error visualization, but got shape {image.shape}")
    image = np.transpose(image, (1, 2, 0))
    return build_validation_preview_bgr(image, cfg)


def load_original_preview_image(image_path: Path, cfg: ValidationConfig) -> np.ndarray:
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        image = load_npy_image(image_path)
        return build_validation_preview_bgr(image, cfg)
    if suffix in {".tif", ".tiff"}:
        ok, pages = cv2.imreadmulti(str(image_path), flags=cv2.IMREAD_UNCHANGED)
        if ok and pages:
            if pages[0].ndim == 2:
                stacked = np.stack(pages, axis=2)
                return build_validation_preview_bgr(stacked, cfg)
            return build_validation_preview_bgr(np.ascontiguousarray(pages[0]), cfg)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cv2.imread failed for source image: {image_path}")
    if image.ndim == 2:
        image = image[..., None]
    return build_validation_preview_bgr(np.ascontiguousarray(image.astype(np.float32, copy=False)), cfg)


def draw_raw_gt_polygons_on_original(
    source_image_path: Path, source_label_path: Path, cfg: ValidationConfig
) -> tuple[np.ndarray, int]:
    canvas = load_original_preview_image(source_image_path, cfg)
    image_h, image_w = canvas.shape[:2]
    class_to_id = {name: idx for idx, name in enumerate(DEFAULT_CONFIG.class_names)}
    raw_annotations = parse_raw_label_file(source_label_path, class_to_id, cfg.raw_include_difficult)
    gt_count = 0
    for class_id, points in raw_annotations:
        sanitized = sanitize_annotation(points, image_w, image_h)
        if sanitized is None:
            continue
        polygon = sanitized.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        class_name = (
            DEFAULT_CONFIG.class_names[class_id] if 0 <= class_id < len(DEFAULT_CONFIG.class_names) else str(class_id)
        )
        text_origin = _resolve_label_origin(sanitized.astype(np.int32), canvas.shape)
        _draw_box_label(canvas, class_name, text_origin, (255, 0, 0))
        gt_count += 1
    return canvas, gt_count


def greedy_match_by_iou(iou_matrix: np.ndarray, threshold: float) -> np.ndarray:
    matches = np.argwhere(iou_matrix >= threshold)
    if matches.shape[0] == 0:
        return np.zeros((0, 2), dtype=int)
    pair_iou = iou_matrix[matches[:, 0], matches[:, 1]]
    order = pair_iou.argsort()[::-1]
    matches = matches[order]
    pair_iou = pair_iou[order]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    pair_iou = iou_matrix[matches[:, 0], matches[:, 1]]
    order = pair_iou.argsort()[::-1]
    matches = matches[order]
    return matches[np.unique(matches[:, 0], return_index=True)[1]]


def append_analysis_warning(logger: logging.Logger, warnings: list[str], message: str) -> None:
    warnings.append(message)
    logger.warning(message)


def resolve_target_class_map(
    target_specs: tuple[str | int, ...], class_names: dict[int, str], logger: logging.Logger, warnings: list[str]
) -> dict[int, str]:
    name_to_id = {name: class_id for class_id, name in class_names.items()}
    resolved: dict[int, str] = {}
    for spec in target_specs:
        try:
            if isinstance(spec, int):
                class_id = spec
            else:
                class_id = name_to_id[str(spec)]
            if class_id not in class_names:
                raise KeyError(spec)
            resolved[class_id] = class_names[class_id]
        except Exception:
            append_analysis_warning(logger, warnings, f"忽略不存在的目标类别配置：{spec!r}")
    if not resolved:
        raise ValueError(f"None of target classes {target_specs!r} can be resolved from dataset names {class_names}.")
    return resolved


def scale_obb_boxes_to_original(
    boxes_xywhr: torch.Tensor | np.ndarray, imgsz: tuple[int, int], ori_shape: tuple[int, int], ratio_pad: Any
) -> np.ndarray:
    if isinstance(boxes_xywhr, np.ndarray):
        boxes_tensor = torch.from_numpy(boxes_xywhr).float()
    else:
        boxes_tensor = boxes_xywhr.detach().float().cpu()
    if boxes_tensor.numel() == 0:
        return np.zeros((0, 5), dtype=np.float32)
    scaled = ops.scale_boxes(imgsz, boxes_tensor.clone(), ori_shape, ratio_pad=ratio_pad, xywh=True)
    return scaled.cpu().numpy().astype(np.float32, copy=False)


def analyze_target_class_errors(
    gt_classes: np.ndarray,
    gt_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_boxes: np.ndarray,
    target_class_id: int,
    iou_threshold: float,
) -> dict[str, np.ndarray | int]:
    gt_target_idx = np.flatnonzero(gt_classes == target_class_id).astype(int)
    pred_target_idx = np.flatnonzero(pred_classes == target_class_id).astype(int)
    matched_gt_idx = np.zeros(0, dtype=int)
    matched_pred_idx = np.zeros(0, dtype=int)
    missed_gt_idx = gt_target_idx.copy()
    non_target_class_fp_idx = np.zeros(0, dtype=int)
    background_fp_idx = np.zeros(0, dtype=int)

    if gt_target_idx.size and pred_target_idx.size:
        target_iou = (
            batch_probiou(
                torch.from_numpy(gt_boxes[gt_target_idx]).float(), torch.from_numpy(pred_boxes[pred_target_idx]).float()
            )
            .cpu()
            .numpy()
        )
        matches = greedy_match_by_iou(target_iou, iou_threshold)
        if matches.shape[0]:
            matched_gt_idx = gt_target_idx[matches[:, 0]].astype(int)
            matched_pred_idx = pred_target_idx[matches[:, 1]].astype(int)
            missed_gt_idx = np.setdiff1d(gt_target_idx, matched_gt_idx, assume_unique=False)

    if pred_target_idx.size:
        unmatched_pred_idx = np.setdiff1d(pred_target_idx, matched_pred_idx, assume_unique=False)
        if unmatched_pred_idx.size:
            if gt_boxes.shape[0]:
                iou_all = (
                    batch_probiou(
                        torch.from_numpy(gt_boxes).float(), torch.from_numpy(pred_boxes[unmatched_pred_idx]).float()
                    )
                    .cpu()
                    .numpy()
                )
            else:
                iou_all = np.zeros((0, unmatched_pred_idx.size), dtype=np.float32)
            non_target_class_fp: list[int] = []
            background_fp: list[int] = []
            for local_pred_idx, pred_idx in enumerate(unmatched_pred_idx.tolist()):
                if iou_all.shape[0] == 0:
                    background_fp.append(pred_idx)
                    continue
                best_gt_idx = int(iou_all[:, local_pred_idx].argmax())
                best_iou = float(iou_all[best_gt_idx, local_pred_idx])
                if best_iou >= iou_threshold and gt_classes[best_gt_idx] != target_class_id:
                    non_target_class_fp.append(pred_idx)
                else:
                    # Include duplicate same-class predictions in the background-style FP bucket because they
                    # do not correspond to a new valid target instance.
                    background_fp.append(pred_idx)
            non_target_class_fp_idx = np.array(non_target_class_fp, dtype=int)
            background_fp_idx = np.array(background_fp, dtype=int)

    return {
        "matched_gt_idx": matched_gt_idx,
        "matched_pred_idx": matched_pred_idx,
        "missed_gt_idx": missed_gt_idx.astype(int),
        "non_target_class_fp_idx": non_target_class_fp_idx.astype(int),
        "background_fp_idx": background_fp_idx.astype(int),
        "false_negative_count": int(missed_gt_idx.size),
        "false_positive_count": int(non_target_class_fp_idx.size + background_fp_idx.size),
    }


def _draw_text_banner(
    image: np.ndarray, lines: list[str], text_color: tuple[int, int, int], origin: tuple[int, int] = (12, 16)
) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 1
    line_gap = 10
    heights: list[int] = []
    baselines: list[int] = []
    for line in lines:
        (_, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        heights.append(text_h)
        baselines.append(baseline)
    x0, y0 = origin
    cursor_y = y0 + 4
    for line, text_h, baseline in zip(lines, heights, baselines):
        text_y = min(image.shape[0] - baseline - 2, cursor_y + text_h)
        # Draw a thin dark outline first so colored text remains readable on bright backgrounds.
        cv2.putText(image, line, (x0, text_y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, line, (x0, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cursor_y = text_y + line_gap


def render_false_positive_image(
    image: np.ndarray,
    pred_boxes: np.ndarray,
    non_target_class_fp_idx: np.ndarray,
    background_fp_idx: np.ndarray,
) -> np.ndarray:
    canvas = image.copy()
    for idx in non_target_class_fp_idx.tolist():
        draw_obb_polygon(canvas, pred_boxes[idx], (255, 0, 255))
    for idx in background_fp_idx.tolist():
        draw_obb_polygon(canvas, pred_boxes[idx], (128, 128, 128))
    _draw_text_banner(
        canvas,
        [
            f"Non-target FP as target: {int(non_target_class_fp_idx.size)}",
            f"Background FP as target: {int(background_fp_idx.size)}",
        ],
        text_color=(0, 0, 255),
    )
    return canvas


def render_false_negative_image(image: np.ndarray, gt_boxes: np.ndarray, missed_gt_idx: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for idx in missed_gt_idx.tolist():
        draw_obb_polygon(canvas, gt_boxes[idx], (255, 0, 0))
    _draw_text_banner(canvas, [f"Missed target GT count: {int(missed_gt_idx.size)}"], text_color=(255, 0, 0))
    return canvas


def sanitize_filename_component(text: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    return sanitized or "unknown"


def should_export_more(current_count: int, limit: int | None) -> bool:
    return limit is None or current_count < limit


def export_error_analysis_image(
    output_dir: Path,
    image_stem: str,
    class_name: str,
    error_type: str,
    canvas: np.ndarray,
) -> Path:
    class_dir = output_dir / sanitize_filename_component(class_name)
    class_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{sanitize_filename_component(image_stem)}_{sanitize_filename_component(class_name)}_{error_type}.jpg"
    output_path = class_dir / file_name
    ok = cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for error-analysis image: {output_path}")
    return output_path


def load_visualization_canvas(
    batch_img: torch.Tensor,
    pbatch: dict[str, Any],
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    cfg: ValidationConfig,
    logger: logging.Logger,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        canvas = load_original_preview_image(Path(pbatch["im_file"]), cfg)
        scaled_pred_boxes = scale_obb_boxes_to_original(
            pred_boxes, pbatch["imgsz"], pbatch["ori_shape"], pbatch["ratio_pad"]
        )
        scaled_gt_boxes = scale_obb_boxes_to_original(
            gt_boxes, pbatch["imgsz"], pbatch["ori_shape"], pbatch["ratio_pad"]
        )
        return canvas, scaled_pred_boxes, scaled_gt_boxes
    except Exception as exc:
        append_analysis_warning(
            logger,
            warnings,
            f"读取原图失败，回退到验证输入可视化：image={pbatch['im_file']}, reason={exc}",
        )
        fallback_canvas = tensor_to_preview_bgr(batch_img, cfg)
        return fallback_canvas, pred_boxes.astype(np.float32, copy=False), gt_boxes.astype(np.float32, copy=False)


class ErrorAnalysisOBBValidator(OBBValidator):
    """按指定类别导出漏检/误检图片与统计的独立 OBB 验证器。."""

    def __init__(
        self,
        runtime_cfg: ValidationConfig,
        analysis_cfg: ErrorAnalysisConfig,
        target_classes: dict[int, str],
        run_dir: Path,
        logger: logging.Logger,
        warnings: list[str],
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks: dict | None = None,
    ) -> None:
        self.runtime_cfg = runtime_cfg
        self.analysis_cfg = analysis_cfg
        self.target_classes = target_classes
        self.run_dir = run_dir
        self.logger = logger
        self.warning_messages = warnings
        self.false_positive_dir = run_dir / "false_positives"
        self.false_negative_dir = run_dir / "false_negatives"
        self.false_positive_dir.mkdir(parents=True, exist_ok=True)
        self.false_negative_dir.mkdir(parents=True, exist_ok=True)
        self.exported_image_count = 0
        self.total_image_count = 0
        self.stats_by_class: dict[int, dict[str, int]] = {
            class_id: {
                "false_negative_total": 0,
                "false_positive_total": 0,
                "false_negative_images": 0,
                "false_positive_images": 0,
                "exported_false_negative_images": 0,
                "exported_false_positive_images": 0,
            }
            for class_id in target_classes
        }
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        for si, pred in enumerate(preds):
            self.seen += 1
            self.total_image_count += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "im_name": Path(pbatch["im_file"]).name,
                }
            )

            gt_classes = pbatch["cls"].detach().cpu().numpy().astype(int)
            gt_boxes = pbatch["bboxes"].detach().cpu().numpy().astype(np.float32, copy=False)
            pred_classes = predn["cls"].detach().cpu().numpy().astype(int)
            pred_boxes = predn["bboxes"].detach().cpu().numpy().astype(np.float32, copy=False)

            for class_id, class_name in self.target_classes.items():
                analysis = analyze_target_class_errors(
                    gt_classes=gt_classes,
                    gt_boxes=gt_boxes,
                    pred_classes=pred_classes,
                    pred_boxes=pred_boxes,
                    target_class_id=class_id,
                    iou_threshold=self.runtime_cfg.error_match_iou,
                )
                self._accumulate_stats(class_id, analysis)
                self._maybe_export_images(
                    batch["img"][si], pbatch, class_id, class_name, analysis, pred_boxes, gt_boxes
                )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        save_validation_batch_labels_plot(
            batch=batch,
            ni=ni,
            cfg=self.runtime_cfg,
            names=self.names,
            save_dir=Path(self.save_dir),
            logger=self.logger,
            prepare_batch_fn=self._prepare_batch,
        )

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        save_validation_batch_predictions_plot(
            batch=batch,
            preds=preds,
            ni=ni,
            cfg=self.runtime_cfg,
            names=self.names,
            save_dir=Path(self.save_dir),
            logger=self.logger,
            prepare_batch_fn=self._prepare_batch,
            prepare_pred_fn=self._prepare_pred,
        )

    def _accumulate_stats(self, class_id: int, analysis: dict[str, np.ndarray | int]) -> None:
        fn_count = int(analysis["false_negative_count"])
        fp_count = int(analysis["false_positive_count"])
        self.stats_by_class[class_id]["false_negative_total"] += fn_count
        self.stats_by_class[class_id]["false_positive_total"] += fp_count
        if fn_count > 0:
            self.stats_by_class[class_id]["false_negative_images"] += 1
        if fp_count > 0:
            self.stats_by_class[class_id]["false_positive_images"] += 1

    def _maybe_export_images(
        self,
        batch_img: torch.Tensor,
        pbatch: dict[str, Any],
        class_id: int,
        class_name: str,
        analysis: dict[str, np.ndarray | int],
        pred_boxes: np.ndarray,
        gt_boxes: np.ndarray,
    ) -> None:
        wants_fp = ERROR_TYPE_FALSE_POSITIVE in self.analysis_cfg.error_types_to_export
        wants_fn = ERROR_TYPE_FALSE_NEGATIVE in self.analysis_cfg.error_types_to_export
        has_fp = int(analysis["false_positive_count"]) > 0
        has_fn = int(analysis["false_negative_count"]) > 0
        if (not wants_fp or not has_fp) and (not wants_fn or not has_fn):
            return
        if not should_export_more(self.exported_image_count, self.analysis_cfg.max_export_images):
            return

        try:
            canvas, scaled_pred_boxes, scaled_gt_boxes = load_visualization_canvas(
                batch_img=batch_img,
                pbatch=pbatch,
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
                cfg=self.runtime_cfg,
                logger=self.logger,
                warnings=self.warning_messages,
            )
            image_stem = Path(pbatch["im_file"]).stem
            if (
                wants_fp
                and has_fp
                and should_export_more(self.exported_image_count, self.analysis_cfg.max_export_images)
            ):
                fp_canvas = render_false_positive_image(
                    image=canvas,
                    pred_boxes=scaled_pred_boxes,
                    non_target_class_fp_idx=np.asarray(analysis["non_target_class_fp_idx"], dtype=int),
                    background_fp_idx=np.asarray(analysis["background_fp_idx"], dtype=int),
                )
                export_error_analysis_image(
                    self.false_positive_dir, image_stem, class_name, ERROR_TYPE_FALSE_POSITIVE, fp_canvas
                )
                self.exported_image_count += 1
                self.stats_by_class[class_id]["exported_false_positive_images"] += 1
            if (
                wants_fn
                and has_fn
                and should_export_more(self.exported_image_count, self.analysis_cfg.max_export_images)
            ):
                fn_canvas = render_false_negative_image(
                    image=canvas,
                    gt_boxes=scaled_gt_boxes,
                    missed_gt_idx=np.asarray(analysis["missed_gt_idx"], dtype=int),
                )
                export_error_analysis_image(
                    self.false_negative_dir, image_stem, class_name, ERROR_TYPE_FALSE_NEGATIVE, fn_canvas
                )
                self.exported_image_count += 1
                self.stats_by_class[class_id]["exported_false_negative_images"] += 1
        except Exception as exc:
            append_analysis_warning(
                self.logger,
                self.warning_messages,
                f"导出错误分析图片失败：image={pbatch['im_file']}, class={class_name}, reason={exc}",
            )

    def print_results(self) -> None:
        self.logger.info(
            "Error analysis processed images=%d, exported_images=%d, target_classes=%s",
            self.total_image_count,
            self.exported_image_count,
            list(self.target_classes.values()),
        )


def build_gt_overlay_for_error_samples(
    cfg: ValidationConfig, run_dir: Path, logger: logging.Logger
) -> tuple[int, int, Path]:
    error_dir = run_dir / "error_samples"
    summary_path = run_dir / "error_samples_gt_summary.json"
    if not error_dir.exists():
        summary = {
            "status": "skipped",
            "reason": f"error_samples directory does not exist: {error_dir}",
            "success_count": 0,
            "failure_count": 0,
            "processed_files": [],
            "failed_files": [],
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        logger.warning(summary["reason"])
        return 0, 0, summary_path

    success_count = 0
    failure_count = 0
    processed_files: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []

    for error_sample_path in sorted(error_dir.glob("*.jpg")):
        if error_sample_path.stem.endswith("_GT"):
            continue
        try:
            source_stem = extract_original_stem_from_error_sample(error_sample_path)
            source_image_path = find_original_image_path(cfg.raw_image_dir, source_stem)
            source_label_path = cfg.raw_label_dir / f"{source_stem}.txt"
            if not source_label_path.exists():
                raise FileNotFoundError(f"Could not find original GT label: {source_label_path}")
            canvas, gt_count = draw_raw_gt_polygons_on_original(source_image_path, source_label_path, cfg)
            output_path = error_dir / f"{error_sample_path.stem}_GT.jpg"
            ok = cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for GT overlay output: {output_path}")
            processed_files.append(
                {
                    "error_sample": str(error_sample_path),
                    "source_image": str(source_image_path),
                    "source_label": str(source_label_path),
                    "gt_overlay_image": str(output_path),
                    "gt_count": gt_count,
                }
            )
            success_count += 1
            logger.info(
                "Generated GT overlay image: "
                f"error_sample={error_sample_path.name}, source={source_image_path.name}, "
                f"gt_count={gt_count}, output={output_path.name}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            failure_count += 1
            failed_files.append({"error_sample": str(error_sample_path), "reason": str(exc)})
            logger.error(f"Failed to build GT overlay for {error_sample_path.name}: {exc}")

    summary = {
        "status": "completed",
        "error_samples_dir": str(error_dir),
        "success_count": success_count,
        "failure_count": failure_count,
        "processed_files": processed_files,
        "failed_files": failed_files,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info(
        "Completed GT overlay generation for error samples: "
        f"success={success_count}, failed={failure_count}, summary={summary_path}"
    )
    return success_count, failure_count, summary_path


def save_markdown_report(
    cfg: ValidationConfig,
    run_dir: Path,
    official_stats: dict[str, float],
    custom_metrics: dict[str, float | int | None],
    error_csv_path: Path,
    gt_overlay_summary: tuple[int, int, Path],
    class_distribution_chart_path: Path,
) -> Path:
    report_path = run_dir / "validation_report.md"
    gt_overlay_success, gt_overlay_failed, gt_overlay_summary_path = gt_overlay_summary
    false_alarm_rate_text = format_false_alarm_rate(custom_metrics["false_alarm_rate"])
    lines = [
        "# Validation Report",
        "",
        "## Config",
        "",
        f"- weights: `{cfg.weights}`",
        f"- data: `{cfg.data}`",
        f"- imgsz: `{cfg.imgsz}`",
        f"- conf: `{cfg.conf}`",
        f"- nms_iou: `{cfg.iou}`",
        f"- batch: `{cfg.batch}`",
        f"- device: `{cfg.device}`",
        f"- max_det: `{cfg.max_det}`",
        f"- agnostic_nms: `{cfg.agnostic_nms}`",
        f"- split: `{cfg.split}`",
        "",
        "## Official Metrics",
        "",
        "- 统计口径: Ultralytics 官方 PR 曲线口径，按类别统计后再做均值汇总。",
        f"- P: `{official_stats.get('metrics/precision(B)', 0.0):.6f}`",
        f"- R: `{official_stats.get('metrics/recall(B)', 0.0):.6f}`",
        f"- mAP50: `{official_stats.get('metrics/mAP50(B)', 0.0):.6f}`",
        f"- mAP50-95: `{official_stats.get('metrics/mAP50-95(B)', 0.0):.6f}`",
        "",
        "## Fixed-Threshold Error Metrics",
        "",
        "- 统计口径: 固定 `VAL_CONF` 与 `error_match_iou` 的业务点指标，基于全验证集累加后的 `TP/FP/FN` 计算。",
        "误检率 = FP / (TP + FP)",
        "漏检率 = FN / (TP + FN)",
        f"- tp_iou50: `{custom_metrics['tp_iou50']}`",
        f"- fp_iou50: `{custom_metrics['fp_iou50']}`",
        f"- fn_iou50: `{custom_metrics['fn_iou50']}`",
        f"- 误检率: `{float(custom_metrics['false_detection_rate']):.6f}`",
        f"- 漏检率: `{float(custom_metrics['missed_detection_rate']):.6f}`",
        f"- 告警图占比: `{float(custom_metrics['alarm_image_ratio']):.6f}`",
        f"- 平均每图误报框数: `{float(custom_metrics['avg_false_positive_boxes_per_image']):.6f}`",
        "",
        "## Candidate-Level False Alarm Metrics",
        "",
        "- 统计口径: 自定义候选级虚警率，额外使用阈值前候选预测数 `N` 与构造得到的 `TN`。",
        "虚警率 = FP / (FP + TN)",
        "单图口径: M=GT数, N=候选预测数, K=max(N-M, 0), L=通过当前conf的预测数, J=其中TP数, FP'=L-J, TN'=max(K-FP', 0)",
        "全数据集口径: 累加所有图片的FP'得到FP, 累加所有图片的TN'得到TN, 总虚警率=FP/(FP+TN)",
        f"- 候选预测总数 N: `{custom_metrics['candidate_predictions_total']}`",
        f"- 负样本总数 K: `{custom_metrics['negative_candidates_total']}`",
        f"- 通过conf的预测总数 L: `{custom_metrics['predictions_above_conf_total']}`",
        f"- 通过conf且预测正确总数 J: `{custom_metrics['tp_after_conf_total']}`",
        f"- 虚警率分子 FP: `{custom_metrics['false_alarm_fp_total']}`",
        f"- 虚警率分母中的 TN: `{custom_metrics['false_alarm_tn_total']}`",
        f"- 虚警率有效图片数: `{custom_metrics['false_alarm_valid_image_count']}`",
        f"- 候选级虚警率(自定义): `{false_alarm_rate_text}`",
        "",
        "## Artifacts",
        "",
        f"- 错误样例CSV: `{error_csv_path}`",
        f"- 错误样例归档数: `{custom_metrics['archived_error_samples']}`",
        f"- GT标注图生成成功数: `{gt_overlay_success}`",
        f"- GT标注图生成失败数: `{gt_overlay_failed}`",
        f"- GT标注图汇总: `{gt_overlay_summary_path}`",
        f"- 类别实例分布图: `{class_distribution_chart_path}`",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def save_run_metadata_log(
    run_dir: Path,
    timestamp: str,
    cfg: ValidationConfig,
    mode_name: str,
    started_at: datetime,
    finished_at: datetime,
    extra_lines: list[str] | None = None,
) -> Path:
    log_path = run_dir / f"{mode_name}_run_log_{timestamp}.txt"
    duration_seconds = (finished_at - started_at).total_seconds()
    lines = [
        f"运行模式: {mode_name}",
        f"模型权重完整路径: {cfg.weights}",
        f"数据集配置路径: {cfg.data}",
        f"功能启动时间: {started_at.isoformat(timespec='seconds')}",
        f"功能结束时间: {finished_at.isoformat(timespec='seconds')}",
        f"总运行时长(秒): {duration_seconds:.3f}",
    ]
    if extra_lines:
        lines.extend(["", *extra_lines])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def run_validation(cfg: ValidationConfig) -> Path:
    run_dir = cfg.results_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now()
    write_config_json(cfg, run_dir)
    effective_data_yaml = resolve_validation_data_yaml(cfg, run_dir, logger)
    with effective_data_yaml.open("r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f) or {}
    if "names" not in data_yaml or not data_yaml["names"]:
        raise ValueError(f"Dataset yaml must define a non-empty 'names' field, but got: {effective_data_yaml}")

    logger.info("Starting validation run.")
    logger.info(
        "Effective validation config: "
        f"weights={cfg.weights}, data={effective_data_yaml}, dataset_mode={cfg.dataset_mode}, imgsz={cfg.imgsz}, conf={cfg.conf}, "
        f"iou={cfg.iou}, batch={cfg.batch}, device={cfg.device}, max_det={cfg.max_det}, "
        f"agnostic_nms={cfg.agnostic_nms}, split={cfg.split}"
    )

    try:
        # pt权重文件保存了权重和模型结构，可以直接恢复模型，
        # 前提是init.py和tasks.py已经把新模型的块都已经注册
        model = YOLO(str(cfg.weights))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load model/weights from {cfg.weights}: {exc}") from exc
    dataset_names = load_dataset_names(effective_data_yaml)
    model_names = {int(k): str(v) for k, v in model.model.names.items()}
    if len(model_names) != len(dataset_names):
        raise RuntimeError(
            "Model/data class-count mismatch: "
            f"weights provide {len(model_names)} classes {model_names}, "
            f"but dataset yaml defines {len(dataset_names)} classes {dataset_names}. "
            "Please switch to the matching weights or matching dataset yaml."
        )

    validator_args = build_validator_args(cfg)
    validator_args["data"] = str(effective_data_yaml)
    validator = ExtendedOBBValidator(
        runtime_cfg=cfg, run_dir=run_dir, logger=logger, save_dir=run_dir, args=validator_args
    )
    try:
        official_stats = validator(model=model.model)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Validation execution failed. Please check weight compatibility, dataset yaml, device setting, "
            f"and memory usage. Root cause: {exc}"
        ) from exc

    custom_metrics = validator.get_custom_metrics()
    log_validation_summary(logger, official_stats, custom_metrics)
    metrics_csv_path, metrics_json_path = save_metrics_files(cfg, run_dir, logger, official_stats, custom_metrics)
    error_csv_path = save_error_records_csv(run_dir, validator.error_records)
    class_instance_counts = count_dataset_instances_by_class(effective_data_yaml, cfg.split, dataset_names)
    class_distribution_chart_path = generate_class_distribution_chart(
        run_dir, dataset_names, class_instance_counts, logger
    )
    gt_overlay_summary = build_gt_overlay_for_error_samples(cfg, run_dir, logger)
    report_path = save_markdown_report(
        cfg,
        run_dir,
        official_stats,
        custom_metrics,
        error_csv_path,
        gt_overlay_summary,
        class_distribution_chart_path,
    )
    finished_at = datetime.now()
    run_log_path = save_run_metadata_log(
        run_dir=run_dir,
        timestamp=timestamp,
        cfg=cfg,
        mode_name="validation",
        started_at=started_at,
        finished_at=finished_at,
        extra_lines=[
            f"本次实际生效数据集路径: {effective_data_yaml}",
            f"验证结果目录: {run_dir}",
            f"Markdown报告路径: {report_path}",
            f"指标CSV路径: {metrics_csv_path}",
            f"指标JSON路径: {metrics_json_path}",
        ],
    )
    logger.info(f"Saved error sample CSV: {error_csv_path}")
    logger.info(f"Saved markdown report: {report_path}")
    logger.info(f"Saved validation run log: {run_log_path}")
    logger.info(f"Validation completed successfully. Results saved to: {run_dir}")
    return run_dir


def save_error_analysis_log(
    run_dir: Path,
    timestamp: str,
    cfg: ValidationConfig,
    total_images: int,
    stats_by_class: dict[int, dict[str, int]],
    target_classes: dict[int, str],
    warnings: list[str],
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    log_path = run_dir / f"error_analysis_log_{timestamp}.txt"
    duration_seconds = (finished_at - started_at).total_seconds()
    lines = [
        f"模型权重完整路径: {cfg.weights}",
        f"功能启动时间: {started_at.isoformat(timespec='seconds')}",
        f"功能结束时间: {finished_at.isoformat(timespec='seconds')}",
        f"总运行时长(秒): {duration_seconds:.3f}",
        f"验证集总图片数: {total_images}",
        "",
        "各目标类导出统计:",
    ]
    for class_id, class_name in target_classes.items():
        stats = stats_by_class[class_id]
        lines.extend(
            [
                f"- {class_name} (id={class_id})",
                f"  漏检总数: {stats['false_negative_total']}",
                f"  误检总数: {stats['false_positive_total']}",
                f"  导出的漏检图片数: {stats['exported_false_negative_images']}",
                f"  导出的误检图片数: {stats['exported_false_positive_images']}",
            ]
        )
    lines.extend(["", "异常告警信息:"])
    if warnings:
        lines.extend(f"- {message}" for message in warnings)
    else:
        lines.append("- 无")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def save_error_analysis_report(
    run_dir: Path,
    timestamp: str,
    cfg: ValidationConfig,
    total_images: int,
    target_classes: dict[int, str],
    stats_by_class: dict[int, dict[str, int]],
) -> Path:
    report_path = run_dir / f"error_analysis_report_{timestamp}.md"
    total_fn = 0
    total_fp = 0
    total_fn_images = 0
    total_fp_images = 0
    lines = [
        "# Error Analysis Report",
        "",
        "## Config",
        "",
        f"- weights: `{cfg.weights}`",
        f"- data: `{cfg.data}`",
        f"- imgsz: `{cfg.imgsz}`",
        f"- conf: `{cfg.conf}`",
        f"- nms_iou: `{cfg.iou}`",
        f"- error_match_iou: `{cfg.error_match_iou}`",
        f"- total_images: `{total_images}`",
        "",
        "## Per-Class Statistics",
        "",
    ]
    for class_id, class_name in target_classes.items():
        stats = stats_by_class[class_id]
        total_fn += stats["false_negative_total"]
        total_fp += stats["false_positive_total"]
        total_fn_images += stats["false_negative_images"]
        total_fp_images += stats["false_positive_images"]
        lines.extend(
            [
                f"### {class_name}",
                "",
                f"- 目标类名称: `{class_name}`",
                f"- 目标类ID: `{class_id}`",
                f"- 总漏检数量: `{stats['false_negative_total']}`",
                f"- 总误检数量: `{stats['false_positive_total']}`",
                f"- 漏检图片数: `{stats['false_negative_images']}`",
                f"- 误检图片数: `{stats['false_positive_images']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            f"- 所有目标类总漏检数量: `{total_fn}`",
            f"- 所有目标类总误检数量: `{total_fp}`",
            f"- 所有目标类漏检图片数汇总: `{total_fn_images}`",
            f"- 所有目标类误检图片数汇总: `{total_fp_images}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_error_analysis(cfg: ValidationConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.error_analysis.output_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir, log_filename=f"error_analysis_console_{timestamp}.log")
    warnings: list[str] = []
    started_at = datetime.now()
    write_config_json(cfg, run_dir)
    effective_data_yaml = resolve_validation_data_yaml(cfg, run_dir, logger)
    dataset_names = load_dataset_names(effective_data_yaml)
    target_classes = resolve_target_class_map(cfg.error_analysis.target_classes, dataset_names, logger, warnings)

    logger.info("Starting target-class error analysis export run.")
    logger.info(
        "Error-analysis config: weights=%s, data=%s, target_classes=%s, error_types=%s, max_export_images=%s",
        cfg.weights,
        effective_data_yaml,
        list(target_classes.values()),
        list(cfg.error_analysis.error_types_to_export),
        cfg.error_analysis.max_export_images,
    )

    try:
        model = YOLO(str(cfg.weights))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load model/weights from {cfg.weights}: {exc}") from exc
    model_names = {int(k): str(v) for k, v in model.model.names.items()}
    if len(model_names) != len(dataset_names):
        raise RuntimeError(
            "Model/data class-count mismatch: "
            f"weights provide {len(model_names)} classes {model_names}, "
            f"but dataset yaml defines {len(dataset_names)} classes {dataset_names}. "
            "Please switch to the matching weights or matching dataset yaml."
        )

    validator_args = build_validator_args(cfg)
    validator_args.update({"data": str(effective_data_yaml), "plots": False, "save_json": False})
    validator = ErrorAnalysisOBBValidator(
        runtime_cfg=cfg,
        analysis_cfg=cfg.error_analysis,
        target_classes=target_classes,
        run_dir=run_dir,
        logger=logger,
        warnings=warnings,
        save_dir=run_dir,
        args=validator_args,
    )
    try:
        validator(model=model.model)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Error-analysis execution failed. Please check weight compatibility, dataset yaml, device setting, "
            f"and memory usage. Root cause: {exc}"
        ) from exc

    finished_at = datetime.now()
    report_path = save_error_analysis_report(
        run_dir=run_dir,
        timestamp=timestamp,
        cfg=cfg,
        total_images=validator.total_image_count,
        target_classes=target_classes,
        stats_by_class=validator.stats_by_class,
    )
    log_path = save_error_analysis_log(
        run_dir=run_dir,
        timestamp=timestamp,
        cfg=cfg,
        total_images=validator.total_image_count,
        stats_by_class=validator.stats_by_class,
        target_classes=target_classes,
        warnings=warnings,
        started_at=started_at,
        finished_at=finished_at,
    )
    logger.info("Saved error-analysis report: %s", report_path)
    logger.info("Saved error-analysis log: %s", log_path)
    logger.info("Error-analysis export completed successfully. Results saved to: %s", run_dir)
    return run_dir


def run_conf_sweep(cfg: ValidationConfig) -> Path:
    run_dir = cfg.conf_sweep.output_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir, log_filename=None)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now()
    conf_values = generate_conf_sweep_values(cfg.conf_sweep.start, cfg.conf_sweep.end, cfg.conf_sweep.step)
    metrics_by_name = {
        "P": [],
        "R": [],
        "mAP50": [],
        "mAP50-95": [],
        "误检率": [],
        "漏检率": [],
        "告警图占比": [],
        "候选级虚警率(自定义)": [],
    }
    target_selections: tuple[ConfSweepTargetSelection, ...] = tuple()
    target_metrics_by_key: dict[str, dict[str, list[float]]] = {}

    logger.info(
        "Starting VAL_CONF sweep: weights=%s, data=%s, conf_start=%.6f, conf_end=%.6f, conf_step=%.6f, points=%d",
        cfg.weights,
        cfg.data,
        cfg.conf_sweep.start,
        cfg.conf_sweep.end,
        cfg.conf_sweep.step,
        len(conf_values),
    )

    try:
        model = YOLO(str(cfg.weights))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load model/weights from {cfg.weights}: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="trained_obb_conf_sweep_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        effective_data_yaml = resolve_validation_data_yaml(cfg, temp_dir, logger)
        dataset_names = load_dataset_names(effective_data_yaml)
        target_selections = resolve_conf_sweep_target_selections(cfg.conf_sweep.target_classes, dataset_names)
        target_metrics_by_key = {
            selection.key: create_official_metric_curve_buffer() for selection in target_selections
        }
        model_names = {int(k): str(v) for k, v in model.model.names.items()}
        if len(model_names) != len(dataset_names):
            raise RuntimeError(
                "Model/data class-count mismatch: "
                f"weights provide {len(model_names)} classes {model_names}, "
                f"but dataset yaml defines {len(dataset_names)} classes {dataset_names}. "
                "Please switch to the matching weights or matching dataset yaml."
            )

        for conf_value in conf_values:
            sweep_cfg = replace(cfg, conf=conf_value, plots=False, save_json=False, max_error_samples=0)
            validator_args = build_validator_args(sweep_cfg)
            validator_args.update(
                {
                    "data": str(effective_data_yaml),
                    "conf": sweep_cfg.conf,
                    "plots": False,
                    "save_json": False,
                }
            )
            validator_save_dir = temp_dir / f"conf_{conf_value:.6f}".replace(".", "_")
            validator_save_dir.mkdir(parents=True, exist_ok=True)
            validator = ExtendedOBBValidator(
                runtime_cfg=sweep_cfg,
                run_dir=validator_save_dir,
                logger=logger,
                save_dir=validator_save_dir,
                args=validator_args,
            )
            try:
                official_stats = validator(model=model.model)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    "VAL_CONF sweep execution failed. Please check weight compatibility, dataset yaml, device setting, "
                    f"and memory usage. Root cause: {exc}"
                ) from exc

            custom_metrics = validator.get_custom_metrics()
            class_metrics_by_id = extract_official_metrics_by_class(validator, dataset_names)
            precision = float(official_stats.get("metrics/precision(B)", 0.0))
            recall = float(official_stats.get("metrics/recall(B)", 0.0))
            map50 = float(official_stats.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(official_stats.get("metrics/mAP50-95(B)", 0.0))
            metrics_by_name["P"].append(precision)
            metrics_by_name["R"].append(recall)
            metrics_by_name["mAP50"].append(map50)
            metrics_by_name["mAP50-95"].append(map50_95)
            metrics_by_name["误检率"].append(float(custom_metrics["false_detection_rate"]))
            metrics_by_name["漏检率"].append(float(custom_metrics["missed_detection_rate"]))
            metrics_by_name["告警图占比"].append(float(custom_metrics["alarm_image_ratio"]))
            metrics_by_name["候选级虚警率(自定义)"].append(
                0.0 if custom_metrics["false_alarm_rate"] is None else float(custom_metrics["false_alarm_rate"])
            )
            for target_selection in target_selections:
                target_snapshot = get_conf_sweep_target_metric_snapshot(
                    target_selection, official_stats, class_metrics_by_id
                )
                target_buffer = target_metrics_by_key[target_selection.key]
                for metric_name, metric_value in target_snapshot.items():
                    target_buffer[metric_name].append(float(metric_value))
            logger.info(
                "VAL_CONF=%.6f -> P=%.6f, R=%.6f, mAP50=%.6f, mAP50-95=%.6f, "
                "误检率=%.6f, 漏检率=%.6f, 告警图占比=%.6f, 候选级虚警率(自定义)=%s",
                conf_value,
                precision,
                recall,
                map50,
                map50_95,
                float(custom_metrics["false_detection_rate"]),
                float(custom_metrics["missed_detection_rate"]),
                float(custom_metrics["alarm_image_ratio"]),
                format_false_alarm_rate(custom_metrics["false_alarm_rate"]),
            )

    output_paths = save_conf_sweep_plots(run_dir, conf_values, metrics_by_name, logger)
    for target_selection in target_selections:
        output_paths.extend(
            save_conf_sweep_target_plots(
                run_dir=run_dir,
                conf_values=conf_values,
                target_selection=target_selection,
                metrics_by_name=target_metrics_by_key[target_selection.key],
                logger=logger,
            )
        )
    finished_at = datetime.now()
    run_log_path = save_run_metadata_log(
        run_dir=run_dir,
        timestamp=timestamp,
        cfg=cfg,
        mode_name="conf_sweep",
        started_at=started_at,
        finished_at=finished_at,
        extra_lines=[
            f"VAL_CONF起始值: {cfg.conf_sweep.start:.6f}",
            f"VAL_CONF终止值: {cfg.conf_sweep.end:.6f}",
            f"VAL_CONF步长: {cfg.conf_sweep.step:.6f}",
            f"扫描点数量: {len(conf_values)}",
            f"额外目标绘图配置: {list(cfg.conf_sweep.target_classes)}",
            f"输出图像数量: {len(output_paths)}",
        ],
    )
    logger.info("Saved VAL_CONF sweep run log: %s", run_log_path)
    logger.info("VAL_CONF sweep completed successfully. Plot directory: %s", run_dir)
    return run_dir


def run_selected_mode(cfg: ValidationConfig) -> Path:
    if cfg.error_analysis.enabled:
        return run_error_analysis(cfg)
    if cfg.conf_sweep.enabled:
        return run_conf_sweep(cfg)
    return run_validation(cfg)


def main() -> None:
    try:
        cfg = build_config(parse_args())
        validate_config(cfg)
        run_dir = run_selected_mode(cfg)
        print(json.dumps({"status": "success", "run_dir": str(run_dir)}, ensure_ascii=True, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=True, indent=2), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
