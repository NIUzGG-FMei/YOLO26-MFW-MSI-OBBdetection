"""8-channel OBB dataset preparation and training entry point.

本文件支持 ``legacy`` 回退流程和 ``balanced_multiscale`` 新流程。balanced
模式的完整使用手册紧随导入区，包含 70/20/10 视图比例、整图 OBB 标签变换、
显存保护、训练命令和回退方式；配置区与命令行参数均在定义处保留中文说明。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import shutil
import statistics
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Ensure the local repository root is importable when this script is launched directly from the `examples` folder.
# 中文：当脚本从 `examples` 目录被直接运行时，先把仓库根目录加入 `sys.path`，确保能导入本地源码版 `ultralytics`。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.multiscale_dataset_utils import (  # noqa: E402
    IDENTITY_TRANSFORM,
    ViewCandidate,
    ViewRatios,
    generate_candidates,
    load_multichannel_tiff,
    manifest_view_counts,
    maximum_feasible_total,
    select_candidates,
    validate_manifest_rows,
    validate_view_ratios,
    write_manifest,
    write_selected_candidates,
)
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.obb.train import OBBTrainer  # noqa: E402
from ultralytics.models.yolo.obb.val import OBBValidator  # noqa: E402
from ultralytics.utils import LOCAL_RANK, LOGGER, RANK  # noqa: E402
from ultralytics.utils.patches import imread  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, unwrap_model  # noqa: E402

"""
新数据集方案使用手册（balanced_multiscale）
==============================================

本脚本支持两种互相隔离的数据处理模式：

1. legacy
   保留当前旧流程：单一 `patch_size`、原有 overlap/keep_empty 配置、原有
   `AUGMENTED_DATASET` 合并方式。旧数据目录不会被新模式删除或改写。

2. balanced_multiscale
   生成可审计的多视图训练集：
   - 70% `patch256`：256x256 局部滑窗；
   - 20% `patch512`：512x512 更大上下文滑窗；
   - 10% `full_scaled`：整张 1200x900 图像按比例缩放并 letterbox 到
     `full_view_size`，同时对 OBB 四个顶点执行同样的缩放和平移；
   - patch 滑窗使用重叠步长（256 -> 128，512 -> 256）；
   - 空 patch 会进入候选池，再由 manifest 配额采样，不会无限制地把所有
     背景窗口复制进训练集；
   - 默认训练输入尺寸为 256，512 patch 在训练 loader 中统一缩放到 256，
     因而保留“大上下文视图”而不把 GPU 激活显存提高到 512 输入的水平。

推荐运行顺序（首次运行不要直接训练）：

    python examples/custom_obb_prepare_and_train.py --mode prepare \
        --preprocess-profile balanced_multiscale \
        --prepared-dataset-dir /path/to/prepared_balanced \
        --no-use-augmented-dataset

检查 `prepare_summary.json`、`train_manifest.csv`、`dataset_validation.json`：
三类视图、8 通道、uint8、OBB 坐标范围和数据源划分都通过后，再执行：

    python examples/custom_obb_prepare_and_train.py --mode train \
        --preprocess-profile balanced_multiscale \
        --prepared-dataset-dir /path/to/prepared_balanced \
        --batch 1 --val-batch 1 --amp \
        --use-augmented-dataset \
        --augmented-dataset-dir /path/to/augmented_target_patch_dataset

显存安全原则：
 - 新模式默认 `train_imgsz=256`、`val_batch=1`、AMP 开启；
 - 训练前预检训练/验证前向显存；
 - 训练开始前默认关闭 1200x900 整图 profile，避免尚未训练就发生显存峰值；
 - 每个 epoch 的验证使用独立的小 batch；
 - 额外的整图评估和通道相关性统计放到训练结束后的独立步骤；
 - 训练前会保存 `pre_val` checkpoint，验证 OOM 时可以清理显存并恢复；
 - 任何新模式失败都可以把 `--preprocess-profile legacy` 作为回退入口。

比例说明：70/20/10 是最终 train manifest 的样本视图比例，不是把 Mosaic
等多图增强后的 GPU 张量强行解释成该比例。若必须保持输入视图比例，建议新模式
关闭 Mosaic/MixUp/CutMix；否则 manifest 比例仍然准确，但一个训练样本可能由
多个视图拼接而成。

IDE 路径说明：直接点击运行时，legacy 默认使用 `prepared_Dataset` 和
`augmented_target_patch_dataset`；balanced_multiscale 自动切换到带有
`_balanced_multiscale` 后缀的两套独立目录。命令行显式传入
`--prepared-dataset-dir` 或 `--augmented-dataset-dir` 时，以显式路径为准。
"""


# =========================
# IDE Quick Config
# 顶部快速配置区
# 直接修改这里的几个值，然后点击右上角运行即可。
# =========================

# IDE_RUN_MODE:
# - "prepare": 只做数据集切片和 `data.yaml` 生成
# - "train": 直接使用已有切片数据开始训练
# - "prepare_and_train": 先切片，再立即训练
# 中文：最常改的运行阶段开关。
IDE_RUN_MODE = "prepare"

# IDE_DEVICE:
# - "0": 使用第 0 张 GPU
# - "0,1": 使用多卡
# - "cpu": 使用 CPU
# - None: 交给 Ultralytics 自动选择
# 中文：最常改的训练设备开关。
IDE_DEVICE: str | None = "0"

# IDE_EPOCHS:
# 中文：最常改的训练轮数开关。
IDE_EPOCHS = 100

# IDE_BATCH:  -1 是自动
# 中文：最常改的 batch size 开关。
IDE_BATCH = 16

# IDE_SEED:
# 中文：训练随机种子，固定后更利于结果复现。
IDE_SEED = 0

# IDE_DETERMINISTIC:
# 中文：是否启用确定性训练，True 更可复现但可能略慢。
IDE_DETERMINISTIC = True

# IDE_PRETRAINED:
# - "yolo26n-obb.pt": 使用官方预训练权重，若本地不存在可能会尝试联网下载
# - "/abs/path/to/xxx.pt": 使用本地预训练权重文件
# "/home/mofengwei/ultralytics/yolo26n-obb.pt"
# - None: 不加载任何预训练权重，直接从头训练
# 中文：最常改的预训练权重开关。
IDE_PRETRAINED: str | None = "/home/mofengwei/ultralytics/yolo26n-obb.pt"

# IDE_RESUME:
# - True: 从当前 `run_name` 或 `run_name-序号` 对应目录下的 `weights/last.pt` 继续训练
# - False: 不启用断点续训
# 中文：是否启用断点续训。
IDE_RESUME = False

# IDE_RESUME_RUN_INDEX:
# - None: 使用当前 `run_name`
# - 7: 使用 `run_name-7`
# 中文：当 `IDE_RESUME=True` 且未显式设置 `IDE_RESUME_FROM` 时，用这个序号自动拼接断点续训目录。
IDE_RESUME_RUN_INDEX: int | None = 7

# IDE_RESUME_FROM:
# - None: 不指定额外路径；若 `IDE_RESUME=True`，则默认使用当前 `run_name` 或 `run_name-序号` 对应 run 的 `weights/last.pt`
# - "/abs/path/to/last.pt": 从指定 checkpoint 继续训练
# - "/abs/path/to/run_dir": 从指定 run 目录中的 `weights/last.pt` 继续训练
# 中文：显式指定断点续训来源路径。
IDE_RESUME_FROM: str | None = None

# IDE_NPY_LAYOUT:
# - "CHW": 原始数组格式为 (channel, height, width)
# - "CWH": 原始数组格式为 (channel, width, height)
# - "HWC": 原始数组格式为 (height, width, channel)
# 中文：根据你当前标签坐标范围（x 约到 1200，y 约到 900），这批数据默认按 "CWH" 解析。
IDE_NPY_LAYOUT = "CWH"

# IDE_PREVIEW_SAMPLES:
# 中文：prepare 完成后，自动抽样多少张 patch 做标签拼图预览。
IDE_PREVIEW_SAMPLES = 8

# IDE_LEGACY_PREPARED_DATASET_DIR:
# 中文：legacy 模式使用的原有切片目录。保留原目录名，避免破坏现有实验。
IDE_LEGACY_PREPARED_DATASET_DIR = Path("/mnt/d/Vscode work_place/datasetObjectDetection/prepared_Dataset")

# IDE_BALANCED_PREPARED_DATASET_DIR:
# 中文：balanced_multiscale 模式专用切片目录。切换 profile 后会自动使用该目录。
IDE_BALANCED_PREPARED_DATASET_DIR = Path(
    "/mnt/d/Vscode work_place/datasetObjectDetection/prepared_Dataset_balanced_multiscale"
)

# IDE_PREPARED_DATASET_DIR:
# 中文：legacy 兼容别名；如果你原来手动修改过这个变量，legacy 模式仍会读取它。
IDE_PREPARED_DATASET_DIR = IDE_LEGACY_PREPARED_DATASET_DIR

# IDE_USE_AUGMENTED_DATASET:
# 中文：训练时是否把额外增强数据集 `augmented_target_patch_dataset` 一并加入 train。
IDE_USE_AUGMENTED_DATASET = False

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
# 中文：legacy 兼容别名；legacy 模式仍使用该变量作为额外增强目录。
IDE_AUGMENTED_DATASET_DIR = IDE_LEGACY_AUGMENTED_DATASET_DIR

# IDE_PREPROCESS_PROFILE:
# - "legacy": 使用当前旧数据处理流程，不创建多尺度 manifest。
# - "balanced_multiscale": 创建 70% patch256、20% patch512、10% full_scaled 的新数据集。
# 中文：新方案默认建议先用 prepare 模式检查，再切换到 train；旧数据可随时用 legacy 回退。
IDE_PREPROCESS_PROFILE = "balanced_multiscale"


def resolve_profile_prepared_dataset_dir(profile: str) -> Path:
    """Return the IDE-default prepared directory for one preprocessing profile."""
    if profile == "balanced_multiscale":
        return IDE_BALANCED_PREPARED_DATASET_DIR
    return IDE_PREPARED_DATASET_DIR


def resolve_profile_augmented_dataset_dir(profile: str) -> Path:
    """Return the IDE-default augmented directory for one preprocessing profile."""
    if profile == "balanced_multiscale":
        return IDE_BALANCED_AUGMENTED_DATASET_DIR
    return IDE_AUGMENTED_DATASET_DIR


# IDE_VIEW_RATIOS:
# 中文：balanced_multiscale 最终训练清单的视图比例，顺序固定为 256 patch、512 patch、整图缩放。
IDE_VIEW_RATIOS = (0.70, 0.20, 0.10)

# IDE_MULTISCALE_PATCH_SIZES:
# 中文：balanced_multiscale 候选滑窗尺寸；必须正好是 256 和 512，分别对应两类 patch。
IDE_MULTISCALE_PATCH_SIZES = (256, 512)

# IDE_FULL_VIEW_SIZE:
# 中文：整图缩放视图输出的方形尺寸。推荐与 IDE_TRAIN_IMGSZ 相同，避免二次 resize。
IDE_FULL_VIEW_SIZE = 256

# IDE_TRAIN_IMGSZ:
# 中文：训练和验证的 GPU 输入尺寸。新模式默认 256，以控制 8 通道激活显存；512 需要重新预检。
IDE_TRAIN_IMGSZ = 256

# IDE_VAL_BATCH:
# 中文：每个 epoch 验证的 batch，独立于训练 batch；建议固定为 1，避免验证阶段显存峰值。
IDE_VAL_BATCH = 1

# IDE_AMP:
# 中文：是否启用自动混合精度；GPU 训练建议开启，可明显降低训练和验证显存。
IDE_AMP = True

# IDE_MULTISCALE_TOTAL_TRAIN_SAMPLES:
# 中文：balanced_multiscale 每个数据集最终 train 样本数；0 表示根据三类候选容量自动取最大可行值。
IDE_MULTISCALE_TOTAL_TRAIN_SAMPLES = 0

# IDE_MULTISCALE_KEEP_EMPTY:
# 中文：是否把空 patch 放入 balanced_multiscale 候选池；最终是否入选由 70/20/10 配额决定。
IDE_MULTISCALE_KEEP_EMPTY = True

# IDE_STRICT_VIEW_RATIO:
# 中文：是否在生成后严格校验三类样本比例；建议开启，比例不合格时禁止进入训练。
IDE_STRICT_VIEW_RATIO = True

# IDE_DISABLE_HEAVY_AUGMENTATION:
# 中文：是否关闭 Mosaic/MixUp/CutMix/CopyPaste，使 manifest 的视图比例与单张模型输入更一致。
IDE_DISABLE_HEAVY_AUGMENTATION = True

# IDE_ENABLE_FULL_IMAGE_PROFILE:
# 中文：是否在训练开始前执行 1200x900 整图 GPU profile；默认关闭，防止 profile 先触发 OOM。
IDE_ENABLE_FULL_IMAGE_PROFILE = False

# IDE_ENABLE_POST_TRAIN_FEATURE_CORRELATION:
# 中文：训练结束后是否额外执行 head 特征通道相关性验证；默认关闭，避免重复验证造成 OOM。
IDE_ENABLE_POST_TRAIN_FEATURE_CORRELATION = False

# IDE_SAVE_BEFORE_VALIDATION:
# 中文：是否在每个 epoch 验证前保存 pre_val checkpoint，便于验证 OOM 后恢复。
IDE_SAVE_BEFORE_VALIDATION = True

# IDE_MEMORY_SAFETY_FRACTION:
# 中文：显存预检允许使用的显存比例；例如 0.75 表示至少保留 25% 余量。
IDE_MEMORY_SAFETY_FRACTION = 0.75

# PATCH_IOF_THRESHOLD:
# 中文：边缘目标与当前 patch 的 IoF 达到该阈值时才保留。
PATCH_IOF_THRESHOLD = 0.6
MODEL_STATS_WARMUP_RUNS = 10
MODEL_STATS_TIMED_RUNS = 30

# =========================
# Metric Eval Config
# 指标计算参数配置
# 这些参数会影响训练过程中和训练后额外验证时输出的 Box 类指标
# （P / R / mAP50 / mAP50-95）。
#
# Ultralytics 官方默认值说明：
# - `default.yaml` 中 `val` 的 `conf` 默认留空，对常规检测会落到 0.001
# - 对 OBB 任务，`BaseValidator` 会在 `conf is None` 时进一步兜底为 0.01
# - `iou` 官方默认值为 0.7（用于 NMS）
# - `max_det` 官方默认值为 300
# - mAP IoU 阈值范围在官方验证器中固定为 0.50:0.95，共 10 个点
# 中文：这里统一集中管理指标计算相关阈值，后续训练与验证优先使用这里的值。
# =========================

# METRIC_EVAL_CONF:
# - 含义：验证时的置信度阈值，低于该分数的预测不会参与后续统计
# - 取值范围：[0.0, 1.0]
# - 官方 OBB 实际默认值：0.01
METRIC_EVAL_CONF = 0.01

# METRIC_EVAL_NMS_IOU:
# - 含义：验证时 NMS 的 IoU 阈值
# - 取值范围：[0.0, 1.0]
# - 官方默认值：0.7
METRIC_EVAL_NMS_IOU = 0.7

# METRIC_EVAL_MAX_DET:
# - 含义：每张图保留的最多预测框数量
# - 取值范围：正整数，通常 >= 1
# - 官方默认值：300
METRIC_EVAL_MAX_DET = 300

# METRIC_EVAL_AGNOSTIC_NMS:
# - 含义：是否使用类别无关 NMS
# - 取值范围：True / False
# - 官方默认值：False
METRIC_EVAL_AGNOSTIC_NMS = False

# METRIC_EVAL_MAP_IOU_START:
# - 含义：mAP50-95 的起始 IoU 阈值
# - 取值范围：[0.0, 1.0]
# - 官方默认值：0.50
METRIC_EVAL_MAP_IOU_START = 0.50

# METRIC_EVAL_MAP_IOU_END:
# - 含义：mAP50-95 的结束 IoU 阈值
# - 取值范围：[0.0, 1.0]，且必须 >= 起始值
# - 官方默认值：0.95
METRIC_EVAL_MAP_IOU_END = 0.95

# METRIC_EVAL_MAP_IOU_POINTS:
# - 含义：mAP50-95 使用多少个 IoU 采样点
# - 取值范围：正整数，通常 >= 2
# - 官方默认值：10
METRIC_EVAL_MAP_IOU_POINTS = 10


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Describe one dataset split and how to handle its difficult samples.

    中文：描述一个数据集划分，以及该划分中 difficult 目标是否保留。
    """

    # image_dir: original NPY image directory for this split
    # 中文：该划分原始 NPY 图像所在目录
    image_dir: Path
    # label_dir: original raw label directory for this split
    # 中文：该划分原始标签文件所在目录
    label_dir: Path
    # include_difficult: whether to keep objects whose last label column is difficult=1
    # 中文：是否保留标签最后一列中 difficult=1 的目标
    include_difficult: bool


@dataclass(frozen=True)
class MetricEvalConfig:
    """Collect validation-time metric thresholds and related evaluation settings.

    中文：集中管理验证阶段用于计算指标的阈值和评估相关配置。
    """

    conf: float
    iou: float
    max_det: int
    agnostic_nms: bool
    map_iou_start: float
    map_iou_end: float
    map_iou_points: int


@dataclass(frozen=True)
class PrepareConfig:
    """Collect all preprocessing and training settings in one object.

    中文：把数据预处理和训练相关的配置集中到一个对象里管理。
    """

    # train: source configuration for the training split
    # 中文：训练集的源数据配置
    train: DatasetSplitConfig
    # val: source configuration for the validation split
    # 中文：验证集的源数据配置
    val: DatasetSplitConfig
    # save_dir: training project/checkpoint root used by YOLO
    # 中文：YOLO 训练输出目录根路径，用于保存 checkpoint 和日志
    save_dir: Path
    # prepared_dataset_dir: output directory for the prepared patch dataset
    # 中文：切片后数据集输出目录，与训练权重保存目录分离
    prepared_dataset_dir: Path
    # use_augmented_dataset: whether to merge the extra augmented patch dataset into the training split
    # 中文：训练时是否把额外增强 patch 数据集合并进训练集
    use_augmented_dataset: bool
    # augmented_dataset_dir: root directory of the extra augmented patch dataset
    # 中文：额外增强 patch 数据集根目录
    augmented_dataset_dir: Path
    # class_names: classes to keep from the raw label files, in YOLO class-id order
    # 中文：从原始标签中过滤并保留的类别，顺序就是 YOLO 的类别 id 顺序
    class_names: tuple[str, ...]
    # patch_size: sliding crop size as (height, width)
    # 中文：滑窗切片大小，格式为 (高, 宽)
    patch_size: tuple[int, int]
    # overlap: if True, use half-patch stride; otherwise use non-overlapping windows
    # 中文：是否启用重叠滑窗；True 表示步长为 patch 一半
    overlap: bool
    # keep_empty_patches: whether to keep cropped patches with no target objects
    # 中文：是否保留没有目标的空 patch
    keep_empty_patches: bool
    # preprocess_profile: legacy or balanced_multiscale dataset pipeline
    # 中文：数据处理模式；legacy 保持旧流程，balanced_multiscale 启用 70/20/10 视图清单
    preprocess_profile: str
    # view_ratios: ratios for patch256, patch512 and full_scaled in the final train manifest
    # 中文：最终训练清单中三类视图的比例
    view_ratios: tuple[float, float, float]
    # multiscale_patch_sizes: candidate crop sizes for the two patch views
    # 中文：多尺度候选 patch 尺寸，固定对应 256 和 512
    multiscale_patch_sizes: tuple[int, int]
    # full_view_size: square output size for full-image letterbox views
    # 中文：整图缩放视图的方形输出尺寸
    full_view_size: int
    # train_imgsz: GPU input size used by Ultralytics training/validation
    # 中文：训练和验证的统一 GPU 输入尺寸
    train_imgsz: int
    # val_batch: validation batch independent from the training batch
    # 中文：独立的验证 batch，默认 1 以降低 epoch 末尾显存峰值
    val_batch: int
    # amp: automatic mixed precision switch
    # 中文：是否使用自动混合精度
    amp: bool
    # total_train_samples: optional deterministic cap for each balanced dataset
    # 中文：每个 balanced 数据集的 train 样本上限，0 表示按候选容量自动计算
    total_train_samples: int
    # strict_view_ratio: fail preparation when the final manifest misses the target ratio
    # 中文：是否严格验证最终三类视图比例
    strict_view_ratio: bool
    # disable_heavy_augmentation: disable multi-image transforms in the new profile
    # 中文：是否关闭 Mosaic/MixUp/CutMix/CopyPaste，保持视图比例语义
    disable_heavy_augmentation: bool
    # enable_full_image_profile: run the optional full-image GPU timing profile
    # 中文：是否启用训练前整图 GPU profile
    enable_full_image_profile: bool
    # enable_post_train_feature_correlation: run the optional post-training validation pass
    # 中文：是否启用训练结束后的特征相关性验证
    enable_post_train_feature_correlation: bool
    # save_before_validation: save a recovery checkpoint before each validation
    # 中文：是否在每轮验证前保存恢复 checkpoint
    save_before_validation: bool
    # memory_safety_fraction: maximum allowed GPU memory fraction in preflight
    # 中文：显存预检的最大允许占用比例
    memory_safety_fraction: float
    # model: YOLO model config or checkpoint passed to `yolo obb train`
    # 中文：传给 `yolo obb train` 的模型结构配置或模型文件
    model: str
    # pretrained: pretrained weights used for transfer learning, or None for training from scratch
    # 中文：用于迁移学习的预训练权重；如果为 None，则表示从头训练
    pretrained: str | None
    # epochs: total training epochs
    # 中文：总训练轮数
    epochs: int
    # batch: training batch size
    # 中文：训练 batch size
    batch: int
    # workers: dataloader worker count
    # 中文：DataLoader 的并行 worker 数量
    workers: int
    # device: training device, such as "0", "0,1", or "cpu"
    # 中文：训练设备，例如 "0"、"0,1" 或 "cpu"
    device: str | None
    # cache: Ultralytics cache mode, e.g. "disk", "ram", or "False"
    # 中文：Ultralytics 数据缓存方式，例如 "disk"、"ram" 或 "False"
    cache: str
    # seed: random seed used for training reproducibility
    # 中文：训练随机种子，用于提高复现性
    seed: int
    # deterministic: whether to request deterministic training behavior
    # 中文：是否启用确定性训练行为
    deterministic: bool
    # metric_eval: validation-time thresholds for Box metrics
    # 中文：验证阶段用于计算 Box 指标的阈值与相关设置
    metric_eval: MetricEvalConfig
    # run_name: experiment name under save_dir
    # 中文：本次实验在 save_dir 下的运行名
    run_name: str

    @property
    def dataset_root(self) -> Path:
        """Backward-compatible alias of the prepared dataset directory.

        中文：兼容旧字段名，返回切片后数据集保存目录。
        """
        return self.prepared_dataset_dir

    @property
    def data_yaml(self) -> Path:
        """Return the default `data.yaml` path for the prepared dataset.

        中文：返回默认的 `data.yaml` 文件路径。
        """
        return self.dataset_root / "data.yaml"


# Default settings matched to the user's dataset paths and training preferences.
# 中文：这里定义了与你当前数据集路径和训练偏好对应的默认配置。
# 说明：如果你主要是在 IDE 里点运行，优先改文件顶部的 IDE 快速配置区即可。
DEFAULT_CONFIG = PrepareConfig(
    train=DatasetSplitConfig(
        # Training images in NPY format.
        # 中文：训练集原始 NPY 图像目录
        image_dir=Path("/mnt/d/Vscode work_place/datasetObjectDetection/train/images"),
        # Raw training labels in "8 points + class_name + difficult" format.
        # 中文：训练集原始标签目录，标签格式为“8个点 + 类名 + difficult”
        label_dir=Path("/mnt/d/Vscode work_place/datasetObjectDetection/train/labels"),
        # Keep difficult objects during training.
        # 中文：训练时保留困难目标
        include_difficult=True,
    ),
    val=DatasetSplitConfig(
        # Validation images in NPY format.
        # 中文：验证集原始 NPY 图像目录
        image_dir=Path("/mnt/d/Vscode work_place/datasetObjectDetection/test/images"),
        # Raw validation labels in "8 points + class_name + difficult" format.
        # 中文：验证集原始标签目录，标签格式为“8个点 + 类名 + difficult”
        label_dir=Path("/mnt/d/Vscode work_place/datasetObjectDetection/test/labels"),
        # Drop difficult objects during validation.
        # 中文：验证时是否保留困难目标作为GT
        include_difficult=True,
    ),
    # Checkpoint/output root for YOLO runs.
    # 中文：YOLO 训练输出和 checkpoint 保存根目录
    save_dir=Path("/mnt/d/Vscode work_place/datasetObjectDetection/checkpoints"),
    # Prepared patch dataset root.
    # 中文：切片后数据集独立保存目录
    prepared_dataset_dir=resolve_profile_prepared_dataset_dir(IDE_PREPROCESS_PROFILE),
    # Whether to merge the extra augmented patch dataset into train during training.
    # 中文：训练时是否把额外增强 patch 数据集并入 train
    use_augmented_dataset=IDE_USE_AUGMENTED_DATASET,
    # Extra augmented patch dataset root.
    # 中文：额外增强 patch 数据集根目录
    augmented_dataset_dir=resolve_profile_augmented_dataset_dir(IDE_PREPROCESS_PROFILE),
    # Only these classes are kept and remapped to YOLO class ids 0..N-1.
    # 中文：仅保留...类别，并映射为 YOLO 类别 id 0/1/2...
    class_names=("car", "bus", "van", "awning-bike", "truck", "tricycle", "bike", "pedestrian"),
    # Crop each source image into 200x200 patches.
    # 中文：将每张原图切成 200x200 的 patch
    #  这里不是CWH   例如patchsize=(900,1200)，即为 900x1200x8  是为HWC   H=Y，W-X
    patch_size=(256, 256),
    # Use half-patch stride for sliding-window cropping.
    # 中文：使用半个 patch 尺寸作为滑窗步长，形成重叠切片
    overlap=False,
    # Discard patches without any kept object.
    # 中文：丢弃没有目标的空 patch
    keep_empty_patches=IDE_MULTISCALE_KEEP_EMPTY,
    # Keep the old pipeline as the safe default; pass --preprocess-profile balanced_multiscale to use the new one.
    # 中文：默认仍使用旧流程，显式传 balanced_multiscale 才启用新数据集方案。
    preprocess_profile=IDE_PREPROCESS_PROFILE,
    # Ratios for patch256, patch512 and full_scaled in balanced_multiscale mode.
    # 中文：新方案三类视图比例。
    view_ratios=IDE_VIEW_RATIOS,
    # Candidate crop sizes used only by balanced_multiscale mode.
    # 中文：新方案候选 patch 尺寸。
    multiscale_patch_sizes=IDE_MULTISCALE_PATCH_SIZES,
    # Square size used when letterboxing a complete source image.
    # 中文：整图缩放后的输出尺寸。
    full_view_size=IDE_FULL_VIEW_SIZE,
    # GPU input size; kept at 256 by default to reduce 8-channel memory use.
    # 中文：GPU 训练输入尺寸。
    train_imgsz=IDE_TRAIN_IMGSZ,
    # Independent validation batch.
    # 中文：独立验证 batch。
    val_batch=IDE_VAL_BATCH,
    # Enable AMP by default on supported devices.
    # 中文：默认启用自动混合精度。
    amp=IDE_AMP,
    # 0 means automatically choose the largest feasible balanced manifest.
    # 中文：新方案每个数据集的样本总数上限。
    total_train_samples=IDE_MULTISCALE_TOTAL_TRAIN_SAMPLES,
    # Fail closed when view ratios do not pass validation.
    # 中文：严格验证视图比例。
    strict_view_ratio=IDE_STRICT_VIEW_RATIO,
    # Disable multi-image transforms in the new profile.
    # 中文：新方案是否关闭多图增强。
    disable_heavy_augmentation=IDE_DISABLE_HEAVY_AUGMENTATION,
    # Full-image profiling is disabled by default to avoid a pre-training VRAM spike.
    # 中文：训练前整图 profile 开关。
    enable_full_image_profile=IDE_ENABLE_FULL_IMAGE_PROFILE,
    # Post-training feature correlation is optional and uses a separate safe validation pass.
    # 中文：训练后特征相关性统计开关。
    enable_post_train_feature_correlation=IDE_ENABLE_POST_TRAIN_FEATURE_CORRELATION,
    # Save a recovery checkpoint before validation.
    # 中文：验证前 checkpoint 开关。
    save_before_validation=IDE_SAVE_BEFORE_VALIDATION,
    # Keep a memory margin during preflight.
    # 中文：显存预检安全比例。
    memory_safety_fraction=IDE_MEMORY_SAFETY_FRACTION,
    # Build an OBB model that can adapt to custom input channels.
    # 中文：使用模型结构文件构建网络，以便适配自定义输入通道数
    model="yolo26n-obb-5.yaml",
    # Transfer weights from the official pretrained OBB model.
    # 中文：默认预训练权重，默认取自顶部 IDE 快速配置区；None 表示不加载预训练
    pretrained=IDE_PRETRAINED,
    # Default training epochs.
    # 中文：默认训练轮数，默认取自顶部 IDE 快速配置区
    epochs=IDE_EPOCHS,
    # Default batch size.
    # 中文：默认 batch size，默认取自顶部 IDE 快速配置区
    batch=IDE_BATCH,
    # Default dataloader workers.
    # 中文：默认数据加载 worker 数
    workers=4,
    # Use Ultralytics automatic device selection unless overridden.
    # 中文：默认训练设备，默认取自顶部 IDE 快速配置区
    device=IDE_DEVICE,
    # Cache prepared images on disk.
    # 中文：默认将训练数据缓存到磁盘
    cache="disk",
    # Default random seed.
    # 中文：默认随机种子
    seed=IDE_SEED,
    # Default deterministic training switch.
    # 中文：默认启用确定性训练
    deterministic=IDE_DETERMINISTIC,
    # Validation-time metric thresholds and related settings.
    # 中文：验证阶段用于计算 Box 指标的阈值与相关设置
    metric_eval=MetricEvalConfig(
        conf=METRIC_EVAL_CONF,
        iou=METRIC_EVAL_NMS_IOU,
        max_det=METRIC_EVAL_MAX_DET,
        agnostic_nms=METRIC_EVAL_AGNOSTIC_NMS,
        map_iou_start=METRIC_EVAL_MAP_IOU_START,
        map_iou_end=METRIC_EVAL_MAP_IOU_END,
        map_iou_points=METRIC_EVAL_MAP_IOU_POINTS,
    ),
    # Experiment/run name.
    # 中文：实验运行名
    # run_name="yolo26_obb_car_bike_pedestrian_8ch",
    run_name="yolo26n_obb_5_8ch",
    # run_name="yolo26_obb_rgb124_8ch",
)

# Default run mode used when clicking "Run" directly in the IDE without extra CLI arguments.
# 中文：直接在 IDE 里点击运行时使用的默认模式，实际值来自顶部 IDE 快速配置区。
DEFAULT_RUN_MODE = IDE_RUN_MODE


def parse_args() -> argparse.Namespace:
    """Define command-line arguments used to override the default config.

    中文：定义命令行参数，用于覆盖默认配置。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an 8-channel custom OBB dataset from NPY images and optionally launch YOLO26-OBB training."
        )
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help=(
            "Deprecated compatibility flag. If set, it behaves like `--mode prepare_and_train`. "
            "中文：兼容旧用法，等价于 `--mode prepare_and_train`。"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_RUN_MODE,
        choices=("prepare", "train", "prepare_and_train"),
        help="Execution mode. 中文：执行模式，可选仅准备数据、仅训练、或先准备再训练。",
    )
    parser.add_argument(
        "--preprocess-profile",
        type=str,
        default=DEFAULT_CONFIG.preprocess_profile,
        choices=("legacy", "balanced_multiscale"),
        help="Dataset pipeline profile. 中文：legacy 保持旧流程，balanced_multiscale 启用 70/20/10 多视图方案。",
    )
    parser.add_argument(
        "--view-ratios",
        type=str,
        default=",".join(f"{value:.6f}" for value in DEFAULT_CONFIG.view_ratios),
        help="Ratios for patch256,patch512,full_scaled. 中文：新方案三类视图比例，例如 0.7,0.2,0.1。",
    )
    parser.add_argument(
        "--multiscale-patch-sizes",
        type=str,
        default=",".join(str(value) for value in DEFAULT_CONFIG.multiscale_patch_sizes),
        help="Two candidate crop sizes. 中文：新方案 patch 尺寸，必须为 256,512。",
    )
    parser.add_argument(
        "--full-view-size",
        type=int,
        default=DEFAULT_CONFIG.full_view_size,
        help="Square output size for full_scaled views. 中文：整图缩放视图输出尺寸。",
    )
    parser.add_argument(
        "--train-imgsz",
        type=int,
        default=DEFAULT_CONFIG.train_imgsz,
        help="Unified GPU input size. 中文：训练/验证统一输入尺寸，默认 256 以降低显存。",
    )
    parser.add_argument(
        "--val-batch",
        type=int,
        default=DEFAULT_CONFIG.val_batch,
        help="Validation batch independent of train batch. 中文：每轮验证 batch，建议 1。",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.amp,
        help="Enable automatic mixed precision. 中文：是否启用 AMP。",
    )
    parser.add_argument(
        "--total-train-samples",
        type=int,
        default=DEFAULT_CONFIG.total_train_samples,
        help="Balanced manifest sample cap; 0 means automatic. 中文：新方案样本总数上限，0 表示自动。",
    )
    parser.add_argument(
        "--strict-view-ratio",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.strict_view_ratio,
        help="Fail when view ratios are invalid. 中文：是否严格校验 70/20/10 比例。",
    )
    parser.add_argument(
        "--disable-heavy-augmentation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.disable_heavy_augmentation,
        help="Disable Mosaic/MixUp/CutMix/CopyPaste in the new profile. 中文：是否关闭多图增强。",
    )
    parser.add_argument(
        "--enable-full-image-profile",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.enable_full_image_profile,
        help="Run optional full-image GPU profiling before training. 中文：是否启用训练前整图 profile。",
    )
    parser.add_argument(
        "--enable-post-train-feature-correlation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.enable_post_train_feature_correlation,
        help="Run optional post-training feature correlation validation. 中文：是否执行训练后特征统计。",
    )
    parser.add_argument(
        "--save-before-validation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.save_before_validation,
        help="Save a recovery checkpoint before validation. 中文：是否在验证前保存恢复点。",
    )
    parser.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=DEFAULT_CONFIG.memory_safety_fraction,
        help="Maximum GPU memory fraction in preflight. 中文：显存预检最大占用比例。",
    )
    parser.add_argument(
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use overlapping source windows. 中文：是否使用重叠滑窗；balanced_multiscale 强制为 True。",
    )
    parser.add_argument(
        "--keep-empty-patches",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep empty patches in the candidate pool. 中文：是否保留空 patch 候选；新方案默认开启。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG.epochs,
        help="Number of training epochs. 中文：训练总轮数。",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_CONFIG.batch,
        help="Batch size for training. 中文：训练批大小。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG.workers,
        help="Number of dataloader workers. 中文：数据加载并行 worker 数。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG.device,
        help="Training device, e.g. 0, 0,1, or cpu. 中文：训练设备，如 0、0,1 或 cpu。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG.model,
        help="YOLO model definition or checkpoint path passed to the training command. 中文：传给训练命令的模型结构文件或模型路径。",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=DEFAULT_CONFIG.pretrained,
        help=(
            "Pretrained weights used for transfer learning. Use 'None' to disable pretrained weights. "
            "中文：迁移学习使用的预训练权重，传入 'None' 表示不加载预训练。"
        ),
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=DEFAULT_CONFIG.cache,
        choices=("ram", "disk", "False", "false"),
        help="Ultralytics cache mode for training data. 中文：训练数据缓存方式。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.seed,
        help="Random seed used for training reproducibility. 中文：训练随机种子。",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.deterministic,
        help="Enable deterministic training behavior. 中文：是否启用确定性训练。",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_CONFIG.run_name,
        help="Experiment name under save_dir. 中文：保存在 save_dir 下的实验名称。",
    )
    parser.add_argument(
        "--prepared-dataset-dir",
        type=str,
        default=None,
        help="Prepared patch dataset output directory. 中文：切片后数据集输出目录。",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Deprecated alias of --prepared-dataset-dir. 中文：旧参数名，等价于 --prepared-dataset-dir。",
    )
    parser.add_argument(
        "--use-augmented-dataset",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.use_augmented_dataset,
        help="Whether to merge the extra augmented patch dataset into training. 中文：是否把额外增强数据集并入训练。",
    )
    parser.add_argument(
        "--augmented-dataset-dir",
        type=str,
        default=None,
        help=(
            "Root directory of the extra augmented patch dataset. If omitted, the IDE-default directory for "
            "--preprocess-profile is selected automatically. 中文：不指定时按模式自动选择额外增强目录。"
        ),
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        default=IDE_RESUME,
        help=(
            "Resume training from the current run's `weights/last.pt`. "
            "Only supported with `--mode train`. 中文：从当前 run 的 `weights/last.pt` 继续训练，仅支持 `--mode train`。"
        ),
    )
    resume_group.add_argument(
        "--resume-from",
        type=str,
        default=IDE_RESUME_FROM,
        help=(
            "Resume training from a specific checkpoint path or run directory. "
            "Only supported with `--mode train`. 中文：从指定 checkpoint 路径或 run 目录继续训练，仅支持 `--mode train`。"
        ),
    )
    parser.add_argument(
        "--resume-run-index",
        type=int,
        default=IDE_RESUME_RUN_INDEX,
        help=(
            "Optional run index appended as `name-index` when `--resume` is used without `--resume-from`. "
            "中文：当仅使用 `--resume` 且未指定 `--resume-from` 时，自动按 `run_name-序号` 形式拼接 run 目录。"
        ),
    )
    return parser.parse_args()


def load_npy_image(path: Path) -> np.ndarray:
    """Load one NPY image and convert it to contiguous HWC uint8 format.

    中文：读取单个 NPY 图像，并转换成连续存储的 HWC uint8 格式。
    """
    image = np.load(path, allow_pickle=False)
    if image.ndim == 2:
        image = image[..., None]
    elif image.ndim == 3:
        # Normalize configured raw NPY layout to HWC for later cropping/writing.
        # 中文：按照顶部配置的 NPY 轴顺序，把原始数组统一转换为后续切片使用的 HWC。
        if IDE_NPY_LAYOUT == "CHW":
            image = np.transpose(image, (1, 2, 0))
        elif IDE_NPY_LAYOUT == "CWH":
            image = np.transpose(image, (2, 1, 0))
        elif IDE_NPY_LAYOUT == "HWC":
            pass
        else:
            raise ValueError(f"Unsupported IDE_NPY_LAYOUT={IDE_NPY_LAYOUT!r}")
    else:
        raise ValueError(f"Unsupported array rank for {path}: shape={image.shape}")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image = image.astype(np.float32)
    max_value = float(image.max())
    min_value = float(image.min())
    if max_value <= 1.0 and min_value >= 0.0:
        image *= 255.0
    elif max_value > 255.0 or min_value < 0.0:
        denom = max(max_value - min_value, 1e-6)
        image = (image - min_value) * (255.0 / denom)
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def save_multichannel_tiff(path: Path, image: np.ndarray) -> None:
    """Save an HWC multi-channel patch as a multi-page TIFF file.

    中文：把 HWC 多通道 patch 保存成多页 TIFF 文件。
    """
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    ok = cv2.imwritemulti(str(path), chw)
    if not ok:
        raise RuntimeError(f"Failed to write TIFF patch: {path}")


def parse_raw_label_file(
    label_path: Path, class_to_id: dict[str, int], include_difficult: bool
) -> list[tuple[int, np.ndarray]]:
    """Read one raw label file and keep only wanted classes and difficulty settings.

    中文：读取单个原始标签文件，并按类别和 difficult 设置筛选目标。
    """
    annotations: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return annotations

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


def polygon_area(points: np.ndarray) -> float:
    """Return the area of a 2D polygon.

    中文：计算二维多边形面积。
    """
    if len(points) < 3:
        return 0.0
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def clip_polygon_to_rect(points: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray | None:
    """Clip one convex polygon against an axis-aligned rectangle.

    中文：将一个凸多边形裁剪到给定的轴对齐矩形范围内。
    """
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    area, clipped = cv2.intersectConvexConvex(points.astype(np.float32), rect)
    if area <= 1e-6 or clipped is None or len(clipped) < 3:
        return None
    return clipped.reshape(-1, 2)


def rebuild_obb_from_polygon(points: np.ndarray) -> np.ndarray:
    """Rebuild an oriented box from a clipped polygon using the minimum-area rectangle.

    中文：基于裁剪后的多边形，用最小外接旋转矩形重建 OBB。
    """
    rect = cv2.minAreaRect(points.astype(np.float32))
    return cv2.boxPoints(rect).astype(np.float32)


def clip_points_to_bounds(points: np.ndarray, width: float, height: float) -> np.ndarray:
    """Clamp polygon coordinates into image or patch bounds.

    中文：将多边形坐标钳制到图像或 patch 的有效范围内。
    """
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, height)
    return clipped


def sanitize_annotation(points: np.ndarray, image_w: int, image_h: int) -> np.ndarray | None:
    """Fix legacy out-of-bounds OBB annotations by clipping them to the image canvas.

    中文：兼容历史越界标注，将其裁剪到图像有效区域后再重建 OBB。
    """
    if not np.isfinite(points).all():
        return None

    clipped = clip_polygon_to_rect(points, 0.0, 0.0, float(image_w), float(image_h))
    if clipped is None:
        return None

    if polygon_area(clipped) <= 1e-6:
        return None

    # Rebuild after clipping so historical negative coordinates and other out-of-image corners
    # become a valid OBB inside the current image canvas.
    rebuilt = rebuild_obb_from_polygon(clipped)
    rebuilt = clip_points_to_bounds(rebuilt, float(image_w), float(image_h))
    return rebuilt if polygon_area(rebuilt) > 1e-6 else None


def compute_patch_iof(points: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[float, np.ndarray | None]:
    """Compute IoF between one annotation and one patch rectangle.

    中文：计算一个目标与当前 patch 矩形之间的 IoF，并返回交集多边形。
    """
    object_area = polygon_area(points)
    if object_area <= 1e-6:
        return 0.0, None

    clipped = clip_polygon_to_rect(points, float(x0), float(y0), float(x1), float(y1))
    if clipped is None:
        return 0.0, None

    return polygon_area(clipped) / object_area, clipped


def project_annotation_to_patch(
    points: np.ndarray, x0: int, y0: int, x1: int, y1: int, min_iof: float = PATCH_IOF_THRESHOLD
) -> tuple[np.ndarray | None, float]:
    """Clip one annotation into a patch and rebuild the kept region as a new OBB.

    中文：将目标裁剪到 patch 内，并在 IoF 达标时重建新的 OBB。
    """
    iof, clipped = compute_patch_iof(points, x0, y0, x1, y1)
    if clipped is None or iof < min_iof:
        return None, iof

    shifted = clipped.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    rebuilt = rebuild_obb_from_polygon(shifted)
    rebuilt = clip_points_to_bounds(rebuilt, float(x1 - x0), float(y1 - y0))
    return (rebuilt if polygon_area(rebuilt) > 1e-6 else None), iof


def get_starts(length: int, patch: int, step: int) -> list[int]:
    """Generate sliding-window start positions that fully cover one image dimension.

    中文：生成某一个维度上的滑窗起始位置，确保整张图都能被覆盖。
    """
    if length <= patch:
        return [0]

    starts = list(range(0, length - patch + 1, step))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def write_label_file(path: Path, labels: list[tuple[int, np.ndarray]], patch_h: int, patch_w: int) -> None:
    """Write one patch label file in Ultralytics YOLO OBB format with normalized coordinates.

    中文：将单个 patch 的标签写成 Ultralytics 所需的 YOLO OBB 归一化格式。
    """
    lines = []
    for class_id, points in labels:
        normalized = points.copy()
        normalized[:, 0] /= patch_w
        normalized[:, 1] /= patch_h
        coords = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
        lines.append(f"{class_id} {coords}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_split(
    split_name: str,
    split_cfg: DatasetSplitConfig,
    output_root: Path,
    class_to_id: dict[str, int],
    patch_size: tuple[int, int],
    overlap: bool,
    keep_empty_patches: bool,
) -> dict[str, int]:
    """Convert one split from raw NPY+TXT data into cropped TIFF patches and YOLO OBB labels.

    中文：把某个数据划分从原始 NPY+TXT 格式转换成切片后的 TIFF 图像和 YOLO OBB 标签。
    """
    patch_h, patch_w = patch_size
    # Half-patch stride implements overlapping sliding windows.
    # 中文：半个 patch 的步长表示使用重叠滑窗切片。
    step_h = max(patch_h // 2, 1) if overlap else patch_h
    step_w = max(patch_w // 2, 1) if overlap else patch_w

    out_image_dir = output_root / "images" / split_name
    out_label_dir = output_root / "labels" / split_name
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(split_cfg.image_dir.glob("*.npy"))
    stats = {
        "source_images": 0,
        "saved_patches": 0,
        "saved_labels": 0,
        "skipped_empty_patches": 0,
        "kept_objects": 0,
    }

    for image_path in image_paths:
        stats["source_images"] += 1
        label_path = split_cfg.label_dir / f"{image_path.stem}.txt"
        image = load_npy_image(image_path)
        image_h, image_w = image.shape[:2]
        raw_annotations = parse_raw_label_file(label_path, class_to_id, split_cfg.include_difficult)
        annotations = []
        for class_id, points in raw_annotations:
            sanitized = sanitize_annotation(points, image_w, image_h)
            if sanitized is not None:
                annotations.append((class_id, sanitized))

        y_starts = get_starts(image_h, patch_h, step_h)
        x_starts = get_starts(image_w, patch_w, step_w)

        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + patch_h, image_h)
                x1 = min(x0 + patch_w, image_w)
                patch = image[y0:y1, x0:x1]
                kept_labels: list[tuple[int, np.ndarray]] = []

                for class_id, points in annotations:
                    projected, _ = project_annotation_to_patch(points, x0, y0, x1, y1)
                    # Keep intersecting targets only when enough of the object falls into the patch.
                    # 中文：与 patch 相交的目标只有在 IoF 达到阈值后才保留。
                    if projected is None:
                        continue
                    kept_labels.append((class_id, projected))

                if not kept_labels and not keep_empty_patches:
                    stats["skipped_empty_patches"] += 1
                    continue

                patch_name = f"{image_path.stem}__x{x0}_y{y0}"
                patch_image_path = out_image_dir / f"{patch_name}.tiff"
                patch_label_path = out_label_dir / f"{patch_name}.txt"

                save_multichannel_tiff(patch_image_path, patch)
                stats["saved_patches"] += 1

                if kept_labels:
                    write_label_file(patch_label_path, kept_labels, patch.shape[0], patch.shape[1])
                    stats["saved_labels"] += 1
                    stats["kept_objects"] += len(kept_labels)

    return stats


def parse_source_annotations(
    image_path: Path, split_cfg: DatasetSplitConfig, class_to_id: dict[str, int]
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Load one normalized source image and its sanitized pixel-coordinate OBBs."""
    image = load_npy_image(image_path)
    image_h, image_w = image.shape[:2]
    raw = parse_raw_label_file(split_cfg.label_dir / f"{image_path.stem}.txt", class_to_id, split_cfg.include_difficult)
    annotations = [
        (class_id, sanitized)
        for class_id, points in raw
        if (sanitized := sanitize_annotation(points, image_w, image_h)) is not None
    ]
    return image, annotations


def make_multiscale_source_loader(split_cfg: DatasetSplitConfig, class_to_id: dict[str, int]):
    """Return a bounded-memory source loader for balanced candidate materialization."""

    def load_source(image_path: Path) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
        image = load_npy_image(image_path)
        image_h, image_w = image.shape[:2]
        label_path = split_cfg.label_dir / f"{image_path.stem}.txt"
        raw = parse_raw_label_file(label_path, class_to_id, split_cfg.include_difficult)
        annotations = []
        for class_id, points in raw:
            sanitized = sanitize_annotation(points, image_w, image_h)
            if sanitized is not None:
                annotations.append((class_id, sanitized))
        return image, annotations

    return load_source


def prepare_balanced_multiscale_split(
    split_name: str,
    split_cfg: DatasetSplitConfig,
    output_root: Path,
    class_to_id: dict[str, int],
    cfg: PrepareConfig,
    target_class_ids: set[int] | None = None,
    require_target: bool = False,
    transforms: tuple[str, ...] = (IDENTITY_TRANSFORM,),
) -> tuple[dict[str, object], Path]:
    """Generate and validate one deterministic balanced multiscale split."""
    if cfg.multiscale_patch_sizes != (256, 512):
        raise ValueError(
            f"balanced_multiscale requires multiscale_patch_sizes=(256, 512), got {cfg.multiscale_patch_sizes}."
        )
    if not cfg.overlap or not cfg.keep_empty_patches:
        raise ValueError("balanced_multiscale requires overlap=True and keep_empty_patches=True.")
    ratios = ViewRatios(*cfg.view_ratios)
    validate_view_ratios(ratios)
    image_paths = sorted(split_cfg.image_dir.glob("*.npy"))
    if not image_paths:
        raise FileNotFoundError(f"No source NPY images found in {split_cfg.image_dir}.")

    candidates: list[ViewCandidate] = []
    loader = make_multiscale_source_loader(split_cfg, class_to_id)
    for image_path in image_paths:
        image, annotations = loader(image_path)
        candidates.extend(
            generate_candidates(
                image_path=image_path,
                split=split_name,
                image=image,
                annotations=annotations,
                patch_sizes=cfg.multiscale_patch_sizes,
                full_view_size=cfg.full_view_size,
                overlap=True,
                keep_empty_patches=cfg.keep_empty_patches,
                min_iof=PATCH_IOF_THRESHOLD,
                target_class_ids=target_class_ids,
                require_target=require_target,
                transforms=transforms,
            )
        )
        del image, annotations

    selected, selected_counts, capacities = select_candidates(
        candidates,
        ratios=ratios,
        seed=cfg.seed,
        total_samples=cfg.total_train_samples if split_name == "train" else 0,
    )
    image_paths_out, rows = write_selected_candidates(
        selected=selected,
        output_root=output_root,
        source_loader=loader,
        min_iof=PATCH_IOF_THRESHOLD,
        padding_value=114,
        seed=cfg.seed,
    )
    if cfg.strict_view_ratio:
        actual_counts = validate_manifest_rows(rows, ratios)
    else:
        actual_counts = manifest_view_counts(rows)
    manifest_path = write_manifest(rows, output_root / f"{split_name}_manifest.csv")
    stats: dict[str, object] = {
        "source_images": len(image_paths),
        "candidate_counts": capacities,
        "selected_counts": selected_counts,
        "actual_counts": actual_counts,
        "selected_total": len(image_paths_out),
        "view_ratios": ratios.as_dict(),
        "overlap": True,
        "keep_empty_patches": cfg.keep_empty_patches,
        "manifest": str(manifest_path),
        "max_feasible_total": maximum_feasible_total(capacities, ratios),
        "require_target": require_target,
    }
    return stats, manifest_path


def validate_balanced_dataset_files(
    manifest_path: Path,
    expected_channels: int,
    ratios: ViewRatios,
    strict_ratio: bool = True,
    expected_num_classes: int | None = None,
) -> dict[str, object]:
    """Validate every materialized balanced sample and its normalized OBB labels."""
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Balanced manifest is empty: {manifest_path}")
    counts = validate_manifest_rows(rows, ratios) if strict_ratio else manifest_view_counts(rows)
    checked = 0
    empty_labels = 0
    seen_image_paths: set[Path] = set()
    seen_label_paths: set[Path] = set()
    for row in rows:
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Manifest references missing files: {image_path}, {label_path}")
        image_path = image_path.resolve()
        label_path = label_path.resolve()
        if image_path in seen_image_paths or label_path in seen_label_paths:
            raise ValueError(f"Manifest contains duplicate materialized paths: {image_path}, {label_path}")
        seen_image_paths.add(image_path)
        seen_label_paths.add(label_path)
        image = load_multichannel_tiff(image_path)
        if image.ndim != 3 or image.shape[2] != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels for {image_path}, got shape={image.shape}.")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 balanced sample, got {image.dtype} for {image_path}.")
        if image.shape[0] != image.shape[1]:
            raise ValueError(f"Balanced sample is not square: {image_path} shape={image.shape[:2]}.")
        view_type = str(row.get("view_type", ""))
        expected_size = {"patch256": 256, "patch512": 512}.get(view_type)
        if view_type == "full_scaled":
            expected_size = int(row.get("size", image.shape[0]))
        if expected_size is not None and image.shape[:2] != (expected_size, expected_size):
            raise ValueError(
                f"Unexpected {view_type} output size for {image_path}: "
                f"got {image.shape[:2]}, expected {(expected_size, expected_size)}."
            )
        label_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not label_lines:
            empty_labels += 1
        for line in label_lines:
            values = line.split()
            if len(values) != 9:
                raise ValueError(f"Invalid OBB label at {label_path}: expected 9 fields, got {line!r}")
            class_id = int(values[0])
            coordinates = np.asarray([float(value) for value in values[1:]], dtype=np.float32)
            if expected_num_classes is not None and class_id >= expected_num_classes:
                raise ValueError(f"Class id {class_id} exceeds {expected_num_classes} classes at {label_path}.")
            if (
                not np.isfinite(coordinates).all()
                or class_id < 0
                or np.any(coordinates < -1e-6)
                or np.any(coordinates > 1.000001)
            ):
                raise ValueError(f"Out-of-range OBB label at {label_path}: {line!r}")
        if "label_count" in row and int(row["label_count"]) != len(label_lines):
            raise ValueError(
                f"Manifest label_count disagrees with label file {label_path}: "
                f"{row['label_count']} != {len(label_lines)}."
            )
        checked += 1
    result = {
        "manifest": str(manifest_path),
        "checked_samples": checked,
        "empty_label_samples": empty_labels,
        "view_counts": counts,
        "view_ratios": {name: counts[name] / checked for name in counts},
        "expected_channels": expected_channels,
    }
    validation_path = manifest_path.with_name("dataset_validation.json")
    validation_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    return result


def validate_manifest_matches_image_directory(manifest_path: Path, image_dir: Path) -> None:
    """Reject balanced datasets whose materialized files differ from the audited manifest."""
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    manifest_images = {Path(row["image_path"]).resolve() for row in rows}
    actual_images = {path.resolve() for path in get_patch_image_paths(image_dir)}
    if manifest_images != actual_images:
        raise ValueError(
            f"Balanced manifest/image directory mismatch for {image_dir}: "
            f"manifest={len(manifest_images)}, actual={len(actual_images)}."
        )


def write_data_yaml(output_root: Path, class_names: tuple[str, ...], channels: int) -> Path:
    """Generate the Ultralytics dataset YAML that points training to the prepared patch directories.

    中文：生成 Ultralytics 训练所需的 `data.yaml`，指向切片后的数据目录。
    """
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        f"path: {output_root}\ntrain: images/train\nval: images/val\nchannels: {channels}\nnames:\n{names_block}\n"
    )
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def write_custom_data_yaml(
    output_root: Path,
    class_names: tuple[str, ...],
    channels: int,
    train_spec: str,
    val_spec: str,
    output_name: str,
) -> Path:
    """Generate one dataset YAML with custom train/val entries.

    中文：生成一个可自定义 train/val 指向的数据集 YAML。
    """
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        f"path: {output_root}\ntrain: {train_spec}\nval: {val_spec}\nchannels: {channels}\nnames:\n{names_block}\n"
    )
    yaml_path = output_root / output_name
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def get_patch_image_paths(image_dir: Path) -> list[Path]:
    """Collect TIFF patch image paths from one prepared-like split directory.

    中文：从一个 prepared 风格的划分目录中收集 TIFF patch 图像路径。
    """
    return sorted(list(image_dir.glob("*.tiff")) + list(image_dir.glob("*.tif")))


def write_image_list_file(image_paths: list[Path], output_path: Path) -> Path:
    """Write one text file containing absolute image paths, one per line.

    中文：写出一个图像路径列表文件，每行一个绝对路径。
    """
    lines = [str(path.resolve()) for path in image_paths]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def infer_dataset_channels(prepared_dataset_dir: Path, fallback_yaml_path: Path) -> int:
    """Infer channel count from data.yaml or one prepared patch image.

    中文：优先从 data.yaml 读取通道数，读不到时再从任意一张 patch 图像推断。
    """
    if fallback_yaml_path.exists():
        for raw_line in fallback_yaml_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("channels:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    break

    sample_patch = next((prepared_dataset_dir / "images" / "train").glob("*.tiff"), None)
    if sample_patch is None:
        sample_patch = next((prepared_dataset_dir / "images" / "train").glob("*.tif"), None)
    if sample_patch is None:
        raise FileNotFoundError(
            "Failed to infer channels because no training patch image was found under "
            f"{prepared_dataset_dir / 'images' / 'train'}."
        )

    ok, pages = cv2.imreadmulti(str(sample_patch), flags=cv2.IMREAD_UNCHANGED)
    if ok and pages:
        if pages[0].ndim == 2:
            return len(pages)
        return int(pages[0].shape[2]) if pages[0].ndim == 3 else 1

    image = imread(str(sample_patch), flags=cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read prepared patch image: {sample_patch}")
    return int(image.shape[2]) if image.ndim == 3 else 1


def build_training_data_yaml(
    prepared_dataset_dir: Path,
    class_names: tuple[str, ...],
    use_augmented_dataset: bool,
    augmented_dataset_dir: Path,
    preprocess_profile: str = "legacy",
    view_ratios: tuple[float, float, float] = (0.70, 0.20, 0.10),
    strict_view_ratio: bool = True,
) -> Path:
    """Resolve the dataset YAML actually used for training.

    中文：生成并返回训练阶段真正使用的数据集 YAML。
    """
    base_yaml_path = prepared_dataset_dir / "data.yaml"
    if not use_augmented_dataset:
        return base_yaml_path

    prepared_train_dir = prepared_dataset_dir / "images" / "train"
    prepared_val_dir = prepared_dataset_dir / "images" / "val"
    augmented_train_dir = augmented_dataset_dir / "images" / "train"
    if not prepared_train_dir.exists() or not prepared_val_dir.exists():
        raise FileNotFoundError(
            f"Prepared dataset train/val directories are missing under {prepared_dataset_dir}. "
            "Run prepare first or provide a valid prepared dataset path."
        )
    if not augmented_train_dir.exists():
        raise FileNotFoundError(
            f"Augmented dataset train directory not found: {augmented_train_dir}. "
            "Please check --augmented-dataset-dir or disable --use-augmented-dataset."
        )

    prepared_train_images = get_patch_image_paths(prepared_train_dir)
    augmented_train_images = get_patch_image_paths(augmented_train_dir)
    prepared_val_images = get_patch_image_paths(prepared_val_dir)
    if not prepared_train_images:
        raise RuntimeError(f"No prepared training patches found under {prepared_train_dir}.")
    if not augmented_train_images:
        raise RuntimeError(f"No augmented training patches found under {augmented_train_dir}.")
    if not prepared_val_images:
        raise RuntimeError(f"No prepared validation patches found under {prepared_val_dir}.")

    channels = infer_dataset_channels(prepared_dataset_dir, base_yaml_path)
    if preprocess_profile == "balanced_multiscale":
        augmented_manifest = augmented_dataset_dir / "train_manifest.csv"
        if not augmented_manifest.exists():
            raise FileNotFoundError(
                f"Balanced augmented manifest not found: {augmented_manifest}. "
                "Run build_target_class_augmented_obb_dataset.py with --preprocess-profile balanced_multiscale "
                "and --output-mode reset."
            )
        validate_balanced_dataset_files(
            manifest_path=augmented_manifest,
            expected_channels=channels,
            ratios=ViewRatios(*view_ratios),
            strict_ratio=strict_view_ratio,
            expected_num_classes=len(class_names),
        )
        validate_manifest_matches_image_directory(augmented_manifest, augmented_train_dir)
    combined_train_list = write_image_list_file(
        prepared_train_images + augmented_train_images,
        prepared_dataset_dir / "train_with_augmented.txt",
    )
    combined_val_list = write_image_list_file(
        prepared_val_images,
        prepared_dataset_dir / "val_prepared_only.txt",
    )
    return write_custom_data_yaml(
        output_root=prepared_dataset_dir,
        class_names=class_names,
        channels=channels,
        train_spec=combined_train_list.name,
        val_spec=combined_val_list.name,
        output_name="data_with_augmented.yaml",
    )


def load_yolo_obb_label_file(label_path: Path, image_h: int, image_w: int) -> list[tuple[int, np.ndarray]]:
    """Load one YOLO OBB label file and denormalize it back to pixel coordinates.

    中文：读取单个 YOLO OBB 标签文件，并把归一化坐标还原成像素坐标。
    """
    labels: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return labels

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(float(parts[0]))
        coords = np.array([float(v) for v in parts[1:9]], dtype=np.float32).reshape(4, 2)
        coords[:, 0] *= image_w
        coords[:, 1] *= image_h
        labels.append((class_id, coords))
    return labels


def get_obb_size_metrics(points: np.ndarray) -> tuple[float, float, float]:
    """Return long side, short side, and polygon area for one OBB.

    中文：返回单个 OBB 的长边、短边以及像素面积。
    """
    _, (w, h), _ = cv2.minAreaRect(points.astype(np.float32))
    long_side = float(max(w, h))
    short_side = float(min(w, h))
    area = polygon_area(points)
    return long_side, short_side, area


def collect_prepared_split_class_stats(
    dataset_root: Path, split_name: str, class_names: tuple[str, ...]
) -> list[dict[str, int | float | str]]:
    """Collect per-class object distribution and average OBB size from one prepared split.

    中文：从一个切片后的数据划分中统计每类目标分布和平均 OBB 尺寸。
    """
    image_dir = dataset_root / "images" / split_name
    label_dir = dataset_root / "labels" / split_name
    image_paths = sorted(image_dir.glob("*.tiff"))
    per_class = [
        {
            "object_count": 0,
            "long_side_sum": 0.0,
            "short_side_sum": 0.0,
            "area_sum": 0.0,
        }
        for _ in class_names
    ]
    labeled_image_count = 0
    total_objects = 0

    for image_path in image_paths:
        image = imread(str(image_path), flags=cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read prepared patch image: {image_path}")
        image_h, image_w = image.shape[:2]
        labels = load_yolo_obb_label_file(label_dir / f"{image_path.stem}.txt", image_h, image_w)
        if labels:
            labeled_image_count += 1

        for class_id, points in labels:
            if not 0 <= class_id < len(class_names):
                continue
            long_side, short_side, area = get_obb_size_metrics(points)
            stat = per_class[class_id]
            stat["object_count"] += 1
            stat["long_side_sum"] += long_side
            stat["short_side_sum"] += short_side
            stat["area_sum"] += area
            total_objects += 1

    rows: list[dict[str, int | float | str]] = []
    for class_id, class_name in enumerate(class_names):
        stat = per_class[class_id]
        object_count = int(stat["object_count"])
        divisor = max(object_count, 1)
        rows.append(
            {
                "split": split_name,
                "split_image_count": len(image_paths),
                "labeled_image_count": labeled_image_count,
                "class_id": class_id,
                "class_name": class_name,
                "object_count": object_count,
                "object_fraction_in_split": (object_count / total_objects) if total_objects else 0.0,
                "avg_long_side_px": float(stat["long_side_sum"]) / divisor if object_count else 0.0,
                "avg_short_side_px": float(stat["short_side_sum"]) / divisor if object_count else 0.0,
                "avg_area_px2": float(stat["area_sum"]) / divisor if object_count else 0.0,
            }
        )
    return rows


def write_prepared_dataset_stats_csv(dataset_root: Path, class_names: tuple[str, ...]) -> Path:
    """Write a CSV summary of train/val class distribution and average OBB size.

    中文：生成一个 CSV 文件，保存 train/val 每类目标分布和平均 OBB 尺寸。
    """
    output_path = dataset_root / "dataset_class_stats.csv"
    rows: list[dict[str, int | float | str]] = []
    for split_name in ("train", "val"):
        rows.extend(collect_prepared_split_class_stats(dataset_root, split_name, class_names))

    fieldnames = [
        "split",
        "split_image_count",
        "labeled_image_count",
        "class_id",
        "class_name",
        "object_count",
        "object_fraction_in_split",
        "avg_long_side_px",
        "avg_short_side_px",
        "avg_area_px2",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "object_fraction_in_split": f"{float(row['object_fraction_in_split']):.6f}",
                    "avg_long_side_px": f"{float(row['avg_long_side_px']):.6f}",
                    "avg_short_side_px": f"{float(row['avg_short_side_px']):.6f}",
                    "avg_area_px2": f"{float(row['avg_area_px2']):.6f}",
                }
            )
    return output_path


def count_module_parameters(module: torch.nn.Module) -> int:
    """Count the number of parameters in one module.

    中文：统计一个模块的参数总量。
    """
    return sum(parameter.numel() for parameter in module.parameters())


def count_leaf_layers(module: torch.nn.Module) -> int:
    """Count leaf modules, matching Ultralytics' model summary layer definition.

    中文：统计叶子模块数量，与 Ultralytics 模型 summary 中的层数定义保持一致。
    """
    layers = OrderedDict((name, child) for name, child in module.named_modules() if len(child._modules) == 0)
    return len(layers)


def estimate_module_storage_bytes(module: torch.nn.Module) -> int:
    """Estimate in-memory model size from parameters and buffers.

    中文：根据参数和 buffer 估算模型在内存中的大小。
    """
    return sum(tensor.numel() * tensor.element_size() for tensor in (*module.parameters(), *module.buffers()))


def write_model_parameter_stats_csv(torch_model: torch.nn.Module, output_path: Path) -> Path:
    """Write total, backbone and final-head parameter counts to a CSV.

    中文：写出总参数量、排除最终检测头后的参数量以及最终检测头参数量，供训练前后对比。
    """
    base_model = unwrap_model(torch_model)
    modules = getattr(base_model, "model", None)
    if modules is None or len(modules) == 0:
        raise TypeError("Expected a parsed model with a non-empty `.model` module list.")
    total = count_module_parameters(base_model)
    head = count_module_parameters(modules[-1])
    backbone = max(total - head, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["component", "param_count", "param_ratio"])
        writer.writeheader()
        for component, count in (
            ("total", total),
            ("backbone_excluding_final_head", backbone),
            ("detect_head", head),
        ):
            writer.writerow(
                {
                    "component": component,
                    "param_count": int(count),
                    "param_ratio": f"{count / max(total, 1):.6f}",
                }
            )
    return output_path


def resolve_model_size_bytes(torch_model: torch.nn.Module, pretrained: str | None) -> tuple[int, str, str | None]:
    """Resolve model size, preferring an existing pretrained checkpoint file when available.

    中文：解析模型大小；若预训练权重文件存在，则优先使用其实际文件大小。
    """
    pretrained = normalize_pretrained_arg(pretrained)
    if pretrained:
        pretrained_path = Path(pretrained)
        if pretrained_path.exists():
            return pretrained_path.stat().st_size, "pretrained_weight_file", str(pretrained_path)
    return estimate_module_storage_bytes(torch_model), "in_memory_parameters_and_buffers", None


def build_padded_inference_tensor(
    image_path: Path, device: torch.device, stride: int
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int]]:
    """Load one full image and pad it to the model stride for stable whole-image inference profiling.

    中文：读取一张整图，并补齐到模型步长的整数倍，便于稳定统计整图推理耗时。
    """
    image = load_npy_image(image_path)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, but got shape={image.shape} from {image_path}")
    image_h, image_w, image_c = image.shape
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float() / 255.0
    padded_h = int(math.ceil(image_h / stride) * stride)
    padded_w = int(math.ceil(image_w / stride) * stride)
    pad_h = padded_h - image_h
    pad_w = padded_w - image_w
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)
    return tensor.to(device), (image_h, image_w, image_c), (padded_h, padded_w)


def build_patch_inference_tensor(
    image_path: Path, device: torch.device, patch_size: tuple[int, int]
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Load one image and build a single patch tensor for profiling patch-level inference.

    中文：读取一张图像，并裁出单个 patch 张量用于统计 patch 级推理耗时。
    """
    image = load_npy_image(image_path)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, but got shape={image.shape} from {image_path}")
    patch_h, patch_w = patch_size
    image_h, image_w, image_c = image.shape
    crop_h = min(image_h, patch_h)
    crop_w = min(image_w, patch_w)
    patch = image[:crop_h, :crop_w]
    if crop_h != patch_h or crop_w != patch_w:
        pad_h = patch_h - crop_h
        pad_w = patch_w - crop_w
        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    tensor = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device), (patch_h, patch_w, image_c)


def measure_inference_time_ms(
    model: torch.nn.Module, input_tensor: torch.Tensor, warmup_runs: int, timed_runs: int
) -> tuple[float, float]:
    """Measure mean and median forward time for one input tensor.

    中文：统计单个输入张量前向推理的平均和中位耗时。
    """
    timings_ms: list[float] = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(max(warmup_runs, 0)):
            _ = model(input_tensor)
        for _ in range(max(timed_runs, 1)):
            if input_tensor.is_cuda:
                torch.cuda.synchronize(input_tensor.device)
            start = time.perf_counter()
            _ = model(input_tensor)
            if input_tensor.is_cuda:
                torch.cuda.synchronize(input_tensor.device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    if was_training:
        model.train()
    return statistics.mean(timings_ms), statistics.median(timings_ms)


def write_model_profile_csv(
    torch_model: torch.nn.Module, output_path: Path, cfg: PrepareConfig, pretrained: str | None
) -> Path:
    """Write one pre-training CSV with model structure, size, FLOPs and whole-image inference timing.

    中文：训练开始前写出一个 CSV，包含模型层数、参数量、GFLOPS、模型大小和整图推理耗时。
    """
    base_model = unwrap_model(torch_model)
    if not hasattr(base_model, "model") or len(base_model.model) == 0:
        raise TypeError("Expected a parsed Ultralytics model with a non-empty .model module list.")

    head_module = base_model.model[-1]
    device = next(base_model.parameters()).device
    stride = max(int(base_model.stride.max()), 32) if hasattr(base_model, "stride") else 32
    layer_count = count_leaf_layers(base_model)
    total_params = count_module_parameters(base_model)
    head_params = count_module_parameters(head_module)
    backbone_params = max(total_params - head_params, 0)
    total_divisor = max(total_params, 1)
    train_imgsz = int(cfg.train_imgsz)
    gflops_train = float(get_flops(base_model, imgsz=train_imgsz))
    model_size_bytes, model_size_source, model_size_path = resolve_model_size_bytes(base_model, pretrained)

    row: dict[str, str | int | float | None] = {
        "model_name": base_model.__class__.__name__,
        "head_module_name": head_module.__class__.__name__,
        "device": str(device),
        "model_stride": stride,
        "train_patch_size_h": int(cfg.patch_size[0]),
        "train_patch_size_w": int(cfg.patch_size[1]),
        "layer_count": int(layer_count),
        "parameter_count": int(total_params),
        "parameter_count_millions": total_params / 1e6,
        "backbone_parameter_count": int(backbone_params),
        "backbone_parameter_ratio": backbone_params / total_divisor,
        "detect_head_parameter_count": int(head_params),
        "detect_head_parameter_ratio": head_params / total_divisor,
        "gflops_at_train_imgsz": gflops_train,
        "single_patch_height": int(cfg.patch_size[0]),
        "single_patch_width": int(cfg.patch_size[1]),
        "single_patch_channels": None,
        "single_patch_forward_ms_mean": None,
        "single_patch_forward_ms_median": None,
        "full_image_sample_path": None,
        "full_image_height": None,
        "full_image_width": None,
        "full_image_channels": None,
        "full_image_infer_height": None,
        "full_image_infer_width": None,
        "gflops_at_full_image": None,
        "full_image_forward_ms_mean": None,
        "full_image_forward_ms_median": None,
        "model_size_bytes": int(model_size_bytes),
        "model_size_mib": model_size_bytes / (1024 * 1024),
        "model_size_source": model_size_source,
        "model_size_path": model_size_path,
        "warmup_runs": int(MODEL_STATS_WARMUP_RUNS),
        "timed_runs": int(MODEL_STATS_TIMED_RUNS),
    }

    sample_image_path = next(cfg.train.image_dir.glob("*.npy"), None)
    if sample_image_path is not None:
        try:
            patch_tensor, (_patch_h, _patch_w, patch_c) = build_patch_inference_tensor(
                sample_image_path, device, (train_imgsz, train_imgsz)
            )
            patch_forward_mean_ms, patch_forward_median_ms = measure_inference_time_ms(
                base_model, patch_tensor, MODEL_STATS_WARMUP_RUNS, MODEL_STATS_TIMED_RUNS
            )
            row.update(
                {
                    "single_patch_channels": int(patch_c),
                    "single_patch_forward_ms_mean": patch_forward_mean_ms,
                    "single_patch_forward_ms_median": patch_forward_median_ms,
                }
            )
            if cfg.enable_full_image_profile:
                input_tensor, (image_h, image_w, image_c), (infer_h, infer_w) = build_padded_inference_tensor(
                    sample_image_path, device, stride
                )
                forward_mean_ms, forward_median_ms = measure_inference_time_ms(
                    base_model, input_tensor, MODEL_STATS_WARMUP_RUNS, MODEL_STATS_TIMED_RUNS
                )
                row.update(
                    {
                        "full_image_sample_path": str(sample_image_path),
                        "full_image_height": int(image_h),
                        "full_image_width": int(image_w),
                        "full_image_channels": int(image_c),
                        "full_image_infer_height": int(infer_h),
                        "full_image_infer_width": int(infer_w),
                        "gflops_at_full_image": float(get_flops(base_model, imgsz=[infer_h, infer_w])),
                        "full_image_forward_ms_mean": forward_mean_ms,
                        "full_image_forward_ms_median": forward_median_ms,
                    }
                )
        except Exception as exc:
            row["full_image_sample_path"] = f"{sample_image_path} (profiling_failed: {exc})"
        finally:
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "head_module_name",
                "device",
                "model_stride",
                "train_patch_size_h",
                "train_patch_size_w",
                "layer_count",
                "parameter_count",
                "parameter_count_millions",
                "backbone_parameter_count",
                "backbone_parameter_ratio",
                "detect_head_parameter_count",
                "detect_head_parameter_ratio",
                "gflops_at_train_imgsz",
                "single_patch_height",
                "single_patch_width",
                "single_patch_channels",
                "single_patch_forward_ms_mean",
                "single_patch_forward_ms_median",
                "full_image_sample_path",
                "full_image_height",
                "full_image_width",
                "full_image_channels",
                "full_image_infer_height",
                "full_image_infer_width",
                "gflops_at_full_image",
                "full_image_forward_ms_mean",
                "full_image_forward_ms_median",
                "model_size_bytes",
                "model_size_mib",
                "model_size_source",
                "model_size_path",
                "warmup_runs",
                "timed_runs",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                **row,
                "parameter_count_millions": f"{float(row['parameter_count_millions']):.6f}",
                "backbone_parameter_ratio": f"{float(row['backbone_parameter_ratio']):.6f}",
                "detect_head_parameter_ratio": f"{float(row['detect_head_parameter_ratio']):.6f}",
                "gflops_at_train_imgsz": f"{float(row['gflops_at_train_imgsz']):.6f}",
                "single_patch_forward_ms_mean": (
                    f"{float(row['single_patch_forward_ms_mean']):.6f}"
                    if row["single_patch_forward_ms_mean"] is not None
                    else ""
                ),
                "single_patch_forward_ms_median": (
                    f"{float(row['single_patch_forward_ms_median']):.6f}"
                    if row["single_patch_forward_ms_median"] is not None
                    else ""
                ),
                "gflops_at_full_image": (
                    f"{float(row['gflops_at_full_image']):.6f}" if row["gflops_at_full_image"] is not None else ""
                ),
                "full_image_forward_ms_mean": (
                    f"{float(row['full_image_forward_ms_mean']):.6f}"
                    if row["full_image_forward_ms_mean"] is not None
                    else ""
                ),
                "full_image_forward_ms_median": (
                    f"{float(row['full_image_forward_ms_median']):.6f}"
                    if row["full_image_forward_ms_median"] is not None
                    else ""
                ),
                "model_size_mib": f"{float(row['model_size_mib']):.6f}",
            }
        )
    return output_path


def compute_avg_abs_channel_correlation(feature_map: torch.Tensor) -> float:
    """Compute mean absolute Pearson correlation across channels of one feature map.

    中文：计算单个特征图在通道维上的平均绝对皮尔逊相关系数。
    """
    if feature_map.ndim != 4 or feature_map.shape[1] < 2:
        return 0.0

    flattened = feature_map.detach().float().permute(1, 0, 2, 3).reshape(feature_map.shape[1], -1)
    flattened = flattened - flattened.mean(dim=1, keepdim=True)
    std = flattened.std(dim=1, unbiased=False, keepdim=True)
    valid = (std.squeeze(1) > 1e-12).nonzero(as_tuple=False).flatten()
    if valid.numel() < 2:
        return 0.0

    normalized = flattened[valid] / std[valid].clamp_min(1e-12)
    corr = normalized @ normalized.T / normalized.shape[1]
    mask = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
    if not mask.any():
        return 0.0
    return float(corr.abs()[mask].mean().item())


def write_head_feature_correlation_csv(
    model: YOLO, data_yaml: Path, save_dir: Path, cfg: PrepareConfig, args: argparse.Namespace
) -> Path:
    """Run a final validation pass and save head-input feature-map channel correlations to CSV.

    中文：在训练结束后额外跑一次验证，并把检测头输入特征图的通道相关性写入 CSV。
    """
    base_model = unwrap_model(model.model)
    if not hasattr(base_model, "model") or len(base_model.model) == 0:
        raise TypeError("Expected a parsed Ultralytics model with a non-empty .model module list.")
    head_module = base_model.model[-1]
    stats: list[dict[str, float]] = []
    batch_count = 0

    def _pre_hook(_module, inputs) -> None:
        nonlocal batch_count
        features = inputs[0] if inputs else None
        if not isinstance(features, list):
            return
        if not stats:
            stats.extend(
                {"corr_sum": 0.0, "channel_sum": 0.0, "height_sum": 0.0, "width_sum": 0.0} for _ in range(len(features))
            )

        batch_count += 1
        for idx, feature_map in enumerate(features):
            stats[idx]["corr_sum"] += compute_avg_abs_channel_correlation(feature_map)
            stats[idx]["channel_sum"] += float(feature_map.shape[1])
            stats[idx]["height_sum"] += float(feature_map.shape[2])
            stats[idx]["width_sum"] += float(feature_map.shape[3])

    hook = head_module.register_forward_pre_hook(_pre_hook)
    try:
        val_kwargs = {
            "data": str(data_yaml),
            "imgsz": cfg.train_imgsz,
            "batch": cfg.val_batch,
            "workers": args.workers,
            "split": "val",
            "plots": False,
            "project": str(save_dir),
            "name": "post_train_feature_corr_val",
            "exist_ok": True,
        }
        val_kwargs.update(build_metric_eval_kwargs(cfg.metric_eval))
        if args.device:
            val_kwargs["device"] = args.device
        model.val(validator=make_metric_eval_validator(cfg.metric_eval), **val_kwargs)
    finally:
        hook.remove()

    if batch_count == 0 or not stats:
        raise RuntimeError("No validation feature maps were captured for head correlation analysis.")

    output_path = save_dir / "head_feature_channel_correlation.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_map_index",
                "batch_count",
                "avg_channels",
                "avg_height",
                "avg_width",
                "avg_abs_channel_corr",
            ],
        )
        writer.writeheader()
        for idx, stat in enumerate(stats):
            writer.writerow(
                {
                    "feature_map_index": idx,
                    "batch_count": batch_count,
                    "avg_channels": f"{stat['channel_sum'] / batch_count:.6f}",
                    "avg_height": f"{stat['height_sum'] / batch_count:.6f}",
                    "avg_width": f"{stat['width_sum'] / batch_count:.6f}",
                    "avg_abs_channel_corr": f"{stat['corr_sum'] / batch_count:.6f}",
                }
            )
    return output_path


def to_preview_bgr(image: np.ndarray) -> np.ndarray:
    """Convert a multi-channel patch to a 3-channel image for preview only.

    中文：把多通道 patch 转成仅用于可视化检查的 3 通道图像。
    """
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[2] == 1:
        preview = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 2:
        zero = np.zeros_like(image[..., :1])
        preview = np.concatenate((image, zero), axis=2)
    else:
        preview = image[..., :3]
    return np.ascontiguousarray(preview.astype(np.uint8))


def draw_obb_labels(
    image: np.ndarray, labels: list[tuple[int, np.ndarray]], class_names: tuple[str, ...]
) -> np.ndarray:
    """Draw OBB polygons and class names on one preview image.

    中文：在单张预览图上绘制 OBB 多边形和类别名称。
    """
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
        text_y = max(y - 6, 14)
        cv2.putText(canvas, class_name, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def create_label_preview_grid(
    dataset_root: Path, split_name: str, class_names: tuple[str, ...], sample_count: int = IDE_PREVIEW_SAMPLES
) -> Path | None:
    """Sample prepared patches, draw labels, and save a single mosaic preview image.

    中文：从切片后的数据集中抽样若干 patch，画出标签后拼成一张总览图并保存。
    """
    image_dir = dataset_root / "images" / split_name
    label_dir = dataset_root / "labels" / split_name
    image_paths = sorted(image_dir.glob("*.tiff"))
    if not image_paths:
        return None

    sample_count = min(sample_count, len(image_paths))
    sampled_paths = random.Random(0).sample(image_paths, sample_count)
    preview_tiles: list[np.ndarray] = []

    for image_path in sampled_paths:
        image = imread(str(image_path), flags=cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        preview = to_preview_bgr(image)
        image_h, image_w = preview.shape[:2]
        labels = load_yolo_obb_label_file(label_dir / f"{image_path.stem}.txt", image_h, image_w)
        preview = draw_obb_labels(preview, labels, class_names)
        cv2.putText(
            preview,
            image_path.stem,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        preview_tiles.append(preview)

    if not preview_tiles:
        return None

    tile_h, tile_w = preview_tiles[0].shape[:2]
    grid_cols = max(1, math.ceil(math.sqrt(sample_count)))
    grid_rows = max(1, math.ceil(sample_count / grid_cols))
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    while len(preview_tiles) < grid_rows * grid_cols:
        preview_tiles.append(blank.copy())

    row_images = []
    for row_idx in range(grid_rows):
        row_tiles = preview_tiles[row_idx * grid_cols : (row_idx + 1) * grid_cols]
        row_images.append(np.concatenate(row_tiles, axis=1))
    mosaic = np.concatenate(row_images, axis=0)

    output_path = dataset_root / f"{split_name}_labels_preview_grid.jpg"
    ok = cv2.imwrite(str(output_path), mosaic)
    if not ok:
        raise RuntimeError(f"Failed to write label preview grid: {output_path}")
    return output_path


def build_train_command(cfg: PrepareConfig, args: argparse.Namespace, data_yaml: Path) -> list[str]:
    """Assemble a readable equivalent training command for display.

    中文：根据配置和命令行覆盖项，拼出等价的训练命令字符串用于显示。
    """
    cache = "False" if str(args.cache).lower() == "false" else str(args.cache)
    command = [
        "python",
        "examples/custom_obb_prepare_and_train.py",
        "--mode",
        "train",
        "--preprocess-profile",
        cfg.preprocess_profile,
        "--prepared-dataset-dir",
        str(cfg.prepared_dataset_dir),
        "--model",
        str(args.model),
        "--pretrained",
        str(args.pretrained),
        "--epochs",
        str(args.epochs),
        "--train-imgsz",
        str(cfg.train_imgsz),
        "--val-batch",
        str(cfg.val_batch),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--cache",
        cache,
        "--seed",
        str(args.seed),
        "--run-name",
        str(args.run_name),
        "--view-ratios",
        ",".join(f"{value:.6f}" for value in cfg.view_ratios),
        "--multiscale-patch-sizes",
        ",".join(str(value) for value in cfg.multiscale_patch_sizes),
        "--full-view-size",
        str(cfg.full_view_size),
        "--total-train-samples",
        str(cfg.total_train_samples),
        "--memory-safety-fraction",
        str(cfg.memory_safety_fraction),
    ]
    command.append("--amp" if cfg.amp else "--no-amp")
    command.append("--deterministic" if args.deterministic else "--no-deterministic")
    command.append("--strict-view-ratio" if cfg.strict_view_ratio else "--no-strict-view-ratio")
    command.append(
        "--disable-heavy-augmentation" if cfg.disable_heavy_augmentation else "--no-disable-heavy-augmentation"
    )
    command.append("--enable-full-image-profile" if cfg.enable_full_image_profile else "--no-enable-full-image-profile")
    command.append(
        "--enable-post-train-feature-correlation"
        if cfg.enable_post_train_feature_correlation
        else "--no-enable-post-train-feature-correlation"
    )
    command.append("--save-before-validation" if cfg.save_before_validation else "--no-save-before-validation")
    command.append("--overlap" if cfg.overlap else "--no-overlap")
    command.append("--keep-empty-patches" if cfg.keep_empty_patches else "--no-keep-empty-patches")
    command.append("--use-augmented-dataset" if bool(args.use_augmented_dataset) else "--no-use-augmented-dataset")
    command.extend(["--augmented-dataset-dir", str(cfg.augmented_dataset_dir)])
    if args.device:
        command.extend(["--device", str(args.device)])
    return command


def resolve_mode(args: argparse.Namespace) -> str:
    """Resolve the final execution mode, keeping backward compatibility with `--train`.

    中文：解析最终执行模式，并兼容旧的 `--train` 参数。
    """
    if args.train and args.mode == DEFAULT_RUN_MODE:
        return "prepare_and_train"
    return args.mode


def parse_numeric_tuple(text: str, length: int, cast, name: str) -> tuple:
    """Parse a comma-separated numeric tuple with an exact length."""
    try:
        values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric comma-separated values, got {text!r}.") from exc
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values, got {values}.")
    return values


def build_runtime_config(args: argparse.Namespace) -> PrepareConfig:
    """Build and validate the immutable runtime configuration from CLI values."""
    profile = str(args.preprocess_profile)
    profile_prepared_dataset_dir = resolve_profile_prepared_dataset_dir(profile)
    profile_augmented_dataset_dir = resolve_profile_augmented_dataset_dir(profile)
    augmented_dataset_dir = (
        Path(args.augmented_dataset_dir) if args.augmented_dataset_dir else profile_augmented_dataset_dir
    )
    ratios = parse_numeric_tuple(args.view_ratios, 3, float, "view_ratios")
    validate_view_ratios(ViewRatios(*ratios))
    multiscale_sizes = parse_numeric_tuple(args.multiscale_patch_sizes, 2, int, "multiscale_patch_sizes")
    if profile == "balanced_multiscale" and multiscale_sizes != (256, 512):
        raise ValueError(f"balanced_multiscale requires patch sizes 256,512, got {multiscale_sizes}.")
    if any(value <= 0 for value in multiscale_sizes):
        raise ValueError(f"multiscale_patch_sizes must be positive, got {multiscale_sizes}.")
    if args.full_view_size <= 0 or args.train_imgsz <= 0:
        raise ValueError("full_view_size and train_imgsz must be positive.")
    if profile == "balanced_multiscale" and args.full_view_size != args.train_imgsz:
        raise ValueError(
            "balanced_multiscale requires full_view_size == train_imgsz to avoid a second spatial transform; "
            f"got {args.full_view_size} and {args.train_imgsz}."
        )
    if args.val_batch < 1:
        raise ValueError(f"val_batch must be >= 1, got {args.val_batch}.")
    if args.total_train_samples < 0:
        raise ValueError(f"total_train_samples must be >= 0, got {args.total_train_samples}.")
    if not 0.0 < args.memory_safety_fraction <= 1.0:
        raise ValueError(f"memory_safety_fraction must be in (0, 1], got {args.memory_safety_fraction}.")

    if profile == "balanced_multiscale":
        # ``None`` means "use the profile default".  The balanced profile's
        # default is True even though legacy keeps the historical setting.
        overlap = True if args.overlap is None else bool(args.overlap)
        keep_empty = True if args.keep_empty_patches is None else bool(args.keep_empty_patches)
        if not overlap:
            raise ValueError("balanced_multiscale requires overlap=True; remove --no-overlap.")
        if not keep_empty:
            raise ValueError("balanced_multiscale requires keep_empty_patches=True; remove --no-keep-empty-patches.")
        overlap = True
        keep_empty = True
    else:
        overlap = DEFAULT_CONFIG.overlap if args.overlap is None else bool(args.overlap)
        keep_empty = (
            DEFAULT_CONFIG.keep_empty_patches if args.keep_empty_patches is None else bool(args.keep_empty_patches)
        )

    return PrepareConfig(
        train=DEFAULT_CONFIG.train,
        val=DEFAULT_CONFIG.val,
        save_dir=DEFAULT_CONFIG.save_dir,
        prepared_dataset_dir=profile_prepared_dataset_dir,
        use_augmented_dataset=bool(args.use_augmented_dataset),
        augmented_dataset_dir=augmented_dataset_dir,
        class_names=DEFAULT_CONFIG.class_names,
        patch_size=DEFAULT_CONFIG.patch_size,
        overlap=overlap,
        keep_empty_patches=keep_empty,
        preprocess_profile=profile,
        view_ratios=ratios,
        multiscale_patch_sizes=multiscale_sizes,
        full_view_size=int(args.full_view_size),
        train_imgsz=int(args.train_imgsz),
        val_batch=int(args.val_batch),
        amp=bool(args.amp),
        total_train_samples=int(args.total_train_samples),
        strict_view_ratio=bool(args.strict_view_ratio),
        disable_heavy_augmentation=bool(args.disable_heavy_augmentation),
        enable_full_image_profile=bool(args.enable_full_image_profile),
        enable_post_train_feature_correlation=bool(args.enable_post_train_feature_correlation),
        save_before_validation=bool(args.save_before_validation),
        memory_safety_fraction=float(args.memory_safety_fraction),
        model=args.model,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=args.cache,
        seed=args.seed,
        deterministic=args.deterministic,
        metric_eval=DEFAULT_CONFIG.metric_eval,
        run_name=args.run_name,
    )


def normalize_pretrained_arg(pretrained: str | None) -> str | None:
    """Normalize pretrained argument so both Python `None` and string 'None' disable weight loading.

    中文：规范化 pretrained 参数，使 Python 的 None 和字符串 'None' 都表示不加载权重。
    """
    if pretrained is None:
        return None
    if isinstance(pretrained, str) and pretrained.strip().lower() == "none":
        return None
    return pretrained


def resolve_resume_run_name(args: argparse.Namespace) -> str:
    """Resolve the run directory name used for automatic resume.

    中文：解析自动断点续训时使用的 run 目录名。
    """
    run_index = getattr(args, "resume_run_index", None)
    if run_index is None:
        return args.run_name
    if run_index < 0:
        raise ValueError(f"`resume_run_index` must be >= 0, but got {run_index}.")
    return f"{args.run_name}-{run_index}"


def resolve_resume_checkpoint(cfg: PrepareConfig, args: argparse.Namespace, mode: str) -> Path | None:
    """Resolve the resumable checkpoint path from CLI arguments.

    中文：根据命令行参数解析用于断点续训的 checkpoint 路径。
    """
    if not (args.resume or args.resume_from):
        return None
    if mode != "train":
        raise ValueError(
            "Resume is only supported with `--mode train` to avoid rebuilding prepared datasets. "
            "Please switch to `--mode train` before resuming."
        )

    resume_run_name = resolve_resume_run_name(args)
    candidate = Path(args.resume_from) if args.resume_from else (cfg.save_dir / resume_run_name / "weights" / "last.pt")
    if candidate.is_dir():
        run_last = candidate / "weights" / "last.pt"
        direct_last = candidate / "last.pt"
        if run_last.exists():
            candidate = run_last
        elif direct_last.exists():
            candidate = direct_last

    if not candidate.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {candidate}. "
            "Please provide `--resume-from /path/to/last.pt` or make sure the resolved run directory "
            "contains `weights/last.pt`."
        )
    return candidate.resolve()


def validate_metric_eval_config(metric_eval: MetricEvalConfig) -> MetricEvalConfig:
    """Validate metric-evaluation thresholds before training or manual validation starts.

    中文：在训练或额外验证开始前校验指标计算相关阈值是否合法。
    """
    if not 0.0 <= metric_eval.conf <= 1.0:
        raise ValueError(f"METRIC_EVAL_CONF must be within [0, 1], but got {metric_eval.conf}.")
    if not 0.0 <= metric_eval.iou <= 1.0:
        raise ValueError(f"METRIC_EVAL_NMS_IOU must be within [0, 1], but got {metric_eval.iou}.")
    if metric_eval.max_det < 1:
        raise ValueError(f"METRIC_EVAL_MAX_DET must be >= 1, but got {metric_eval.max_det}.")
    if not 0.0 <= metric_eval.map_iou_start <= 1.0:
        raise ValueError(f"METRIC_EVAL_MAP_IOU_START must be within [0, 1], but got {metric_eval.map_iou_start}.")
    if not 0.0 <= metric_eval.map_iou_end <= 1.0:
        raise ValueError(f"METRIC_EVAL_MAP_IOU_END must be within [0, 1], but got {metric_eval.map_iou_end}.")
    if metric_eval.map_iou_start > metric_eval.map_iou_end:
        raise ValueError(
            "METRIC_EVAL_MAP_IOU_START must be <= METRIC_EVAL_MAP_IOU_END, "
            f"but got {metric_eval.map_iou_start} > {metric_eval.map_iou_end}."
        )
    if metric_eval.map_iou_points < 2:
        raise ValueError(f"METRIC_EVAL_MAP_IOU_POINTS must be >= 2, but got {metric_eval.map_iou_points}.")
    return metric_eval


def build_metric_eval_kwargs(metric_eval: MetricEvalConfig) -> dict[str, float | int | bool]:
    """Build validation kwargs from the configured metric-evaluation settings.

    中文：根据指标计算配置生成验证阶段的参数字典。
    """
    return {
        "conf": metric_eval.conf,
        "iou": metric_eval.iou,
        "max_det": metric_eval.max_det,
        "agnostic_nms": metric_eval.agnostic_nms,
    }


def apply_metric_eval_config_to_validator(validator: OBBValidator, metric_eval: MetricEvalConfig) -> None:
    """Apply configured metric thresholds to one validator instance.

    中文：把指标计算配置应用到具体的验证器实例上。
    """
    validator.args.conf = metric_eval.conf
    validator.args.iou = metric_eval.iou
    validator.args.max_det = metric_eval.max_det
    validator.args.agnostic_nms = metric_eval.agnostic_nms
    validator.iouv = torch.linspace(metric_eval.map_iou_start, metric_eval.map_iou_end, metric_eval.map_iou_points)
    validator.niou = validator.iouv.numel()


def build_metric_eval_log_values(
    metric_eval: MetricEvalConfig, validator: OBBValidator | None = None
) -> dict[str, str]:
    """Return concise logging values for the currently effective metric thresholds.

    中文：返回当前生效指标阈值的简洁日志文本。
    """
    conf = metric_eval.conf
    iou = metric_eval.iou
    max_det = metric_eval.max_det
    agnostic_nms = metric_eval.agnostic_nms
    map_iou_start = metric_eval.map_iou_start
    map_iou_end = metric_eval.map_iou_end
    map_iou_points = metric_eval.map_iou_points

    if validator is not None:
        conf = float(getattr(validator.args, "conf", conf))
        iou = float(getattr(validator.args, "iou", iou))
        max_det = int(getattr(validator.args, "max_det", max_det))
        agnostic_nms = bool(getattr(validator.args, "agnostic_nms", agnostic_nms))
        if getattr(validator, "iouv", None) is not None and validator.iouv.numel():
            map_iou_start = float(validator.iouv[0].item())
            map_iou_end = float(validator.iouv[-1].item())
            map_iou_points = int(validator.iouv.numel())

    return {
        "val_conf": f"{conf:.3f}",
        "nms_iou": f"{iou:.3f}",
        "max_det": str(max_det),
        "agn_nms": str(agnostic_nms),
        "map_iou": f"{map_iou_start:.2f}:{map_iou_end:.2f}/{map_iou_points}",
    }


def make_metric_eval_validator(metric_eval: MetricEvalConfig) -> type[OBBValidator]:
    """Create an OBB validator class that applies the configured metric thresholds on init.

    中文：创建一个会在初始化时自动应用指标阈值配置的 OBB 验证器类。
    """

    class ConfiguredOBBValidator(OBBValidator):
        def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
            super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
            apply_metric_eval_config_to_validator(self, metric_eval)

    return ConfiguredOBBValidator


def load_or_build_model(model_name: str, pretrained: str | None, resume_checkpoint: Path | None = None) -> YOLO:
    """Build a YOLO model and optionally load pretrained weights.

    中文：构建 YOLO 模型，并在提供预训练权重时加载权重。
    """
    if resume_checkpoint is not None:
        return YOLO(str(resume_checkpoint))
    model = YOLO(model_name)
    pretrained = normalize_pretrained_arg(pretrained)
    if pretrained:
        model = model.load(pretrained)
    return model


class SafeOBBTrainer(OBBTrainer):
    """OBB trainer with an independent validation batch and validation-OOM recovery."""

    safe_val_batch = 1
    save_before_validation = True

    def _build_train_pipeline(self) -> None:
        """Build the normal pipeline, then replace its validation loader with a safer loader."""
        super()._build_train_pipeline()
        validation_batch = max(int(self.safe_val_batch), 1)
        validation_batch = max(validation_batch // max(self.world_size, 1), 1)
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=validation_batch,
            rank=LOCAL_RANK,
            mode="val",
        )
        LOGGER.info(f"Safe OBB validation batch={validation_batch}; train batch={self.batch_size}.")

    def _save_pre_validation_checkpoint(self) -> None:
        """Save a checkpoint before validation so a validation OOM can be resumed."""
        if not self.save_before_validation or RANK not in {-1, 0}:
            return
        try:
            self.save_model()
            if self.last.exists():
                shutil.copy2(self.last, self.last.with_name("pre_val.pt"))
        except Exception as exc:
            LOGGER.warning(f"Could not save pre-validation checkpoint: {exc}")

    def validate(self):
        """Run validation and continue training if validation exhausts CUDA memory."""
        if getattr(self, "_validation_oom", False):
            return self.metrics, self.best_fitness if self.best_fitness is not None else 0.0
        self._save_pre_validation_checkpoint()
        try:
            return super().validate()
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            LOGGER.warning("Validation ran out of CUDA memory; skipping subsequent in-epoch validation safely.")
            self._validation_oom = True
            self._clear_memory()
            return self.metrics, self.best_fitness if self.best_fitness is not None else 0.0

    def final_eval(self) -> None:
        """Protect final evaluation from turning a completed training run into a failure."""
        if getattr(self, "_validation_oom", False):
            LOGGER.warning("Final validation skipped because validation previously hit CUDA OOM.")
            self._clear_memory()
            return
        try:
            super().final_eval()
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            LOGGER.warning("Final validation ran out of CUDA memory; checkpoints remain available.")
            self._clear_memory()


def is_cuda_oom_error(error: BaseException) -> bool:
    """Return whether an exception represents a CUDA out-of-memory failure.

    Some PyTorch/driver combinations raise ``RuntimeError`` with an OOM message instead of the dedicated
    ``torch.cuda.OutOfMemoryError`` class. Treat both forms alike so an epoch-end validation failure cannot silently
    bypass the recovery path.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return isinstance(error, RuntimeError) and "out of memory" in message


def make_safe_obb_trainer(val_batch: int, save_before_validation: bool) -> type[SafeOBBTrainer]:
    """Create a per-run safe trainer class without leaking settings globally."""
    safe_val_batch_value = max(int(val_batch), 1)
    safe_save_before_validation_value = bool(save_before_validation)

    class ConfiguredSafeOBBTrainer(SafeOBBTrainer):
        safe_val_batch = safe_val_batch_value
        save_before_validation = safe_save_before_validation_value

    ConfiguredSafeOBBTrainer.__name__ = "ConfiguredSafeOBBTrainer"
    return ConfiguredSafeOBBTrainer


def resolve_preflight_device(device: str | None) -> torch.device | None:
    """Resolve a single CUDA device for the lightweight pre-training forward."""
    if not torch.cuda.is_available() or str(device).lower() == "cpu":
        return None
    device_text = str(device).strip().lower() if device is not None else ""
    if "," in device_text:
        # DDP creates its own processes and device assignment; a parent
        # process must not allocate a competing probe tensor.
        return None
    if device_text in {"", "none", "auto"}:
        return torch.device("cuda:0")
    if device_text.isdigit():
        return torch.device(f"cuda:{device_text}")
    if device_text.startswith("cuda"):
        return torch.device(device_text)
    return None


def run_training_memory_preflight(
    model: YOLO,
    cfg: PrepareConfig,
    data_yaml: Path,
    device: str | None,
    requested_batch: int,
) -> int:
    """Probe a real autograd forward and conservatively adjust batch size.

    The probe uses the dataset channel count and the exact training image size, so a mismatch between a 3-channel
    assumption and the actual 8-channel model is detected before the trainer starts. It deliberately does not probe the
    full 1200x900 image.
    """
    if cfg.preprocess_profile != "balanced_multiscale":
        return requested_batch
    probe_device = resolve_preflight_device(device)
    if probe_device is None:
        return requested_batch
    channels = infer_dataset_channels(data_yaml.parent, data_yaml)
    torch_model = unwrap_model(model.model)
    was_training = torch_model.training
    requested_probe_batch = max(int(requested_batch), 1)
    probe_batch = requested_probe_batch

    def tensor_values(value):
        """Yield tensors nested in the model's train-mode output dictionaries."""
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from tensor_values(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from tensor_values(child)

    while True:
        probe = None
        output = None
        loss = None
        try:
            torch_model.to(probe_device)
            probe = torch.zeros(
                (1 if requested_batch < 1 else probe_batch, channels, cfg.train_imgsz, cfg.train_imgsz),
                dtype=torch.float32,
                device=probe_device,
                requires_grad=True,
            )
            # Eval mode avoids changing BatchNorm running statistics, while
            # the explicit backward still exercises activation/gradient memory.
            torch_model.eval()
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if cfg.amp else nullcontext()
            with torch.enable_grad(), autocast_context:
                output = torch_model(probe)
                tensors = [tensor for tensor in tensor_values(output) if tensor.requires_grad]
                if not tensors:
                    raise RuntimeError("Balanced preflight model output contains no differentiable tensors.")
                loss = sum(tensor.float().mean() for tensor in tensors)
                loss.backward()
            break
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            if probe_batch <= 1:
                gc.collect()
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"The balanced training preflight already OOMed at batch=1, imgsz={cfg.train_imgsz}, "
                    f"channels={channels}. Reduce --train-imgsz or use a smaller model."
                ) from exc
            probe_batch = max(probe_batch // 2, 1)
            LOGGER.warning(f"Preflight CUDA OOM; retrying the training memory probe with batch={probe_batch}.")
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            torch_model.zero_grad(set_to_none=True)
            del probe, output, loss
            if probe_device.type == "cuda":
                torch.cuda.empty_cache()

    if was_training:
        torch_model.train()

    if probe_device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(probe_device).total_memory
        reserved_fraction = torch.cuda.memory_reserved(probe_device) / max(total_memory, 1)
        LOGGER.info(
            f"Balanced memory preflight passed: imgsz={cfg.train_imgsz}, channels={channels}, "
            f"reserved_fraction={reserved_fraction:.3f}."
        )
        if reserved_fraction > cfg.memory_safety_fraction and requested_batch > 1:
            LOGGER.warning(
                f"Reserved GPU memory fraction {reserved_fraction:.3f} exceeds safety limit "
                f"{cfg.memory_safety_fraction:.3f}; reducing training batch to 1."
            )
            return 1
    return probe_batch if requested_batch > 0 else requested_batch


def register_epoch_tqdm_callbacks(model: YOLO, metric_eval: MetricEvalConfig) -> None:
    """Register a stable epoch-level tqdm progress bar for IDE-friendly training output.

    中文：注册一个按 epoch 更新的 tqdm 进度条，适合在 IDE 控制台中稳定显示训练进度。
    """
    state = {"pbar": None, "completed": 0}

    def _format_postfix(trainer) -> dict[str, str]:
        postfix: dict[str, str] = {}

        if trainer.tloss is not None:
            tloss = trainer.tloss.tolist() if hasattr(trainer.tloss, "tolist") else trainer.tloss
            if not isinstance(tloss, list):
                tloss = [float(tloss)]
            loss_names = (
                trainer.loss_names if len(trainer.loss_names) == len(tloss) else [f"loss{i}" for i in range(len(tloss))]
            )
            for name, value in zip(loss_names, tloss):
                postfix[name] = f"{float(value):.4f}"

        if isinstance(trainer.metrics, dict):
            for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                if key in trainer.metrics:
                    short_key = key.split("/")[-1].replace("(B)", "")
                    postfix[short_key] = f"{float(trainer.metrics[key]):.4f}"

        validator = getattr(trainer, "validator", None)
        postfix.update(build_metric_eval_log_values(metric_eval, validator))

        return postfix

    def on_train_start(trainer) -> None:
        total_epochs = int(trainer.epochs)
        state["completed"] = 0
        state["pbar"] = tqdm(total=total_epochs, desc="Training", unit="epoch", dynamic_ncols=True)

    def on_fit_epoch_end(trainer) -> None:
        pbar = state["pbar"]
        if pbar is None:
            return
        target_completed = int(trainer.epoch) + 1
        delta = max(target_completed - state["completed"], 0)
        if delta:
            pbar.update(delta)
            state["completed"] = target_completed
        pbar.set_description(f"Epoch {target_completed}/{int(trainer.epochs)}")
        postfix = _format_postfix(trainer)
        if postfix:
            pbar.set_postfix(postfix, refresh=False)

    def on_train_end(trainer) -> None:
        pbar = state["pbar"]
        if pbar is not None:
            if state["completed"] < int(trainer.epochs):
                pbar.update(int(trainer.epochs) - state["completed"])
            pbar.close()
            state["pbar"] = None

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def register_model_stats_callback(model: YOLO, cfg: PrepareConfig, pretrained: str | None) -> None:
    """Register a callback that saves one consolidated model-profile CSV when training starts.

    中文：注册训练开始回调，在 run 目录中保存一份合并后的模型统计 CSV。
    """

    def on_train_start(trainer) -> None:
        write_model_profile_csv(
            trainer.model, Path(trainer.save_dir) / "model_profile_before_train.csv", cfg, pretrained
        )

    model.add_callback("on_train_start", on_train_start)


def register_metric_eval_callbacks(model: YOLO, metric_eval: MetricEvalConfig) -> None:
    """Register callbacks that apply and log metric-evaluation thresholds used by validation.

    中文：注册回调，在训练内验证前应用并记录指标计算相关阈值。
    """

    def on_pretrain_routine_end(trainer) -> None:
        validator = getattr(trainer, "validator", None)
        if isinstance(validator, OBBValidator):
            apply_metric_eval_config_to_validator(validator, metric_eval)

    def on_train_start(trainer) -> None:
        validator = getattr(trainer, "validator", None)
        metric_log_values = build_metric_eval_log_values(metric_eval, validator)
        print("Metric eval config: " + ", ".join(f"{key}={value}" for key, value in metric_log_values.items()))

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_start", on_train_start)


def train_with_python_api(cfg: PrepareConfig, args: argparse.Namespace, data_yaml: Path) -> None:
    """Start training through the Ultralytics Python API instead of subprocess CLI.

    中文：通过 Ultralytics 的 Python API 启动训练，不再依赖外部 CLI 命令。
    """
    validate_metric_eval_config(cfg.metric_eval)
    cache: str | bool = False if str(args.cache).lower() == "false" else args.cache
    mode = resolve_mode(args)
    resume_checkpoint = resolve_resume_checkpoint(cfg, args, mode)
    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
    model = load_or_build_model(args.model, args.pretrained, resume_checkpoint=resume_checkpoint)
    safe_mode = cfg.preprocess_profile == "balanced_multiscale"
    trainer_class = make_safe_obb_trainer(cfg.val_batch, cfg.save_before_validation) if safe_mode else None
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": cfg.train_imgsz if safe_mode else max(cfg.patch_size),
        "batch": args.batch,
        "workers": args.workers,
        "cache": cache,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "project": str(cfg.save_dir),
        "name": args.run_name,
    }
    if safe_mode:
        train_kwargs["amp"] = cfg.amp
    if cfg.preprocess_profile == "balanced_multiscale" and cfg.disable_heavy_augmentation:
        # Multi-image augmentation changes the meaning of the 70/20/10
        # manifest ratio because one tensor then contains several views.
        train_kwargs.update({"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "multi_scale": 0.0})
    train_kwargs.update(build_metric_eval_kwargs(cfg.metric_eval))
    if args.device:
        train_kwargs["device"] = args.device
    if resume_checkpoint is not None:
        train_kwargs["resume"] = str(resume_checkpoint)
    if safe_mode:
        train_kwargs["batch"] = run_training_memory_preflight(
            model=model,
            cfg=cfg,
            data_yaml=data_yaml,
            device=args.device,
            requested_batch=int(args.batch),
        )

    # The base trainer already retries first-epoch training OOMs.  This outer
    # loop additionally resumes from the most recent checkpoint if a later
    # epoch fails, reducing the batch size before trying again.  Validation
    # OOMs are handled inside SafeOBBTrainer and do not enter this loop.
    retry_count = 0
    while True:
        register_epoch_tqdm_callbacks(model, cfg.metric_eval)
        register_model_stats_callback(model, cfg, args.pretrained)
        register_metric_eval_callbacks(model, cfg.metric_eval)
        try:
            if trainer_class is None:
                model.train(**train_kwargs)
            else:
                model.train(trainer=trainer_class, **train_kwargs)
            break
        except Exception as exc:
            if not safe_mode or not is_cuda_oom_error(exc):
                raise
            trainer = getattr(model, "trainer", None)
            checkpoint = getattr(trainer, "last", None)
            actual_batch = int(getattr(trainer, "batch_size", train_kwargs["batch"]))
            checkpoint = Path(checkpoint) if checkpoint is not None else None
            if RANK != -1 or checkpoint is None or not checkpoint.exists() or actual_batch <= 1 or retry_count >= 3:
                raise RuntimeError(
                    "Training still runs out of CUDA memory after the built-in retries. "
                    "Use --batch 1, a smaller --train-imgsz, or a smaller model."
                ) from exc
            retry_count += 1
            next_batch = max(actual_batch // 2, 1)
            LOGGER.warning(
                f"Training CUDA OOM; resuming from {checkpoint} with batch={next_batch} (retry {retry_count}/3)."
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_or_build_model(args.model, args.pretrained, resume_checkpoint=checkpoint)
            train_kwargs["batch"] = next_batch
            train_kwargs["resume"] = str(checkpoint)

    train_save_dir = Path(model.trainer.save_dir)
    run_feature_correlation = cfg.enable_post_train_feature_correlation or not safe_mode
    if run_feature_correlation:
        try:
            feature_corr_csv = write_head_feature_correlation_csv(model, data_yaml, train_save_dir, cfg, args)
            print(f"Saved head feature correlation CSV: {feature_corr_csv}")
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                raise
            LOGGER.warning("Post-training feature correlation skipped because it ran out of CUDA memory.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        LOGGER.info("Post-training feature correlation is disabled by configuration.")


def main() -> None:
    """Run preprocessing, write metadata, and optionally start YOLO training.

    中文：执行数据预处理、写出元信息，并在需要时启动 YOLO 训练。
    """
    args = parse_args()
    cfg = build_runtime_config(args)
    mode = resolve_mode(args)

    prepared_dataset_dir_arg = args.prepared_dataset_dir or args.dataset_root
    prepared_dataset_dir = Path(prepared_dataset_dir_arg) if prepared_dataset_dir_arg else cfg.prepared_dataset_dir
    cfg = replace(cfg, prepared_dataset_dir=prepared_dataset_dir)
    yaml_path = prepared_dataset_dir / "data.yaml"
    training_data_yaml_path = yaml_path
    dataset_stats_csv_path: Path | None = None

    if mode in {"prepare", "prepare_and_train"}:
        # Recreate the prepared dataset directory to avoid mixing old and new patch outputs.
        # 中文：每次重建输出目录，避免旧切片和新切片混在一起。
        if prepared_dataset_dir.exists():
            shutil.rmtree(prepared_dataset_dir)
        prepared_dataset_dir.mkdir(parents=True, exist_ok=True)
        class_to_id = {name: i for i, name in enumerate(cfg.class_names)}

        balanced_validation: dict[str, object] | None = None
        train_manifest_path: Path | None = None
        if cfg.preprocess_profile == "balanced_multiscale":
            train_stats, train_manifest_path = prepare_balanced_multiscale_split(
                split_name="train",
                split_cfg=cfg.train,
                output_root=prepared_dataset_dir,
                class_to_id=class_to_id,
                cfg=cfg,
            )
            sample_image = load_npy_image(next(cfg.train.image_dir.glob("*.npy")))
            channels = sample_image.shape[2]
            balanced_validation = validate_balanced_dataset_files(
                train_manifest_path,
                expected_channels=channels,
                ratios=ViewRatios(*cfg.view_ratios),
                strict_ratio=cfg.strict_view_ratio,
                expected_num_classes=len(cfg.class_names),
            )
            validate_manifest_matches_image_directory(train_manifest_path, prepared_dataset_dir / "images" / "train")
            # Keep validation deterministic and small; full-image evaluation is separate.
            val_stats = prepare_split(
                split_name="val",
                split_cfg=cfg.val,
                output_root=prepared_dataset_dir,
                class_to_id=class_to_id,
                patch_size=(256, 256),
                overlap=False,
                keep_empty_patches=True,
            )
        else:
            train_stats = prepare_split(
                split_name="train",
                split_cfg=cfg.train,
                output_root=prepared_dataset_dir,
                class_to_id=class_to_id,
                patch_size=cfg.patch_size,
                overlap=cfg.overlap,
                keep_empty_patches=cfg.keep_empty_patches,
            )
            val_stats = prepare_split(
                split_name="val",
                split_cfg=cfg.val,
                output_root=prepared_dataset_dir,
                class_to_id=class_to_id,
                patch_size=cfg.patch_size,
                overlap=cfg.overlap,
                keep_empty_patches=cfg.keep_empty_patches,
            )
            sample_image = load_npy_image(next(cfg.train.image_dir.glob("*.npy")))

        sample_train_patch = next((prepared_dataset_dir / "images" / "train").glob("*.tiff"), None)
        if sample_train_patch is None:
            raise RuntimeError("No training patches were generated. Please check label filtering and patch settings.")

        # Read one source image to infer the number of channels written into `data.yaml`.
        # 中文：读取一张原图来推断通道数，并写入 `data.yaml`。
        channels = sample_image.shape[2]
        yaml_path = write_data_yaml(prepared_dataset_dir, cfg.class_names, channels)
        if mode == "prepare_and_train":
            dataset_stats_csv_path = write_prepared_dataset_stats_csv(prepared_dataset_dir, cfg.class_names)
        train_preview_path = create_label_preview_grid(
            prepared_dataset_dir, "train", cfg.class_names, IDE_PREVIEW_SAMPLES
        )
        val_preview_path = create_label_preview_grid(prepared_dataset_dir, "val", cfg.class_names, IDE_PREVIEW_SAMPLES)

        training_data_yaml_path = build_training_data_yaml(
            prepared_dataset_dir=prepared_dataset_dir,
            class_names=cfg.class_names,
            use_augmented_dataset=cfg.use_augmented_dataset,
            augmented_dataset_dir=cfg.augmented_dataset_dir,
            preprocess_profile=cfg.preprocess_profile,
            view_ratios=cfg.view_ratios,
            strict_view_ratio=cfg.strict_view_ratio,
        )
        summary = {
            "mode": mode,
            "class_names": list(cfg.class_names),
            "patch_size": list(cfg.patch_size),
            "preprocess_profile": cfg.preprocess_profile,
            "view_ratios": dict(zip(("patch256", "patch512", "full_scaled"), cfg.view_ratios)),
            "multiscale_patch_sizes": list(cfg.multiscale_patch_sizes),
            "full_view_size": cfg.full_view_size,
            "train_imgsz": cfg.train_imgsz,
            "val_batch": cfg.val_batch,
            "overlap": cfg.overlap,
            "keep_empty_patches": cfg.keep_empty_patches,
            "strict_view_ratio": cfg.strict_view_ratio,
            "train_manifest": str(train_manifest_path) if train_manifest_path else None,
            "balanced_validation": balanced_validation,
            "patch_iof_threshold": PATCH_IOF_THRESHOLD,
            "use_augmented_dataset": cfg.use_augmented_dataset,
            "augmented_dataset_dir": str(cfg.augmented_dataset_dir),
            "train_include_difficult": cfg.train.include_difficult,
            "val_include_difficult": cfg.val.include_difficult,
            "channels": channels,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "prepared_dataset_dir": str(prepared_dataset_dir),
            "dataset_root": str(prepared_dataset_dir),
            "data_yaml": str(yaml_path),
            "training_data_yaml": str(training_data_yaml_path),
            "dataset_stats_csv": str(dataset_stats_csv_path) if dataset_stats_csv_path else None,
            "train_preview_grid": str(train_preview_path) if train_preview_path else None,
            "val_preview_grid": str(val_preview_path) if val_preview_path else None,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        summary_path = prepared_dataset_dir / "prepare_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

        train_command = build_train_command(cfg, args, training_data_yaml_path)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        print("\nRecommended train command:\n")
        print(" ".join(f'"{item}"' if " " in item else item for item in train_command))

    if mode == "train":
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Prepared dataset yaml not found: {yaml_path}. "
                "Run with --mode prepare first, or switch to --mode prepare_and_train."
            )
        if cfg.preprocess_profile == "balanced_multiscale":
            train_manifest_path = prepared_dataset_dir / "train_manifest.csv"
            if not train_manifest_path.exists():
                raise FileNotFoundError(
                    f"Balanced train manifest not found: {train_manifest_path}. "
                    "Run --mode prepare with the balanced profile first."
                )
            channels = infer_dataset_channels(prepared_dataset_dir, yaml_path)
            validate_balanced_dataset_files(
                train_manifest_path,
                expected_channels=channels,
                ratios=ViewRatios(*cfg.view_ratios),
                strict_ratio=cfg.strict_view_ratio,
                expected_num_classes=len(cfg.class_names),
            )
            validate_manifest_matches_image_directory(train_manifest_path, prepared_dataset_dir / "images" / "train")
        training_data_yaml_path = build_training_data_yaml(
            prepared_dataset_dir=prepared_dataset_dir,
            class_names=cfg.class_names,
            use_augmented_dataset=cfg.use_augmented_dataset,
            augmented_dataset_dir=cfg.augmented_dataset_dir,
            preprocess_profile=cfg.preprocess_profile,
            view_ratios=cfg.view_ratios,
            strict_view_ratio=cfg.strict_view_ratio,
        )
        print(f"Training from dataset yaml: {training_data_yaml_path}")

    if mode in {"train", "prepare_and_train"}:
        if dataset_stats_csv_path is None:
            dataset_stats_csv_path = write_prepared_dataset_stats_csv(prepared_dataset_dir, cfg.class_names)
        print(f"Saved dataset class stats CSV: {dataset_stats_csv_path}")
        if cfg.use_augmented_dataset:
            print(f"Using augmented dataset train split from: {cfg.augmented_dataset_dir}")
        train_with_python_api(cfg, args, training_data_yaml_path)


if __name__ == "__main__":
    main()
