"""Whole-image YOLO26-OBB training entry for the prepared 8-channel dataset (IDE-friendly).

本文件是 ``examples/prepare_obb_dataset.py`` 生成数据集的配套训练入口：
整图训练（不切 patch），直接消费 ``prepared_obb_dataset/data.yaml``。

设计说明
========

1. 整图训练，imgsz 与部署一致
   - 数据预处理阶段保持 1200x900 原图，训练时由官方 dataloader letterbox
     到 ``IDE_IMGSZ``（默认 1184，32 的整数倍，接近原生分辨率）。
   - 训练与部署（整图验证）共用同一输入口径，消除"256 patch 训练 vs
     1200 整图推理"的尺度鸿沟与置信度漂移。

2. 训练内验证口径 = data.yaml（labels/，含 difficult）
   - 训练过程中每个 epoch 的验证跟随 ``IDE_DATA_YAML``；
   - 训练结束后若 ``IDE_CLEAN_VAL_AFTER_TRAIN=True``，自动用
     ``data_clean.yaml``（labels_clean/，剔除 difficult）再跑一次官方验证，
     输出 clean 口径的 P/R/mAP，供与含 difficult 口径对比。

3. 8 通道与预训练
   - OBBTrainer 会用 ``data.yaml`` 的 ``channels: 8`` 重建模型首层，
     并从 3 通道预训练权重部分迁移首层卷积（仓库 fork 已实现
     ``model.0.conv.weight`` 的前 c1/c2 通道拷贝），其余层按名匹配加载。

4. IDE 使用方式
   - 直接改顶部配置区，然后点击 IDE 的运行按钮即可；脚本也会在启动时
     打印等价的 CLI 命令，方便复现与存档。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.obb.train import OBBTrainer  # noqa: E402


# =========================
# 1. 数据与运行控制
# =========================

# IDE_DATA_YAML:
# - 训练数据集 YAML（由 prepare_obb_dataset.py 生成）。
# - data.yaml      : labels/ 含 difficult 目标（训练 + 训练内标准评估）
# - data_clean.yaml: labels_clean/ 剔除 difficult（仅用于训练后 clean 指标评估）
# 中文：训练主数据文件；默认用 data.yaml（difficult 参与训练，避免难例变隐式负样本）。
IDE_DATA_YAML = "/home/mofengwei/datasetObjectDetection/prepared_obb_dataset/data.yaml"

# IDE_DATA_CLEAN_YAML:
# - 训练结束后自动验证的 clean 口径数据文件（difficult 已剔除）。
# - 若设为 None，则跳过训练后 clean 验证。
# 中文：训练后 clean 口径验证数据；None 表示不执行。
IDE_DATA_CLEAN_YAML = "/home/mofengwei/datasetObjectDetection/prepared_obb_dataset_clean/data.yaml"

# IDE_CLASS_NAMES:
# - 类别名列表，顺序必须与 prepare_obb_dataset.py 的 IDE_CLASS_NAMES 完全一致
#   （即 data.yaml 中 names 的 id 映射），稀有类过采样与类别加权都依赖它。
# 中文：类别名顺序（必须与预处理脚本一致）。
IDE_CLASS_NAMES = ("car", "bus", "van", "awning-bike", "truck", "tricycle", "bike", "pedestrian")

# IDE_RUN_MODE:
# - "train": 仅训练
# - "train_and_clean_val": 训练 + 训练结束后用 clean 口径再验证一次
# 中文：最常改的运行阶段开关。
IDE_RUN_MODE = "train_and_clean_val"

# IDE_DEVICE:
# - "0"    : 使用第 0 张 GPU（本工程只有一张显卡，多卡训练已禁用）
# - "cpu"  : CPU（很慢，仅调试）
# - None   : 自动选择
# - 注意：不要填写 "0,1" 等多卡写法，脚本会直接拒绝启动。
# 中文：训练设备开关（仅支持单卡或 CPU）。
IDE_DEVICE: str | None = "0"

# IDE_EPOCHS:
# - 训练总轮数。数据量较大（约 76000 目标）时 100 轮起步，
#   配合 IDE_PATIENCE 让官方早停兜底。
# 中文：训练轮数。
IDE_EPOCHS = 150

# IDE_PATIENCE:
# - 验证指标连续多少轮无改善就提前停止；-1 表示不早停。
# 中文：早停耐心值。
IDE_PATIENCE = 30

# IDE_BATCH:
# - 训练批大小。-1 表示自动：脚本会先做真实显存预检（forward+backward），
#   从 16 开始逐级下降直到放得下，输出安全 batch 再开始训练。
# - 整图 1184 输入 + 8 通道在 8GB 显存（RTX 3060 Ti）下实测安全范围：
#   imgsz=1184 -> batch 2-4；imgsz=1024 -> batch 4-6；imgsz=640 -> batch 8-16。
# 中文：训练 batch size；-1 = 显存预检自动选择（推荐）。
IDE_BATCH = -1

# IDE_IMGSZ:
# - 训练/验证输入边长（正方形 letterbox）。
# - 1184 = 37x32，最接近 1200 原生宽度，目标像素损失最小；
#   8GB 显存下建议先用 1024（letterbox 缩放 0.85，目标仅缩小 15%），
#   显存仍不足再降到 896 / 640 / 512（目标会变小，小目标更难）。
# 中文：训练输入尺寸，必须为 32 的倍数。
IDE_IMGSZ = 1184

# IDE_SEED / IDE_DETERMINISTIC:
# - 随机种子与确定性开关；固定后结果可复现，便于不同模型公平对比。
# 中文：随机种子与确定性训练开关。
IDE_SEED = 0
IDE_DETERMINISTIC = True

# =========================
# 2. 模型与预训练权重
# =========================

# IDE_MODEL:
# - 模型结构文件：填 ultralytics/cfg/models/26/ 下的 YAML 文件名即可。
# - 可选：yolo26n-obb.yaml（基线）、yolo26-obb-p2.yaml（加 P2 头）、
#   yolo26-obb-4.yaml（C3k2_PC）、yolo26-obb-p2-4.yaml（P2+C3k2_PC）、
#   yolo26-obb-9.yaml（backbone 特征注入）。
# - 也支持直接填 .pt 权重路径（从该权重继续训练，结构跟随权重）。
# 中文：模型结构 YAML 名或权重路径。
IDE_MODEL = "yolo26n-obb-bifpn-add.yaml"

# IDE_PRETRAINED_SCOPE:
# - 预训练权重加载范围开关，层编号以 ultralytics/cfg/models/26/yolo26-obb.yaml 为准
#   （每行 YAML 对应模型 model.<index> 一层，第 23 层为 OBB26 检测头）：
#   "backbone"      : 只加载 backbone 预训练权重（第 0~10 层），用于修改 neck 的实验对比。
#   "backbone+neck" : 只加载 backbone+neck 预训练权重（第 0~22 层），用于修改 OBB26 检测头的实验对比。
#   "model"         : 加载全部预训练权重（第 0~23 层，含 OBB26 检测头），用于修改损失函数等训练侧逻辑的实验对比。
# - 注意：backbone+neck 按层号过滤源权重，目标 YAML 的第 11~22 行需与 baseline 对齐，
#   该模式只在“仅替换第 23 层检测头”的实验里语义严格成立；neck 中增删层会导致索引错位。
# 中文：预训练权重加载范围开关；当前默认配置（bifpn 改 neck 实验）用 backbone。
IDE_PRETRAINED_SCOPE = "backbone"

# IDE_PRETRAINED:
# - 预训练权重：官方 yolo26n-obb.pt（3 通道 80 类，首层自动适配 8 通道）。
# - None 表示从头训练（不推荐，8 通道浅层特征将完全随机初始化）。
# 中文：预训练权重路径；None 表示从头训练。
IDE_PRETRAINED: str | None = "/home/mofengwei/ultralytics/yolo26n-obb.pt"

# IDE_RESUME:
# - True: 从 IDE_RESUME_RUN_NAME 对应 run 目录的 weights/last.pt 断点续训。
# - False: 新开一轮训练。
# 中文：是否从上次中断点继续训练。
IDE_RESUME = False

# IDE_RESUME_RUN_NAME:
# - 仅 IDE_RESUME=True 时生效：续训的 run 目录名（不含序号）。
# - 例如 "yolo26n_obb_whole_image-1" 的序号在 IDE_RESUME_RUN_INDEX 指定。
# 中文：断点续训的 run 名。
IDE_RESUME_RUN_NAME = "yolo26n_obb_whole_image"

# IDE_RESUME_RUN_INDEX:
# - None: 直接使用 IDE_RESUME_RUN_NAME；
# - 7:    使用 f"{IDE_RESUME_RUN_NAME}-7"。
# 中文：续训 run 的序号后缀。
IDE_RESUME_RUN_INDEX: int | None = None

# =========================
# 3. 输出目录
# =========================

# IDE_PROJECT:
# - 训练输出（weights/results）的根目录，每个 run 一个子目录。
# 中文：checkpoint 保存根目录。
IDE_PROJECT = "/home/mofengwei/datasetObjectDetection/checkpoints"

# IDE_RUN_NAME:
# - 本次实验的 run 名，可读且区分实验即可，建议包含模型名+数据口径。
# 中文：本次运行的实验名。
IDE_RUN_NAME = "yolo26n_obb_bifpn-add_backbone_pretrain"

# =========================
# 4. 数据加载与精度
# =========================

# IDE_WORKERS:
# - DataLoader 子进程数；CPU 核数充足时可调大（如 8-16）。
# 中文：数据加载 worker 数。
IDE_WORKERS = 8

# IDE_PIN_MEMORY:
# - 训练 dataloader 是否使用固定内存（pinned memory）。
# - 默认 False：本机为 WSL2 环境，pin_memory 线程在显存紧张时会出现
#   虚假的 "CUDA error: out of memory"（实测 batch=16 时仅用 4.3G/8G 就崩溃），
#   关闭后可稳定复现 batch=8@640 完整训练（峰值 2.9G）。
# - 影响：关闭后 CPU->GPU 拷贝略慢（通常 <5% 训练耗时），稳定性优先。
# 中文：训练 dataloader 是否启用固定内存（WSL2 下建议 False）。
IDE_PIN_MEMORY = False

# IDE_CACHE:
# - "disk": 图像缓存到磁盘（推荐，8 通道 TIFF 解码慢，缓存可显著提速）
# - "ram" : 缓存到内存（显存之外还要吃内存）
# - False : 不缓存
# 中文：数据缓存模式。
IDE_CACHE = "disk"

# IDE_VAL_BATCH:
# - 仅用于训练后 clean 口径验证的 batch（模型训练内的验证 batch 跟随训练 batch）。
# 中文：训练后验证 batch。
IDE_VAL_BATCH = 1

# IDE_AMP:
# - 自动混合精度，GPU 训练建议开启（省显存且更快）。
# 中文：是否启用 AMP。
IDE_AMP = True

# =========================
# 5. 数据增强（OBB 重点）
# =========================
# 说明：OBB 标签会随图像同步旋转/翻转/缩放（官方 loader 自动处理），
# 因此下列增强对 OBB 是几何安全的。

# IDE_DEGREES:
# - 随机旋转角度（度）。OBB 模型角度自由度大，强烈建议开 5~15 度；
#   本数据集目标多为道路车辆（朝向分布有限），建议从 5 度起步。
# 中文：随机旋转增强（0=关闭）。
IDE_DEGREES = 5.0

# IDE_FLIPLR / IDE_FLIPUD:
# - 水平/垂直翻转概率。水平翻转对行车目标安全且几乎免费。
# 中文：水平/垂直翻转概率。
IDE_FLIPLR = 0.5
IDE_FLIPUD = 0.0

# IDE_SCALE / IDE_TRANSLATE:
# - 随机缩放与平移。缩放本身兼具多尺度效果（对小目标友好）。
# 中文：随机缩放比例与平移比例。
IDE_SCALE = 0.5
IDE_TRANSLATE = 0.1

# IDE_MOSAIC / IDE_CLOSE_MOSAIC:
# - Mosaic 多图拼接（默认 1.0，现在我改为0，最后 10 轮关闭）。
# - 注意：mosaic 会改变单样本的"整图语义"，若想严格保持整图分布可设 0。
# 中文：mosaic 概率与关闭轮数。
IDE_MOSAIC = 0
IDE_CLOSE_MOSAIC = 10

# IDE_MIXUP / IDE_CUTMIX / IDE_COPY_PASTE:
# - 多图混合增强，OBB 任务官方默认全 0，保持默认即可。
# 中文：mixup/cutmix/copy-paste 概率（OBB 默认 0）。
IDE_MIXUP = 0.0
IDE_CUTMIX = 0.0
IDE_COPY_PASTE = 0.0

# IDE_HSV_H / IDE_HSV_S / IDE_HSV_V:
# - 颜色扰动。多光谱数据各通道物理意义不同，过强的 HSV 扰动可能失真，
#   建议保持官方默认或调低。
# 中文：HSV 颜色扰动幅度。
IDE_HSV_H = 0.015
IDE_HSV_S = 0.4
IDE_HSV_V = 0.4

# IDE_ERASING:
# - 随机擦除（官方默认 0.4），对小目标可能误删，可按需调低。
# 中文：随机擦除概率。
IDE_ERASING = 0.2

# IDE_MULTI_SCALE:
# - 多尺度训练区间（0=关闭；0.5~1.5 表示每 10 轮在 [0.5, 1.5]*imgsz 间随机）。
# - 整图部署口径下建议先 0，确认基线后再开多尺度。
# 中文：多尺度训练开关。
IDE_MULTI_SCALE = 0.0

# IDE_RECT:
# - 是否按长宽比分批（rect 训练）。整图部署口径建议 False 保持固定方形输入。
# 中文：rect 分批开关。
IDE_RECT = False

# =========================
# 6. 类别不平衡处理（本数据集核心痛点）
# =========================

# IDE_CLS_PW:
# - 类别不平衡的 cls 损失加权指数（本仓库 fork 的语义，与官方不同！）：
#   0.0 = 关闭；0.5 = 半强度逆频率加权；1.0 = 完整逆频率加权。
# - 权重自动从训练集标签统计（每类出现频次的倒数）并归一化（均值=1.0），
#   无需手工指定各类权重，实现见 DetectionTrainer.set_class_weights。
# - 建议从 0.5 起步：car/pedestrian/bike 占 87%，bus/truck/tricycle 仅 1-2%，
#   完整逆频率（1.0）可能过度压低高频类，0.5 是常见的稳妥起点。
# 中文：cls 损失逆频率加权指数（0=关闭，1=完整逆频率）。
IDE_CLS_PW = 0.5

# IDE_RARE_OVERSAMPLE_CLASSES:
# - 稀有类帧过采样（训练侧实现，原理与夜间过采样相同：重复图片列表行）。
# - 训练开始时扫描训练集标签，凡是"帧内稀有类目标数 >= IDE_RARE_MIN_OBJECTS"
#   的帧，在图片列表里重复 IDE_RARE_OVERSAMPLE_REPEATS 次。
# - 只影响 train 的采样分布；val/test 不受影响；内存零增量
#   （cache 按唯一路径缓存，重复行只增加迭代时间）。
# - 空元组 = 关闭。
# 中文：稀有类目标集合；含这些目标的帧才会被过采样；空元组=关闭。
IDE_RARE_OVERSAMPLE_CLASSES: tuple[str, ...] = ("bus", "truck", "tricycle", "van")

# IDE_RARE_MIN_OBJECTS:
# - 帧内稀有类目标数阈值：>= 该值才过采样。
# - 注意：本数据集场景密集（约 40 目标/帧），阈值=1 时会命中约 85% 的 train 帧
#   （实测），过采样退化为整体加倍、毫无意义。建议 2-3。
# 中文：帧内稀有类目标数阈值（>= 该值才过采样）。
IDE_RARE_MIN_OBJECTS = 3

# IDE_RARE_OVERSAMPLE_REPEATS:
# - 每个稀有类帧额外重复的次数（1 = 翻倍）。
# 中文：稀有类帧重复倍数。
IDE_RARE_OVERSAMPLE_REPEATS = 1

# =========================
# 7. 损失权重与优化器（一般保持默认）
# =========================

# 中文：loss 权重与优化器参数，默认与官方一致，一般无需修改。
IDE_BOX = 7.5
IDE_CLS = 0.5
IDE_DFL = 1.5
IDE_OPTIMIZER = "auto"  # auto/SGD/Adam/AdamW
IDE_LR0 = 0.01
IDE_LRF = 0.01
IDE_WEIGHT_DECAY = 0.0005
IDE_COS_LR = False

# =========================
# 8. 训练后自动验证
# =========================

# IDE_POST_VAL_CONF:
# - 训练后（含 clean 口径）验证使用的置信度阈值；None=官方默认（OBB 0.01）。
# 中文：训练后验证的 conf 阈值。
IDE_POST_VAL_CONF: float | None = 0.01

# IDE_POST_VAL_IOU:
# - 训练后验证的 NMS IoU 阈值。
# 中文：训练后验证的 NMS IoU。
IDE_POST_VAL_IOU = 0.7


def resolve_resume_path() -> Path | None:
    """Resolve the resume checkpoint path from the IDE config."""
    if not IDE_RESUME:
        return None
    run_name = IDE_RESUME_RUN_NAME
    if IDE_RESUME_RUN_INDEX is not None:
        run_name = f"{run_name}-{IDE_RESUME_RUN_INDEX}"
    candidate = Path(IDE_PROJECT) / run_name / "weights" / "last.pt"
    if not candidate.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
    return candidate


def build_clean_val_kwargs(model: YOLO, train_kwargs: dict[str, object]) -> dict[str, object]:
    """Build kwargs for the post-training clean-metric validation.

    Saves results inside the training run directory (``<project>/<name>/clean_val``)
    instead of the default ``runs/obb/val-N`` location, and reuses the trainer's
    device so a device-less training run never silently falls back to CPU.

    中文：构建训练后 clean 口径验证参数：输出到 <project>/<name>/clean_val，
    并复用训练实际使用的 device，避免自动选卡训练后验证退到 CPU。
    """
    project = train_kwargs.get("project")
    run_name = str(train_kwargs.get("name") or "unnamed_run")
    run_dir = Path(str(project)) / run_name if project else REPO_ROOT / "runs" / run_name

    trainer_args = getattr(getattr(model, "trainer", None), "args", None)
    device = getattr(trainer_args, "device", None) or train_kwargs.get("device")

    val_kwargs: dict[str, object] = {
        "data": IDE_DATA_CLEAN_YAML,
        "imgsz": IDE_IMGSZ,
        "batch": IDE_VAL_BATCH,
        "split": "val",
        "plots": True,
        "save_json": False,
        "project": str(run_dir),
        "name": "clean_val",
        "exist_ok": True,
    }
    if device is not None:
        val_kwargs["device"] = device
    return val_kwargs


def preflight_train_batch(
    model: YOLO,
    channels: int,
    imgsz: int,
    requested_batch: int,
    device: str,
    amp: bool,
    num_classes: int,
    memory_fraction: float = 0.40,
) -> int:
    """Probe real forward+backward memory and return a safe training batch size.

    The probe runs in train mode (BatchNorm behavior matches real training) and accepts a
    batch only if its reserved memory stays below ``memory_fraction`` of total VRAM, leaving
    headroom for the real OBB loss, EMA, optimizer states and pinned transfer buffers.
    Candidates start from the requested batch (or 16 when ``requested_batch`` is -1).
    If batch=1 still exceeds the threshold, it raises with guidance to reduce imgsz.

    中文：训练模式显存预检。只接受"预留显存 <= total*fraction"的 batch，
    为真实 OBB loss / EMA / 优化器 / pin 缓冲留足余量；OOM 或超阈自动降 batch。
    """
    import gc

    import torch

    from ultralytics.nn.tasks import OBBModel

    if not torch.cuda.is_available():
        return max(requested_batch, 1)

    device = device.strip().lower()
    if "," in device:
        raise ValueError(
            f"Multi-GPU training is disabled in this script (got device={device!r}). "
            "Use a single GPU id (e.g. '0'), a 'cuda:N' device, or 'cpu'."
        )
    if device.startswith("cuda"):
        pass
    elif device.isdigit():
        device = f"cuda:{device}"
    else:
        return max(requested_batch, 1)

    def _iter_tensors(value):
        """Yield all tensors nested in dicts/tuples/lists (recursive)."""
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from _iter_tensors(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                yield from _iter_tensors(child)

    explicit_batch = requested_batch and requested_batch > 0
    if explicit_batch:
        candidates = [int(requested_batch)]
    else:
        candidates = [16, 12, 8, 6, 4, 3, 2, 1]

    yaml_dict = getattr(model.model, "yaml", None)
    if not isinstance(yaml_dict, dict):
        raise TypeError("Cannot preflight memory: model does not expose a YOLO yaml dict.")

    total_memory = torch.cuda.get_device_properties(torch.device(device)).total_memory
    probe_model = OBBModel(yaml_dict, ch=channels, nc=num_classes, verbose=False)
    probe_model.train()  # 训练模式：BN/激活行为与真实训练一致
    probe_model.to(device)

    for batch in candidates:
        probe = None
        output = None
        loss = None
        try:
            probe = torch.zeros((batch, channels, imgsz, imgsz), dtype=torch.float32, device=device, requires_grad=True)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else torch.nullcontext()
            with torch.enable_grad(), autocast:
                output = probe_model(probe)
                tensors = [t for t in _iter_tensors(output) if isinstance(t, torch.Tensor) and t.requires_grad]
                if not tensors:
                    raise RuntimeError("Preflight model output contains no differentiable tensors.")
                loss = sum(t.float().mean() for t in tensors)
                loss.backward()
            reserved = torch.cuda.memory_reserved(device)
            fraction = reserved / total_memory
            if fraction <= memory_fraction:
                print(
                    f"[INFO] Preflight OK: batch={batch}, imgsz={imgsz}, ch={channels} "
                    f"-> reserved {reserved / 1e9:.2f}G / {total_memory / 1e9:.2f}G ({100 * fraction:.0f}%), "
                    f"within {100 * memory_fraction:.0f}% safety limit."
                )
                return batch
            if explicit_batch:
                print(
                    f"[WARN] Preflight at requested batch={batch} reserves {100 * fraction:.0f}% VRAM "
                    f"(limit {100 * memory_fraction:.0f}%); aborting because an explicit batch does not auto-fallback."
                )
            else:
                print(
                    f"[WARN] Preflight at batch={batch} reserves {100 * fraction:.0f}% VRAM "
                    f"(limit {100 * memory_fraction:.0f}%); trying smaller batch..."
                )
        except torch.cuda.OutOfMemoryError:
            print(f"[WARN] preflight OOM at batch={batch}, imgsz={imgsz}; trying smaller batch...")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"[WARN] preflight OOM at batch={batch}, imgsz={imgsz}; trying smaller batch...")
        finally:
            probe_model.zero_grad(set_to_none=True)
            del probe, output, loss
            gc.collect()
            torch.cuda.empty_cache()

    if explicit_batch:
        raise RuntimeError(
            f"Preflight failed at requested batch={requested_batch}, imgsz={imgsz}, channels={channels}. "
            "Set a smaller IDE_BATCH, use -1 for automatic probing, or reduce IDE_IMGSZ."
        )
    raise RuntimeError(
        f"Preflight OOM even at batch=1, imgsz={imgsz}, channels={channels} (8GB GPU). "
        "Reduce IDE_IMGSZ (e.g. 1184 -> 1024 -> 896 -> 640) and try again."
    )


def build_rare_oversampled_yaml(data_yaml: Path) -> Path | None:
    """Build a derived data YAML whose train split oversamples frames with rare-class objects.

    Train-side implementation: the current train image list (from ``data_yaml``) is read,
    frames containing any ``IDE_RARE_OVERSAMPLE_CLASSES`` object are repeated in a new list
    file, and a derived YAML pointing at that list is returned. val/test are untouched.

    中文：训练侧稀有类帧过采样。若 data.yaml 的 train 已是列表文件（如 prepare 侧
    夜间过采样生成的 train_oversampled.txt），以它为基础继续叠加，二者自动合并；
    否则以 images/train 目录为基准。返回派生 YAML 路径；未配置时返回 None。
    """
    if not IDE_RARE_OVERSAMPLE_CLASSES or IDE_RARE_OVERSAMPLE_REPEATS < 1:
        return None
    import yaml

    from ultralytics.data.utils import img2label_paths

    with data_yaml.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    dataset_root = Path(payload.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    def resolve_dataset_path(spec: str) -> Path:
        """Resolve a data.yaml path field against the dataset root."""
        path = Path(spec)
        return path if path.is_absolute() else dataset_root / path

    train_spec = str(payload.get("train") or "images/train")
    train_path = resolve_dataset_path(train_spec)

    if train_path.is_file():
        entries = [ln.strip() for ln in train_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        image_paths: list[Path] = []
        for entry in entries:
            path = Path(entry)
            if path.is_absolute():
                image_paths.append(path)
            else:
                candidate = dataset_root / path
                image_paths.append(candidate if candidate.exists() else train_path.parent / path)
        base_source = train_path.name
    elif train_path.is_dir():
        image_paths = sorted(train_path.glob("*.tiff"))
        base_source = train_spec
    else:
        raise FileNotFoundError(f"Train split not found for oversampling scan: {train_path}")

    if not image_paths:
        print(f"[WARN] No training images found under {train_path}; skipping rare oversampling.")
        return None

    labels_spec = payload.get("labels")
    labels_root = resolve_dataset_path(str(labels_spec)) if labels_spec else None

    def label_path_for(image_path: Path) -> Path:
        """Map an image path to its label path, preserving nested directories."""
        if labels_root is None:
            return Path(img2label_paths([str(image_path)])[0])
        try:
            relative = image_path.relative_to(dataset_root)
        except ValueError:
            relative = Path(image_path.name)
        parts = relative.parts
        if parts and parts[0] == "images":
            parts = parts[1:]
        return labels_root / Path(*parts).with_suffix(".txt")

    class_to_id = {name: idx for idx, name in enumerate(IDE_CLASS_NAMES)}
    rare_ids = {class_to_id[name] for name in IDE_RARE_OVERSAMPLE_CLASSES if name in class_to_id}
    if not rare_ids:
        raise ValueError(
            f"None of IDE_RARE_OVERSAMPLE_CLASSES {IDE_RARE_OVERSAMPLE_CLASSES} exists in {IDE_CLASS_NAMES}."
        )

    rare_entries: list[str] = []
    rare_counts: list[int] = []
    for image_path in image_paths:
        label_path = label_path_for(image_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Train label file missing for oversampling scan: {label_path}")
        count = sum(
            1
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and int(float(line.split()[0])) in rare_ids
        )
        if count >= IDE_RARE_MIN_OBJECTS:
            rare_entries.append(str(image_path))
            rare_counts.append(count)

    if not rare_entries:
        print(
            f"[WARN] No train frames contain >= {IDE_RARE_MIN_OBJECTS} objects of "
            f"{IDE_RARE_OVERSAMPLE_CLASSES}; skipping rare oversampling."
        )
        return None

    base_entries = [str(p) for p in image_paths]
    extra = [entry for entry in rare_entries for _ in range(IDE_RARE_OVERSAMPLE_REPEATS)]
    list_path = dataset_root / "train_rare_oversampled.txt"
    list_path.write_text("\n".join(base_entries + extra) + "\n", encoding="utf-8")

    derived_yaml = data_yaml.with_name(f"{data_yaml.stem}_oversampled.yaml")
    payload["train"] = list_path.name
    with derived_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    total = len(base_entries)
    print(
        f"[INFO] Rare-class oversampling (train-side): base={total} entries (from {base_source}), "
        f"rare-class-rich frames={len(rare_entries)} ({100 * len(rare_entries) / total:.1f}%) "
        f"with >= {IDE_RARE_MIN_OBJECTS} rare objects, +{len(extra)} repeated -> {list_path.name}"
    )
    print(
        f"[INFO] Rare-object count distribution in selected frames: "
        f"min={min(rare_counts)}, max={max(rare_counts)}, mean={sum(rare_counts) / len(rare_counts):.1f}"
    )
    print(f"[INFO] Derived training YAML: {derived_yaml}")
    return derived_yaml


def build_train_kwargs() -> dict[str, object]:
    """Assemble the training kwargs from the IDE config area."""
    kwargs: dict[str, object] = {
        "data": IDE_DATA_YAML,
        "epochs": IDE_EPOCHS,
        "patience": IDE_PATIENCE,
        "imgsz": IDE_IMGSZ,
        "batch": IDE_BATCH,
        "workers": IDE_WORKERS,
        "cache": IDE_CACHE,
        "seed": IDE_SEED,
        "deterministic": IDE_DETERMINISTIC,
        "project": IDE_PROJECT,
        "name": IDE_RUN_NAME,
        "amp": IDE_AMP,
        "degrees": IDE_DEGREES,
        "fliplr": IDE_FLIPLR,
        "flipud": IDE_FLIPUD,
        "scale": IDE_SCALE,
        "translate": IDE_TRANSLATE,
        "mosaic": IDE_MOSAIC,
        "close_mosaic": IDE_CLOSE_MOSAIC,
        "mixup": IDE_MIXUP,
        "cutmix": IDE_CUTMIX,
        "copy_paste": IDE_COPY_PASTE,
        "hsv_h": IDE_HSV_H,
        "hsv_s": IDE_HSV_S,
        "hsv_v": IDE_HSV_V,
        "erasing": IDE_ERASING,
        "multi_scale": IDE_MULTI_SCALE,
        "rect": IDE_RECT,
        "box": IDE_BOX,
        "cls": IDE_CLS,
        "dfl": IDE_DFL,
        "optimizer": IDE_OPTIMIZER,
        "lr0": IDE_LR0,
        "lrf": IDE_LRF,
        "weight_decay": IDE_WEIGHT_DECAY,
        "cos_lr": IDE_COS_LR,
        "cls_pw": IDE_CLS_PW,
    }
    if IDE_DEVICE is not None:
        kwargs["device"] = IDE_DEVICE
    return kwargs


def print_cli_equivalent(
    kwargs: dict[str, object],
    model: str | None = None,
    pretrained: str | None = None,
    pretrained_scope: str | None = None,
) -> None:
    """Print the equivalent `yolo` CLI command for reproducibility."""
    parts = ["yolo", "obb", "train"]
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, bool):
            parts.append(f"{key}={'true' if value else 'false'}")
        else:
            parts.append(f"{key}={value}")
    if model:
        parts.append(f"model={model}")
    if pretrained and str(pretrained).lower() != "none":
        parts.append(f"pretrained={pretrained}")
    print("=" * 70)
    print("Equivalent CLI command:")
    print("  " + " ".join(parts))
    if pretrained_scope and pretrained_scope != "model":
        print(
            f"  # NOTE: script-only pretrained-scope={pretrained_scope}; "
            "official yolo CLI has no scoped-weight flag, use this script to reproduce it."
        )
    print("=" * 70)


def print_run_cli(args: argparse.Namespace, kwargs: dict[str, object], resume_checkpoint: Path | None) -> None:
    """Print the CLI command matching the actual run (new training vs true resume).

    For a resumed run the command must point at ``last.pt`` and include the
    ``resume=True`` entry already present in ``kwargs``; it must not advertise
    the YAML + pretrained path used only for a fresh start.

    中文：打印与实际执行一致的 CLI：续训时用 last.pt + resume=True，不再打印
    YAML 与 pretrained，避免复制出“新训练”命令。
    """
    if resume_checkpoint is not None:
        print_cli_equivalent(kwargs, model=str(resume_checkpoint), pretrained=None, pretrained_scope=None)
    else:
        print_cli_equivalent(
            kwargs, model=args.model, pretrained=args.pretrained, pretrained_scope=args.pretrained_scope
        )


def _layer_index_within_scope(key: str, max_layer: int) -> bool:
    """Return True when a state-dict key belongs to model layers 0..max_layer.

    Non-``model.*`` keys (rare compatibility buffers) are kept by default, while
    ``model.<index>...`` keys are kept only when ``0 <= index <= max_layer``.

    中文：判断 state_dict 键是否属于第 0~max_layer 层；非 model.* 键默认保留。
    """
    parts = key.split(".")
    if parts[0] != "model":
        return True
    if len(parts) < 2 or not parts[1].isdigit():
        return False
    return 0 <= int(parts[1]) <= max_layer


_PRETRAINED_SCOPE_MAX_LAYER = {
    "backbone": 10,  # yolo26-obb.yaml layers 0..10
    "backbone+neck": 22,  # yolo26-obb.yaml layers 0..22
    "model": 23,  # yolo26-obb.yaml layers 0..23 (OBB26 detection head)
}


def load_pretrained_by_scope(model: YOLO, weights: str | Path, scope: str) -> None:
    """Load pretrained weights restricted to the selected yolo26-obb.yaml layer range.

    ``scope`` follows the baseline ``ultralytics/cfg/models/26/yolo26-obb.yaml``
    layer numbering: ``backbone`` loads layers 0..10, ``backbone+neck`` loads
    layers 0..22, and ``model`` loads everything (0..23). All modes mirror
    ``BaseModel.load`` so the 3-channel first Conv is still partially transferred
    onto the 8-channel stem, and the checkpoint is registered on ``model.ckpt``
    exactly like ``Model.load`` so ``Model.train()`` reuses this in-memory model
    instead of silently rebuilding a randomly initialized one in ``setup_model()``.

    中文：按 backbone / backbone+neck / model 范围加载预训练权重。与官方
    Model.load 一样写入 model.ckpt 与 overrides["pretrained"]，保证 Model.train()
    复用已过滤的内存模型而不是在 setup_model() 里重建随机模型；权重来源与
    load_checkpoint 一致，优先 checkpoint["ema"]，不存在时用 checkpoint["model"]。
    """
    if scope not in _PRETRAINED_SCOPE_MAX_LAYER:
        raise ValueError(
            f"Unsupported IDE_PRETRAINED_SCOPE: {scope!r}. Expected one of {tuple(_PRETRAINED_SCOPE_MAX_LAYER)}."
        )

    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import intersect_dicts

    # load_checkpoint 返回的 source_model 已按官方逻辑选择 ema/model，并做了兼容性处理。
    source_model, checkpoint = load_checkpoint(weights)
    model.ckpt = checkpoint  # 关键：Model.train() 依赖 truthy ckpt 复用内存模型（镜像官方 Model.load）。
    model.overrides["pretrained"] = weights  # 与官方 Model.load 一致，供 DDP 子进程与复现记录使用。
    source_state_dict = source_model.float().state_dict()

    max_layer = _PRETRAINED_SCOPE_MAX_LAYER[scope]
    if scope == "model":
        filtered_state_dict = source_state_dict  # 全量加载，不按层号过滤。
    else:
        filtered_state_dict = {
            key: value for key, value in source_state_dict.items() if _layer_index_within_scope(key, max_layer)
        }

    updated_state_dict = intersect_dicts(filtered_state_dict, model.model.state_dict())
    model.model.load_state_dict(updated_state_dict, strict=False)
    len_updated = len(updated_state_dict)

    # 与官方 BaseModel.load 相同：首层 3 通道 -> 8 通道时按 min(c1, c2) 部分迁移。
    first_conv = "model.0.conv.weight"
    target_state_dict = model.model.state_dict()
    if first_conv not in updated_state_dict and first_conv in target_state_dict and first_conv in filtered_state_dict:
        c1, c2, h, w = target_state_dict[first_conv].shape
        cc1, cc2, ch, cw = filtered_state_dict[first_conv].shape
        if ch == h and cw == w:
            c1, c2 = min(c1, cc1), min(c2, cc2)
            target_state_dict[first_conv][:c1, :c2] = filtered_state_dict[first_conv][:c1, :c2]
            len_updated += 1

    print(
        f"[INFO] Transferred {len_updated}/{len(model.model.state_dict())} items "
        f"from pretrained weights (scope={scope}, layers 0..{max_layer})."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides (optional); IDE config values are the defaults."""
    parser = argparse.ArgumentParser(
        description="Train YOLO26-OBB on the prepared 8-channel whole-image dataset (IDE defaults in config area)."
    )
    parser.add_argument("--data", type=str, default=IDE_DATA_YAML)
    parser.add_argument("--model", type=str, default=IDE_MODEL)
    parser.add_argument("--pretrained", type=str, default=IDE_PRETRAINED)
    parser.add_argument(
        "--pretrained-scope",
        type=str,
        default=IDE_PRETRAINED_SCOPE,
        choices=tuple(_PRETRAINED_SCOPE_MAX_LAYER),
        help="Pretrained weight scope: backbone (layers 0-10), backbone+neck (0-22), or model (0-23).",
    )
    parser.add_argument("--epochs", type=int, default=IDE_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IDE_IMGSZ)
    parser.add_argument("--batch", type=int, default=IDE_BATCH)
    parser.add_argument("--device", type=str, default=IDE_DEVICE)
    parser.add_argument("--run-name", type=str, default=IDE_RUN_NAME)
    parser.add_argument("--project", type=str, default=IDE_PROJECT)
    parser.add_argument(
        "--mode",
        type=str,
        default=IDE_RUN_MODE,
        choices=("train", "train_and_clean_val"),
        help="train or train_and_clean_val (post-training clean-metric validation).",
    )
    return parser.parse_args()


class StableOBBTrainer(OBBTrainer):
    """OBB trainer with training-side pinned memory disabled (WSL2 stability).

    The pin_memory thread can raise spurious 'CUDA error: out of memory' on this
    rig even with ample free VRAM; disabling pinned memory avoids that failure mode.

    中文：关闭训练侧固定内存的 OBB 训练器，规避 WSL2 下 pin_memory 线程的虚假 OOM。
    """

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
        pin_memory: bool | None = None,
    ):
        """Build a dataloader with pinned memory disabled when requested."""
        effective_pin_memory = False if mode == "train" and not IDE_PIN_MEMORY else pin_memory
        return super().get_dataloader(
            dataset_path, batch_size=batch_size, rank=rank, mode=mode, pin_memory=effective_pin_memory
        )


def main() -> None:
    """Run training (and optional clean-metric validation) from the IDE config."""
    args = parse_args()
    if args.device and "," in str(args.device):
        raise SystemExit(
            f"Multi-GPU training is disabled in this script (got device={args.device!r}). "
            "Use a single GPU id (e.g. '0'), a 'cuda:N' device, or 'cpu'."
        )
    kwargs = build_train_kwargs()
    kwargs["data"] = args.data
    kwargs["project"] = args.project
    kwargs["name"] = args.run_name
    if args.device:
        kwargs["device"] = args.device
    if args.epochs != IDE_EPOCHS:
        kwargs["epochs"] = args.epochs
    if args.imgsz != IDE_IMGSZ:
        kwargs["imgsz"] = args.imgsz
    if args.batch != IDE_BATCH:
        kwargs["batch"] = args.batch
    if args.mode != IDE_RUN_MODE:
        kwargs["mode"] = args.mode

    if IDE_RARE_OVERSAMPLE_CLASSES:
        oversampled_yaml = build_rare_oversampled_yaml(Path(kwargs["data"]))
        if oversampled_yaml is not None:
            kwargs["data"] = str(oversampled_yaml)

    print(f"[INFO] data: {kwargs['data']}")
    print(f"[INFO] model: {args.model}, pretrained: {args.pretrained}, scope: {args.pretrained_scope}")
    print(f"[INFO] cls_pw (inverse-frequency power): {IDE_CLS_PW}")

    resume_checkpoint = resolve_resume_path()
    if resume_checkpoint is not None:
        model = YOLO(str(resume_checkpoint))
        kwargs["resume"] = True  # 真断点续训：从 last.pt 恢复 optimizer/epoch/LR/close_mosaic 状态。
        print(f"[INFO] Resuming from {resume_checkpoint} (resume=True)")
    else:
        model = YOLO(args.model)
        if args.pretrained and str(args.pretrained).lower() != "none":
            load_pretrained_by_scope(model, args.pretrained, args.pretrained_scope)
            print(f"[INFO] Loaded pretrained weights: {args.pretrained}")

    # 显存预检：用真实 batch/imgsz/通道数做 forward+backward，OOM 自动降 batch。
    # 除预检自身的显存/配置错误外，其余异常不再吞掉，避免掩盖真实问题。
    import yaml as yaml_module

    with open(kwargs["data"], encoding="utf-8") as f:
        data_payload = yaml_module.safe_load(f) or {}
    channels = int(data_payload.get("channels", 3))
    device = str(kwargs.get("device") or "0")
    safe_batch = preflight_train_batch(
        model=model,
        channels=channels,
        imgsz=int(kwargs["imgsz"]),
        requested_batch=int(kwargs["batch"]),
        device=device,
        amp=bool(kwargs.get("amp", True)),
        num_classes=len(IDE_CLASS_NAMES),
    )
    if safe_batch != int(kwargs["batch"]):
        print(f"[INFO] Preflight: safe batch = {safe_batch} (requested {kwargs['batch']}).")
    kwargs["batch"] = safe_batch

    print_run_cli(args, kwargs, resume_checkpoint)
    model.train(trainer=StableOBBTrainer, **kwargs)
    print(
        f"[INFO] Training finished. best.pt: {Path(model.trainer.best).resolve() if hasattr(model.trainer, 'best') else 'n/a'}"
    )

    run_clean_val = args.mode == "train_and_clean_val" and IDE_DATA_CLEAN_YAML
    if run_clean_val:
        print(f"[INFO] Post-training clean-metric validation on {IDE_DATA_CLEAN_YAML}")
        val_kwargs = build_clean_val_kwargs(model, kwargs)
        if IDE_POST_VAL_CONF is not None:
            val_kwargs["conf"] = IDE_POST_VAL_CONF
        if IDE_POST_VAL_IOU is not None:
            val_kwargs["iou"] = IDE_POST_VAL_IOU
        clean_results = model.val(**val_kwargs)
        print("[INFO] Clean-metric results (difficult excluded):")
        print(
            f"  P={float(np.mean(clean_results.box.p)):.4f}  R={float(np.mean(clean_results.box.r)):.4f}  "
            f"mAP50={clean_results.box.map50:.4f}  mAP50-95={clean_results.box.map:.4f}"
        )


if __name__ == "__main__":
    main()
