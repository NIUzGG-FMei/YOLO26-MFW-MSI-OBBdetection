"""Official-metric evaluation of a trained OBB model on the prepared dataset TEST split.

本文件用 ``prepare_obb_dataset.py`` 生成的独立 test 子集（从未参与训练与选模）对训练
好的权重做终评，输出官方口径 P / R / mAP50 / mAP50-95，以及新增的**分尺度指标**：

1. 双口径同时评估
   - data.yaml        : labels/ 含 difficult（标准口径）
   - data_clean.yaml  : labels_clean/ 剔除 difficult（clean 口径）
   - 两套结果都会打印并写入 JSON，difficult 对指标的影响可量化。

2. 与训练/验证的一致性
   - 推理链与训练脚本一致：end2end（无 NMS）、imgsz=1184、conf=0.01、iou=0.7；
   - test 子集与 train/val 按场景组完全隔离（prepare 时已保证）。

3. 分尺度指标（ScaleStratifiedOBBValidator，官方匹配逻辑的分解视图）
   - 面积分桶 AP / AR / TP-FP-FN 归因：mini / small / medium / large
     （默认阈值 16²/64²/256² px²，按原图 1200x900 像素口径，可用 --area-edges 改）；
   - 长边分桶：<16 / 16-32 / 32-64 / 64-128 / >=128 px（--long-side-edges 可改）；
   - 每桶 AP50 与 AP(mAP50-95)，与官方总体 mAP 同源（同一套 tp 矩阵按桶拆分，
     未匹配预测在所有桶计为 FP，匹配到其它桶的预测在该桶忽略，与 COCO 口径一致）；
   - 每类 × 每尺度的 AP50 矩阵与各桶 GT 数；
   - 部署操作点复评：--deploy-conf（默认 0.25）再跑一遍标准口径，看小目标掉多少。

4. IDE 使用方式
   - 改顶部配置区后直接点运行；脚本打印等价 CLI 命令并输出结果 JSON。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.obb.val import OBBValidator  # noqa: E402
from ultralytics.utils.metrics import ap_per_class, batch_probiou  # noqa: E402


# =========================
# IDE Quick Config
# 直接修改这里的值然后点击运行即可。
# =========================

# IDE_WEIGHTS:
# - 待评估的权重文件（best.pt / last.pt 均可）。
# 中文：待评估权重路径。
IDE_WEIGHTS = "/home/mofengwei/datasetObjectDetection/checkpoints/yolo26n_obb_bifpn_whole_image/weights/best.pt"

# IDE_DATA_YAML / IDE_DATA_CLEAN_YAML:
# - 由 prepare_obb_dataset.py 生成的测试口径数据文件。
# - data.yaml      : labels/ 含 difficult（标准口径）
# - data_clean.yaml: labels_clean/ 剔除 difficult（clean 口径）
# - 设为 None 可跳过对应口径。
# 中文：标准与 clean 口径的数据 YAML。
IDE_DATA_YAML = "/home/mofengwei/datasetObjectDetection/prepared_obb_dataset/data.yaml"
IDE_DATA_CLEAN_YAML = "/home/mofengwei/datasetObjectDetection/prepared_obb_dataset_clean/data.yaml"

# IDE_SPLIT:
# - 评估哪个子集："test" 终评（默认）、"val" 复现训练内验证。
# 中文：评估子集。
IDE_SPLIT = "test"

# IDE_IMGSZ:
# - 评估输入尺寸，必须与训练一致（默认 1184）。
# 中文：评估输入尺寸。
IDE_IMGSZ = 1184

# IDE_CONF / IDE_IOU:
# - 置信度阈值与 NMS IoU（end2end 模型下 iou 不参与抑制，仅作记录）。
# - 与既往整图验证报告一致使用 0.01；想看部署操作点可改 0.25-0.3。
# 中文：评估 conf 阈值与 NMS IoU。
IDE_CONF = 0.01
IDE_IOU = 0.7

# IDE_BATCH:
# - 评估 batch；8GB 显存 + 8 通道建议 1-2。
# 中文：评估 batch。
IDE_BATCH = 1

# IDE_DEVICE:
# - 评估设备："0"（单卡）/ "cpu"；本工程只有一张显卡，多卡验证已禁用。
# 中文：评估设备（仅支持单卡或 CPU）。
IDE_DEVICE = "0"

# IDE_WORKERS:
# - DataLoader worker 数。
# 中文：数据加载 worker 数。
IDE_WORKERS = 4

# IDE_HALF:
# - 是否用 FP16 推理（更快，指标几乎无差异）。
# 中文：FP16 推理开关。
IDE_HALF = True

# IDE_PLOTS:
# - 是否保存 PR/F1/混淆矩阵等曲线图，以及分尺度折线图 scale_metrics.png。
# 中文：保存曲线图开关。
IDE_PLOTS = True

# IDE_OUTPUT_ROOT:
# - 结果输出根目录，每次运行自动建 val_test_YYYYmmdd_HHMMSS 子目录。
# 中文：结果输出根目录。
IDE_OUTPUT_ROOT = REPO_ROOT / "runs" / "obb_test_validation"

# =========================
# 分尺度指标配置（新增）
# =========================

# IDE_SCALE_METRICS:
# - 是否输出分尺度指标（面积分桶 + 长边分桶 + 每类x每尺度矩阵）。
# - 关闭后行为与旧版完全一致，仅输出官方 P/R/mAP。
# 中文：分尺度指标开关。
IDE_SCALE_METRICS = True

# IDE_AREA_EDGES:
# - 面积分桶边界（边长阈值，单位=原图像素 px），生成 mini/small/medium/large 四桶：
#   16/64/256 -> <256px²、256-4096px²、4096-65536px²、>=65536px²。
# - 面积按 OBB 多边形在**原图 1200x900** 上的像素面积计算（与训练 imgsz 无关，
#   也不受 letterbox 影响），保证不同 imgsz 实验可比较。
# 中文：面积分桶边界（px）。
IDE_AREA_EDGES = (16.0, 64.0, 256.0)

# IDE_LONG_SIDE_EDGES:
# - 长边分桶边界（px）：<16 / 16-32 / 32-64 / 64-128 / >=128。
# - 车辆是细长目标，长边比面积更能解释"这个车在图上有多大"。
# 中文：长边分桶边界（px）。
IDE_LONG_SIDE_EDGES = (16.0, 32.0, 64.0, 128.0)

# IDE_DEPLOY_CONF:
# - 部署操作点复评的置信度阈值：用该 conf 再跑一遍标准口径，输出分尺度指标，
#   看 conf 提高后小目标 AP 掉多少。None 表示跳过复评。
# 中文：部署操作点 conf；None=跳过。
IDE_DEPLOY_CONF = 0.25


# =========================
# 分尺度验证器（官方匹配逻辑的分解视图）
# =========================


class ScaleStratifiedOBBValidator(OBBValidator):
    """OBB validator that additionally records per-object scale data for size-bucketed AP/AR.

    中文：在官方 OBB 验证器基础上，逐图记录每个 GT 的原图像素面积/长边与
    0.5 IoU 官方两步匹配（与 match_predictions 完全一致）得到的 GT 序号；
    最终把官方 tp 矩阵按"匹配 GT 所在桶"拆分，
    输出与官方 mAP 同源的分尺度 AP/AR/TP-FP-FN（COCO 口径）。
    """

    _area_edges: tuple[float, ...] = (16.0, 64.0, 256.0)
    _long_side_edges: tuple[float, ...] = (16.0, 32.0, 64.0, 128.0)

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize the validator and prepare per-image scale record storage."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self._scale_records: list[dict[str, np.ndarray]] = []

    def update_metrics(self, preds, batch) -> None:
        """Record per-image scale data first, then delegate to the official metric update."""
        for si, pred in enumerate(preds):
            self._record_scale(si, pred, batch)
        super().update_metrics(preds, batch)

    def _record_scale(self, si: int, pred: dict[str, torch.Tensor], batch: dict) -> None:
        """Record one image: GT sizes in original pixels, official tp matrix, and 0.5-IoU GT matches."""
        idx = batch["batch_idx"] == si
        gt_cls = batch["cls"][idx].squeeze(-1)
        gt_boxes = batch["bboxes"][idx]  # normalized xywhr (long-side-first, rad)
        imgsz = batch["img"].shape[2:]
        predn = self._prepare_pred(pred)
        n_preds = predn["cls"].shape[0]

        gt_boxes_px = gt_boxes.clone()
        if gt_boxes_px.shape[0]:
            gt_boxes_px[..., :4].mul_(torch.tensor(imgsz, device=gt_boxes_px.device)[[1, 0, 1, 0]])

        # Same tp matrix as the official matching (identical inputs and matching code).
        tp = self._process_batch(predn, {"cls": gt_cls, "bboxes": gt_boxes_px})["tp"]

        # GT sizes in ORIGINAL-image pixels: the loader's `ratio_pad` records the actual
        # resize ratio applied to the image (rect mode applies a second batch-shape
        # resize whose label transform does not follow the letterbox math, so the
        # simple min(imgsz/ori) ratio is unreliable). Undo it with ratio_pad:
        #   w_orig = w_norm * batch_W / rw,  h_orig = h_norm * batch_H / rh.
        if gt_boxes.shape[0]:
            ratio_pad = batch["ratio_pad"][si]
            rh = float(ratio_pad[0][0])
            rw = float(ratio_pad[0][1])
            w_orig = gt_boxes[:, 2].cpu().numpy() * imgsz[1] / rw
            h_orig = gt_boxes[:, 3].cpu().numpy() * imgsz[0] / rh
            areas = w_orig * h_orig
            longs = np.maximum(w_orig, h_orig)
        else:
            areas = np.zeros(0, dtype=np.float64)
            longs = np.zeros(0, dtype=np.float64)

        matched = self._greedy_match_gt_idx(gt_cls, gt_boxes_px, predn, iou_threshold=float(self.iouv[0]))

        self._scale_records.append(
            {
                "gt_cls": gt_cls.cpu().numpy().astype(np.int64),
                "gt_area_px": areas,
                "gt_long_px": longs,
                "tp": tp,
                "conf": np.zeros(0) if n_preds == 0 else predn["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if n_preds == 0 else predn["cls"].cpu().numpy().astype(np.int64),
                "matched_gt": matched,
            }
        )

    @staticmethod
    def _greedy_match_gt_idx(gt_cls, gt_boxes_px, predn, iou_threshold: float) -> np.ndarray:
        """Replicate Ultralytics' official two-step matching at one IoU threshold.

        Mirrors ``BaseValidator.match_predictions`` exactly: pairs are sorted by IoU
        descending, then the first pair per DETECTION is kept, then the first pair per
        GT. Returns the matched GT index per prediction (-1 = unmatched).
        """
        n_preds = predn["cls"].shape[0]
        if n_preds == 0 or gt_boxes_px.shape[0] == 0:
            return np.full(n_preds, -1, dtype=np.int64)
        iou = batch_probiou(gt_boxes_px, predn["bboxes"])
        correct_class = gt_cls[:, None] == predn["cls"][None, :]
        iou_np = (iou * correct_class).cpu().numpy()
        matched = np.full(n_preds, -1, dtype=np.int64)
        matches = np.array(np.nonzero(iou_np >= iou_threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matched[matches[:, 1].astype(int)] = matches[:, 0].astype(int)
        return matched

    def finalize_metrics(self) -> None:
        """Finalize official metrics and attach the size-stratified report to the metrics object."""
        super().finalize_metrics()
        self.metrics.scale_report = self.compute_scale_report()

    def compute_scale_report(self) -> dict[str, object]:
        """Build the full size-stratified report from per-image records."""
        records = self._scale_records
        gt_cls = _concat_records(records, "gt_cls").astype(np.int64)
        areas = _concat_records(records, "gt_area_px").astype(np.float64)
        longs = _concat_records(records, "gt_long_px").astype(np.float64)
        tp = _concat_records(records, "tp").astype(bool)
        conf = _concat_records(records, "conf").astype(np.float64)
        pred_cls = _concat_records(records, "pred_cls").astype(np.int64)
        # Globalize per-image local GT indices so bucket masks over the concatenated
        # GT arrays use consistent positions (unmatched stays -1).
        matched_parts: list[np.ndarray] = []
        offset = 0
        for record in records:
            local = record["matched_gt"]
            matched_parts.append(np.where(local >= 0, local + offset, -1).astype(np.int64))
            offset += int(record["gt_cls"].size)
        matched = np.concatenate(matched_parts) if matched_parts else np.zeros(0, dtype=np.int64)

        return {
            "conf": float(self.args.conf),
            "imgsz": int(self.args.imgsz),
            "area_edges_px2": [float(e) ** 2 for e in self._area_edges],
            "area": _bucket_ap(areas, self._area_edges, gt_cls, tp, conf, pred_cls, matched, area_mode=True),
            "long_side_edges_px": [float(e) for e in self._long_side_edges],
            "long_side": _bucket_ap(longs, self._long_side_edges, gt_cls, tp, conf, pred_cls, matched, area_mode=False),
        }


def _concat_records(records: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    """Concatenate one per-image array across all records (empty-safe)."""
    arrays = [record[key] for record in records]
    if not arrays:
        return np.zeros(0)
    if all(array.size == 0 for array in arrays):
        return arrays[0]
    return np.concatenate(arrays)


def _bucket_names(edges: tuple[float, ...], area_mode: bool) -> list[str]:
    """Human-readable bucket names from edge thresholds."""
    if area_mode and edges == (16.0, 64.0, 256.0):
        return ["mini", "small", "medium", "large"]
    if area_mode:
        bounds = [f"<{edges[0] ** 2:g}px2"] + [
            f"{edges[i - 1] ** 2:g}-{edges[i] ** 2:g}px2" for i in range(1, len(edges))
        ]
        return bounds + [f">={edges[-1] ** 2:g}px2"]
    bounds = [f"<{edges[0]:g}px"] + [f"{edges[i - 1]:g}-{edges[i]:g}px" for i in range(1, len(edges))]
    return bounds + [f">={edges[-1]:g}px"]


def _bucket_ap(
    values: np.ndarray,
    edges: tuple[float, ...],
    gt_cls: np.ndarray,
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    matched: np.ndarray,
    area_mode: bool,
) -> dict[str, object]:
    """Compute per-bucket AP/AR/attribution from one global official matching (COCO-style).

    Rules: unmatched predictions count as FP in every bucket; predictions matched to a GT
    of another bucket are ignored in this bucket (neither TP nor FP); bucket GT without a
    match are FN. This is the pycocotools ``evalImg`` convention.
    """
    if area_mode:
        # Edges are SIDE lengths in px (e.g. 16 -> 16x16 = 256 px²); areas are in px².
        squared = tuple(float(e) ** 2 for e in edges)
    else:
        squared = edges
    bounds = [(0.0, squared[0])]
    bounds += [(squared[i - 1], squared[i]) for i in range(1, len(squared))]
    bounds += [(squared[-1], float("inf"))]
    names = _bucket_names(edges, area_mode)

    matched_gt_idx = np.unique(matched[matched >= 0]).astype(np.int64)
    n_gt_total = int(values.size)
    n_fp_total = int((matched == -1).sum())
    ar50_total = (matched_gt_idx.size / n_gt_total) if n_gt_total else None

    buckets: list[dict[str, object]] = []
    for name, (lo, hi) in zip(names, bounds):
        gt_in = (values >= lo) & (values < hi)
        n_gt = int(gt_in.sum())
        gt_positions = np.nonzero(gt_in)[0]
        n_tp = int(np.isin(matched_gt_idx, gt_positions).sum())
        n_fn = n_gt - n_tp

        if n_gt == 0:
            buckets.append(
                {
                    "name": name,
                    "n_gt": 0,
                    "n_tp": 0,
                    "n_fn": 0,
                    "ar50": None,
                    "ap50": None,
                    "ap": None,
                    "p": None,
                    "r": None,
                    "per_class": [],
                }
            )
            continue

        keep = np.zeros(matched.shape[0], dtype=bool)
        keep[matched == -1] = True
        positive = matched >= 0
        keep[positive] = gt_in[matched[positive]]

        tp_b, conf_b, pred_cls_b, gt_cls_b = tp[keep], conf[keep], pred_cls[keep], gt_cls[gt_in]
        ap50 = ap_mean = p_mean = r_mean = None
        per_class: list[dict[str, object]] = []
        if tp_b.size and gt_cls_b.size:
            res = ap_per_class(tp_b, conf_b, pred_cls_b, gt_cls_b)
            p, r, ap, unique_classes = res[2], res[3], res[5], res[6]
            if ap.shape[0]:
                ap50 = float(ap[:, 0].mean())
                ap_mean = float(ap.mean())
            if p.size:
                p_mean, r_mean = float(p.mean()), float(r.mean())
            for ci, cls_id in enumerate(unique_classes):
                per_class.append(
                    {
                        "class_id": int(cls_id),
                        "n_gt": int(np.sum(gt_cls_b == cls_id)),
                        "ap50": float(ap[ci, 0]),
                        "ap": float(ap[ci].mean()),
                    }
                )

        buckets.append(
            {
                "name": name,
                "n_gt": n_gt,
                "n_tp": n_tp,
                "n_fn": n_fn,
                "ar50": n_tp / n_gt,
                "ap50": ap50,
                "ap": ap_mean,
                "p": p_mean,
                "r": r_mean,
                "per_class": per_class,
            }
        )
    return {"buckets": buckets, "n_gt_total": n_gt_total, "n_fp_total": n_fp_total, "ar50_total": ar50_total}


def make_scale_stratified_validator(area_edges: tuple[float, ...], long_side_edges: tuple[float, ...]):
    """Create a per-run validator class with the requested bucket edges (no global leakage)."""

    class ConfiguredValidator(ScaleStratifiedOBBValidator):
        _area_edges = tuple(area_edges)
        _long_side_edges = tuple(long_side_edges)

    ConfiguredValidator.__name__ = "ConfiguredScaleStratifiedOBBValidator"
    return ConfiguredValidator


# =========================
# 报告打印与图表
# =========================


def print_per_class(results, names: dict[int, str]) -> None:
    """Print the per-class official metrics table."""
    print(f"{'class':14s} {'P':>8s} {'R':>8s} {'mAP50':>8s} {'mAP50-95':>8s}")
    for i, cls_id in enumerate(results.box.ap_class_index):
        print(
            f"{names[cls_id]:14s} {results.box.p[i]:8.4f} {results.box.r[i]:8.4f} "
            f"{results.box.ap50[i]:8.4f} {results.box.ap[i]:8.4f}"
        )


def print_scale_report(report: dict[str, object], class_names: dict[int, str]) -> None:
    """Print size-bucketed AP/AR tables and the per-class x per-scale AP50 matrices."""
    for axis, label in (("area", "AREA (px², original image)"), ("long_side", "LONG SIDE (px, original image)")):
        section = report.get(axis)
        if not section:
            continue
        buckets = section["buckets"]
        print("-" * 100)
        print(f"Scale buckets [{label}] (conf={report['conf']}):")
        print(f"{'bucket':>12s} {'n_GT':>7s} {'n_TP':>7s} {'n_FN':>7s} {'AR50':>8s} {'AP50':>8s} {'AP':>8s}")
        for bucket in buckets:
            fmt = lambda v: "    n/a" if v is None else f"{v:8.4f}"  # noqa: E731
            print(
                f"{bucket['name']:>12s} {bucket['n_gt']:7d} {bucket['n_tp']:7d} {bucket['n_fn']:7d} "
                f"{fmt(bucket['ar50'])} {fmt(bucket['ap50'])} {fmt(bucket['ap'])}"
            )
        print(
            f"total: n_GT={section['n_gt_total']}  n_FP={section['n_fp_total']}  AR50={section['ar50_total']:.4f}"
            if section["ar50_total"] is not None
            else "total: no GT"
        )
        # per-class x per-scale AP50 matrix
        classes = sorted({entry["class_id"] for bucket in buckets for entry in bucket["per_class"]})
        if not classes:
            continue
        header = f"{'class':14s}" + "".join(f"{bucket['name']:>10s}" for bucket in buckets)
        print(f"Per-class AP50 x {axis} buckets:")
        print(header)
        for cls_id in classes:
            row = f"{class_names[cls_id]:14s}"
            for bucket in buckets:
                entry = next((e for e in bucket["per_class"] if e["class_id"] == cls_id), None)
                row += f"{entry['ap50']:10.4f}" if entry else f"{'-':>10s}"
            print(row)


def plot_scale_metrics(summaries: list[dict[str, object]], run_dir: Path) -> None:
    """Save a line chart of bucketed AP50/AP across evaluated variants.

    每个口径一条实线（AP50）+ 一条同色虚线（AP）；x 轴为尺度桶，
    折线比柱状更能同时展示多个口径与桶之间的趋势而不互相遮挡。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping scale metrics chart.")
        return

    variants = [s for s in summaries if s.get("scale_metrics")]
    if not variants:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = zip(axes, ("area", "long_side"), ("Area buckets (AP50 / AP)", "Long-side buckets (AP50 / AP)"))
    colors = plt.cm.tab10.colors
    for ax, axis, title in panels:
        labels = [bucket["name"] for bucket in variants[0]["scale_metrics"][axis]["buckets"]]
        x = np.arange(len(labels))
        for group, variant in enumerate(variants):
            buckets = variant["scale_metrics"][axis]["buckets"]
            color = colors[group % len(colors)]
            for key, linestyle, label_suffix in (
                ("ap50", "-", "AP50"),
                ("ap", "--", "AP"),
            ):
                values = [b[key] if b[key] is not None else 0.0 for b in buckets]
                ax.plot(
                    x,
                    values,
                    linestyle=linestyle,
                    color=color,
                    marker="o" if linestyle == "-" else "s",
                    markersize=4,
                    linewidth=1.6,
                    alpha=0.9,
                    label=f"{variant['label']} {label_suffix}",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    chart_path = run_dir / "scale_metrics.png"
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] scale metrics chart saved: {chart_path}")


# =========================
# 评估流程
# =========================


def eval_one(
    model: YOLO,
    data_yaml: str,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    batch: int,
    device: str,
    workers: int,
    half: bool,
    plots: bool,
    run_dir: Path,
    label: str,
    scale_metrics: bool = True,
    area_edges: tuple[float, ...] = IDE_AREA_EDGES,
    long_side_edges: tuple[float, ...] = IDE_LONG_SIDE_EDGES,
) -> dict[str, object]:
    """Run one official-metric evaluation and return a JSON-serializable summary."""
    print("=" * 70)
    print(f"[{label}] evaluating on {data_yaml} (split={split})")
    validator = make_scale_stratified_validator(area_edges, long_side_edges) if scale_metrics else None
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        batch=batch,
        device=device,
        workers=workers,
        half=half,
        plots=plots,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        validator=validator,
    )
    summary: dict[str, object] = {
        "label": label,
        "data_yaml": data_yaml,
        "split": split,
        "conf": conf,
        "P": float(np.mean(results.box.p)),
        "R": float(np.mean(results.box.r)),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "per_class": [],
    }
    for i, cls_id in enumerate(results.box.ap_class_index):
        summary["per_class"].append(
            {
                "class": model.names[cls_id],
                "class_id": int(cls_id),
                "P": float(results.box.p[i]),
                "R": float(results.box.r[i]),
                "mAP50": float(results.box.ap50[i]),
                "mAP50-95": float(results.box.ap[i]),
            }
        )
    scale_report = getattr(results, "scale_report", None)
    if scale_report:
        summary["scale_metrics"] = scale_report
    print(
        f"[{label}] P={summary['P']:.4f}  R={summary['R']:.4f}  "
        f"mAP50={summary['mAP50']:.4f}  mAP50-95={summary['mAP50-95']:.4f}"
    )
    print_per_class(results, model.names)
    if scale_report:
        print_scale_report(scale_report, model.names)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides; IDE config values are the defaults."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OBB model on the prepared dataset test split "
        "(official metrics + size-stratified AP/AR)."
    )
    parser.add_argument("--weights", type=str, default=IDE_WEIGHTS)
    parser.add_argument("--data", type=str, default=IDE_DATA_YAML)
    parser.add_argument("--data-clean", type=str, default=IDE_DATA_CLEAN_YAML)
    parser.add_argument("--split", type=str, default=IDE_SPLIT, choices=("test", "val"))
    parser.add_argument("--imgsz", type=int, default=IDE_IMGSZ)
    parser.add_argument("--conf", type=float, default=IDE_CONF)
    parser.add_argument("--iou", type=float, default=IDE_IOU)
    parser.add_argument("--batch", type=int, default=IDE_BATCH)
    parser.add_argument("--device", type=str, default=IDE_DEVICE)
    parser.add_argument("--workers", type=int, default=IDE_WORKERS)
    parser.add_argument("--output-root", type=Path, default=IDE_OUTPUT_ROOT)
    parser.add_argument(
        "--scale-metrics",
        action=argparse.BooleanOptionalAction,
        default=IDE_SCALE_METRICS,
        help="Enable size-stratified AP/AR metrics. 中文：分尺度指标开关。",
    )
    parser.add_argument(
        "--area-edges",
        type=str,
        default=",".join(str(e) for e in IDE_AREA_EDGES),
        help="Comma-separated area bucket edges in px (sqrt thresholds), e.g. 16,64,256. 中文：面积分桶边界。",
    )
    parser.add_argument(
        "--long-side-edges",
        type=str,
        default=",".join(str(e) for e in IDE_LONG_SIDE_EDGES),
        help="Comma-separated long-side bucket edges in px, e.g. 16,32,64,128. 中文：长边分桶边界。",
    )
    parser.add_argument(
        "--deploy-conf",
        type=float,
        default=IDE_DEPLOY_CONF,
        help="Optional deployment operating-point conf for an extra standard-split pass; use -1 to skip. "
        "中文：部署操作点 conf 复评；-1 跳过。",
    )
    return parser.parse_args()


def parse_edges(text: str, name: str) -> tuple[float, ...]:
    """Parse a comma-separated edge list into a sorted tuple."""
    edges = tuple(sorted(float(part) for part in text.split(",") if part.strip()))
    if len(edges) < 1:
        raise ValueError(f"--{name} requires at least one comma-separated number, got {text!r}.")
    if any(edge <= 0 for edge in edges):
        raise ValueError(f"--{name} edges must be positive, got {edges}.")
    return edges


def main() -> None:
    """Evaluate the trained weights on the prepared dataset test split (dual policy + scale metrics)."""
    args = parse_args()
    if args.device and "," in str(args.device):
        raise SystemExit(
            f"Multi-GPU validation is not supported in this script (got device={args.device!r}). "
            "Use a single GPU id (e.g. '0') or 'cpu'."
        )
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not args.data and not args.data_clean:
        raise ValueError("Both --data and --data-clean are None; nothing to evaluate.")
    area_edges = parse_edges(args.area_edges, "area-edges")
    long_side_edges = parse_edges(args.long_side_edges, "long-side-edges")
    deploy_conf = None if args.deploy_conf is not None and args.deploy_conf < 0 else args.deploy_conf

    run_dir = args.output_root / f"val_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"weights: {weights}")
    print(f"split  : {args.split}  imgsz={args.imgsz}  conf={args.conf}  iou={args.iou}  batch={args.batch}")
    print(f"scale  : area_edges={area_edges}  long_side_edges={long_side_edges}  deploy_conf={deploy_conf}")
    print(f"output : {run_dir}")

    model = YOLO(str(weights))

    summaries: list[dict[str, object]] = []
    if args.data:
        summaries.append(
            eval_one(
                model=model,
                data_yaml=args.data,
                split=args.split,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                half=IDE_HALF,
                plots=IDE_PLOTS,
                run_dir=run_dir / "standard",
                label="standard (difficult included)",
                scale_metrics=args.scale_metrics,
                area_edges=area_edges,
                long_side_edges=long_side_edges,
            )
        )
    if args.data_clean:
        summaries.append(
            eval_one(
                model=model,
                data_yaml=args.data_clean,
                split=args.split,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                half=IDE_HALF,
                plots=IDE_PLOTS,
                run_dir=run_dir / "clean",
                label="clean (difficult excluded)",
                scale_metrics=args.scale_metrics,
                area_edges=area_edges,
                long_side_edges=long_side_edges,
            )
        )
    if args.data and deploy_conf is not None:
        summaries.append(
            eval_one(
                model=model,
                data_yaml=args.data,
                split=args.split,
                imgsz=args.imgsz,
                conf=deploy_conf,
                iou=args.iou,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                half=IDE_HALF,
                plots=False,
                run_dir=run_dir / "standard_deploy_conf",
                label=f"standard deploy conf={deploy_conf}",
                scale_metrics=args.scale_metrics,
                area_edges=area_edges,
                long_side_edges=long_side_edges,
            )
        )

    if IDE_PLOTS:
        plot_scale_metrics(summaries, run_dir)

    payload = {
        "weights": str(weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "area_edges_px": list(area_edges),
        "long_side_edges_px": list(long_side_edges),
        "results": summaries,
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print("=" * 70)
    print("Summary (official metrics):")
    for s in summaries:
        print(f"  [{s['label']}] P={s['P']:.4f}  R={s['R']:.4f}  mAP50={s['mAP50']:.4f}  mAP50-95={s['mAP50-95']:.4f}")
    print(f"Saved: {run_dir / 'metrics_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
