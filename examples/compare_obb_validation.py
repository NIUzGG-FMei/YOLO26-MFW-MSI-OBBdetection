"""Compare two OBB validation runs and plot comprehensive metric fluctuations.

本脚本读取两次 ``validate_obb_test.py`` 生成的 ``metrics_summary.json``，做全面横向对比：

1. 官方口径指标 P / R / mAP50 / mAP50-95 的浮动（Δ 与相对变化 %）；
2. 分尺度指标（面积分桶 / 长边分桶）：n_GT / n_TP / n_FN / AR50 / AP50 / AP 的浮动；
3. difficult（标准口径）与 clean（剔除 difficult）两套口径之间的影响对比；
4. 生成 4x3 折线图网格（共 12 个子图）：

   - 第 1 行：官方指标，一列一个口径（standard / clean / deploy conf）；
   - 第 2 行：面积桶，一列一个口径；每个 run 三条线（AP50 实线 / AP 虚线 / AR50 点线）；
   - 第 3 行：长边桶，同上；
   - 第 4 行：difficult vs clean 口径对比（官方指标 / 面积桶 AP50 / 长边桶 AP50）。

   图题标注两个 checkpoint 名（如 ``yolo26n_obb_10_whole_image`` vs
   ``yolo26n_obb_whole_image-2``），图例风格与 ``scale_metrics.png`` 一致
   （每个 run 一种颜色，AP50 实线 / AP 虚线），桶上方标注 ΔAP50 与 ΔAR50。

用法：直接改顶部 IDE 配置后运行，或命令行传 ``--run-a`` / ``--run-b``。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =========================
# IDE Quick Config
# 直接修改这里的值然后点击运行即可。
# =========================

# IDE_RUN_A / IDE_RUN_B:
# - 两次验证的输出目录（含 metrics_summary.json），A 为基准（旧），B 为待比较（新）。
# 中文：两次验证的输出目录。
IDE_RUN_A = Path("runs/obb_test_validation/val_test_20260813_232802")
IDE_RUN_B = Path("runs/obb_test_validation/val_test_20260817_175019")

# IDE_OUTPUT_ROOT:
# 中文：对比折线图输出根目录（每次自动建时间戳子目录）。
IDE_OUTPUT_ROOT = REPO_ROOT / "runs" / "obb_validation_compare"

# 口径标签（按 validate_obb_test.py 的 eval_one label 命名）
STANDARD_LABEL = "standard (difficult included)"
CLEAN_LABEL = "clean (difficult excluded)"


def load_run(run_dir: Path) -> dict:
    """Load one metrics_summary.json and validate its structure."""
    json_path = run_dir / "metrics_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"metrics_summary.json not found: {json_path}")
    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"Invalid metrics_summary.json (missing 'results'): {json_path}")
    return payload


_COMPARABILITY_KEYS = ("split", "imgsz", "conf", "iou", "area_edges_px", "long_side_edges_px")


def validate_comparability(payload_a: dict, payload_b: dict) -> None:
    """Reject runs that were evaluated under different validation settings.

    Comparing runs with different split / imgsz / conf / IoU / bucket edges would
    produce silently misleading deltas; these fields must match exactly.

    中文：校验两次验证 run 的 split/imgsz/conf/iou/分桶边界一致，不一致直接拒绝。
    """
    mismatches = [
        f"{key}: A={payload_a.get(key)!r} vs B={payload_b.get(key)!r}"
        for key in _COMPARABILITY_KEYS
        if payload_a.get(key) != payload_b.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Runs are not comparable; refusing to compute deltas. Differences:\n  " + "\n  ".join(mismatches)
        )


def checkpoint_name(payload: dict, run_dir: Path) -> str:
    """Derive the checkpoint/run name from the weights path or fall back to the run dir name."""
    weights = payload.get("weights")
    if weights:
        candidate = Path(str(weights)).parent.parent.name
        if candidate and candidate not in {"", "checkpoints"}:
            return candidate
    return run_dir.name


def _fmt_delta(value: float | None, digits: int = 4) -> str:
    """Format a signed delta, e.g. +0.0123 / -0.0123 / '     n/a'."""
    if value is None:
        return "     n/a"
    return f"{value:+.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    """Format a signed relative change in percent."""
    if value is None:
        return "     n/a"
    return f"{value * 100:+.2f}%"


def print_comparison_table(a: dict, b: dict) -> None:
    """Print official + scale metric deltas (B - A) for one aligned variant."""
    print(f"{'metric':22s} {'A':>10s} {'B':>10s} {'Δ':>10s} {'Δ%':>10s}")
    for key, name in (("P", "P"), ("R", "R"), ("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")):
        va, vb = a.get(key), b.get(key)
        delta = (vb - va) if (va is not None and vb is not None) else None
        rel = (delta / va) if (va not in {None, 0.0} and delta is not None) else None
        print(
            f"{name:22s} {va if va is None else f'{va:.4f}':>10s} "
            f"{vb if vb is None else f'{vb:.4f}':>10s} "
            f"{_fmt_delta(delta):>10s} {_fmt_pct(rel):>10s}"
        )

    for axis in ("area", "long_side"):
        sa, sb = a.get("scale_metrics", {}).get(axis), b.get("scale_metrics", {}).get(axis)
        if not sa or not sb:
            print(f"  (no scale_metrics for axis {axis})")
            continue
        names_a = {bucket["name"]: bucket for bucket in sa["buckets"]}
        names_b = {bucket["name"]: bucket for bucket in sb["buckets"]}
        print(f"  {axis} buckets (Δ = B - A):")
        print(f"    {'bucket':12s} {'n_GT':>6s} {'n_TP':>7s} {'n_FN':>7s} {'AR50':>8s} {'AP50':>8s} {'AP':>8s}")
        for name in list(names_a) + [name for name in names_b if name not in names_a]:
            ba, bb = names_a.get(name), names_b.get(name)
            if ba is None or bb is None:
                print(f"    {name:12s} (only in {'A' if ba is not None else 'B'})")
                continue
            n_gt = (bb["n_gt"] - ba["n_gt"]) if (ba["n_gt"] is not None and bb["n_gt"] is not None) else None
            n_tp = (bb["n_tp"] - ba["n_tp"]) if (ba["n_tp"] is not None and bb["n_tp"] is not None) else None
            n_fn = (bb["n_fn"] - ba["n_fn"]) if (ba["n_fn"] is not None and bb["n_fn"] is not None) else None
            ar50 = (bb["ar50"] - ba["ar50"]) if (ba["ar50"] is not None and bb["ar50"] is not None) else None
            ap50 = (bb["ap50"] - ba["ap50"]) if (ba["ap50"] is not None and bb["ap50"] is not None) else None
            ap = (bb["ap"] - ba["ap"]) if (ba["ap"] is not None and bb["ap"] is not None) else None
            print(
                f"    {name:12s} {n_gt if n_gt is None else f'{n_gt:+d}':>6s} "
                f"{n_tp if n_tp is None else f'{n_tp:+d}':>7s} "
                f"{n_fn if n_fn is None else f'{n_fn:+d}':>7s} "
                f"{_fmt_delta(ar50, 3):>8s} {_fmt_delta(ap50, 3):>8s} {_fmt_delta(ap, 3):>8s}"
            )


def _safe_delta(a: float | None, b: float | None) -> float | None:
    """Return b - a when both values exist, else None."""
    return (b - a) if (a is not None and b is not None) else None


def print_policy_impact(payload_a: dict, payload_b: dict) -> None:
    """Print the difficult-vs-clean policy impact (standard - clean) for both runs."""
    results_a = {s["label"]: s for s in payload_a["results"]}
    results_b = {s["label"]: s for s in payload_b["results"]}
    missing_a = [label for label in (STANDARD_LABEL, CLEAN_LABEL) if label not in results_a]
    missing_b = [label for label in (STANDARD_LABEL, CLEAN_LABEL) if label not in results_b]
    if missing_a or missing_b:
        print(
            f"[WARN] standard/clean variants missing "
            f"(A missing={missing_a or 'none'}, B missing={missing_b or 'none'}); "
            "skipping difficult-vs-clean comparison."
        )
        return

    standard_a, clean_a = results_a[STANDARD_LABEL], results_a[CLEAN_LABEL]
    standard_b, clean_b = results_b[STANDARD_LABEL], results_b[CLEAN_LABEL]

    print("-" * 100)
    print("[difficult vs clean 口径影响]  (Δ = standard - clean，每列一个 run)")
    print(f"{'metric':22s} {'A_Δ':>9s} {'A_Δ%':>9s} {'B_Δ':>9s} {'B_Δ%':>9s}")
    for key, name in (("P", "P"), ("R", "R"), ("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")):
        rows = []
        for standard, clean in ((standard_a, clean_a), (standard_b, clean_b)):
            delta = _safe_delta(clean.get(key), standard.get(key))
            rel = (delta / clean.get(key)) if (delta is not None and clean.get(key) not in {None, 0.0}) else None
            rows.append((delta, rel))
        print(
            f"{name:22s} {_fmt_delta(rows[0][0]):>9s} {_fmt_pct(rows[0][1]):>9s} "
            f"{_fmt_delta(rows[1][0]):>9s} {_fmt_pct(rows[1][1]):>9s}"
        )

    scale_variants = (standard_a, clean_a, standard_b, clean_b)
    if not all(variant.get("scale_metrics") for variant in scale_variants):
        print("[WARN] scale_metrics missing in one or more standard/clean variants; skipping bucket impact table.")
        return

    for axis in ("area", "long_side"):
        buckets_a = {b["name"]: b for b in standard_a["scale_metrics"][axis]["buckets"]}
        buckets_c = {b["name"]: b for b in clean_a["scale_metrics"][axis]["buckets"]}
        buckets_b = {b["name"]: b for b in standard_b["scale_metrics"][axis]["buckets"]}
        buckets_d = {b["name"]: b for b in clean_b["scale_metrics"][axis]["buckets"]}
        names = list(buckets_a) + [name for name in buckets_b if name not in buckets_a]
        names += [name for name in buckets_c if name not in names] + [name for name in buckets_d if name not in names]
        print(f"  {axis} buckets AP50 / AR50 (Δ = standard - clean):")
        print(f"    {'bucket':12s} {'A_AP50':>9s} {'B_AP50':>9s} {'A_AR50':>9s} {'B_AR50':>9s}")
        for name in names:
            row = [buckets_a.get(name), buckets_c.get(name), buckets_b.get(name), buckets_d.get(name)]
            if any(bucket is None for bucket in row):
                print(f"    {name:12s} (bucket missing in one or more variants)")
                continue
            d_ap50_a = _safe_delta(buckets_c[name].get("ap50"), buckets_a[name].get("ap50"))
            d_ap50_b = _safe_delta(buckets_d[name].get("ap50"), buckets_b[name].get("ap50"))
            d_ar50_a = _safe_delta(buckets_c[name].get("ar50"), buckets_a[name].get("ar50"))
            d_ar50_b = _safe_delta(buckets_d[name].get("ar50"), buckets_b[name].get("ar50"))
            print(
                f"    {name:12s} {_fmt_delta(d_ap50_a, 3):>9s} {_fmt_delta(d_ap50_b, 3):>9s} "
                f"{_fmt_delta(d_ar50_a, 3):>9s} {_fmt_delta(d_ar50_b, 3):>9s}"
            )


def plot_comparison(
    a: dict,
    b: dict,
    name_a: str,
    name_b: str,
    dir_a: Path,
    dir_b: Path,
    out_path: Path,
) -> None:
    """Plot the 4x3 grid: official / area / long-side per variant + difficult-vs-clean panel."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping comparison chart.")
        return

    results_a = {s["label"]: s for s in a["results"]}
    results_b = {s["label"]: s for s in b["results"]}
    shared_labels = [label for label in results_a if label in results_b]
    if len(shared_labels) > 3:
        print(f"[WARN] More than 3 shared variants ({len(shared_labels)}); only the first 3 will be plotted.")
    labels = shared_labels[:3]
    if not labels:
        print("[WARN] No shared variant labels between the two runs; skipping chart.")
        return

    color_a, color_b = plt.cm.tab10.colors[0], plt.cm.tab10.colors[1]
    official_cats = ["P", "R", "mAP50", "mAP50-95"]

    def annotate_delta(
        ax, x_pos: float, delta: float | None, y_max: float, offset: float, fontsize: float = 6.5
    ) -> None:
        """Annotate one signed delta above the lines, colored by sign."""
        if delta is None or np.isnan(delta):
            return
        color = "tab:red" if delta < 0 else "tab:green"
        ax.text(
            x_pos,
            y_max + offset,
            f"{delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )

    def bucket_values(sm: dict, key: str, bucket_names: list[str]) -> list[float]:
        """Extract one metric per bucket in the requested name order, missing -> NaN."""
        names = {bucket["name"]: bucket for bucket in sm["buckets"]}
        return [names.get(name, {}).get(key, np.nan) for name in bucket_names]

    fig = plt.figure(figsize=(21, 16))
    grid = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.22)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(4)])
    fig.suptitle(f"{name_a}  vs  {name_b}", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.965,
        f"A: {dir_a}    B: {dir_b}",
        ha="center",
        va="top",
        fontsize=8,
        color="gray",
    )

    # ---- Rows 1-3: one column per variant ----
    for i, variant_label in enumerate(labels):
        va, vb = results_a[variant_label], results_b[variant_label]
        short = variant_label.split(" (")[0]
        axes[0, i].set_title(f"Official metrics [{short}]", fontsize=9)
        x = np.arange(len(official_cats))
        axes[0, i].plot(
            x, [va[c] for c in official_cats], "-o", color=color_a, linewidth=1.6, markersize=4, label=name_a
        )
        axes[0, i].plot(
            x, [vb[c] for c in official_cats], "-o", color=color_b, linewidth=1.6, markersize=4, label=name_b
        )
        for xi, c in enumerate(official_cats):
            if va[c] is not None and vb[c] is not None:
                annotate_delta(axes[0, i], xi, vb[c] - va[c], max(va[c], vb[c]), 0.035)
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(official_cats)
        axes[0, i].set_ylim(0, 1.12)
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(axis="y", alpha=0.3)

        for row, axis, axis_title in ((1, "area", "Area buckets"), (2, "long_side", "Long-side buckets")):
            sm_a, sm_b = va.get("scale_metrics", {}).get(axis), vb.get("scale_metrics", {}).get(axis)
            ax = axes[row, i]
            if not sm_a or not sm_b:
                ax.set_title(f"{axis_title} [{short}] (no data)", fontsize=9)
                continue
            ax.set_title(f"{axis_title} [{short}] (AP50 / AP / AR50)", fontsize=9)
            buckets = sm_a["buckets"]
            xlabels = [bucket["name"] for bucket in buckets]
            xlabels += [bucket["name"] for bucket in sm_b["buckets"] if bucket["name"] not in xlabels]
            x = np.arange(len(xlabels))
            for run_name, sm, color in ((name_a, sm_a, color_a), (name_b, sm_b, color_b)):
                for key, linestyle, marker, suffix in (
                    ("ap50", "-", "o", "AP50"),
                    ("ap", "--", "s", "AP"),
                    ("ar50", ":", "^", "AR50"),
                ):
                    ax.plot(
                        x,
                        bucket_values(sm, key, xlabels),
                        linestyle=linestyle,
                        color=color,
                        marker=marker,
                        markersize=3.5,
                        linewidth=1.5,
                        alpha=0.9,
                        label=f"{run_name} {suffix}",
                    )
            names_a = {bucket["name"]: bucket for bucket in sm_a["buckets"]}
            names_b = {bucket["name"]: bucket for bucket in sm_b["buckets"]}
            for xi, bucket_name in enumerate(xlabels):
                v50a = names_a.get(bucket_name, {}).get("ap50")
                v50b = names_b.get(bucket_name, {}).get("ap50")
                if v50a is not None and v50b is not None and not np.isnan(v50a) and not np.isnan(v50b):
                    annotate_delta(ax, xi, v50b - v50a, max(v50a, v50b), 0.02)
                ra = names_a.get(bucket_name, {}).get("ar50")
                rb = names_b.get(bucket_name, {}).get("ar50")
                if ra is not None and rb is not None and not np.isnan(ra) and not np.isnan(rb):
                    annotate_delta(ax, xi, rb - ra, max(ra, rb), 0.06)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=30, ha="right")
            ax.set_ylim(0, 1.2)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(axis="y", alpha=0.3)

    # ---- Row 4: difficult vs clean policy impact ----
    std_label = STANDARD_LABEL if STANDARD_LABEL in labels else None
    clean_label = CLEAN_LABEL if CLEAN_LABEL in labels else None
    policy_labels_present = std_label is not None and clean_label is not None
    policy_scale_available = policy_labels_present and all(
        results.get(label, {}).get("scale_metrics") is not None
        for results in (results_a, results_b)
        for label in (std_label, clean_label)
    )
    if policy_labels_present:
        ax = axes[3, 0]
        ax.set_title("difficult vs clean: official metrics", fontsize=9)
        x = np.arange(len(official_cats))
        for run_name, results, color in ((name_a, results_a, color_a), (name_b, results_b, color_b)):
            vs, vc = results[std_label], results[clean_label]
            ax.plot(
                x,
                [vs[c] for c in official_cats],
                "-o",
                color=color,
                linewidth=1.6,
                markersize=4,
                label=f"{run_name} difficult",
            )
            ax.plot(
                x,
                [vc[c] for c in official_cats],
                "--o",
                color=color,
                linewidth=1.4,
                markersize=4,
                alpha=0.8,
                label=f"{run_name} clean",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(official_cats)
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

        if not policy_scale_available:
            axes[3, 1].set_title("difficult vs clean: scale buckets unavailable", fontsize=9)
            axes[3, 2].set_title("difficult vs clean: scale buckets unavailable", fontsize=9)
        for col, axis, panel_title in (
            (1, "area", "difficult vs clean: area buckets AP50"),
            (2, "long_side", "difficult vs clean: long-side buckets AP50"),
        ):
            ax = axes[3, col]
            if not policy_scale_available:
                continue
            ax.set_title(panel_title, fontsize=9)
            sm_s = results_a[std_label]["scale_metrics"][axis]
            xlabels = [bucket["name"] for bucket in sm_s["buckets"]]
            for results in (results_a, results_b):
                for label in (std_label, clean_label):
                    sm = results[label]["scale_metrics"][axis]
                    xlabels += [bucket["name"] for bucket in sm["buckets"] if bucket["name"] not in xlabels]
            x = np.arange(len(xlabels))
            for run_name, results, color in ((name_a, results_a, color_a), (name_b, results_b, color_b)):
                vs = results[std_label]["scale_metrics"][axis]
                vc = results[clean_label]["scale_metrics"][axis]
                ax.plot(
                    x,
                    bucket_values(vs, "ap50", xlabels),
                    "-o",
                    color=color,
                    linewidth=1.6,
                    markersize=4,
                    label=f"{run_name} difficult",
                )
                ax.plot(
                    x,
                    bucket_values(vc, "ap50", xlabels),
                    "--o",
                    color=color,
                    linewidth=1.4,
                    markersize=4,
                    alpha=0.8,
                    label=f"{run_name} clean",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=30, ha="right")
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)
    else:
        axes[3, 0].set_title("difficult vs clean (unavailable)", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] comparison chart saved: {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides; IDE config values are the defaults."""
    parser = argparse.ArgumentParser(
        description="Compare two OBB validation runs (metrics_summary.json) and plot comprehensive metric fluctuations."
    )
    parser.add_argument(
        "--run-a", type=Path, default=IDE_RUN_A, help="Baseline validation run dir. 中文：基准验证目录。"
    )
    parser.add_argument(
        "--run-b", type=Path, default=IDE_RUN_B, help="Compared validation run dir. 中文：待比较验证目录。"
    )
    parser.add_argument(
        "--output-root", type=Path, default=IDE_OUTPUT_ROOT, help="Chart output root. 中文：图表输出根目录。"
    )
    return parser.parse_args()


def main() -> None:
    """Compare two runs comprehensively and save the fluctuation chart."""
    args = parse_args()
    for run_dir in (args.run_a, args.run_b):
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    payload_a = load_run(args.run_a)
    payload_b = load_run(args.run_b)
    validate_comparability(payload_a, payload_b)
    name_a = checkpoint_name(payload_a, args.run_a)
    name_b = checkpoint_name(payload_b, args.run_b)

    print("=" * 100)
    print(f"A: {name_a}  <-  {args.run_a}")
    print(f"B: {name_b}  <-  {args.run_b}")

    results_a = {s["label"]: s for s in payload_a["results"]}
    results_b = {s["label"]: s for s in payload_b["results"]}
    shared = [label for label in results_a if label in results_b]
    missing = [label for label in results_a if label not in results_b] + [
        label for label in results_b if label not in results_a
    ]
    if missing:
        print(f"[WARN] variants present in only one run (skipped): {missing}")
    for label in shared:
        print("-" * 100)
        print(f"[{label}]  (Δ = B - A)")
        print_comparison_table(results_a[label], results_b[label])

    print_policy_impact(payload_a, payload_b)

    out_dir = args.output_root / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path = out_dir / "metrics_comparison.png"
    plot_comparison(payload_a, payload_b, name_a, name_b, args.run_a, args.run_b, out_path)
    print("=" * 100)
    print(f"Chart saved: {out_path}")


if __name__ == "__main__":
    main()
