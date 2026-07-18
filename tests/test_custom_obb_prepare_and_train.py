import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "custom_obb_prepare_and_train.py"
SPEC = importlib.util.spec_from_file_location("custom_obb_prepare_and_train", MODULE_PATH)
custom_obb_prepare_and_train = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = custom_obb_prepare_and_train
SPEC.loader.exec_module(custom_obb_prepare_and_train)


def _rect_metrics(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return center and sorted side lengths for order-invariant OBB comparisons."""
    (cx, cy), (w, h), _ = cv2.minAreaRect(points.astype(np.float32))
    return np.array([cx, cy], dtype=np.float32), np.sort(np.array([w, h], dtype=np.float32))


def test_edge_object_iof_filter_and_obb_rebuild():
    """Keep an edge OBB only when its IoF against the patch reaches the configured threshold."""
    points = cv2.boxPoints(((42.0, 25.0), (36.0, 20.0), 28.0)).astype(np.float32)

    iof, clipped = custom_obb_prepare_and_train.compute_patch_iof(points, 0, 0, 50, 50)
    projected, returned_iof = custom_obb_prepare_and_train.project_annotation_to_patch(points, 0, 0, 50, 50)

    assert clipped is not None
    assert np.isclose(iof, returned_iof, atol=1e-6)
    assert iof >= custom_obb_prepare_and_train.PATCH_IOF_THRESHOLD
    assert projected is not None
    assert np.all(projected >= -1e-4)
    assert np.all(projected[:, 0] <= 50.0 + 1e-4)
    assert np.all(projected[:, 1] <= 50.0 + 1e-4)

    expected_center, expected_size = _rect_metrics(clipped)
    actual_center, actual_size = _rect_metrics(projected)
    assert np.allclose(actual_center, expected_center, atol=1e-3)
    assert np.allclose(actual_size, expected_size, atol=1e-3)


def test_fully_inside_object_keeps_full_geometry():
    """Retain a full in-patch OBB with IoF=1 and unchanged geometry."""
    points = cv2.boxPoints(((25.0, 25.0), (24.0, 12.0), 33.0)).astype(np.float32)

    iof, _ = custom_obb_prepare_and_train.compute_patch_iof(points, 0, 0, 50, 50)
    projected, returned_iof = custom_obb_prepare_and_train.project_annotation_to_patch(points, 0, 0, 50, 50)

    assert np.isclose(iof, 1.0, atol=1e-6)
    assert np.isclose(returned_iof, 1.0, atol=1e-6)
    assert projected is not None

    expected_center, expected_size = _rect_metrics(points)
    actual_center, actual_size = _rect_metrics(projected)
    assert np.allclose(actual_center, expected_center, atol=1e-3)
    assert np.allclose(actual_size, expected_size, atol=1e-3)


def test_legacy_negative_label_is_sanitized_and_written(tmp_path):
    """Clip legacy negative coordinates to the image canvas before patch export."""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_root = tmp_path / "prepared"
    image_dir.mkdir()
    label_dir.mkdir()

    original_layout = custom_obb_prepare_and_train.IDE_NPY_LAYOUT
    try:
        custom_obb_prepare_and_train.IDE_NPY_LAYOUT = "HWC"
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        np.save(image_dir / "sample.npy", image)
        (label_dir / "sample.txt").write_text("-10 10 30 10 30 40 -10 40 car 0\n", encoding="utf-8")

        split_cfg = custom_obb_prepare_and_train.DatasetSplitConfig(
            image_dir=image_dir,
            label_dir=label_dir,
            include_difficult=True,
        )
        stats = custom_obb_prepare_and_train.prepare_split(
            split_name="train",
            split_cfg=split_cfg,
            output_root=output_root,
            class_to_id={"car": 0},
            patch_size=(100, 100),
            overlap=False,
            keep_empty_patches=False,
        )
    finally:
        custom_obb_prepare_and_train.IDE_NPY_LAYOUT = original_layout

    assert stats["saved_patches"] == 1
    assert stats["saved_labels"] == 1
    assert stats["kept_objects"] == 1

    label_path = output_root / "labels" / "train" / "sample__x0_y0.txt"
    assert label_path.exists()

    parts = label_path.read_text(encoding="utf-8").strip().split()
    coords = np.array([float(v) for v in parts[1:9]], dtype=np.float32).reshape(4, 2)
    assert np.all(coords >= -1e-6)
    assert np.all(coords <= 1.0 + 1e-6)

    pixel_coords = coords.copy()
    pixel_coords[:, 0] *= 100.0
    pixel_coords[:, 1] *= 100.0
    assert np.isclose(pixel_coords[:, 0].min(), 0.0, atol=1e-3)
    assert np.isclose(pixel_coords[:, 0].max(), 30.0, atol=1e-3)
    assert np.isclose(pixel_coords[:, 1].min(), 10.0, atol=1e-3)
    assert np.isclose(pixel_coords[:, 1].max(), 40.0, atol=1e-3)


def test_write_prepared_dataset_stats_csv(tmp_path):
    """Write class-distribution/size CSV from prepared labels and verify key metrics."""
    dataset_root = tmp_path / "prepared"
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    original_layout = custom_obb_prepare_and_train.IDE_NPY_LAYOUT
    try:
        custom_obb_prepare_and_train.IDE_NPY_LAYOUT = "HWC"
        # 100x100x8 image to mimic multispectral patches.
        sample = np.zeros((100, 100, 8), dtype=np.uint8)
        custom_obb_prepare_and_train.save_multichannel_tiff(dataset_root / "images" / "train" / "a.tiff", sample)
        custom_obb_prepare_and_train.save_multichannel_tiff(dataset_root / "images" / "val" / "b.tiff", sample)
    finally:
        custom_obb_prepare_and_train.IDE_NPY_LAYOUT = original_layout

    (dataset_root / "labels" / "train" / "a.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.4 0.1 0.4\n1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n",
        encoding="utf-8",
    )
    (dataset_root / "labels" / "val" / "b.txt").write_text(
        "1 0.2 0.2 0.4 0.2 0.4 0.5 0.2 0.5\n",
        encoding="utf-8",
    )

    csv_path = custom_obb_prepare_and_train.write_prepared_dataset_stats_csv(dataset_root, ("car", "bike"))
    assert csv_path.exists()

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    assert len(rows) == 4  # 2 classes x 2 splits

    train_car = next(r for r in rows if r["split"] == "train" and r["class_name"] == "car")
    train_bike = next(r for r in rows if r["split"] == "train" and r["class_name"] == "bike")
    val_car = next(r for r in rows if r["split"] == "val" and r["class_name"] == "car")
    val_bike = next(r for r in rows if r["split"] == "val" and r["class_name"] == "bike")

    assert train_car["object_count"] == "1"
    assert train_bike["object_count"] == "1"
    assert val_car["object_count"] == "0"
    assert val_bike["object_count"] == "1"

    # train split has two objects in total, one per class.
    assert np.isclose(float(train_car["object_fraction_in_split"]), 0.5, atol=1e-6)
    assert np.isclose(float(train_bike["object_fraction_in_split"]), 0.5, atol=1e-6)
    # val split has one bike object only.
    assert np.isclose(float(val_car["object_fraction_in_split"]), 0.0, atol=1e-6)
    assert np.isclose(float(val_bike["object_fraction_in_split"]), 1.0, atol=1e-6)


def test_compute_avg_abs_channel_correlation():
    """Return 1.0 for perfectly correlated non-constant channels."""
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    feature_map = torch.stack((base, base * 2), dim=0).unsqueeze(0)
    corr = custom_obb_prepare_and_train.compute_avg_abs_channel_correlation(feature_map)
    assert np.isclose(corr, 1.0, atol=1e-6)


def test_write_model_parameter_stats_csv(tmp_path):
    """Write total/backbone/head parameter counts for a simple parsed-like model."""

    class DummyParsedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList(
                [
                    nn.Conv2d(3, 4, kernel_size=1, bias=True),
                    nn.Conv2d(4, 2, kernel_size=1, bias=False),
                ]
            )

    dummy = DummyParsedModel()
    csv_path = custom_obb_prepare_and_train.write_model_parameter_stats_csv(
        dummy, tmp_path / "model_parameter_stats.csv"
    )
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    assert [row["component"] for row in rows] == ["total", "backbone_excluding_final_head", "detect_head"]

    total = sum(parameter.numel() for parameter in dummy.parameters())
    head = sum(parameter.numel() for parameter in dummy.model[-1].parameters())
    backbone = total - head
    assert rows[0]["param_count"] == str(total)
    assert rows[1]["param_count"] == str(backbone)
    assert rows[2]["param_count"] == str(head)
    assert np.isclose(float(rows[2]["param_ratio"]), head / total, atol=1e-6)
