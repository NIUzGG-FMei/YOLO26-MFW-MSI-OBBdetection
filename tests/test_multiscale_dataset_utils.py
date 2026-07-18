from pathlib import Path

import cv2
import numpy as np

from examples.multiscale_dataset_utils import (
    VIEW_TYPES,
    ViewRatios,
    allocate_view_counts,
    generate_candidates,
    letterbox_full_image,
    load_multichannel_tiff,
    manifest_view_counts,
    maximum_feasible_total,
    select_candidates,
    validate_manifest_rows,
    write_selected_candidates,
)


def test_integer_view_quota_always_sums_to_total():
    ratios = ViewRatios()
    for total in range(3, 101):
        counts = allocate_view_counts(total, ratios)
        assert set(counts) == set(VIEW_TYPES)
        assert sum(counts.values()) == total
        assert all(count >= 1 for count in counts.values())


def test_maximum_feasible_total_accounts_for_integer_rounding():
    ratios = ViewRatios()
    capacities = {"patch256": 100, "patch512": 20, "full_scaled": 10}
    total = maximum_feasible_total(capacities, ratios)
    assert total == 100
    counts = allocate_view_counts(total, ratios)
    assert all(counts[name] <= capacities[name] for name in VIEW_TYPES)


def test_full_image_letterbox_transforms_obb_points():
    image = np.zeros((900, 1200, 8), dtype=np.uint8)
    points = cv2.boxPoints(((600.0, 450.0), (120.0, 60.0), 25.0)).astype(np.float32)
    transformed, labels, metadata = letterbox_full_image(image, [(3, points)], target_size=256)

    assert transformed.shape == (256, 256, 8)
    assert metadata["pad_top"] == 32
    assert np.all(transformed == 0) | np.any(transformed == 114)
    assert len(labels) == 1
    transformed_center, transformed_size, _ = cv2.minAreaRect(labels[0][1])
    assert np.allclose(transformed_center, (128.0, 128.0), atol=1.0)
    assert np.allclose(np.sort(transformed_size), np.sort((25.6, 12.8)), atol=1.0)
    assert np.all(labels[0][1] >= -1e-5)
    assert np.all(labels[0][1] <= 256.0 + 1e-5)


def test_empty_candidates_are_kept_and_manifest_is_balanced(tmp_path: Path):
    image = np.zeros((900, 1200, 8), dtype=np.uint8)
    source = tmp_path / "empty.npy"
    candidates = generate_candidates(
        image_path=source,
        split="train",
        image=image,
        annotations=[],
        patch_sizes=(256, 512),
        full_view_size=256,
        overlap=True,
        keep_empty_patches=True,
        min_iof=0.6,
    )
    selected, expected_counts, _ = select_candidates(candidates, ViewRatios(), seed=7, total_samples=10)
    _, rows = write_selected_candidates(
        selected,
        tmp_path / "prepared",
        source_loader=lambda _: (image, []),
        min_iof=0.6,
        seed=7,
    )

    assert manifest_view_counts(rows) == expected_counts == {"patch256": 7, "patch512": 2, "full_scaled": 1}
    assert all(int(row["label_count"]) == 0 for row in rows)
    assert validate_manifest_rows(rows, ViewRatios()) == expected_counts
    assert load_multichannel_tiff(Path(rows[0]["image_path"])).shape == (256, 256, 8)


def test_overlapping_windows_include_final_image_edge():
    image = np.zeros((900, 1200, 8), dtype=np.uint8)
    candidates = generate_candidates(
        image_path=Path("image.npy"),
        split="train",
        image=image,
        annotations=[],
        patch_sizes=(256, 512),
        full_view_size=256,
        overlap=True,
        keep_empty_patches=True,
        min_iof=0.6,
    )
    patch256 = [candidate for candidate in candidates if candidate.view_type == "patch256"]
    assert max(candidate.x0 for candidate in patch256) == 944
    assert max(candidate.y0 for candidate in patch256) == 644
