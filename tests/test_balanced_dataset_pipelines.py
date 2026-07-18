import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import examples.build_target_class_augmented_obb_dataset as augmented
import examples.custom_obb_prepare_and_train as custom
from examples.multiscale_dataset_utils import ViewRatios


def _make_source_dataset(root: Path, class_name: str = "bus") -> tuple[Path, Path]:
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(3):
        image = np.zeros((900, 1200, 8), dtype=np.uint8)
        image[..., index] = 20 + index
        image_path = image_dir / f"sample_{index}.npy"
        np.save(image_path, image)
        points = cv2.boxPoints(((300.0 + index * 250.0, 420.0), (100.0, 50.0), 20.0)).reshape(-1)
        (label_dir / f"sample_{index}.txt").write_text(
            " ".join(f"{value:.3f}" for value in points) + f" {class_name} 0\n", encoding="utf-8"
        )
    return image_dir, label_dir


def test_custom_balanced_preparation_has_exact_quota_and_validates_tiff(tmp_path: Path, monkeypatch):
    image_dir, label_dir = _make_source_dataset(tmp_path / "source", "car")
    output_dir = tmp_path / "prepared"
    monkeypatch.setattr(custom, "IDE_NPY_LAYOUT", "HWC")
    split = custom.DatasetSplitConfig(image_dir=image_dir, label_dir=label_dir, include_difficult=True)
    cfg = replace(
        custom.DEFAULT_CONFIG,
        train=split,
        prepared_dataset_dir=output_dir,
        class_names=("car",),
        use_augmented_dataset=False,
        preprocess_profile="balanced_multiscale",
        view_ratios=(0.70, 0.20, 0.10),
        multiscale_patch_sizes=(256, 512),
        full_view_size=256,
        train_imgsz=256,
        total_train_samples=10,
        strict_view_ratio=True,
        overlap=True,
        keep_empty_patches=True,
    )

    stats, manifest_path = custom.prepare_balanced_multiscale_split("train", split, output_dir, {"car": 0}, cfg)
    validation = custom.validate_balanced_dataset_files(
        manifest_path, expected_channels=8, ratios=ViewRatios(), strict_ratio=True, expected_num_classes=1
    )

    assert stats["selected_counts"] == {"patch256": 7, "patch512": 2, "full_scaled": 1}
    assert validation["checked_samples"] == 10
    assert validation["view_counts"] == stats["selected_counts"]
    assert validation["expected_channels"] == 8
    assert Path(stats["manifest"]).exists()


def test_target_augmented_balanced_preparation_is_target_only_and_exact(tmp_path: Path, monkeypatch):
    image_dir, label_dir = _make_source_dataset(tmp_path / "source", "bus")
    output_dir = tmp_path / "augmented"
    monkeypatch.setattr(custom, "IDE_NPY_LAYOUT", "HWC")
    split = custom.DatasetSplitConfig(image_dir=image_dir, label_dir=label_dir, include_difficult=True)
    class_to_id = {name: index for index, name in enumerate(custom.DEFAULT_CONFIG.class_names)}
    cfg = augmented.TargetPatchAugmentConfig(
        source_splits=("train",),
        output_dir=output_dir,
        target_class_names=("bus",),
        patch_size=(256, 256),
        random_seed=0,
        output_mode="reset",
        include_center_crop=True,
        translation_variants_per_gt=1,
        translation_max_offset=(16, 16),
        geometric_variants_per_gt=1,
        translation_geometric_variants_per_gt=1,
        geometric_transforms=("rot90", "flip_h"),
        preprocess_profile="balanced_multiscale",
        view_ratios=(0.70, 0.20, 0.10),
        multiscale_patch_sizes=(256, 512),
        full_view_size=256,
        total_samples=10,
        strict_view_ratio=True,
        overlap=True,
        keep_empty_patches=True,
    )

    stats, manifest_path = augmented.prepare_balanced_multiscale_split(
        "train", split, cfg, class_to_id, {class_to_id["bus"]}
    )
    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    assert stats["selected_counts"] == {"patch256": 7, "patch512": 2, "full_scaled": 1}
    assert len(rows) == 11  # header plus ten materialized samples
    assert stats["require_target"] is True
    assert stats["overlap"] is True
    assert stats["keep_empty_patches"] is True


def test_balanced_augmented_manifest_is_checked_before_yaml_merge(tmp_path: Path, monkeypatch):
    image_dir, label_dir = _make_source_dataset(tmp_path / "source", "bus")
    augmented_root = tmp_path / "augmented"
    prepared_root = tmp_path / "prepared"
    monkeypatch.setattr(custom, "IDE_NPY_LAYOUT", "HWC")
    split = custom.DatasetSplitConfig(image_dir=image_dir, label_dir=label_dir, include_difficult=True)
    class_to_id = {name: index for index, name in enumerate(custom.DEFAULT_CONFIG.class_names)}
    augmented_cfg = augmented.TargetPatchAugmentConfig(
        source_splits=("train",),
        output_dir=augmented_root,
        target_class_names=("bus",),
        patch_size=(256, 256),
        random_seed=0,
        output_mode="reset",
        include_center_crop=True,
        translation_variants_per_gt=0,
        translation_max_offset=(0, 0),
        geometric_variants_per_gt=0,
        translation_geometric_variants_per_gt=0,
        geometric_transforms=("rot90",),
        preprocess_profile="balanced_multiscale",
        view_ratios=(0.70, 0.20, 0.10),
        multiscale_patch_sizes=(256, 512),
        full_view_size=256,
        total_samples=10,
        strict_view_ratio=True,
        overlap=True,
        keep_empty_patches=True,
    )
    augmented.prepare_balanced_multiscale_split("train", split, augmented_cfg, class_to_id, {class_to_id["bus"]})

    prepared_train = prepared_root / "images" / "train"
    prepared_val = prepared_root / "images" / "val"
    prepared_train.mkdir(parents=True)
    prepared_val.mkdir(parents=True)
    (prepared_root / "labels" / "train").mkdir(parents=True)
    (prepared_root / "labels" / "val").mkdir(parents=True)
    sample = np.zeros((256, 256, 8), dtype=np.uint8)
    custom.save_multichannel_tiff(prepared_train / "base.tiff", sample)
    custom.save_multichannel_tiff(prepared_val / "val.tiff", sample)
    (prepared_root / "labels" / "train" / "base.txt").write_text("\n", encoding="utf-8")
    (prepared_root / "labels" / "val" / "val.txt").write_text("\n", encoding="utf-8")
    custom.write_data_yaml(prepared_root, tuple(custom.DEFAULT_CONFIG.class_names), 8)

    yaml_path = custom.build_training_data_yaml(
        prepared_root,
        tuple(custom.DEFAULT_CONFIG.class_names),
        use_augmented_dataset=True,
        augmented_dataset_dir=augmented_root,
        preprocess_profile="balanced_multiscale",
        view_ratios=(0.70, 0.20, 0.10),
        strict_view_ratio=True,
    )
    train_list = (prepared_root / "train_with_augmented.txt").read_text(encoding="utf-8").splitlines()
    assert yaml_path.name == "data_with_augmented.yaml"
    assert len(train_list) == 11


def test_balanced_cli_requires_explicit_safe_output_and_overlap(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_target_class_augmented_obb_dataset.py",
            "--preprocess-profile",
            "balanced_multiscale",
            "--output-mode",
            "reset",
            "--overlap",
            "--keep-empty-patches",
        ],
    )
    cfg = augmented.build_config(augmented.parse_args())
    assert cfg.preprocess_profile == "balanced_multiscale"
    assert cfg.view_ratios == (0.7, 0.2, 0.1)
    assert cfg.overlap is True
    assert cfg.keep_empty_patches is True


def test_ide_profile_selects_separate_dataset_directories(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_obb_prepare_and_train.py",
            "--mode",
            "train",
            "--preprocess-profile",
            "balanced_multiscale",
            "--no-use-augmented-dataset",
        ],
    )
    balanced_args = custom.parse_args()
    balanced_cfg = custom.build_runtime_config(balanced_args)
    assert balanced_cfg.prepared_dataset_dir == custom.IDE_BALANCED_PREPARED_DATASET_DIR
    assert balanced_cfg.augmented_dataset_dir == custom.IDE_BALANCED_AUGMENTED_DATASET_DIR

    monkeypatch.setattr(
        sys,
        "argv",
        ["custom_obb_prepare_and_train.py", "--mode", "train", "--preprocess-profile", "legacy"],
    )
    legacy_args = custom.parse_args()
    legacy_cfg = custom.build_runtime_config(legacy_args)
    assert legacy_cfg.prepared_dataset_dir == custom.IDE_PREPARED_DATASET_DIR
    assert legacy_cfg.augmented_dataset_dir == custom.IDE_AUGMENTED_DATASET_DIR


def test_ide_augmented_script_selects_profile_directory_and_safe_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_target_class_augmented_obb_dataset.py", "--preprocess-profile", "balanced_multiscale"],
    )
    balanced_cfg = augmented.build_config(augmented.parse_args())
    assert balanced_cfg.output_dir == augmented.IDE_BALANCED_AUGMENTED_DATASET_DIR
    assert balanced_cfg.output_mode == "reset"

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_target_class_augmented_obb_dataset.py", "--preprocess-profile", "legacy"],
    )
    legacy_cfg = augmented.build_config(augmented.parse_args())
    assert legacy_cfg.output_dir == augmented.IDE_AUGMENTED_DATASET_DIR
    assert legacy_cfg.output_mode == augmented.IDE_OUTPUT_MODE


def test_safe_trainer_skips_repeated_validation_after_oom(monkeypatch):
    calls = {"validate": 0}

    def raise_oom(_self):
        calls["validate"] += 1
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(custom.OBBTrainer, "validate", raise_oom)
    trainer = object.__new__(custom.SafeOBBTrainer)
    trainer.save_before_validation = False
    trainer.metrics = {}
    trainer.best_fitness = 0.0
    trainer._clear_memory = lambda: None

    result = trainer.validate()
    repeated = trainer.validate()
    assert calls["validate"] == 1
    assert trainer._validation_oom is True
    assert result == ({}, 0.0)
    assert repeated == ({}, 0.0)


def test_safe_trainer_factory_keeps_per_run_settings():
    trainer_class = custom.make_safe_obb_trainer(val_batch=3, save_before_validation=False)
    assert trainer_class.safe_val_batch == 3
    assert trainer_class.save_before_validation is False


def test_feature_probe_obb_models_accept_eight_channel_256_input():
    from ultralytics.nn.tasks import OBBModel

    for model_name in ("yolo26-obb-F.yaml", "yolo26-obb-4-F.yaml"):
        model = OBBModel(
            cfg=str(Path("ultralytics/cfg/models/26") / model_name),
            nc=1,
            ch=8,
            verbose=False,
        ).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 8, 256, 256))
        assert isinstance(output, tuple)
        assert len(output) == 2


def test_ultralytics_obb_dataset_loader_reads_all_eight_tiff_pages(tmp_path: Path):
    from ultralytics.data.dataset import YOLODataset

    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    custom.save_multichannel_tiff(image_dir / "sample.tiff", np.zeros((256, 256, 8), dtype=np.uint8))
    (label_dir / "sample.txt").write_text("0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4\n", encoding="utf-8")
    dataset = YOLODataset(
        str(image_dir),
        imgsz=256,
        augment=False,
        cache=False,
        data={"names": {0: "car"}, "nc": 1, "channels": 8},
        task="obb",
    )
    sample = dataset[0]
    assert sample["img"].shape == (8, 256, 256)
    assert sample["cls"].shape == (1, 1)
    assert sample["bboxes"].shape == (1, 5)


def test_training_api_uses_safe_trainer_and_memory_safe_overrides(tmp_path: Path, monkeypatch):
    class DummyModel:
        def __init__(self):
            self.callbacks = []
            self.trainer = SimpleNamespace(save_dir=tmp_path)
            self.train_kwargs = None

        def add_callback(self, name, callback):
            self.callbacks.append((name, callback))

        def train(self, **kwargs):
            self.train_kwargs = kwargs

    model = DummyModel()
    monkeypatch.setattr(custom, "load_or_build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(custom, "write_model_profile_csv", lambda *args, **kwargs: tmp_path / "profile.csv")
    monkeypatch.setattr(custom, "resolve_resume_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["custom_obb_prepare_and_train.py", "--mode", "train"])
    args = custom.parse_args()
    cfg = replace(
        custom.build_runtime_config(args),
        preprocess_profile="balanced_multiscale",
        save_dir=tmp_path,
        train_imgsz=256,
        val_batch=1,
        amp=True,
        disable_heavy_augmentation=True,
        enable_post_train_feature_correlation=False,
    )

    custom.train_with_python_api(cfg, args, tmp_path / "data.yaml")
    kwargs = model.train_kwargs
    assert kwargs is not None
    assert kwargs["trainer"].safe_val_batch == 1
    assert kwargs["imgsz"] == 256
    assert kwargs["amp"] is True
    assert kwargs["batch"] == args.batch
    assert kwargs["mosaic"] == 0.0
    assert kwargs["mixup"] == 0.0
    assert kwargs["cutmix"] == 0.0
    assert kwargs["copy_paste"] == 0.0
