"""Minimal regression tests for the whole-image OBB train/val/compare workflows.

Covers the three example scripts used by the project's main experiment loop:

- ``examples/train_obb_dataset.py``: scoped pretrained loading, multi-GPU rejection,
  rare-class oversampling path resolution, post-training clean-validation kwargs,
  reproducible resume CLI.
- ``examples/validate_obb_test.py``: scale-bucket helpers and multi-GPU rejection.
- ``examples/compare_obb_validation.py``: run-comparability validation and defensive
  handling of missing variants/scale metrics.

All tests are offline and do not require CUDA or prepared datasets.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


def _load_example(name: str):
    """Import an example script by path without executing its ``main()``."""
    spec = importlib.util.spec_from_file_location(f"obb_example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_mod():
    return _load_example("train_obb_dataset")


@pytest.fixture(scope="module")
def validate_mod():
    return _load_example("validate_obb_test")


@pytest.fixture(scope="module")
def compare_mod():
    return _load_example("compare_obb_validation")


def _result(label: str, *, scale: bool = True) -> dict:
    """Build a minimal metrics_summary.json result entry."""
    result = {"label": label, "P": 0.1, "R": 0.2, "mAP50": 0.3, "mAP50-95": 0.4}
    if scale:
        result["scale_metrics"] = {
            "area": {"buckets": [{"name": "small", "ap50": 0.1, "ar50": 0.2}]},
            "long_side": {"buckets": [{"name": "<16px", "ap50": 0.1, "ar50": 0.2}]},
        }
    return result


def _payload(results: list[dict]) -> dict:
    """Build a minimal metrics_summary.json payload with current comparable settings."""
    return {
        "split": "test",
        "imgsz": 1184,
        "conf": 0.01,
        "area_edges_px": [16.0, 64.0, 256.0],
        "long_side_edges_px": [16.0, 32.0, 64.0, 128.0],
        "results": results,
    }


# -------------------------------
# train_obb_dataset.py
# -------------------------------


def test_layer_index_within_scope(train_mod) -> None:
    assert train_mod._layer_index_within_scope("model.0.conv.weight", 10)
    assert train_mod._layer_index_within_scope("model.10.cv2.weight", 10)
    assert not train_mod._layer_index_within_scope("model.11.upsample.weight", 10)
    assert not train_mod._layer_index_within_scope("model.-1.weight", 10)
    assert train_mod._layer_index_within_scope("unexpected.buffer", 10)


def test_load_pretrained_by_scope_sets_ckpt_and_filters_layers(train_mod, monkeypatch) -> None:
    """Scoped loading must register ckpt and leave out-of-scope layers untouched."""
    import ultralytics.nn.tasks as nn_tasks

    source = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    with torch.no_grad():
        source[0].weight.fill_(1.0)
        source[1].weight.fill_(2.0)
    checkpoint = {"model": source, "epoch": 9, "optimizer": object()}

    target_module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    target_module.load_state_dict(source.state_dict())
    before_layer1 = target_module[1].weight.detach().clone()

    model = SimpleNamespace(model=target_module, ckpt={}, overrides={})
    monkeypatch.setattr(nn_tasks, "load_checkpoint", lambda weights: (source, checkpoint))
    train_mod.load_pretrained_by_scope(model, "fake.pt", "backbone")

    assert model.ckpt is checkpoint
    assert model.overrides.get("pretrained") == "fake.pt"
    assert torch.equal(target_module[0].weight, source[0].weight)
    assert torch.equal(target_module[1].weight, before_layer1)


def test_train_main_rejects_multi_gpu(train_mod, monkeypatch) -> None:
    monkeypatch.setattr(train_mod, "parse_args", lambda: SimpleNamespace(device="0,1"))
    with pytest.raises(SystemExit, match="Multi-GPU training is disabled"):
        train_mod.main()


def test_print_run_cli_resume_uses_last_pt_and_resume_true(train_mod, capsys) -> None:
    """The reproducible CLI for a resumed run must point at last.pt and include resume=True."""
    args = SimpleNamespace(
        model="yolo26n-obb-bifpn.yaml",
        pretrained="/home/user/yolo26n-obb.pt",
        pretrained_scope="backbone",
    )
    kwargs = {"epochs": 10, "resume": True}
    last_pt = Path("/tmp/checkpoints/run/weights/last.pt")

    train_mod.print_run_cli(args, kwargs, last_pt)
    out = capsys.readouterr().out

    assert f"model={last_pt}" in out
    assert "resume=true" in out
    assert "pretrained=" not in out


def test_build_clean_val_kwargs_keeps_results_inside_run_and_device(train_mod) -> None:
    """Post-training clean validation must save under the run dir and reuse the trainer device."""
    model = SimpleNamespace(trainer=SimpleNamespace(args=SimpleNamespace(device="cuda:0")))
    kwargs = {"project": "/tmp/checkpoints", "name": "run1", "device": "0"}

    val_kwargs = train_mod.build_clean_val_kwargs(model, kwargs)

    assert val_kwargs["project"] == "/tmp/checkpoints/run1"
    assert val_kwargs["name"] == "clean_val"
    assert val_kwargs["exist_ok"] is True
    assert val_kwargs["device"] == "cuda:0"

    model_no_device = SimpleNamespace(trainer=SimpleNamespace(args=SimpleNamespace(device=None)))
    val_kwargs = train_mod.build_clean_val_kwargs(model_no_device, {"project": "/tmp/p", "name": "r"})
    assert "device" not in val_kwargs


def test_build_rare_oversampled_yaml_handles_nested_train_list(train_mod, monkeypatch, tmp_path) -> None:
    """Rare oversampling must follow nested image paths instead of flattening stems."""
    monkeypatch.setattr(train_mod, "IDE_RARE_OVERSAMPLE_CLASSES", ("bus",))
    monkeypatch.setattr(train_mod, "IDE_RARE_MIN_OBJECTS", 1)
    monkeypatch.setattr(train_mod, "IDE_RARE_OVERSAMPLE_REPEATS", 1)

    dataset = tmp_path / "ds"
    (dataset / "images" / "train" / "sub").mkdir(parents=True)
    (dataset / "labels" / "train" / "sub").mkdir(parents=True)
    (dataset / "images" / "train" / "sub" / "img.tiff").write_bytes(b"")
    (dataset / "labels" / "train" / "sub" / "img.txt").write_text("1 0.5 0.5 0.1 0.05 0.0\n", encoding="utf-8")
    train_list = dataset / "train_list.txt"
    train_list.write_text("images/train/sub/img.tiff\n", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset}\ntrain: train_list.txt\nnames:\n  0: car\n  1: bus\n",
        encoding="utf-8",
    )

    derived = train_mod.build_rare_oversampled_yaml(data_yaml)

    import yaml

    assert derived is not None
    payload = yaml.safe_load(derived.read_text(encoding="utf-8"))
    entries = (dataset / payload["train"]).read_text(encoding="utf-8").splitlines()
    assert len(entries) == 2  # base + one rare-class repeat


# -------------------------------
# validate_obb_test.py
# -------------------------------


def test_validate_main_rejects_multi_gpu(validate_mod, monkeypatch) -> None:
    monkeypatch.setattr(validate_mod, "parse_args", lambda: SimpleNamespace(device="0,1"))
    with pytest.raises(SystemExit, match="Multi-GPU validation is not supported"):
        validate_mod.main()


def test_bucket_ap_empty_input_returns_empty_buckets(validate_mod) -> None:
    report = validate_mod._bucket_ap(
        np.zeros(0),
        (16.0, 64.0, 256.0),
        np.zeros(0, dtype=np.int64),
        np.zeros((0, 10), dtype=bool),
        np.zeros(0),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        area_mode=True,
    )
    assert report["n_gt_total"] == 0
    assert [bucket["name"] for bucket in report["buckets"]] == ["mini", "small", "medium", "large"]
    assert all(bucket["n_gt"] == 0 for bucket in report["buckets"])


# -------------------------------
# compare_obb_validation.py
# -------------------------------


def test_validate_comparability_rejects_mismatched_edges(compare_mod) -> None:
    a = _payload([_result("standard (difficult included)")])
    b = json.loads(json.dumps(a))
    b["area_edges_px"] = [8.0, 32.0, 128.0]
    with pytest.raises(ValueError, match="not comparable"):
        compare_mod.validate_comparability(a, b)


def test_validate_comparability_accepts_matching_settings(compare_mod) -> None:
    a = _payload([_result("standard (difficult included)")])
    b = json.loads(json.dumps(a))
    compare_mod.validate_comparability(a, b)


def test_print_policy_impact_missing_clean_in_b_does_not_crash(compare_mod, capsys) -> None:
    a = _payload([_result("standard (difficult included)"), _result("clean (difficult excluded)")])
    b = _payload([_result("standard (difficult included)")])

    compare_mod.print_policy_impact(a, b)

    assert "standard/clean variants missing" in capsys.readouterr().out


def test_print_policy_impact_missing_scale_metrics_does_not_crash(compare_mod, capsys) -> None:
    a = _payload(
        [
            _result("standard (difficult included)", scale=False),
            _result("clean (difficult excluded)", scale=False),
        ]
    )
    b = json.loads(json.dumps(a))

    compare_mod.print_policy_impact(a, b)

    assert "scale_metrics missing" in capsys.readouterr().out
