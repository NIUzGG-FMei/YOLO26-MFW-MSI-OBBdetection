import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "validate_trained_obb_model.py"
SPEC = importlib.util.spec_from_file_location("validate_trained_obb_model", MODULE_PATH)
validate_trained_obb_model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validate_trained_obb_model
SPEC.loader.exec_module(validate_trained_obb_model)


def test_run_selected_mode_switches_between_validation_and_error_analysis(monkeypatch, tmp_path):
    calls: list[str] = []

    def _fake_validation(_cfg):
        calls.append("validation")
        return tmp_path / "validation"

    def _fake_analysis(_cfg):
        calls.append("analysis")
        return tmp_path / "analysis"

    def _fake_conf_sweep(_cfg):
        calls.append("conf_sweep")
        return tmp_path / "conf_sweep"

    monkeypatch.setattr(validate_trained_obb_model, "run_validation", _fake_validation)
    monkeypatch.setattr(validate_trained_obb_model, "run_error_analysis", _fake_analysis)
    monkeypatch.setattr(validate_trained_obb_model, "run_conf_sweep", _fake_conf_sweep)

    base_kwargs = dict(
        weights=tmp_path / "best.pt",
        data=tmp_path / "data.yaml",
        imgsz=640,
        conf=0.01,
        iou=0.7,
        batch=1,
        device="cpu",
        results_root=tmp_path / "results",
        max_det=300,
        agnostic_nms=False,
        split="val",
        workers=0,
        half=False,
        plots=False,
        save_json=False,
        error_match_iou=0.5,
        max_error_samples=30,
        run_name="demo",
        dataset_mode="prepared",
        raw_image_dir=tmp_path / "images",
        raw_label_dir=tmp_path / "labels",
        raw_include_difficult=True,
    )

    cfg_off = validate_trained_obb_model.ValidationConfig(
        **base_kwargs,
        error_analysis=validate_trained_obb_model.ErrorAnalysisConfig(
            enabled=False,
            target_classes=("bike",),
            max_export_images=None,
            error_types_to_export=(validate_trained_obb_model.ERROR_TYPE_FALSE_NEGATIVE,),
            output_root=tmp_path / "analysis_root",
        ),
        conf_sweep=validate_trained_obb_model.ConfSweepConfig(
            enabled=False,
            start=0.01,
            end=0.20,
            step=0.01,
            output_root=tmp_path / "conf_sweep_root",
        ),
    )
    cfg_on = validate_trained_obb_model.ValidationConfig(
        **base_kwargs,
        error_analysis=validate_trained_obb_model.ErrorAnalysisConfig(
            enabled=True,
            target_classes=("bike",),
            max_export_images=None,
            error_types_to_export=(validate_trained_obb_model.ERROR_TYPE_FALSE_NEGATIVE,),
            output_root=tmp_path / "analysis_root",
        ),
        conf_sweep=validate_trained_obb_model.ConfSweepConfig(
            enabled=False,
            start=0.01,
            end=0.20,
            step=0.01,
            output_root=tmp_path / "conf_sweep_root",
        ),
    )
    cfg_sweep = validate_trained_obb_model.ValidationConfig(
        **base_kwargs,
        error_analysis=validate_trained_obb_model.ErrorAnalysisConfig(
            enabled=False,
            target_classes=("bike",),
            max_export_images=None,
            error_types_to_export=(validate_trained_obb_model.ERROR_TYPE_FALSE_NEGATIVE,),
            output_root=tmp_path / "analysis_root",
        ),
        conf_sweep=validate_trained_obb_model.ConfSweepConfig(
            enabled=True,
            start=0.01,
            end=0.20,
            step=0.01,
            output_root=tmp_path / "conf_sweep_root",
        ),
    )

    assert validate_trained_obb_model.run_selected_mode(cfg_off) == tmp_path / "validation"
    assert validate_trained_obb_model.run_selected_mode(cfg_on) == tmp_path / "analysis"
    assert validate_trained_obb_model.run_selected_mode(cfg_sweep) == tmp_path / "conf_sweep"
    assert calls == ["validation", "analysis", "conf_sweep"]


def test_compute_image_false_alarm_stats_matches_manual_single_image_math():
    stats = validate_trained_obb_model.compute_image_false_alarm_stats(
        gt_count=2,
        candidate_prediction_count=5,
        passed_conf_count=3,
        true_positive_count=2,
    )

    assert stats.gt_count == 2
    assert stats.candidate_prediction_count == 5
    assert stats.negative_candidate_count == 3
    assert stats.passed_conf_count == 3
    assert stats.true_positive_count == 2
    assert stats.false_positive_count == 1
    assert stats.true_negative_count == 2
    assert stats.false_alarm_rate == 1 / 3


def test_compute_image_false_alarm_stats_handles_empty_and_degenerate_inputs():
    empty_stats = validate_trained_obb_model.compute_image_false_alarm_stats(
        gt_count=0,
        candidate_prediction_count=0,
        passed_conf_count=0,
        true_positive_count=0,
    )
    no_candidate_stats = validate_trained_obb_model.compute_image_false_alarm_stats(
        gt_count=4,
        candidate_prediction_count=0,
        passed_conf_count=0,
        true_positive_count=0,
    )

    assert empty_stats.false_positive_count == 0
    assert empty_stats.true_negative_count == 0
    assert empty_stats.false_alarm_rate is None
    assert no_candidate_stats.negative_candidate_count == 0
    assert no_candidate_stats.true_negative_count == 0
    assert no_candidate_stats.false_alarm_rate is None


def test_summarize_false_alarm_dataset_matches_manual_multi_image_aggregation():
    image_stats = [
        validate_trained_obb_model.compute_image_false_alarm_stats(
            gt_count=2,
            candidate_prediction_count=5,
            passed_conf_count=3,
            true_positive_count=2,
        ),
        validate_trained_obb_model.compute_image_false_alarm_stats(
            gt_count=0,
            candidate_prediction_count=4,
            passed_conf_count=2,
            true_positive_count=0,
        ),
        validate_trained_obb_model.compute_image_false_alarm_stats(
            gt_count=1,
            candidate_prediction_count=3,
            passed_conf_count=1,
            true_positive_count=1,
        ),
    ]

    dataset_stats = validate_trained_obb_model.summarize_false_alarm_dataset(image_stats)

    # Manual baseline:
    # img1 -> FP'=1, TN'=2
    # img2 -> FP'=2, TN'=2
    # img3 -> FP'=0, TN'=2
    # total FP=3, total TN=6, FAR=3/(3+6)=1/3
    assert dataset_stats.image_count == 3
    assert dataset_stats.image_count_with_valid_denominator == 3
    assert dataset_stats.candidate_prediction_total == 12
    assert dataset_stats.negative_candidate_total == 9
    assert dataset_stats.passed_conf_total == 6
    assert dataset_stats.true_positive_total == 3
    assert dataset_stats.false_positive_total == 3
    assert dataset_stats.true_negative_total == 6
    assert dataset_stats.false_alarm_rate == 1 / 3


def test_log_validation_summary_writes_grouped_metric_details(tmp_path):
    logger = validate_trained_obb_model.setup_logger(tmp_path, "false_alarm.log")
    official_stats = {
        "metrics/precision(B)": 0.686017,
        "metrics/recall(B)": 0.636802,
        "metrics/mAP50(B)": 0.685909,
        "metrics/mAP50-95(B)": 0.559981,
    }
    custom_metrics = {
        "tp_iou50": 3086,
        "fp_iou50": 4942,
        "fn_iou50": 90,
        "false_detection_rate": 0.615595,
        "missed_detection_rate": 0.028338,
        "alarm_image_ratio": 0.99,
        "avg_false_positive_boxes_per_image": 49.42,
        "candidate_predictions_total": 12,
        "negative_candidates_total": 9,
        "predictions_above_conf_total": 6,
        "tp_after_conf_total": 3,
        "false_alarm_fp_total": 3,
        "false_alarm_tn_total": 6,
        "false_alarm_rate": 1 / 3,
    }

    validate_trained_obb_model.log_validation_summary(logger, official_stats, custom_metrics)

    log_text = (tmp_path / "false_alarm.log").read_text(encoding="utf-8")
    assert "Official metrics summary: P=0.686017, R=0.636802, mAP50=0.685909, mAP50-95=0.559981" in log_text
    assert "Fixed-threshold error summary: TP=3086, FP=4942, FN=90" in log_text
    assert "candidate_total=12" in log_text
    assert "fp_total=3" in log_text
    assert "tn_total=6" in log_text
    assert "false_alarm_rate=0.333333" in log_text


def test_save_metrics_files_uses_grouped_json_and_prefixed_csv_columns(tmp_path):
    cfg = validate_trained_obb_model.ValidationConfig(
        weights=tmp_path / "best.pt",
        data=tmp_path / "data.yaml",
        imgsz=640,
        conf=0.01,
        iou=0.7,
        batch=1,
        device="cpu",
        results_root=tmp_path / "results",
        max_det=300,
        agnostic_nms=False,
        split="val",
        workers=0,
        half=False,
        plots=False,
        save_json=False,
        error_match_iou=0.5,
        max_error_samples=30,
        run_name="demo",
        dataset_mode="prepared",
        raw_image_dir=tmp_path / "images",
        raw_label_dir=tmp_path / "labels",
        raw_include_difficult=True,
        error_analysis=validate_trained_obb_model.ErrorAnalysisConfig(
            enabled=False,
            target_classes=("bike",),
            max_export_images=None,
            error_types_to_export=(validate_trained_obb_model.ERROR_TYPE_FALSE_NEGATIVE,),
            output_root=tmp_path / "analysis_root",
        ),
        conf_sweep=validate_trained_obb_model.ConfSweepConfig(
            enabled=False,
            start=0.01,
            end=0.20,
            step=0.01,
            output_root=tmp_path / "conf_sweep_root",
        ),
    )
    logger = validate_trained_obb_model.setup_logger(tmp_path, "metrics.log")
    official_stats = {
        "metrics/precision(B)": 0.686017,
        "metrics/recall(B)": 0.636802,
        "metrics/mAP50(B)": 0.685909,
        "metrics/mAP50-95(B)": 0.559981,
    }
    custom_metrics = {
        "tp_iou50": 3086,
        "fp_iou50": 4942,
        "fn_iou50": 90,
        "false_detection_rate": 0.615595,
        "false_alarm_rate": 0.184238,
        "alarm_image_ratio": 0.99,
        "avg_false_positive_boxes_per_image": 49.42,
        "missed_detection_rate": 0.028338,
        "images_with_false_alarm": 99,
        "images_with_missed_detection": 10,
        "candidate_predictions_total": 30000,
        "negative_candidates_total": 26824,
        "predictions_above_conf_total": 8028,
        "tp_after_conf_total": 3086,
        "false_alarm_fp_total": 4942,
        "false_alarm_tn_total": 21882,
        "false_alarm_valid_image_count": 100,
        "archived_error_samples": 30,
    }

    csv_path, json_path = validate_trained_obb_model.save_metrics_files(cfg, tmp_path, logger, official_stats, custom_metrics)

    csv_text = csv_path.read_text(encoding="utf-8")
    json_payload = __import__("json").loads(json_path.read_text(encoding="utf-8"))
    assert "official.precision" in csv_text
    assert "fixed_threshold.false_detection_rate" in csv_text
    assert "candidate_false_alarm.false_alarm_rate" in csv_text
    assert json_payload["official_metrics"]["precision"] == 0.686017
    assert json_payload["fixed_threshold_error_metrics"]["false_detection_rate"] == 0.615595
    assert json_payload["candidate_level_false_alarm_metrics"]["false_alarm_rate"] == 0.184238


def test_save_conf_sweep_plots_exports_official_and_custom_metric_curves(tmp_path):
    logger = validate_trained_obb_model.setup_logger(tmp_path, "conf_sweep.log")
    conf_values = [0.01, 0.05, 0.10]
    metrics_by_name = {
        "P": [0.68, 0.72, 0.75],
        "R": [0.64, 0.61, 0.58],
        "mAP50": [0.69, 0.71, 0.70],
        "mAP50-95": [0.56, 0.58, 0.57],
        "误检率": [0.62, 0.55, 0.48],
        "漏检率": [0.03, 0.05, 0.08],
        "告警图占比": [0.99, 0.84, 0.61],
        "候选级虚警率(自定义)": [0.18, 0.12, 0.09],
    }

    output_paths = validate_trained_obb_model.save_conf_sweep_plots(tmp_path, conf_values, metrics_by_name, logger)

    output_names = {path.name for path in output_paths}
    assert len(output_paths) == 8
    assert "precision_vs_val_conf.png" in output_names
    assert "recall_vs_val_conf.png" in output_names
    assert "map50_vs_val_conf.png" in output_names
    assert "map50_95_vs_val_conf.png" in output_names
    assert "false_detection_rate_vs_val_conf.png" in output_names
    assert "missed_detection_rate_vs_val_conf.png" in output_names
    assert "alarm_image_ratio_vs_val_conf.png" in output_names
    assert "candidate_false_alarm_rate_vs_val_conf.png" in output_names
    for output_path in output_paths:
        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_get_metric_plot_display_name_maps_chinese_metrics_to_ascii_labels():
    assert validate_trained_obb_model.get_metric_plot_display_name("误检率") == "False Detection Rate"
    assert validate_trained_obb_model.get_metric_plot_display_name("漏检率") == "Missed Detection Rate"
    assert validate_trained_obb_model.get_metric_plot_display_name("告警图占比") == "Alarm Image Ratio"
    assert (
        validate_trained_obb_model.get_metric_plot_display_name("候选级虚警率(自定义)")
        == "Candidate False Alarm Rate (Custom)"
    )
    assert validate_trained_obb_model.get_metric_plot_display_name("P") == "P"


def test_save_run_metadata_log_records_weight_path(tmp_path):
    cfg = validate_trained_obb_model.ValidationConfig(
        weights=tmp_path / "best.pt",
        data=tmp_path / "data.yaml",
        imgsz=640,
        conf=0.01,
        iou=0.7,
        batch=1,
        device="cpu",
        results_root=tmp_path / "results",
        max_det=300,
        agnostic_nms=False,
        split="val",
        workers=0,
        half=False,
        plots=False,
        save_json=False,
        error_match_iou=0.5,
        max_error_samples=30,
        run_name="demo",
        dataset_mode="prepared",
        raw_image_dir=tmp_path / "images",
        raw_label_dir=tmp_path / "labels",
        raw_include_difficult=True,
        error_analysis=validate_trained_obb_model.ErrorAnalysisConfig(
            enabled=False,
            target_classes=("bike",),
            max_export_images=None,
            error_types_to_export=(validate_trained_obb_model.ERROR_TYPE_FALSE_NEGATIVE,),
            output_root=tmp_path / "analysis_root",
        ),
        conf_sweep=validate_trained_obb_model.ConfSweepConfig(
            enabled=False,
            start=0.01,
            end=0.20,
            step=0.01,
            output_root=tmp_path / "conf_sweep_root",
        ),
    )
    started_at = validate_trained_obb_model.datetime(2026, 6, 30, 16, 0, 0)
    finished_at = validate_trained_obb_model.datetime(2026, 6, 30, 16, 0, 12)

    log_path = validate_trained_obb_model.save_run_metadata_log(
        run_dir=tmp_path,
        timestamp="20260630_160000",
        cfg=cfg,
        mode_name="validation",
        started_at=started_at,
        finished_at=finished_at,
        extra_lines=["验证结果目录: /tmp/demo"],
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert log_path.name == "validation_run_log_20260630_160000.txt"
    assert f"模型权重完整路径: {cfg.weights}" in log_text
    assert f"数据集配置路径: {cfg.data}" in log_text
    assert "总运行时长(秒): 12.000" in log_text
    assert "验证结果目录: /tmp/demo" in log_text


def test_save_markdown_report_separates_three_metric_sections(tmp_path):
    cfg = validate_trained_obb_model.ValidationConfig(
        weights=tmp_path / "best.pt",
        data=tmp_path / "data.yaml",
        imgsz=640,
        conf=0.01,
        iou=0.7,
        batch=1,
        device="cpu",
        results_root=tmp_path / "results",
        max_det=300,
        agnostic_nms=False,
        split="val",
        workers=0,
        half=False,
        plots=False,
        save_json=False,
        error_match_iou=0.5,
        max_error_samples=30,
        run_name="demo",
        dataset_mode="prepared",
        raw_image_dir=tmp_path / "images",
        raw_label_dir=tmp_path / "labels",
        raw_include_difficult=True,
        error_analysis=validate_trained_obb_model.ErrorAnalysisConfig(
            enabled=False,
            target_classes=("bike",),
            max_export_images=None,
            error_types_to_export=(validate_trained_obb_model.ERROR_TYPE_FALSE_NEGATIVE,),
            output_root=tmp_path / "analysis_root",
        ),
        conf_sweep=validate_trained_obb_model.ConfSweepConfig(
            enabled=False,
            start=0.01,
            end=0.20,
            step=0.01,
            output_root=tmp_path / "conf_sweep_root",
        ),
    )
    official_stats = {
        "metrics/precision(B)": 0.686017,
        "metrics/recall(B)": 0.636802,
        "metrics/mAP50(B)": 0.685909,
        "metrics/mAP50-95(B)": 0.559981,
    }
    custom_metrics = {
        "tp_iou50": 3086,
        "fp_iou50": 4942,
        "fn_iou50": 90,
        "false_detection_rate": 0.615595,
        "false_alarm_rate": 0.184238,
        "alarm_image_ratio": 0.99,
        "avg_false_positive_boxes_per_image": 49.42,
        "missed_detection_rate": 0.028338,
        "images_with_false_alarm": 99,
        "images_with_missed_detection": 10,
        "candidate_predictions_total": 30000,
        "negative_candidates_total": 26824,
        "predictions_above_conf_total": 8028,
        "tp_after_conf_total": 3086,
        "false_alarm_fp_total": 4942,
        "false_alarm_tn_total": 21882,
        "false_alarm_valid_image_count": 100,
        "archived_error_samples": 30,
    }

    report_path = validate_trained_obb_model.save_markdown_report(
        cfg=cfg,
        run_dir=tmp_path,
        official_stats=official_stats,
        custom_metrics=custom_metrics,
        error_csv_path=tmp_path / "error_samples.csv",
        gt_overlay_summary=(30, 0, tmp_path / "error_samples_gt_summary.json"),
        class_distribution_chart_path=tmp_path / "class_instance_distribution.jpg",
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Official Metrics" in report_text
    assert "## Fixed-Threshold Error Metrics" in report_text
    assert "## Candidate-Level False Alarm Metrics" in report_text
    assert "## Artifacts" in report_text
    assert "Ultralytics 官方 PR 曲线口径" in report_text
    assert "固定 `VAL_CONF` 与 `error_match_iou` 的业务点指标" in report_text
    assert "候选级虚警率(自定义)" in report_text
    assert "- 候选预测总数 N: `30000`" in report_text
    assert "- 误检率: `0.615595`" in report_text
    assert "- 漏检率: `0.028338`" in report_text


def test_resolve_target_class_map_supports_single_and_multi_specs(tmp_path):
    warnings: list[str] = []
    logger = validate_trained_obb_model.setup_logger(tmp_path, "resolve.log")
    class_names = {0: "car", 1: "bike", 2: "pedestrian"}

    resolved = validate_trained_obb_model.resolve_target_class_map(("bike", 2, "missing"), class_names, logger, warnings)

    assert resolved == {1: "bike", 2: "pedestrian"}
    assert len(warnings) == 1
    assert "missing" in warnings[0]


def test_normalize_target_class_specs_treats_single_string_as_one_class():
    assert validate_trained_obb_model._normalize_target_class_specs("truck") == ("truck",)
    assert validate_trained_obb_model._normalize_target_class_specs(4) == (4,)
    assert validate_trained_obb_model._normalize_target_class_specs(["truck", "bike"]) == ("truck", "bike")


def test_parse_args_default_target_classes_does_not_split_single_string(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate_trained_obb_model.py"])
    args = validate_trained_obb_model.parse_args()

    assert args.target_classes == ["truck"]
    assert args.val_conf_target_classes == []


def test_normalize_conf_sweep_target_class_specs_supports_all_and_disables_on_python_none():
    assert validate_trained_obb_model._normalize_conf_sweep_target_class_specs("all") == ("ALL",)
    assert validate_trained_obb_model._normalize_conf_sweep_target_class_specs(None) == ()
    assert validate_trained_obb_model._normalize_conf_sweep_target_class_specs(["truck", "ALL", 3]) == (
        "truck",
        "ALL",
        3,
    )


def test_resolve_conf_sweep_target_selections_expands_all_to_every_dataset_class():
    class_names = {0: "car", 1: "truck", 2: "bus"}

    resolved = validate_trained_obb_model.resolve_conf_sweep_target_selections(("ALL", "truck", 2), class_names)

    assert [item.key for item in resolved] == ["class:0", "class:1", "class:2"]
    assert [item.folder_name for item in resolved] == [
        "class_0_car",
        "class_1_truck",
        "class_2_bus",
    ]
    assert [item.display_name for item in resolved] == ["car", "truck", "bus"]


def test_analyze_target_class_errors_counts_fn_and_two_fp_types():
    gt_classes = np.array([1, 1, 0], dtype=np.int64)
    gt_boxes = np.array(
        [
            [20.0, 20.0, 8.0, 4.0, 0.0],
            [40.0, 20.0, 8.0, 4.0, 0.0],
            [60.0, 20.0, 8.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    pred_classes = np.array([1, 1, 1], dtype=np.int64)
    pred_boxes = np.array(
        [
            [20.0, 20.0, 8.0, 4.0, 0.0],  # correct target prediction
            [60.0, 20.0, 8.0, 4.0, 0.0],  # matches non-target class GT
            [85.0, 20.0, 8.0, 4.0, 0.0],  # background FP
        ],
        dtype=np.float32,
    )

    result = validate_trained_obb_model.analyze_target_class_errors(
        gt_classes=gt_classes,
        gt_boxes=gt_boxes,
        pred_classes=pred_classes,
        pred_boxes=pred_boxes,
        target_class_id=1,
        iou_threshold=0.5,
    )

    assert result["false_negative_count"] == 1
    assert result["false_positive_count"] == 2
    assert result["matched_gt_idx"].tolist() == [0]
    assert result["matched_pred_idx"].tolist() == [0]
    assert result["missed_gt_idx"].tolist() == [1]
    assert result["non_target_class_fp_idx"].tolist() == [1]
    assert result["background_fp_idx"].tolist() == [2]


def test_render_and_export_error_images(tmp_path):
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    pred_boxes = np.array(
        [
            [20.0, 20.0, 16.0, 8.0, 0.0],
            [60.0, 20.0, 16.0, 8.0, 0.0],
        ],
        dtype=np.float32,
    )
    gt_boxes = np.array([[40.0, 60.0, 18.0, 10.0, 0.0]], dtype=np.float32)

    fp_canvas = validate_trained_obb_model.render_false_positive_image(
        image=image,
        pred_boxes=pred_boxes,
        non_target_class_fp_idx=np.array([0], dtype=int),
        background_fp_idx=np.array([1], dtype=int),
    )
    fn_canvas = validate_trained_obb_model.render_false_negative_image(
        image=image,
        gt_boxes=gt_boxes,
        missed_gt_idx=np.array([0], dtype=int),
    )

    fp_path = validate_trained_obb_model.export_error_analysis_image(
        tmp_path / "false_positives", "sample", "bike", validate_trained_obb_model.ERROR_TYPE_FALSE_POSITIVE, fp_canvas
    )
    fn_path = validate_trained_obb_model.export_error_analysis_image(
        tmp_path / "false_negatives", "sample", "bike", validate_trained_obb_model.ERROR_TYPE_FALSE_NEGATIVE, fn_canvas
    )

    assert fp_path.exists()
    assert fn_path.exists()
    assert fp_path.name == "sample_bike_false_positive.jpg"
    assert fn_path.name == "sample_bike_false_negative.jpg"
    assert fp_path.parent.name == "bike"
    assert fn_path.parent.name == "bike"
    assert int(fp_canvas.sum()) > 0
    assert int(fn_canvas.sum()) > 0


def test_save_error_analysis_log_and_report(tmp_path):
    cfg = validate_trained_obb_model.ValidationConfig(
        weights=tmp_path / "best.pt",
        data=tmp_path / "data.yaml",
        imgsz=640,
        conf=0.01,
        iou=0.7,
        batch=1,
        device="cpu",
        results_root=tmp_path / "results",
        max_det=300,
        agnostic_nms=False,
        split="val",
        workers=0,
        half=False,
        plots=False,
        save_json=False,
        error_match_iou=0.5,
        max_error_samples=30,
        run_name="demo",
        dataset_mode="prepared",
        raw_image_dir=tmp_path / "images",
        raw_label_dir=tmp_path / "labels",
        raw_include_difficult=True,
        error_analysis=validate_trained_obb_model.ErrorAnalysisConfig(
            enabled=True,
            target_classes=("bike", "pedestrian"),
            max_export_images=None,
            error_types_to_export=(
                validate_trained_obb_model.ERROR_TYPE_FALSE_NEGATIVE,
                validate_trained_obb_model.ERROR_TYPE_FALSE_POSITIVE,
            ),
            output_root=tmp_path / "analysis_root",
        ),
        conf_sweep=validate_trained_obb_model.ConfSweepConfig(
            enabled=False,
            start=0.01,
            end=0.20,
            step=0.01,
            output_root=tmp_path / "conf_sweep_root",
        ),
    )
    stats_by_class = {
        1: {
            "false_negative_total": 2,
            "false_positive_total": 3,
            "false_negative_images": 1,
            "false_positive_images": 2,
            "exported_false_negative_images": 1,
            "exported_false_positive_images": 2,
        },
        2: {
            "false_negative_total": 1,
            "false_positive_total": 4,
            "false_negative_images": 1,
            "false_positive_images": 3,
            "exported_false_negative_images": 1,
            "exported_false_positive_images": 3,
        },
    }
    target_classes = {1: "bike", 2: "pedestrian"}
    started_at = validate_trained_obb_model.datetime(2026, 6, 28, 12, 0, 0)
    finished_at = validate_trained_obb_model.datetime(2026, 6, 28, 12, 0, 5)

    log_path = validate_trained_obb_model.save_error_analysis_log(
        run_dir=tmp_path,
        timestamp="20260628_120000",
        cfg=cfg,
        total_images=100,
        stats_by_class=stats_by_class,
        target_classes=target_classes,
        warnings=["demo warning"],
        started_at=started_at,
        finished_at=finished_at,
    )
    report_path = validate_trained_obb_model.save_error_analysis_report(
        run_dir=tmp_path,
        timestamp="20260628_120000",
        cfg=cfg,
        total_images=100,
        target_classes=target_classes,
        stats_by_class=stats_by_class,
    )

    assert log_path.exists()
    assert report_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    assert "模型权重完整路径" in log_text
    assert "demo warning" in log_text
    assert "bike" in report_text
    assert "pedestrian" in report_text
    assert "所有目标类总误检数量" in report_text


def test_save_conf_sweep_target_plots_exports_four_official_metric_curves_into_subfolder(tmp_path):
    logger = validate_trained_obb_model.setup_logger(tmp_path, "conf_sweep_target.log")
    conf_values = [0.01, 0.05, 0.10]
    metrics_by_name = {
        "P": [0.68, 0.72, 0.75],
        "R": [0.64, 0.61, 0.58],
        "mAP50": [0.69, 0.71, 0.70],
        "mAP50-95": [0.56, 0.58, 0.57],
    }
    selection = validate_trained_obb_model.ConfSweepTargetSelection(
        key="class:1",
        kind="class",
        display_name="truck",
        folder_name="class_1_truck",
        class_id=1,
    )

    output_paths = validate_trained_obb_model.save_conf_sweep_target_plots(
        run_dir=tmp_path,
        conf_values=conf_values,
        target_selection=selection,
        metrics_by_name=metrics_by_name,
        logger=logger,
    )

    assert len(output_paths) == 4
    for output_path in output_paths:
        assert output_path.exists()
        assert output_path.parent == tmp_path / "target_class_plots" / "class_1_truck"
        assert output_path.stat().st_size > 0
    assert {path.name for path in output_paths} == {
        "precision_vs_val_conf.png",
        "recall_vs_val_conf.png",
        "map50_vs_val_conf.png",
        "map50_95_vs_val_conf.png",
    }
