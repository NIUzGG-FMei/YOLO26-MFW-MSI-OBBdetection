import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "visualize_single_npy_with_labels.py"
SPEC = importlib.util.spec_from_file_location("visualize_single_npy_with_labels", MODULE_PATH)
visualize = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = visualize
SPEC.loader.exec_module(visualize)


def test_prepare_model_tensor_preserves_pixels_and_only_pads_bottom_right():
    image = np.arange(35 * 65 * 3, dtype=np.uint8).reshape(35, 65, 3)

    tensor, original_hw = visualize.prepare_model_tensor(image, stride=32, pad_value=114)

    assert original_hw == (35, 65)
    assert tuple(tensor.shape) == (1, 3, 64, 96)
    actual_image = tensor[0, :, :35, :65].permute(1, 2, 0).numpy()
    assert np.array_equal(actual_image, image.astype(np.float32) / 255.0)
    assert np.allclose(tensor[0, :, 35:, :].numpy(), 114.0 / 255.0)
    assert np.allclose(tensor[0, :, :, 65:].numpy(), 114.0 / 255.0)


def test_prepare_model_tensor_keeps_stride_aligned_image_at_original_size():
    image = np.full((64, 96, 8), 42, dtype=np.uint8)

    tensor, original_hw = visualize.prepare_model_tensor(image, stride=32)

    assert original_hw == (64, 96)
    assert tuple(tensor.shape) == (1, 8, 64, 96)


def test_resize_heatmap_upsamples_in_model_space_before_cropping_padding():
    heat_map = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = visualize.resize_heatmap_to_image(heat_map, model_input_hw=(64, 96), image_hw=(35, 65))

    normalized = visualize._normalize_to_uint8(heat_map)
    expected = cv2.resize(normalized, (96, 64), interpolation=cv2.INTER_LINEAR)[:35, :65]
    assert result.shape == (35, 65)
    assert np.array_equal(result, expected)
