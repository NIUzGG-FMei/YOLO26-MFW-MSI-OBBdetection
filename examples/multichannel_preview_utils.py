from __future__ import annotations

from typing import Iterable

import numpy as np


def parse_display_channels(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"--display-channels expects exactly 3 comma-separated indices, but got: {text!r}")
    try:
        channels = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Invalid channel index in --display-channels={text!r}") from exc
    return channels  # type: ignore[return-value]


def validate_stretch_percentiles(low: float, high: float) -> tuple[float, float]:
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(f"Expected 0 <= stretch_low < stretch_high <= 100, but got low={low}, high={high}")
    return float(low), float(high)


def validate_channel_wavelengths(channel_wavelengths_nm: Iterable[float], expected_channel_count: int) -> tuple[float, ...]:
    wavelengths = tuple(float(value) for value in channel_wavelengths_nm)
    if len(wavelengths) != expected_channel_count:
        raise ValueError(
            "Automatic preview mode requires image channel count to match channel_wavelengths_nm length, "
            f"but got image_channels={expected_channel_count}, wavelength_count={len(wavelengths)}."
        )
    return wavelengths


def select_nearest_channel_index(wavelengths: tuple[float, ...], target_nm: float) -> int:
    return min(range(len(wavelengths)), key=lambda idx: abs(float(wavelengths[idx]) - target_nm))


def resolve_preview_rgb_channels(
    image: np.ndarray,
    preview_mode: str,
    display_channels: tuple[int, int, int],
    channel_wavelengths_nm: Iterable[float],
) -> tuple[int, int, int]:
    if image.ndim == 2:
        return (0, 0, 0)
    channel_count = image.shape[2]
    if channel_count == 1:
        return (0, 0, 0)
    if channel_count == 2:
        return (0, 1, 1)

    if preview_mode == "first3":
        return (0, 1, 2)
    if preview_mode == "manual":
        for channel_idx in display_channels:
            if not 0 <= channel_idx < channel_count:
                raise ValueError(
                    f"Manual display channel {channel_idx} is out of range for image with {channel_count} channels."
                )
        return display_channels

    wavelengths = validate_channel_wavelengths(channel_wavelengths_nm, channel_count)
    if preview_mode == "rgb_like":
        return (
            select_nearest_channel_index(wavelengths, 650.0),
            select_nearest_channel_index(wavelengths, 550.0),
            select_nearest_channel_index(wavelengths, 450.0),
        )
    if preview_mode == "false_color":
        return (
            select_nearest_channel_index(wavelengths, 850.0),
            select_nearest_channel_index(wavelengths, 650.0),
            select_nearest_channel_index(wavelengths, 550.0),
        )
    raise ValueError(f"Unsupported preview mode: {preview_mode}")


def stretch_channel_to_uint8(channel: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    channel = channel.astype(np.float32, copy=False)
    low_value = float(np.percentile(channel, low_percentile))
    high_value = float(np.percentile(channel, high_percentile))
    if not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        return np.zeros(channel.shape, dtype=np.uint8)
    stretched = (channel - low_value) * (255.0 / (high_value - low_value))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def build_preview_rgb(
    image: np.ndarray,
    preview_mode: str,
    display_channels: tuple[int, int, int],
    stretch_low: float,
    stretch_high: float,
    channel_wavelengths_nm: Iterable[float],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    rgb_channels = resolve_preview_rgb_channels(image, preview_mode, display_channels, channel_wavelengths_nm)
    channel_planes: list[np.ndarray] = []
    channel_count = 1 if image.ndim == 2 else image.shape[2]
    for channel_idx in rgb_channels:
        if image.ndim == 2:
            channel = image
        else:
            channel = image[..., min(channel_idx, channel_count - 1)]
        channel_planes.append(stretch_channel_to_uint8(channel, stretch_low, stretch_high))
    preview_rgb = np.stack(channel_planes, axis=2)
    return np.ascontiguousarray(preview_rgb), rgb_channels


def build_preview_bgr(
    image: np.ndarray,
    preview_mode: str,
    display_channels: tuple[int, int, int],
    stretch_low: float,
    stretch_high: float,
    channel_wavelengths_nm: Iterable[float],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    if image.ndim == 2:
        image = image[..., None]
    preview_rgb, rgb_channels = build_preview_rgb(
        image=image,
        preview_mode=preview_mode,
        display_channels=display_channels,
        stretch_low=stretch_low,
        stretch_high=stretch_high,
        channel_wavelengths_nm=channel_wavelengths_nm,
    )
    preview_bgr = np.ascontiguousarray(preview_rgb[..., ::-1])
    return preview_bgr, rgb_channels
