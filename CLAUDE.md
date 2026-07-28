# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a customized fork of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), extended for **multispectral Oriented Bounding Box (OBB) detection** on 8-channel NPY images. The primary task is training YOLO26-OBB models on custom multi-channel aerial/satellite datasets.

## Commands

```bash
# Install in editable mode (required before running anything)
pip install -e .

# Run all tests (fast only)
pytest tests/ -v

# Run tests including slow ones
pytest tests/ -v --slow

# Run a single test file
pytest tests/test_c3k2_pc.py -v

# Run a single test function
pytest tests/test_c3k2_pc.py::TestC3k2PCShape::test_default_mode -v

# Train with the custom pipeline (edit IDE_* constants at top of file first)
python examples/custom_obb_prepare_and_train.py --mode train --model yolo26n-obb-4.yaml

# Prepare dataset only
python examples/custom_obb_prepare_and_train.py --mode prepare

# Prepare + train in one step
python examples/custom_obb_prepare_and_train.py --mode prepare_and_train
```

Note: `pytest` is configured with `--doctest-modules` in `pyproject.toml`, so any docstrings containing `>>>` examples must be valid and pass.

## Architecture

### Custom Modules (`ultralytics/nn/modules/`)

All custom modules are exported from `__init__.py` and registered in `tasks.py::parse_model`.

**`block_My.py`** — Direction-aware and CSP blocks:

- `DAKConv`: Direction-aware Adaptive Kernel Conv. Uses `PConv` (pinwheel asymmetric padding) branches for each kernel size in `kk=[3,5,7]`, fused via 1×1 Conv or channel-wise attention (`use_attn=True`).
- `C3k2_PC` (extends `C2f_PC`): Drop-in replacement for `C3k2`. Signature after YAML parsing: `(c1, c2, n, c3k, e, attn, kk, g, shortcut)`. Both `c3k=True` and `attn=True` modes are supported.
- `DySample_UP`: Dynamic upsampling with learnable offset prediction.

**`ms_msa_torch.py`** — Spectral attention and fusion modules:

- `MS_MSA`: Channel-wise (spectral) self-attention. Attention matrix is `dim_head × dim_head` (NOT spatial). Designed for inter-channel correlation modeling.
- `SpectralStage`: Wraps `MS_MSA` with channel alignment, depthwise FFN, and residual connections. Used as a `C3k2`-compatible repeat module in YAML.
- `ChannelSelect`: Selects a fixed subset of input channels by index (e.g., `[1,2,4]` for spatial branch splitting).
- `SpectralInputMix`: Lightweight 1×1 Conv+BN+SiLU for learnable spectral band combination before the backbone.
- `GuidedEnhance` / `GuidedEnhanceZeroInit`: Multiplicative spectral-guided spatial enhancement. `ZeroInit` variant starts as identity (safe for training stability).

**`GMSKConv.py`** — Grouped multi-scale kernel convolution (`GMSKConv`, `CKConv`). Requires `einops`.

**`head.py`** — `OBB26`: Extends `OBB` for the YOLO26 end-to-end detection head (`end2end: True` in YAML).

**`block.py`** — Standard `SPPF` extended with `n` (pool repeat count, default=3) and `shortcut` params. YAML: `[SPPF, [1024, 5, 3, True]]` → `k=5, n=3, shortcut=True`.

### YAML Model Parsing (`ultralytics/nn/tasks.py`)

`parse_model()` maps YAML `[from, repeats, module, args]` rows to module constructors:

- **`base_modules`** frozenset: modules that consume `c1` (auto) and `c2 = args[0]` (width-scaled).
- **`repeat_modules`** frozenset: modules where the YAML `repeats` field is inserted as `n` at `args[2]` via `args.insert(2, n)`. Includes `C3k2_PC` and `SpectralStage`.
- For `C3k2` / `C3k2_PC` on M/L/X scales: `args[3]` is forced to `True` (enables `c3k=True`).
- Special handling for non-standard modules: `ChannelSelect`, `SpectralInputMix`, `DySample_UP`, `GMSKConv`, `GuidedEnhance`, `GuidedEnhanceZeroInit` each have their own `elif` branch.

**YAML arg mapping for `C3k2_PC`:**

```
yaml: [-1, 2, C3k2_PC, [256, False, 0.25]]
parsed call: C3k2_PC(c1, c2=256, n=2, c3k=False, e=0.25)
                                           ↑    ↑     ↑
                                         [0]  [1]   [2]  of yaml args list
```

### Data Pipeline for Multispectral Images

Raw inputs are 8-channel NPY files. The custom pipeline in `examples/custom_obb_prepare_and_train.py`:

1. Loads NPY arrays and normalizes layout to HWC (configurable via `IDE_NPY_LAYOUT`: `"CWH"`, `"CHW"`, `"HWC"`).
2. Slices images into fixed-size patches via sliding window.
3. Saves patches as **multi-page TIFF** (`cv2.imwritemulti`), where each page is one channel.
4. Writes YOLO OBB label files with normalized coordinates.
5. Generates `data.yaml` with `channels: 8` — this is read by `YOLODataset` and passed to the base dataset loader.

The `channels` field in `data.yaml` controls how the model's first Conv layer is initialized (channels ≠ 3 triggers weight interpolation from pretrained when loading).

### Model Config Files (`ultralytics/cfg/models/26/`)

| File                         | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| `yolo26-obb.yaml`            | Baseline YOLO26 OBB (C3k2 backbone)                    |
| `yolo26-obb-4.yaml`          | C3k2_PC (DAKConv) backbone                             |
| `yolo26-obb-dual-msmsa.yaml` | Dual-branch: spatial (C3k2) + spectral (SpectralStage) |
| `yolo26-obb-gmsk*.yaml`      | GMSKConv backbone variants                             |
| `yolo26-obb-rgb124.yaml`     | Fixed 3-channel subset [1,2,4] from 8ch input          |

Scale suffix: `yolo26n-obb.yaml` → base file with scale `n`. Scales `[depth, width, max_channels]` are defined inside each YAML.

### Training Script (`examples/custom_obb_prepare_and_train.py`)

Configure via the `IDE_*` constants at the top of the file. Key ones:

- `IDE_RUN_MODE`: `"train"` / `"prepare"` / `"prepare_and_train"`
- `IDE_PRETRAINED`: path to `.pt` file, or `None` for training from scratch
- `IDE_NPY_LAYOUT`: axis order of raw NPY arrays
- `DEFAULT_CONFIG.model`: which YAML to use (e.g., `"yolo26n-obb-4.yaml"`)

When loading pretrained weights onto an architecture-modified model (e.g., `C3k2_PC` instead of `C3k2`), Ultralytics skips non-matching layers silently — only shared layers (Conv stems, head, SPPF) will transfer.
