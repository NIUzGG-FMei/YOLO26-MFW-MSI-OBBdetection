# [OPEN] gmsk-view-error

## Symptom

- Validation crashes in `GMSKConv.get_hog_feature()`
- Error:
  `RuntimeError: view size is not compatible with input tensor's size and stride ... Use .reshape(...) instead.`

## Runtime Evidence

- Failing file: `ultralytics/nn/modules/GMSKConv.py`
- Failing line: `dirs = dirs_crop.view(B, H_cells, W_cells, -1)`
- Upstream path includes `einops.rearrange()` and tensor slicing before the failing `view()`

## Falsifiable Hypotheses

1. `dirs_crop` becomes non-contiguous after `image2patches()` and crop slicing, so `view()` fails while `reshape()` would succeed.
2. Validation uses a tensor memory layout that differs from training at this branch, exposing a latent contiguity issue only in eval/val.
3. Another upstream op in `get_hog_feature()` returns a strided tensor, and the failure is specific to this exact `view()` call rather than to HOG logic itself.
4. The crash is unrelated to `nbins`/`cell_size`; it is purely a tensor layout issue and should reproduce with the same shape even if those hyperparameters change.

## Plan

1. Inspect the failing tensor-transformation path around `dirs_crop`.
2. Add the minimal instrumentation needed if runtime proof is still required.
3. Apply the smallest safe fix.
4. Rebuild/validate the model path that previously crashed.
