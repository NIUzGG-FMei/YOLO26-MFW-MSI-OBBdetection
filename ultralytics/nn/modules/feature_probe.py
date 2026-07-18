from __future__ import annotations

import torch
import torch.nn as nn

"""
Plug-and-play feature probe module for YOLO YAML architectures.

`FeatureProbe` is an identity module that can be dropped in at any position of
a model YAML without altering the forward semantics or the shape/order of any
state dict entry. It exists purely so that external tooling (e.g. the feature
map visualization pipeline in ``examples/visualize_single_npy_with_labels.py``)
can register hooks at specific, human-labelled positions of the graph.

Because it holds no learnable parameters, a checkpoint trained on a model
*without* the probe can be loaded onto the same model with any number of
probes inserted — the probe simply contributes an empty state dict entry.
"""
"""
  用法

  在任意 ultralytics/cfg/models/26/*.yaml 中，在想观察的位置插入即可：
  - [-1, 1, FeatureProbe,["p3_after_c3k2"]]   # 命名
  - [-1, 1, FeatureProbe,[]]                  # 匿名

  examples/visualize_single_npy_with_labels.py
  然后在脚本顶部：
  ENABLE_FEATURE_VISUALIZATION = True
  FEATURE_VIS_MODEL_YAML = Path("../yolo26-obb.yaml")      # 已插入 FeatureProbe
  FEATURE_VIS_CHECKPOINT = Path(".../best.pt")              # 训练时不含 FeatureProbe
  FEATURE_VIS_OUTPUT_DIR = Path(".../feature_maps")


插入 FeatureProbe 后,head 里所有绝对索引都要往后偏移，因为每个前置 probe 各占一个层号。example:
    - [-1, 6] → [-1, 8](cat backbone P4,target=第 5 层 C3k2_PC[512, False, 0.25] + 后面 1 个 probe → 索引
  5→6,但因为它前还有 2 个 probe,最终 → 8)
    - [-1, 4] → [-1, 5](cat backbone P3)
    - [-1, 13] → [-1, 19]
    - [-1, 10] → [-1, 15]
    - [16, 19, 22] → [22, 25, 28](OBB26 头输入)

  ⚠️ 使用提示：
  这就是"在 YAML 里插入独立层"这种方式的固有代价——只要动一个绝对索引前的位置，下游所有 [-1, N] 里的 N
  都得重算。如果你要在其它 YAML(如 yolo26-obb-2.yaml, yolo26-obb-dual-msmsa.yaml
  等)里加probe,请套同样规则：每插一个 probe,它后面所有绝对索引各自 +1。

"""


class FeatureProbe(nn.Module):
    """Identity module that records the tensor it receives when capture is enabled.

    YAML usage examples::

        -[-1, 1, FeatureProbe, ["p3_after_c3k2"]]  # named probe
        -[-1, 1, FeatureProbe, []]  # anonymous probe

    The probe is a no-op during forward — it returns its input unchanged. When ``enable_capture`` is True (typically
    toggled by the visualization script), it also stores a detached copy of the input tensor in ``last_feature`` for
    later retrieval.

    Attributes:
        probe_name (str): Optional human-readable identifier used by the visualization script to label heatmaps and
            output filenames.
        enable_capture (bool): When True, cache the input on every forward.
        last_feature (torch.Tensor | None): The most recently captured tensor (detached, on the same device as the
            input) or None when nothing has been captured yet.
    """

    def __init__(self, name: str | None = None):
        super().__init__()
        self.probe_name: str = "" if name is None else str(name)
        self.enable_capture: bool = False
        self.last_feature: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_capture:
            self.last_feature = x.detach()
        return x

    def clear(self) -> None:
        """Drop the cached tensor to free memory between runs."""
        self.last_feature = None

    def extra_repr(self) -> str:
        return f"name={self.probe_name!r}"
