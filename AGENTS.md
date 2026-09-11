# Agent 通用指南

本文件供所有编码 Agent 使用，是本仓库项目约定的统一入口。
具体参数、接口和默认值以当前源码及配置为准。

## 项目定位与代码导航

这是 Ultralytics YOLO 的定制分支，主要用于 YOLO26 多光谱旋转框（OBB）检测，
输入主要为 8 通道 NPY 遥感影像。修改时保持已有模型和通用功能的兼容性。

| 路径 | 用途 |
| --- | --- |
| `ultralytics/models/` | 模型及各任务训练、验证、预测实现；OBB 位于 `yolo/obb/` |
| `ultralytics/nn/modules/` | 基础网络层、自定义卷积、光谱注意力和融合模块 |
| `ultralytics/nn/tasks.py` | 模型构建、YAML 解析和权重加载 |
| `ultralytics/engine/` | 通用训练、验证、预测、导出流程 |
| `ultralytics/data/` | 数据加载、标签处理和增强 |
| `ultralytics/cfg/models/26/` | YOLO26 基线及自定义 OBB 模型 YAML |
| `ultralytics/cfg/datasets/` | 数据集配置模板 |
| `examples/` | 数据准备、实验脚本和独立示例 |
| `tests/` | 单元测试和工作流回归测试 |
| `docs/` | 上游文档和构建工具 |

## 当前 OBB 工作流

1. `examples/prepare_obb_dataset.py`：准备数据，按来源分层、按场景组划分
   train/val/test，清洗标签并输出审计信息。
2. `examples/train_obb_dataset.py`：消费准备后的 `data.yaml`，进行整图训练。
3. `examples/validate_obb_test.py`：独立 test 集终评，输出标准、clean 和分尺度指标。
4. `examples/compare_obb_validation.py`：读取两次评估的 `metrics_summary.json` 并对比绘图。

脚本顶部 `IDE_*` 常量支持 IDE 直接运行，也提供 CLI 参数。先阅读源码或查看 `--help`。
本机路径和实验名不保证在其他环境可用；运行前核对输入、输出、设备、权重和模型配置。
不要为检查文档或语法直接启动数据准备、完整训练或终评。

`examples/custom_obb_prepare_and_train.py` 和 `examples/validate_trained_obb_model.py`
是已有的另一套流程，涉及切块等配置。维护时先确认用户使用哪个入口，避免与整图流程混用。

### 数据与评估约束

- 核对 NPY 轴顺序；当前准备入口描述的原始格式是 CWH（8 × W × H），不要默认 HWC。
- 数据 YAML 的 `channels: 8` 必须与实际数据和模型输入一致。
- 保持场景组隔离，避免相关帧跨集合泄漏；夜间过采样仅用于 train。
- 标准 `data.yaml` 保留 difficult 目标；clean 口径剔除 difficult，供单独评估。
- 保留独立 clean 数据集根目录及真实标签映射；图像目录符号链接可能被解析回标准标签。
- test 用于终评，不参与训练和选模；比较实验时对齐输入尺寸、标签口径、推理模式和阈值。

## 自定义网络维护

- `block_My.py` 包含 `DAKConv`、`C3k2_PC`、`DySample_UP` 等模块。
- `ms_msa_torch.py` 包含光谱注意力、通道选择和引导融合模块。
  `MS_MSA` 建模通道相关性，不能误作空间注意力。
- `GMSKConv.py` 包含多尺度卷积模块，涉及 `einops` 依赖。
- 修改 YAML 可用模块时，同时检查 `modules/__init__.py` 的导出、
  `tasks.py` 的导入以及 `parse_model()` 分支。
- 核对 `base_modules` 的通道缩放、`repeat_modules` 的重复次数插入及特殊模块的
  输入输出通道推导；YAML 参数不一定等同于构造函数参数。
- 改变结构或输入通道后，核对实际迁移参数和首层处理逻辑，不能假定预训练权重全部加载成功。

## 开发与验证

```bash
# 按需安装开发依赖
python -m pip install -e ".[dev]"

# 整图工作流回归测试
pytest tests/test_obb_examples_workflows.py -v

# 自定义模块与已有流程测试
pytest tests/test_c3k2_pc.py tests/test_custom_obb_prepare_and_train.py tests/test_validate_trained_obb_model.py -v

# 常规测试；显式包含慢测试
pytest tests/
pytest --slow tests/

# 对修改过的文件检查和格式化
ruff check path/to/changed_file.py
ruff format path/to/changed_file.py

# 按需执行文档和打包验证，需相应依赖
python docs/build_docs.py
python -m build
```

pytest 配置在 `pyproject.toml`，共享 fixture 和慢测试开关在 `tests/conftest.py`。
默认启用 `--doctest-modules`，docstring 中的 `>>>` 示例也可能执行。
按改动范围验证；涉及下载、GPU 或真实数据的测试，先检查环境条件。
行为修复优先在最近的现有测试文件加入回归测试，避免无必要的网络依赖。
报告实际结果及未验证部分，不将语法检查等同于训练或数值正确性验证。

## 编码与协作约定

- Python 使用四空格缩进、120 字符行宽，遵循 Ruff 配置。
- 函数与变量用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_CASE`。
- 在有助于理解接口时增加类型标注，使用简洁的 Google 风格 docstring。
- 开始前检查 Git 状态，保留用户已有修改，避免无关的整仓格式化或重构。
- 不提交数据集、权重、训练产物、缓存、环境文件、凭据或本地插件历史。
  `runs/`、`build/`、`dist/` 是生成产物，但不能因此擅自删除用户文件。
- 提交前核对 diff 和暂存文件；忽略规则不会自动排除已跟踪文件。
- 提交信息简洁，如 `Fix ...`、`Improve ...`、`Add ...`。
  推送前核对远程和分支，本 fork 的 `origin` 与上游 `upstream` 用途不同。
- 向 Ultralytics 上游贡献时遵循 `CONTRIBUTING.md` 的 issue、PR、CI 和 CLA 要求。
- 项目约定统一维护在本文件，不再另建重复的 Agent 指南；
  `README.md` 提供项目简介，子目录 README 保留局部说明。
