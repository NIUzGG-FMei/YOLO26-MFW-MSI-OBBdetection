# YOLO26 多光谱 OBB 检测

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的定制分支，
面向 8 通道遥感影像的旋转框目标检测，包含自定义网络模块及数据准备、整图训练和测试评估流程。

## 开始使用

```bash
python -m pip install -e .
```

主要入口依次为：

1. `examples/prepare_obb_dataset.py`：数据准备与场景组划分。
2. `examples/train_obb_dataset.py`：整图训练。
3. `examples/validate_obb_test.py`：独立测试集评估。
4. `examples/compare_obb_validation.py`：评估结果对比。

运行前检查脚本说明、顶部 `IDE_*` 配置和命令行 `--help`，按实际环境设置数据、权重和输出路径。
模型配置位于 `ultralytics/cfg/models/26/`。

开发与 Agent 协作约定统一见 [AGENTS.md](AGENTS.md)。
子目录 README 提供对应示例或模块说明；上游通用功能见 [Ultralytics 文档](https://docs.ultralytics.com/)。

## 许可证

参见 [LICENSE](LICENSE)；上游贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。
