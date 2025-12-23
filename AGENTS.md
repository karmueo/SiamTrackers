# Repository Guidelines

## 项目结构与模块组织
- 每个跟踪器独立放在顶层目录（如 `NanoTrack/`, `LightTrack/`, `SiamRPNpp/`, `SiamBAN/` 等），改动时进入对应子目录，不在仓库根目录直接操作。
- 典型子目录：`bin/` 入口脚本，`models/` 与子包（`nanotrack/`, `pysot/` 等）存放训练/推理代码，`data/` 训练裁剪，`datasets/` 测试数据集，输出放在 `results/` 或 `hp_search_result/`。
- 公共图示在 `image/`，顶层 `README.md` 收集资源与数据集链接。

## 环境、构建与开发命令

- 使用conda虚拟环境`track`
- 训练前在目标跟踪器目录编译扩展：`python setup.py build_ext --inplace`。
- 常用流程（在目标跟踪器目录执行）：
  - 训练：`python bin/train.py`
  - 评估/基准：`python bin/eval.py`
  - 单序列测试：`python bin/test.py`
  - 性能分析：`python cal_speed.py`，`python cal_macs_params.py`
- 数据集不入库；放置于外部路径后软链至 `data/` 或 `datasets/`，避免大文件提交。

## 代码风格与命名
- 遵循 PEP 8，4 空格缩进；模块/函数用 snake_case，类用 CamelCase，常量与配置键用 UPPER_SNAKE_CASE。
- 新增配置保持现有命名模式（如 `configv1.yaml`, `configv2.yaml`），新脚本置于 `bin/` 并用动词+用途命名（如 `train_lighttrack.py`）。
- 能明确时添加类型标注；导入顺序为标准库、第三方、本地。

## 测试指南
- 无统一测试框架；修改后至少在相关数据子集上运行 `bin/test.py`，并用 `bin/eval.py` 在已知基准集验证。
- 新增工具或核心模块时，补充轻量单测/冒烟测试 `test_*.py`（同包或新建 `tests/`），并说明所需样例数据路径。
- 在 PR 描述中记录关键指标（EAO、AO/SR、FPS）变更。

## 提交与 PR 约定
- 提交信息保持简短祈使句并标明跟踪器，如 `NanoTrack: adjust search head init` 或 `SiamBAN: fix dataset loader`。
- PR 需说明涉及的跟踪器、使用的数据集/配置、执行命令、指标或速度变化；若行为变化，附运行日志或截图。
- 不提交数据集、checkpoint、ONNX/NCNN 导出；使用下载链接或 `.gitignore` 忽略生成物。

## 回答说明
所有回答使用中文回复。