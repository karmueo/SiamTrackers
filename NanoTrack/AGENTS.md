# Repository Guidelines

本指南面向贡献者，概述 NanoTrack 仓库的结构、开发命令、代码规范与提交流程。请在对应跟踪器子目录内工作，不要在仓库根目录直接修改。

## 项目结构与模块组织
- 顶层为各跟踪器目录（如 `NanoTrack/`, `LightTrack/`, `SiamRPNpp/`）；修改 NanoTrack 时进入 `NanoTrack/` 内操作。  
- 典型子目录：`bin/` 入口脚本，`models/` 及子包（`nanotrack/`, `pysot/` 等）存放训练与推理代码，`data/` 训练裁剪，`datasets/` 测试数据集，`results/` 或 `hp_search_result/` 输出结果，公共图示在 `image/`。  
- 大型数据集请放置外部路径并软链至 `data/` 或 `datasets/`，避免将数据提交到仓库。

## 构建、测试与开发命令
- 使用 conda 环境：`conda activate track`。  
- 编译扩展（进入目标跟踪器目录）：`python setup.py build_ext --inplace`。  
- 训练：`python bin/train.py`；单序列测试：`python bin/test.py`；评估/基准：`python bin/eval.py`。  
- 性能分析：`python cal_speed.py` 与 `python cal_macs_params.py`。按需在数据子集上先做冒烟运行。

## 代码风格与命名
- 遵循 PEP 8，4 空格缩进；模块与函数用 snake_case，类用 CamelCase，常量与配置键用 UPPER_SNAKE_CASE。  
- 新增配置延续现有命名（如 `configv1.yaml`, `configv2.yaml`），新脚本放在 `bin/` 并用动词+用途命名（例：`train_lighttrack.py`）。  
- 导入顺序：标准库、第三方、本地。能明确时补充类型标注。

## 测试指南
- 无统一测试框架；修改后至少在相关数据子集上运行 `bin/test.py`，并用 `bin/eval.py` 在已知基准集验证。  
- 新增工具或核心模块时，补充轻量 `test_*.py`（同包或 `tests/`）冒烟测试，并注明所需样例数据路径。

## 提交与 Pull Request
- 提交信息用简短祈使句并带跟踪器前缀，例如 `NanoTrack: adjust search head init` 或 `SiamBAN: fix dataset loader`。  
- PR 需说明涉及跟踪器、使用的数据集/配置、执行命令与指标/速度变化；有行为变化时附运行日志或截图。  
- 不提交数据集、checkpoint、ONNX/NCNN 导出；使用下载链接或 `.gitignore` 忽略生成物。

## 安全与配置提示
- 保持数据与模型路径可配置，避免硬编码本地绝对路径。  
- 运行前确认软链有效，避免在仓库内写入大文件；生成物统一放入 `results/` 或 `hp_search_result/`。

## 规则说明
- 所有回答使用中文回复。
- 如果有不明确的信息先提问或查阅网上最新信息，不要臆断后直接修改代码。
