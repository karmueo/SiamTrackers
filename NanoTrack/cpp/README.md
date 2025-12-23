# NanoTrack C++ Demo

使用 ONNXRuntime + OpenCV 的简化版 `bin/demo.py`。支持首帧默认 ROI 或人工框选，后续跟踪绘制 bbox 并打印置信度。

## 依赖
- OpenCV（带视频/GUI 模块）
- ONNXRuntime（设置环境变量 `ONNXRUNTIME_DIR` 指向解压/安装目录，需包含 `include` 与 `lib`）
- 已导出的 ONNX 模型：`models/onnx/nanotrack_backbone.onnx`、`models/onnx/nanotrack_head.onnx`（可选 `nanotrack_backbone_search.onnx`）

## 编译
```bash
cd cpp
mkdir -p build && cd build
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime    # 可用 CMake 变量或环境变量 ONNXRUNTIME_DIR；如需 CUDA EP，加 -DUSE_CUDA_EP=ON
make -j
```

## 运行示例
```bash
./nanotrack_demo \
  --video ../bin/girl_dance.mp4 \
  --backbone ../models/onnx/nanotrack_backbone.onnx \
  --head ../models/onnx/nanotrack_head.onnx \
  --search_backbone ../models/onnx/nanotrack_backbone_search.onnx \
  --manual_roi                                   # 若不加则使用默认 ROI 975,517,35,21
```

可选参数：
- `--roi x y w h` 指定首帧 bbox
- `--save out.mp4` 保存结果视频
- `--use_cuda` 尝试启用 CUDA Execution Provider（需编译时打开 `USE_CUDA_EP` 且运行库支持 CUDA）
