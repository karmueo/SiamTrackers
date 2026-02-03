#!/bin/bash

# YOLO标注生成器启动脚本

cd "$(dirname "$0")" || exit 1

# 设置环境变量
export PYTHONPATH="${PWD}"

# 运行YOLO标注生成器
python bin/track_yolo_labeler.py \
    --config "models/config/configv3.yaml" \
    --snapshot "models/pretrained/nanotrackv3.pth" \
    --input "/home/tl/data_80/data/video/110/RGB/fz_12/good/bird" \
    --output-dir "./results/yolo_labels" \
    --track-conf "0.92" \
    --label-interval "3" \
    --max-cache "300" \
    --yolo-model "/home/tl/work/yolo/ultralytics/runs/detect/yolo26s_110_rgb_v2/weights/best.pt" \
    --yolo-conf "0.25" \
    --yolo-iou "0.5" \
    --yolo-imgsz "640x352" \
    --yolo-skip-frames "1" \
    --auto-detect \
    --use-first-class \
    --headless \
    --label-seq-len "3"
