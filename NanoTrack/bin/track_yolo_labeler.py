#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO数据集标注生成工具 - 使用NanoTrack辅助标注。

该脚本提供交互式界面，通过NanoTrack跟踪辅助生成YOLO格式的目标检测标注数据。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(os.getcwd())

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.model_load import load_pretrain

# 尝试导入YOLO（可选依赖）
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


# 常量定义
ARROW_LEFT_KEYS = (81, 2424832)  # 左方向键键值
ARROW_RIGHT_KEYS = (83, 2555904)  # 右方向键键值
# 类别名称映射（英文用于显示，中文用于命令行提示）
CLASS_NAMES = ['Bird', 'Drone', 'Night-Drone', 'Aircraft', 'Balloon']
CLASS_NAMES_CN = ['鸟', '无人机', '夜间无人机', '飞机', '气球']  # 中文名称


def parse_args():
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description='YOLO标注生成工具')  # 参数解析器
    parser.add_argument('--config', default='./models/config/configv3.yaml', type=str, help='config file')
    parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth', type=str, help='model path')
    parser.add_argument('--input', required=True, type=str, help='输入视频文件或包含视频的文件夹')
    parser.add_argument('--output-dir', default='./results/yolo_labels', type=str, help='YOLO数据集输出目录')
    parser.add_argument('--track-conf', default=0.3, type=float, help='跟踪置信度阈值')
    parser.add_argument('--label-interval', default=5.0, type=float, help='标注生成间隔(秒)')
    parser.add_argument('--label-seq-len', default=1, type=int, help='每次标注保存连续帧数量(仅最后一帧生成标注)')
    parser.add_argument('--max-cache', default=300, type=int, help='最大缓存帧数(用于左方向键回退)')
    # YOLO检测相关参数
    parser.add_argument('--yolo-model', default='', type=str, help='YOLO模型路径 (如: yolov8n.pt)')
    parser.add_argument('--yolo-conf', default=0.5, type=float, help='YOLO检测置信度阈值')
    parser.add_argument('--yolo-iou', default=0.5, type=float, help='检测框与跟踪框IOU阈值 (低于此值认为跟踪失败)')
    parser.add_argument('--yolo-imgsz', default='', type=str, help='YOLO推理输入尺寸 (如: 640 或 640x352)')
    parser.add_argument('--yolo-skip-frames', default=1, type=int, help='每隔多少帧进行一次YOLO检测 (1=每帧检测, 值越大性能越好但检测越不频繁)')
    parser.add_argument('--auto-detect', action='store_true', help='启用自动检测模式 (第一帧自动检测目标)')
    parser.add_argument('--headless', action='store_true', help='无显示模式 (只在需要用户交互时显示画面，其他时候只打印进度以提高速度)')
    parser.add_argument('--use-first-class', action='store_true', help='使用首个类别模式 (第一个视频首次选择类别后，后续所有视频自动使用该类别，无需再次选择)')
    return parser.parse_args()


def build_nano_tracker(snapshot_path, device):
    """构建 NanoTrack 跟踪器。

    Args:
        snapshot_path (str): 模型权重路径。
        device (torch.device): 运行设备。

    Returns:
        object: 构建好的跟踪器实例。
    """
    model = ModelBuilder()  # 模型实例
    model = load_pretrain(model, snapshot_path).to(device).eval()  # 加载权重后的模型
    return build_tracker(model)


def build_yolo_detector(yolo_model_path, device):
    """构建 YOLO 检测器。

    Args:
        yolo_model_path (str): YOLO模型路径。
        device (torch.device): 运行设备。

    Returns:
        object|None: YOLO模型实例，如果不可用则返回None。
    """
    if not yolo_model_path or not YOLO_AVAILABLE:
        return None
    try:
        model = YOLO(yolo_model_path)
        print(f'YOLO模型加载成功: {yolo_model_path}')
        return model
    except Exception as e:
        print(f'YOLO模型加载失败: {e}')
        return None


def calculate_iou(bbox1, bbox2):
    """计算两个边界框的IOU。

    Args:
        bbox1 (list): 第一个边界框 [x, y, w, h]。
        bbox2 (list): 第二个边界框 [x, y, w, h]。

    Returns:
        float: IOU值 (0-1)。
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 计算交集区域
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def parse_imgsz(imgsz_str):
    """解析YOLO输入尺寸字符串。

    Args:
        imgsz_str (str): 尺寸字符串，如 "640", "640x352", "640 352"。

    Returns:
        int|tuple|None: 解析后的尺寸，如 640 或 (640, 352)，失败返回None。
    """
    if not imgsz_str:
        return None

    try:
        # 尝试分割（支持 "640x352" 或 "640 352"）
        if 'x' in imgsz_str.lower():
            parts = imgsz_str.lower().split('x')
        else:
            parts = imgsz_str.split()

        if len(parts) == 1:
            # 单个尺寸（正方形）
            return int(parts[0])
        elif len(parts) == 2:
            # 宽x高
            return (int(parts[0]), int(parts[1]))
        else:
            print(f'警告: 无效的imgsz格式: {imgsz_str}')
            return None
    except ValueError:
        print(f'警告: 无法解析imgsz: {imgsz_str}')
        return None


def yolo_detect(yolo_model, frame, conf_threshold, imgsz=None):
    """使用YOLO模型进行目标检测。

    Args:
        yolo_model: YOLO模型实例。
        frame (np.ndarray): 输入图像。
        conf_threshold (float): 置信度阈值。
        imgsz (int|tuple|None): 推理输入尺寸，None表示使用默认值。

    Returns:
        list: 检测结果列表，每个元素为 (bbox, class_id, confidence)，bbox格式为 [x, y, w, h]。
    """
    if yolo_model is None:
        return []

    try:
        # 构建推理参数
        kwargs = {'verbose': False, 'conf': conf_threshold}
        if imgsz is not None:
            kwargs['imgsz'] = imgsz

        results = yolo_model(frame, **kwargs)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # 获取边界框 (xyxy格式)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # 转换为xywh格式
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                # 获取类别和置信度
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                detections.append(([x, y, w, h], class_id, confidence))

        return detections
    except Exception as e:
        print(f'YOLO检测出错: {e}')
        return []


def clamp_bbox(bbox, frame_shape):
    """裁剪边界框到图像范围内。

    Args:
        bbox (list): 边界框 [x, y, w, h]。
        frame_shape (tuple): 图像形状 (H, W, C)。

    Returns:
        list: 裁剪后的边界框。
    """
    x, y, w, h = bbox  # 边界框参数
    x = max(0, min(x, frame_shape[1] - 1))  # 左上角x
    y = max(0, min(y, frame_shape[0] - 1))  # 左上角y
    w = max(1, min(w, frame_shape[1] - x))  # 宽度
    h = max(1, min(h, frame_shape[0] - y))  # 高度
    return [x, y, w, h]


def bbox_to_yolo(bbox, img_width, img_height):
    """将边界框转换为YOLO格式。

    Args:
        bbox (list): 边界框 [x, y, w, h]。
        img_width (int): 图像宽度。
        img_height (int): 图像高度。

    Returns:
        list: YOLO格式边界框 [class_id, x_center, y_center, width, height]（归一化）。
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]


def select_class_from_console():
    """从命令行选择目标类别。

    Returns:
        int|None: 选中的类别ID，如果取消则返回None。
    """
    print('\n请选择目标类别 (Select target class):')
    for i, (name_cn, name_en) in enumerate(zip(CLASS_NAMES_CN, CLASS_NAMES)):
        print(f'  {i}. {name_cn} ({name_en})')
    print('  输入其他数字或按 Ctrl+C 取消')

    try:
        while True:
            user_input = input('请输入类别编号 (0-4): ').strip()
            if not user_input:
                continue
            try:
                class_id = int(user_input)
                if 0 <= class_id < len(CLASS_NAMES):
                    print(f'已选择: {CLASS_NAMES_CN[class_id]} ({CLASS_NAMES[class_id]})\n')
                    return class_id
                else:
                    print('无效的类别编号，请重新输入')
            except ValueError:
                print('请输入有效的数字')
    except KeyboardInterrupt:
        print('\n已取消选择')
        return None


def select_class_from_console_once(video_name):
    """在headless模式下进行一次性的命令行类别选择（支持跳过）。

    Args:
        video_name (str): 当前视频名称，用于提示信息。

    Returns:
        int|str|None: 返回类别ID；返回'skip_video'表示跳过视频；返回None表示跳过恢复。
    """
    print('\nheadless模式：跟踪失败，YOLO检测到目标，请选择类别进行恢复:')
    for i, (name_cn, name_en) in enumerate(zip(CLASS_NAMES_CN, CLASS_NAMES)):
        print(f'  {i}. {name_cn} ({name_en})')
    print('  回车: 跳过本次恢复')
    print('  n: 跳过当前视频')
    print('  q: 取消恢复（继续尝试跟踪）')

    try:
        user_input = input(f'[{video_name}] 请输入类别编号 (0-4): ').strip()  # 用户输入
    except (EOFError, KeyboardInterrupt):
        print('\n输入被中断，跳过本次恢复')
        return None

    if not user_input:
        return None
    if user_input.lower() == 'n':
        return 'skip_video'
    if user_input.lower() == 'q':
        return None

    try:
        class_id = int(user_input)  # 解析后的类别ID
    except ValueError:
        print('输入无效，跳过本次恢复')
        return None

    if 0 <= class_id < len(CLASS_NAMES):
        print(f'已选择类别: {CLASS_NAMES_CN[class_id]} ({CLASS_NAMES[class_id]})')
        return class_id

    print('类别编号超出范围，跳过本次恢复')
    return None


def get_video_files(input_path):
    """获取输入路径下的所有视频文件。

    Args:
        input_path (str): 输入文件或文件夹路径。

    Returns:
        list: 视频文件路径列表。
    """
    input_path = Path(input_path)
    if input_path.is_file():
        return [str(input_path)]
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f'**/*{ext}'))
        video_files.extend(input_path.glob(f'**/*{ext.upper()}'))
    return [str(f) for f in sorted(video_files)]


def draw_overlay(frame, bbox, score, class_id, status_text):
    """绘制跟踪框、置信度、类别与状态信息。

    Args:
        frame (np.ndarray): 当前帧图像。
        bbox (list|None): 边界框 [x, y, w, h]。
        score (float|None): 置信度分数。
        class_id (int|None): 类别ID。
        status_text (str): 状态提示文本。

    Returns:
        np.ndarray: 绘制后的图像。
    """
    output = frame.copy()
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_text = ''
        if class_id is not None and 0 <= class_id < len(CLASS_NAMES):
            label_text = CLASS_NAMES[class_id]
        if score is not None:
            label_text += ' {:.3f}'.format(score)
        if label_text:
            cv2.putText(output, label_text, (x, max(0, y - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if status_text:
        cv2.putText(output, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return output


def save_label_sequence(frame_cache, cache_start_idx, current_idx, seq_len, frame, visualized_frame,
                        bbox, class_id, img_dir, label_dir, visualized_dir, video_name):
    """保存连续帧图像并生成当前帧的YOLO标注。

    Args:
        frame_cache (list): 帧缓存列表。
        cache_start_idx (int): 缓存起始帧索引。
        current_idx (int): 当前帧索引。
        seq_len (int): 连续保存的帧数。
        frame (np.ndarray): 当前帧原始图像。
        visualized_frame (np.ndarray): 绘制了标注框的可视化图像。
        bbox (list): 边界框 [x, y, w, h]。
        class_id (int): 类别ID。
        img_dir (str): 原始图像保存目录。
        label_dir (str): 标注保存目录。
        visualized_dir (str): 可视化图像保存目录。
        video_name (str): 视频名称。

    Returns:
        str|None: 保存的当前帧文件名，若缓存不足则返回None。
    """
    start_idx = current_idx - (seq_len - 1)  # 连续帧起始索引
    if start_idx < cache_start_idx:
        return None

    for seq_idx in range(start_idx, current_idx):  # 序列帧索引
        seq_frame = frame_cache[seq_idx - cache_start_idx]  # 序列帧图像
        seq_img_name = f'{video_name}_{seq_idx:06d}.jpg'  # 序列帧文件名
        seq_img_path = os.path.join(img_dir, seq_img_name)  # 序列帧输出路径
        if not os.path.exists(seq_img_path):
            cv2.imwrite(seq_img_path, seq_frame)

    img_name = f'{video_name}_{current_idx:06d}.jpg'  # 当前帧文件名
    img_path = os.path.join(img_dir, img_name)  # 当前帧输出路径
    label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))  # 标注输出路径
    visualized_path = os.path.join(visualized_dir, img_name)  # 可视化输出路径

    # 保存当前帧原始图像
    cv2.imwrite(img_path, frame)

    # 保存可视化图像（带标注框）
    cv2.imwrite(visualized_path, visualized_frame)

    # 保存YOLO格式标注
    h, w = frame.shape[:2]  # 当前图像尺寸
    yolo_bbox = bbox_to_yolo(bbox, w, h)  # YOLO格式框
    with open(label_path, 'w') as f:
        f.write(f'{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n')

    return img_name


def process_single_video(video_path, tracker, yolo_model, args, is_first_video=False, global_class_locked=False, global_class_id=None):
    """处理单个视频文件。

    Args:
        video_path (str): 视频文件路径。
        tracker: NanoTrack跟踪器实例。
        yolo_model: YOLO检测器实例（可选）。
        args (argparse.Namespace): 命令行参数。
        is_first_video (bool): 是否为第一个视频（用于use-first-class模式）。
        global_class_locked (bool): 全局类别锁定状态（跨视频保持）。
        global_class_id (int|None): 全局锁定的类别ID（跨视频保持）。

    Returns:
        tuple: (生成的标注数量, 更新后的全局锁定状态, 更新后的全局类别ID)
    """
    headless = args.headless  # 无显示模式标志
    use_first_class = args.use_first_class  # 首个类别模式标志
    lock_permanent = use_first_class and global_class_locked  # 永久锁定标志（禁用解锁）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('无法打开视频: {}'.format(video_path))
        return (0, global_class_locked, global_class_id)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0

    ret, first_frame = cap.read()
    if not ret:
        print('视频为空: {}'.format(video_path))
        cap.release()
        return (0, global_class_locked, global_class_id)

    # 提取视频名称（用于输出目录）
    video_name = Path(video_path).stem

    # 使用安全的窗口名称（避免中文导致cv2.selectROI问题）
    window_name = f"VideoLabeler_{hash(video_name) & 0x7fffffff}"

    # 计算视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    expected_labels = int(video_duration / args.label_interval) + 1

    print(f'\n正在处理: {video_name}')
    print(f'  视频时长: {video_duration:.1f}秒')
    print(f'  标注间隔: {args.label_interval}秒')
    print(f'  预计生成: {expected_labels}个标注文件')

    # 创建输出目录（使用统一的images和labels目录）
    img_dir = os.path.join(args.output_dir, 'images')
    label_dir = os.path.join(args.output_dir, 'labels')
    visualized_dir = os.path.join(args.output_dir, 'visualized')  # 可视化图片目录
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(visualized_dir, exist_ok=True)

    # 解析YOLO输入尺寸
    yolo_imgsz = parse_imgsz(args.yolo_imgsz)
    if yolo_imgsz is not None:
        print(f'  YOLO推理尺寸: {yolo_imgsz}')

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 状态变量
    frame_cache = [first_frame]
    cache_start_idx = 0
    results = {}
    current_idx = 0
    last_processed_idx = -1
    tracking_active = False
    playing = False
    headless_ui_enabled = headless  # headless模式下仅首帧允许一次窗口交互
    waiting_for_single_target = False  # 等待单目标标志
    skip_frame_save = False  # 跳过当前帧保存标志
    last_multi_target_print_idx = -1  # 上次打印多目标的帧索引
    yolo_detect_counter = 0  # YOLO检测计数器
    auto_recovered = False  # YOLO自动恢复标志
    class_locked = global_class_locked  # 使用全局类别锁定状态
    last_bbox = None
    last_score = None
    current_class_id = global_class_id  # 使用全局类别ID
    label_seq_len = args.label_seq_len  # 连续保存帧数
    status_text = 'PAUSED: Space=Play/Pause, R=Select ROI, L=Lock Class, N=Skip Video'
    saved_labels = 0
    last_save_time = -args.label_interval

    # 打印操作提示（首个类别模式的自动初始化时不显示）
    if not (use_first_class and global_class_id is not None):
        print('\n操作提示:')
        print('  空格: 播放/暂停')
        print('  R: 手动框选目标')
        print('  N: 跳过当前视频（直接处理下一个视频）')
        print('  L/Shift+Tab: 锁定/解锁类别（锁定后YOLO自动恢复不再询问类别）')
        print('  D/右方向键: 单步前进')
        print('  A/左方向键: 单步回退')
        print('  Q/ESC: 退出\n')

    # 自动检测模式：在第一帧进行检测
    # 如果开启首个类别模式且已有全局类别，则跳过交互式检测，直接使用后续的自动初始化
    if args.auto_detect and yolo_model is not None and not (use_first_class and global_class_id is not None):
        print('\n进行自动检测...')
        detections = yolo_detect(yolo_model, first_frame, args.yolo_conf, yolo_imgsz)

        if len(detections) == 1:
            # 检测到单个目标
            bbox, _, conf = detections[0]
            bbox = [int(v) for v in clamp_bbox(bbox, first_frame.shape)]

            print(f'检测到目标 (置信度: {conf:.3f})')
            print(f'边界框: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}')

            # 在画面上绘制检测框
            display_frame = first_frame.copy()
            x, y, w, h = bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # 显示类别选择提示
            help_text = [
                'Detected target! Select class:',
                '0: Bird    1: Drone    2: Night-Drone',
                '3: Aircraft  4: Balloon',
                'Press 0-4 | R=Redraw | N=Skip Video | ESC=Cancel'
            ]
            for i, text in enumerate(help_text):
                cv2.putText(display_frame, text, (10, 25 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, display_frame)
            cv2.waitKey(1)

            # 命令行提示用户选择类别
            print('\n' + '='*50)
            print('检测到目标！请选择类别:')
            print('  0: Bird        (鸟)')
            print('  1: Drone       (无人机)')
            print('  2: Night-Drone (夜间无人机)')
            print('  3: Aircraft    (飞机)')
            print('  4: Balloon     (气球)')
            print('='*50)
            print('操作: 按 0-4 选择类别 | R 手动框选 | N 跳过视频 | ESC 取消')
            sys.stdout.flush()

            # 等待用户按键选择类别
            class_id = None
            if class_locked and current_class_id is not None:
                # 使用锁定的类别
                class_id = current_class_id
                print(f'使用锁定类别: {CLASS_NAMES_CN[class_id]} ({CLASS_NAMES[class_id]})')
                print('按 空格 开始跟踪，按 R 重新框选，按 ESC 取消')
                sys.stdout.flush()
                # 等待用户确认或重新框选
                wait_for_input = True
            else:
                wait_for_input = True

            while wait_for_input:
                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC - 取消
                    wait_for_input = False
                elif key in (ord('n'), ord('N')):  # N - 跳过当前视频
                    print(f'\n跳过当前视频: {video_name}\n')
                    cap.release()
                    cv2.destroyAllWindows()
                    return (0, class_locked, current_class_id)  # 直接返回，处理下一个视频
                elif key == ord(' '):  # 空格 - 确认并开始跟踪
                    if class_id is not None:
                        wait_for_input = False
                elif key in (ord('r'), ord('R')):  # R - 手动框选
                    wait_for_input = False
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)
                    roi = cv2.selectROI(window_name, first_frame, False, False)
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

                    init_rect = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
                    if init_rect[2] > 0 and init_rect[3] > 0:
                        init_rect = clamp_bbox(init_rect, first_frame.shape)
                        # 销毁窗口以便在终端进行输入
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)
                        class_id = select_class_from_console()
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

                    if class_id is not None:
                        tracker.init(first_frame, init_rect)
                        tracking_active = True
                        current_class_id = class_id
                        class_locked = True  # 自动锁定类别
                        last_processed_idx = current_idx
                        last_bbox = init_rect
                        last_score = 1.0
                        results[current_idx] = {'bbox': init_rect, 'score': 1.0}
                        playing = True
                        print(f'手动框选并初始化跟踪: {CLASS_NAMES_CN[class_id]} (已锁定)\n')
                    break
                elif ord('0') <= key <= ord('4'):  # 0-4 - 选择类别
                    class_id = key - ord('0')
                    class_locked = True  # 自动锁定类别
                    print(f'\n已选择类别: {class_id} - {CLASS_NAMES_CN[class_id]} ({CLASS_NAMES[class_id]})')
                    print('✓ 类别已自动锁定，后续自动恢复将使用此类别')
                    print('提示: 按 L 键可解锁类别\n')
                    sys.stdout.flush()
                    wait_for_input = False

            if class_id is not None:
                # 使用选择的类别初始化跟踪
                tracker.init(first_frame, bbox)
                tracking_active = True
                current_class_id = class_id
                last_processed_idx = current_idx
                last_bbox = bbox
                last_score = 1.0
                results[current_idx] = {'bbox': bbox, 'score': 1.0}
                playing = True
                print(f'已自动初始化跟踪，当前类别: {CLASS_NAMES_CN[class_id]} ({CLASS_NAMES[class_id]}) (已锁定)\n')
            else:
                print('取消自动检测，请手动框选\n')
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        elif len(detections) > 1:
            print(f'检测到 {len(detections)} 个目标，跳过自动初始化')
            waiting_for_single_target = True
            status_text = f'DETECTED {len(detections)} TARGETS - Waiting for single target'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        else:
            print('未检测到目标，请手动框选')
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, first_frame)

            # 等待用户操作
            print('提示: 按 R 手动框选 | 按 N 跳过视频 | 按 ESC 退出')
            while True:
                key = cv2.waitKeyEx(0)
                if key == 27 or key == ord('q'):  # ESC 或 Q
                    print('退出程序')
                    cap.release()
                    cv2.destroyAllWindows()
                    return (0, global_class_locked, global_class_id)
                elif key in (ord('n'), ord('N')):  # N 键跳过视频
                    print(f'跳过当前视频: {video_name}\n')
                    cap.release()
                    cv2.destroyAllWindows()
                    return (0, global_class_locked, global_class_id)
                elif key in (ord('r'), ord('R')):  # R 键手动框选
                    print('请手动框选目标...')
                    roi = cv2.selectROI(window_name, first_frame, False, False)
                    init_rect = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
                    if init_rect[2] > 0 and init_rect[3] > 0:
                        init_rect = clamp_bbox(init_rect, first_frame.shape)
                        # 手动选择类别（首个类别模式下，如果有全局类别则直接使用）
                        if use_first_class and global_class_id is not None:
                            class_id = global_class_id
                            class_locked = True
                            print(f'使用全局类别: {CLASS_NAMES_CN[class_id]} ({CLASS_NAMES[class_id]}) (已锁定)\n')
                        else:
                            cv2.destroyAllWindows()
                            cv2.waitKey(1)
                            print('\n请选择目标类别:')
                            print('  0: Bird        (鸟)')
                            print('  1: Drone       (无人机)')
                            print('  2: Night-Drone (夜间无人机)')
                            print('  3: Aircraft    (飞机)')
                            print('  4: Balloon     (气球)')
                            while True:
                                try:
                                    choice = input('请输入类别编号 (0-4): ').strip()
                                    if choice in ['0', '1', '2', '3', '4']:
                                        class_id = int(choice)
                                        class_locked = True
                                        print(f'已选择类别: {class_id} - {CLASS_NAMES_CN[class_id]} ({CLASS_NAMES[class_id]}) (已锁定)\n')
                                        break
                                    else:
                                        print('输入无效，请输入 0-4 之间的数字')
                                except (EOFError, KeyboardInterrupt):
                                    print('\n取消选择')
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    return (0, global_class_locked, global_class_id)

                        # 初始化跟踪
                        tracker.init(first_frame, init_rect)
                        tracking_active = True
                        current_class_id = class_id
                        last_processed_idx = 0
                        last_bbox = init_rect
                        last_score = 1.0
                        results[0] = {'bbox': init_rect, 'score': 1.0}
                        playing = True
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        print('已初始化跟踪，按空格开始处理\n')
                        break
                    else:
                        print('框选无效，请重新选择')
                        cv2.imshow(window_name, first_frame)

    # 首个类别模式：如果已有全局类别，自动检测并初始化
    if use_first_class and global_class_id is not None and not tracking_active:
        print(f'\n使用全局类别: {CLASS_NAMES_CN[global_class_id]} ({CLASS_NAMES[global_class_id]})')
        print('正在进行自动检测...')

        if yolo_model is not None:
            detections = yolo_detect(yolo_model, first_frame, args.yolo_conf, yolo_imgsz)

            if len(detections) == 1:
                # 检测到单个目标，自动初始化
                bbox, _, conf = detections[0]
                init_rect = [int(v) for v in clamp_bbox(bbox, first_frame.shape)]

                tracker.init(first_frame, init_rect)
                tracking_active = True
                current_class_id = global_class_id
                class_locked = True
                last_processed_idx = 0
                last_bbox = init_rect
                last_score = 1.0
                results[0] = {'bbox': init_rect, 'score': 1.0}
                playing = True

                print(f'  检测到目标 (置信度: {conf:.3f})')
                print(f'  边界框: x={init_rect[0]}, y={init_rect[1]}, w={init_rect[2]}, h={init_rect[3]}')
                print(f'  已自动初始化跟踪，类别: {CLASS_NAMES_CN[global_class_id]} ({CLASS_NAMES[global_class_id]}) (已锁定)')
                print('  开始自动处理...\n')
            else:
                print(f'  检测到 {len(detections)} 个目标，跳过该视频')
                cap.release()
                cv2.destroyAllWindows()
                return (0, global_class_locked, global_class_id)
        else:
            print('  错误: YOLO模型未加载，无法自动检测')
            cap.release()
            cv2.destroyAllWindows()
            return (0, global_class_locked, global_class_id)

    # headless模式：首帧交互完成后立即销毁窗口，后续不再弹出
    if headless and headless_ui_enabled:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        headless_ui_enabled = False  # 首帧交互已结束，禁用后续窗口
        print('headless模式：首帧交互已完成，后续不再显示窗口')

    while True:
        if current_idx < cache_start_idx:
            current_idx = cache_start_idx

        # 获取当前帧
        if current_idx <= cache_start_idx + len(frame_cache) - 1:
            frame = frame_cache[current_idx - cache_start_idx]
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame_cache.append(frame)
            if len(frame_cache) > args.max_cache:
                frame_cache.pop(0)
                cache_start_idx += 1
            if current_idx < cache_start_idx:
                current_idx = cache_start_idx

        current_time = current_idx / fps

        # 重置跳过标志和恢复标志
        skip_frame_save = False
        auto_recovered = False

        # 执行跟踪
        if tracking_active and current_idx == last_processed_idx + 1:
            outputs = tracker.track(frame)
            score = outputs.get('best_score', 1.0)
            bbox = outputs.get('bbox', None)
            if bbox is not None:
                bbox = clamp_bbox(bbox, frame.shape)
            results[current_idx] = {'bbox': bbox, 'score': score}
            last_processed_idx = current_idx
            last_bbox = bbox
            last_score = score

            # 检查置信度和YOLO辅助恢复
            track_failed = False  # 跟踪失败标志

            if score < args.track_conf:
                track_failed = True
                # 尝试用YOLO检测恢复
                if yolo_model is not None:
                    detections = yolo_detect(yolo_model, frame, args.yolo_conf, yolo_imgsz)
                    if len(detections) == 1:
                        # YOLO检测到单个目标，自动恢复跟踪
                        det_bbox, _, conf = detections[0]
                        det_bbox = clamp_bbox(det_bbox, frame.shape)

                        # 使用锁定的类别或让用户选择
                        recovery_class_id = None
                        if class_locked and current_class_id is not None:
                            recovery_class_id = current_class_id
                        else:
                            if headless and not headless_ui_enabled:
                                # headless模式下不再弹窗，改为命令行一次性选择
                                playing = False
                                console_choice = select_class_from_console_once(video_name)  # 命令行选择结果
                                if console_choice == 'skip_video':
                                    print(f'\n跳过当前视频: {video_name}')
                                    print(f'已保存 {saved_labels} 个标注\n')
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    return (saved_labels, class_locked, current_class_id)
                                recovery_class_id = console_choice if isinstance(console_choice, int) else None
                                if recovery_class_id is not None:
                                    class_locked = True  # 自动锁定类别
                                    print(f'✓ 类别已自动锁定，后续自动恢复将使用此类别\n')
                                    sys.stdout.flush()
                            else:
                                # 在画面上显示检测框并等待用户选择类别
                                display_frame = frame.copy()
                                x, y, w, h = [int(v) for v in det_bbox]
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

                                # 显示类别选择提示
                                help_text = [
                                    f'Tracking LOST! YOLO detected target (conf: {conf:.3f})',
                                    'Select class to recover:',
                                    '0: Bird    1: Drone    2: Night-Drone',
                                    '3: Aircraft  4: Balloon',
                                    'Press 0-4 | SPACE=Skip | N=Skip Video | ESC=Cancel'
                                ]
                                for i, text in enumerate(help_text):
                                    cv2.putText(display_frame, text, (10, 25 + i * 25),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                                cv2.imshow(window_name, display_frame)
                                cv2.waitKey(1)

                                # 命令行提示
                                print('\n' + '='*50)
                                print('跟踪失败！YOLO检测到目标，请选择类别进行恢复:')
                                print('  0: Bird        (鸟)')
                                print('  1: Drone       (无人机)')
                                print('  2: Night-Drone (夜间无人机)')
                                print('  3: Aircraft    (飞机)')
                                print('  4: Balloon     (气球)')
                                print('='*50)
                                print('操作: 按 0-4 恢复跟踪 | 空格跳过 | N 跳过视频 | ESC 取消')
                                sys.stdout.flush()

                                # 等待用户选择
                                playing = False
                                recovery_class_id = None
                                while recovery_class_id is None:
                                    key = cv2.waitKey(0) & 0xFF
                                    if key == 27:  # ESC - 取消恢复
                                        break
                                    elif key in (ord('n'), ord('N')):  # N - 跳过当前视频
                                        print(f'\n跳过当前视频: {video_name}')
                                        print(f'已保存 {saved_labels} 个标注\n')
                                        cap.release()
                                        cv2.destroyAllWindows()
                                        return (saved_labels, class_locked, current_class_id)  # 直接返回，处理下一个视频
                                    elif key == ord(' '):  # 空格 - 跳过恢复
                                        break
                                    elif ord('0') <= key <= ord('4'):  # 0-4 - 选择类别
                                        recovery_class_id = key - ord('0')
                                        class_locked = True  # 自动锁定类别
                                        print(f'\n已选择类别: {recovery_class_id} - {CLASS_NAMES_CN[recovery_class_id]} ({CLASS_NAMES[recovery_class_id]})')
                                        print('✓ 类别已自动锁定，后续自动恢复将使用此类别\n')
                                        sys.stdout.flush()

                        if recovery_class_id is not None:
                            tracker.init(frame, det_bbox)
                            current_class_id = recovery_class_id
                            last_bbox = det_bbox
                            last_score = 1.0
                            results[current_idx] = {'bbox': det_bbox, 'score': 1.0}
                            auto_recovered = True
                            print(f'  帧 {current_idx}: 置信度低，YOLO自动恢复 ({CLASS_NAMES[recovery_class_id]})')
                            track_failed = False

            if track_failed:
                status_text = f'LOW CONF({score:.3f}), Space=Continue, Q=Quit, R=Reselect'
                tracking_active = False
                playing = False

            # 如果有YOLO模型，同步检测并比较IOU（根据--yolo-skip-frames参数控制频率）
            if yolo_model is not None and tracking_active and bbox is not None:
                yolo_detect_counter += 1
                should_detect = (yolo_detect_counter % args.yolo_skip_frames == 0)

                if should_detect:
                    detections = yolo_detect(yolo_model, frame, args.yolo_conf, yolo_imgsz)

                    if len(detections) == 0:
                        # 未检测到目标，继续跟踪（YOLO可能漏检）
                        pass
                    elif len(detections) == 1:
                        # 检测到单个目标，计算IOU
                        det_bbox, det_class_id, _ = detections[0]
                        iou = calculate_iou(bbox, det_bbox)

                        if iou < args.yolo_iou:
                            # IOU过低，直接用YOLO检测结果恢复（已检测到单个目标）
                            det_bbox_clamped = clamp_bbox(det_bbox, frame.shape)

                            # 使用用户选择的类别，不使用YOLO检测的类别
                            recovery_class_id = None
                            if class_locked and current_class_id is not None:
                                # 使用锁定的类别
                                recovery_class_id = current_class_id
                            elif current_class_id is not None:
                                # 使用当前类别
                                recovery_class_id = current_class_id
                            else:
                                # 没有当前类别，跳过恢复
                                skip_frame_save = True
                                print(f'  帧 {current_idx}: IOU过低 ({iou:.3f})，但无用户选择的类别，跳过该帧')

                            if recovery_class_id is not None:
                                tracker.init(frame, det_bbox_clamped)
                                current_class_id = recovery_class_id
                                last_bbox = det_bbox_clamped
                                last_score = 1.0
                                results[current_idx] = {'bbox': det_bbox_clamped, 'score': 1.0}
                                auto_recovered = True
                                print(f'  帧 {current_idx}: IOU过低 ({iou:.3f})，YOLO框自动恢复 ({CLASS_NAMES[recovery_class_id]})')
                    else:
                        # 检测到多个目标，跳过该帧但继续跟踪
                        skip_frame_save = True
                        # 只打印一次，避免刷屏
                        if current_idx - last_multi_target_print_idx >= 30:
                            print(f'  帧 {current_idx}: 检测到 {len(detections)} 个目标，跳过该帧')
                            last_multi_target_print_idx = current_idx

        # 从结果中获取当前帧信息
        frame_result = results.get(current_idx, None)
        if frame_result is not None:
            last_bbox = frame_result.get('bbox', None)
            last_score = frame_result.get('score', None)

        # 更新状态文本
        if tracking_active:
            play_text = 'PLAYING' if playing else 'PAUSED'
            lock_status = 'LOCKED' if class_locked else 'UNLOCKED'
            if auto_recovered:
                status_text = f'{play_text} | TRACKING | {CLASS_NAMES[current_class_id]} | AUTO-RECOVERED | N=Skip | {lock_status}'
            elif skip_frame_save:
                status_text = f'{play_text} | TRACKING | {CLASS_NAMES[current_class_id]} | SKIP (Multi-target) | N=Skip | {lock_status}'
            else:
                status_text = f'{play_text} | TRACKING | {CLASS_NAMES[current_class_id]} | Space=Pause, N=Skip | {lock_status}'
        else:
            play_text = 'PLAYING' if playing else 'PAUSED'
            lock_status = 'LOCKED' if class_locked else 'UNLOCKED'
            if waiting_for_single_target:
                status_text = f'{play_text} | MULTI-TARGET | N=Skip, R=Select when single'
            else:
                if current_class_id is not None and class_locked:
                    status_text = f'{play_text} | NO TRACK | N=Skip, R=Select ROI, L=Unlock | {CLASS_NAMES[current_class_id]}'
                else:
                    status_text = f'{play_text} | NO TRACK | N=Skip, R=Select ROI, L=Lock Class'

        # 绘制并显示
        frame_to_show = draw_overlay(frame, last_bbox, last_score, current_class_id, status_text)
        if (not headless) or headless_ui_enabled:
            cv2.imshow(window_name, frame_to_show)

        # 自动保存标注
        if tracking_active and last_bbox is not None and (current_time - last_save_time >= args.label_interval) and not skip_frame_save:
            img_name = save_label_sequence(
                frame_cache, cache_start_idx, current_idx, label_seq_len, frame, frame_to_show,
                last_bbox, current_class_id, img_dir, label_dir, visualized_dir, video_name
            )
            if img_name is not None:
                saved_labels += 1
                last_save_time = current_time
                if headless:
                    print(f'  进度: [{current_idx}/{total_frames}] 保存: {img_name} (时间: {current_time:.2f}s, 类别: {CLASS_NAMES[current_class_id]})')
                else:
                    print(f'  已保存: {img_name} (时间: {current_time:.2f}s, 类别: {CLASS_NAMES[current_class_id]})')

        # 打印进度 (headless模式)
        if headless and tracking_active and current_idx % 30 == 0:
            progress = (current_idx / total_frames) * 100
            print(f'  进度: {current_idx}/{total_frames} ({progress:.1f}%) - 置信度: {last_score:.3f}')

        # 按键处理
        if headless:
            key = -1  # 无显示模式下不等待按键
            # 自动播放，不处理按键
            if not playing:
                playing = True  # 自动开始播放
        else:
            delay = 1 if playing else 0
            key = cv2.waitKeyEx(delay)

        if key == 27 or key == ord('q'):
            break
        elif key != -1:  # 有按键输入时才处理（非headless模式）
            if key in (ord('n'), ord('N')):  # N键 - 跳过当前视频
                print(f'\n跳过当前视频: {video_name}')
                print(f'已保存 {saved_labels} 个标注\n')
                break
            elif key == ord(' '):
                playing = not playing
            elif key in (ord('l'), ord('L')) or key == 257:  # L键 或 Shift+Tab - 切换类别锁定
                if current_class_id is not None:
                    if lock_permanent:
                        # 永久锁定模式，禁止解锁
                        print(f'\n⚠ 首个类别模式已启用，类别无法解锁')
                        print(f'  当前类别: {CLASS_NAMES_CN[current_class_id]} ({CLASS_NAMES[current_class_id]})')
                        print(f'  所有视频将自动使用此类别\n')
                    else:
                        class_locked = not class_locked
                        if class_locked:
                            print(f'\n✓ 类别已锁定: {CLASS_NAMES_CN[current_class_id]} ({CLASS_NAMES[current_class_id]})')
                            print('  后续YOLO自动恢复将自动使用此类别')
                            print('  再次按 L 或 Shift+Tab 解锁\n')
                        else:
                            print(f'\n✗ 类别已解锁: {CLASS_NAMES_CN[current_class_id] if current_class_id else "未设置"}')
                            print('  后续YOLO自动恢复时将询问类别\n')
                else:
                    print('\n请先选择类别后再锁定（按 R 框选目标）\n')
            elif key in (ord('r'), ord('R')):
                playing = False
                waiting_for_single_target = False
                roi = cv2.selectROI(window_name, frame, False, False)
                init_rect = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
                if init_rect[2] > 0 and init_rect[3] > 0:
                    init_rect = clamp_bbox(init_rect, frame.shape)
                    # 销毁窗口以便在终端进行输入
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    # 从命令行选择类别
                    class_id = select_class_from_console()
                    # 重新创建窗口
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.waitKey(1)
                    if class_id is not None:
                        tracker.init(frame, init_rect)
                        tracking_active = True
                        current_class_id = class_id
                        class_locked = True  # 自动锁定类别
                        last_processed_idx = current_idx
                        last_bbox = init_rect
                        last_score = 1.0
                        results[current_idx] = {'bbox': init_rect, 'score': 1.0}
                        playing = True
                        print(f'\n✓ 手动框选并锁定类别: {CLASS_NAMES_CN[class_id]} ({CLASS_NAMES[class_id]})')
                        print('  后续YOLO自动恢复将自动使用此类别\n')
            elif key in ARROW_RIGHT_KEYS or key == ord('d'):
                if not playing:
                    current_idx += 1
            elif key in ARROW_LEFT_KEYS or key == ord('a'):
                if not playing:
                    if current_idx > cache_start_idx:
                        current_idx -= 1
                    else:
                        status_text = 'Cache empty, cannot go back'
            else:
                if playing:
                    current_idx += 1
        elif key == -1:  # headless模式
            if playing:
                current_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f'完成: {video_name}, 生成 {saved_labels} 个标注')
    return (saved_labels, class_locked, current_class_id)


def main():
    """YOLO标注生成工具入口函数。"""
    args = parse_args()

    if args.max_cache <= 0:
        raise ValueError('max-cache must be > 0')
    if args.label_seq_len <= 0:
        raise ValueError('label-seq-len must be > 0')
    if args.label_seq_len > args.max_cache:
        raise ValueError('label-seq-len must be <= max-cache')

    # 检查首个类别模式的依赖条件
    if args.use_first_class:
        if not args.auto_detect:
            print('警告: --use-first-class 需要 --auto-detect 才能正常工作')
            print('  已自动启用 --auto-detect')
            args.auto_detect = True
        if not args.yolo_model:
            raise ValueError('--use-first-class 需要 --yolo-model 参数')

    # 配置模型
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    use_cuda = cfg.CUDA
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.set_num_threads(1)

    # 构建跟踪器
    tracker = build_nano_tracker(args.snapshot, device)

    # 构建YOLO检测器（如果指定了模型）
    yolo_model = None
    if args.yolo_model:
        if not YOLO_AVAILABLE:
            print('警告: 未安装ultralytics库，YOLO检测功能不可用')
            print('请运行: pip install ultralytics')
        else:
            yolo_model = build_yolo_detector(args.yolo_model, device)
            if yolo_model is not None:
                print(f'YOLO检测已启用，置信度阈值: {args.yolo_conf}, IOU阈值: {args.yolo_iou}')
                if args.auto_detect:
                    print('自动检测模式已启用')

    # 获取视频文件列表
    video_files = get_video_files(args.input)
    if not video_files:
        print('未找到视频文件')
        return

    print(f'找到 {len(video_files)} 个视频文件')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理所有视频
    total_labels = 0
    global_class_locked = False  # 全局类别锁定状态
    global_class_id = None  # 全局锁定的类别ID
    for i, video_path in enumerate(video_files, 1):
        print(f'\n[{i}/{len(video_files)}] 开始处理视频...')
        if global_class_locked:
            print(f'使用全局锁定类别: {CLASS_NAMES_CN[global_class_id]} ({CLASS_NAMES[global_class_id]})')
        is_first_video = (i == 1)  # 是否为第一个视频
        num_labels, global_class_locked, global_class_id = process_single_video(
            video_path, tracker, yolo_model, args, is_first_video, global_class_locked, global_class_id
        )
        total_labels += num_labels

    print(f'\n全部完成! 总共生成 {total_labels} 个标注')
    print(f'输出目录: {args.output_dir}')


if __name__ == '__main__':
    main()
