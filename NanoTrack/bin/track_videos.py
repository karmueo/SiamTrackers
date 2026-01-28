from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import sys
import time
import multiprocessing

import cv2
import numpy as np
import torch

sys.path.append(os.getcwd())

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.model_load import load_pretrain


VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg'}


def is_video_file(path):
    """判断文件是否为视频文件。

    Args:
        path: 文件路径

    Returns:
        bool: 如果是视频文件返回True，否则返回False
    """
    _, ext = os.path.splitext(path)
    return ext.lower() in VIDEO_EXTS


def list_videos(root_dir):
    """递归遍历目录，返回所有视频文件。

    Args:
        root_dir: 根目录路径

    Yields:
        str: 视频文件的完整路径
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            full_path = os.path.join(dirpath, name)
            if is_video_file(full_path):
                yield full_path


def parse_init_rect(rect_str):
    """解析初始化边界框字符串。

    Args:
        rect_str: 边界框字符串，格式为 "x,y,w,h"

    Returns:
        list|None: 解析后的边界框 [x, y, w, h]，如果输入为空则返回None

    Raises:
        ValueError: 如果格式不正确
    """
    if not rect_str:
        return None
    parts = rect_str.split(',')
    if len(parts) != 4:
        raise ValueError('init_rect must be "x,y,w,h"')
    return [int(float(p)) for p in parts]


def clamp_bbox(bbox, frame_shape):
    """裁剪边界框到图像范围内。

    Args:
        bbox: 边界框 [x, y, w, h]
        frame_shape: 图像形状 (H, W, C)

    Returns:
        list: 裁剪后的边界框 [x, y, w, h]
    """
    x, y, w, h = bbox
    x = max(0, min(x, frame_shape[1] - 1))
    y = max(0, min(y, frame_shape[0] - 1))
    w = max(1, min(w, frame_shape[1] - x))
    h = max(1, min(h, frame_shape[0] - y))
    return [x, y, w, h]


def crop_and_resize(frame, bbox, roi_size):
    """裁剪边界框区域并调整到指定尺寸。

    Args:
        frame: 输入图像
        bbox: 边界框 [x, y, w, h]
        roi_size: 输出ROI尺寸（正方形）

    Returns:
        np.ndarray: 裁剪并调整大小后的图像
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    bw = max(1.0, float(bw))
    bh = max(1.0, float(bh))
    cx = float(x) + bw / 2.0
    cy = float(y) + bh / 2.0
    side = max(bw, bh, float(roi_size))
    side_int = max(1, int(round(side)))
    x1 = int(round(cx - side_int / 2.0))
    y1 = int(round(cy - side_int / 2.0))
    x2 = x1 + side_int
    y2 = y1 + side_int

    out = np.zeros((side_int, side_int, 3), dtype=frame.dtype)
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    return cv2.resize(out, (roi_size, roi_size))


def get_output_base(output_root, video_path, video_root):
    """获取输出基础目录，保持与输入视频相同的相对路径结构。

    Args:
        output_root: 输出根目录
        video_path: 视频文件完整路径
        video_root: 视频根目录

    Returns:
        str: 输出基础目录路径
    """
    rel_path = os.path.relpath(video_path, video_root)
    rel_dir = os.path.dirname(rel_path)
    base_dir = os.path.join(output_root, rel_dir)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_chunk_dir(base_dir, video_path, chunk_idx):
    """获取视频分块目录路径。

    Args:
        base_dir: 基础输出目录
        video_path: 视频文件路径
        chunk_idx: 分块索引

    Returns:
        str: 分块目录路径
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(base_dir, '{}_{}'.format(video_name, chunk_idx))


def ensure_dir(path):
    """确保目录存在，如不存在则创建。

    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def writer_loop(write_queue):
    """图像写入线程循环函数。

    Args:
        write_queue: 写入队列，包含 (path, image) 元组的批次
    """
    while True:
        batch = write_queue.get()
        if batch is None:
            break
        for path, img in batch:
            cv2.imwrite(path, img)


def select_yolo_init(yolo, frame, conf_thres):
    """使用YOLO检测并选择首个满足置信度要求的目标作为初始化框。

    Args:
        yolo: YOLO模型实例
        frame: 输入图像
        conf_thres: 置信度阈值

    Returns:
        list|None: 检测到的边界框 [x, y, w, h]，未检测到则返回None
    """
    results = yolo.predict(frame, conf=conf_thres, verbose=False)
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    for i in range(len(confs)):
        if confs[i] >= conf_thres:
            x1, y1, x2, y2 = xyxy[i]
            return [x1, y1, x2 - x1, y2 - y1]
    return None


def write_chunk_json(chunk_dir, resolution, tracks):
    """将分块跟踪结果写入JSON文件。

    Args:
        chunk_dir: 分块目录路径
        resolution: 视频分辨率 (width, height)
        tracks: 跟踪结果列表，每个元素包含 image 和 bbox 信息
    """
    json_path = '{}.json'.format(chunk_dir)
    payload = {
        'resolution': {
            'width': int(resolution[0]),
            'height': int(resolution[1]),
        },
        'tracks': tracks,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def safe_print(message, print_lock):
    """线程安全的打印函数。

    Args:
        message: 要打印的消息
        print_lock: 打印锁
    """
    with print_lock:
        print(message)

def print_progress(current, total, video_path, print_lock):
    """打印进度条和当前处理视频信息。

    Args:
        current: 当前处理数量
        total: 总数量
        video_path: 当前视频路径
        print_lock: 打印锁
    """
    bar_len = 30
    filled = int(round(bar_len * float(current) / float(total)))
    bar = '=' * filled + '-' * (bar_len - filled)
    msg = '[{}/{}] [{}] {}'.format(current, total, bar, os.path.basename(video_path))
    with print_lock:
        print(msg)


def load_tracker(snapshot_path, local_device):
    """加载NanoTrack跟踪器模型。

    Args:
        snapshot_path: 模型权重文件路径
        local_device: 运行设备

    Returns:
        跟踪器实例
    """
    if local_device.type == 'cuda':
        torch.cuda.set_device(local_device.index)
    local_model = ModelBuilder()
    local_model = load_pretrain(local_model, snapshot_path).to(local_device).eval()
    return build_tracker(local_model)


def process_video(video_path, args_dict, tracker, yolo_model, write_queue, print_lock):
    """处理单个视频文件，进行目标跟踪并保存结果。

    Args:
        video_path: 视频文件路径
        args_dict: 参数字典
        tracker: NanoTrack跟踪器实例
        yolo_model: YOLO检测器实例（可选）
        write_queue: 异步写入队列
        print_lock: 打印锁
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        safe_print('Failed to open video: {}'.format(video_path), print_lock)
        return

    base_dir = get_output_base(args_dict['output_root'], video_path, args_dict['video_root'])
    chunk_idx = 0
    frame_in_chunk = 0
    tracking = False
    started = False
    fps_start = None
    fps_frames = 0
    fps_last_print = None
    infer_time_sum = 0.0
    infer_frames = 0
    window_name = None
    track_frame_idx = 0
    save_batch = []
    chunk_tracks = []
    resolution = None

    init_rect = parse_init_rect(args_dict['init_rect'])

    safe_print('Processing: {}'.format(video_path), print_lock)

    if args_dict['show']:
        window_name = os.path.basename(video_path)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not tracking:
            current_init = None
            if yolo_model is not None:
                current_init = select_yolo_init(yolo_model, frame, args_dict['yolo_conf'])
            elif init_rect is not None:
                current_init = init_rect

            if current_init is None:
                if args_dict['show']:
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)
                continue

            current_init = clamp_bbox(current_init, frame.shape)
            tracker.init(frame, current_init)
            tracking = True
            started = True
            fps_start = time.time()
            fps_last_print = fps_start
            fps_frames = 0
            infer_time_sum = 0.0
            infer_frames = 0
            track_frame_idx = 0

            if track_frame_idx % args_dict['save_step'] == 0:
                roi = crop_and_resize(frame, current_init, args_dict['roi_size'])
                chunk_dir = get_chunk_dir(base_dir, video_path, chunk_idx)
                ensure_dir(chunk_dir)
                image_name = '{:08d}.jpg'.format(frame_in_chunk)
                save_batch.append((os.path.join(chunk_dir, image_name), roi))
                if resolution is None:
                    resolution = (frame.shape[1], frame.shape[0])
                bbox_int = clamp_bbox(current_init, frame.shape)
                chunk_tracks.append({
                    'image': image_name,
                    'bbox': {
                        'x': int(bbox_int[0]),
                        'y': int(bbox_int[1]),
                        'w': int(bbox_int[2]),
                        'h': int(bbox_int[3]),
                    }
                })
                if len(save_batch) >= args_dict['write_batch_size']:
                    write_queue.put(save_batch[:])
                    save_batch = []
                frame_in_chunk += 1
                if frame_in_chunk >= args_dict['frames_per_folder']:
                    write_chunk_json(chunk_dir, resolution, chunk_tracks)
                    chunk_tracks = []
                    chunk_idx += 1
                    frame_in_chunk = 0
            track_frame_idx += 1

            if args_dict['show']:
                x, y, w, h = [int(v) for v in current_init]
                disp = frame.copy()
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow(window_name, disp)
                cv2.waitKey(1)
            continue

        infer_start = time.time()
        outputs = tracker.track(frame)
        infer_time_sum += time.time() - infer_start
        infer_frames += 1
        score = outputs.get('best_score', 1.0)
        if score < args_dict['track_conf']:
            safe_print('Tracking stopped (score {:.4f} < {:.4f})'.format(
                score, args_dict['track_conf']), print_lock)
            break

        bbox = outputs['bbox']
        if track_frame_idx % args_dict['save_step'] == 0:
            roi = crop_and_resize(frame, bbox, args_dict['roi_size'])
            chunk_dir = get_chunk_dir(base_dir, video_path, chunk_idx)
            ensure_dir(chunk_dir)
            image_name = '{:08d}.jpg'.format(frame_in_chunk)
            save_batch.append((os.path.join(chunk_dir, image_name), roi))
            if resolution is None:
                resolution = (frame.shape[1], frame.shape[0])
            bbox_int = clamp_bbox(bbox, frame.shape)
            chunk_tracks.append({
                'image': image_name,
                'bbox': {
                    'x': int(bbox_int[0]),
                    'y': int(bbox_int[1]),
                    'w': int(bbox_int[2]),
                    'h': int(bbox_int[3]),
                }
            })
            if len(save_batch) >= args_dict['write_batch_size']:
                write_queue.put(save_batch[:])
                save_batch = []
            frame_in_chunk += 1
            if frame_in_chunk >= args_dict['frames_per_folder']:
                write_chunk_json(chunk_dir, resolution, chunk_tracks)
                chunk_tracks = []
                chunk_idx += 1
                frame_in_chunk = 0
        track_frame_idx += 1

        fps_frames += 1
        if args_dict['show_fps'] and fps_start is not None:
            now = time.time()
            if now - fps_last_print >= 3.0:
                avg_fps = fps_frames / max(1e-6, now - fps_start)
                infer_fps = infer_frames / max(1e-6, infer_time_sum)
                safe_print('Avg FPS: {:.2f}, Infer FPS: {:.2f} ({})'.format(
                    avg_fps, infer_fps, os.path.basename(video_path)), print_lock)
                fps_last_print = now

        if args_dict['show']:
            x, y, w, h = [int(v) for v in bbox]
            disp = frame.copy()
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(window_name, disp)
            cv2.waitKey(1)

    cap.release()
    if args_dict['show'] and window_name is not None:
        cv2.destroyWindow(window_name)

    if save_batch:
        write_queue.put(save_batch[:])
        save_batch = []

    if chunk_tracks and resolution is not None:
        chunk_dir = get_chunk_dir(base_dir, video_path, chunk_idx)
        write_chunk_json(chunk_dir, resolution, chunk_tracks)

    if not started:
        safe_print('No init found for video: {}'.format(video_path), print_lock)


def worker_loop(worker_id, args_dict, use_cuda, gpu_count, video_queue, write_queue, print_lock,
                progress_lock, progress_started, total_videos):
    """工作进程循环函数，从队列中获取视频任务并处理。

    Args:
        worker_id: 工作进程ID
        args_dict: 参数字典
        use_cuda: 是否使用CUDA
        gpu_count: GPU数量
        video_queue: 视频任务队列
        write_queue: 异步写入队列
        print_lock: 打印锁
        progress_lock: 进度锁
        progress_started: 已开始处理数量（共享变量）
        total_videos: 视频总数量
    """
    cfg.merge_from_file(args_dict['config'])
    cfg.CUDA = use_cuda
    torch.set_num_threads(1)

    if use_cuda and gpu_count > 0:
        local_device = torch.device('cuda:{}'.format(worker_id % gpu_count))
    else:
        local_device = torch.device('cpu')

    tracker = load_tracker(args_dict['snapshot'], local_device)
    yolo_model = None
    if args_dict['use_yolo']:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError('ultralytics is required when --use-yolo is set')
        yolo_model = YOLO(args_dict['yolo_weights'])
        try:
            yolo_device = local_device if use_cuda else torch.device('cpu')
            yolo_model.to(yolo_device)
            safe_print('YOLO device: {}'.format(yolo_device), print_lock)
        except Exception as exc:
            safe_print('YOLO device set failed, fallback to default: {}'.format(exc), print_lock)
    if use_cuda:
        safe_print('Worker {} uses device {}'.format(worker_id, local_device), print_lock)

    while True:
        video_path = video_queue.get()
        if video_path is None:
            break
        with progress_lock:
            progress_started.value += 1
            print_progress(progress_started.value, total_videos, video_path, print_lock)
        process_video(video_path, args_dict, tracker, yolo_model, write_queue, print_lock)


def main():
    """主函数：批量处理视频，使用NanoTrack进行目标跟踪。

    支持功能：
    - 递归扫描指定目录下的所有视频文件
    - 使用YOLO辅助初始化跟踪目标
    - 多进程并行处理
    - 异步图像写入
    - 导出跟踪ROI和标注信息
    """
    parser = argparse.ArgumentParser(description='batch tracking for videos')
    parser.add_argument('--config', default='./models/config/configv3.yaml', type=str, help='config file')
    parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth', type=str, help='model name')
    parser.add_argument('--video-root', required=True, type=str, help='root directory of videos')
    parser.add_argument('--output-root', default='./results/track_rois', type=str, help='output directory')
    parser.add_argument('--use-yolo', action='store_true', help='use ultralytics YOLO for init')
    parser.add_argument('--yolo-weights', default='yolov11n.pt', type=str, help='YOLO weights path')
    parser.add_argument('--yolo-conf', default=0.5, type=float, help='YOLO confidence threshold')
    parser.add_argument('--track-conf', default=0.3, type=float, help='tracking confidence threshold')
    parser.add_argument('--roi-size', default=64, type=int, help='output ROI size')
    parser.add_argument('--frames-per-folder', default=200, type=int, help='frames saved per folder')
    parser.add_argument('--init-rect', default='', type=str, help='manual init rect "x,y,w,h"')
    parser.add_argument('--show', action='store_true', help='show video and tracking results')
    parser.add_argument('--show-fps', action='store_true', help='print average fps every 3 seconds')
    parser.add_argument('--save-step', default=1, type=int, help='save every N frames')
    parser.add_argument('--write-batch-size', default=20, type=int, help='batch size for async writes')
    parser.add_argument('--write-queue-size', default=256, type=int, help='max queue size for async writes')
    parser.add_argument('--num-workers', default=1, type=int, help='number of worker processes')
    args = parser.parse_args()

    if args.frames_per_folder <= 0:
        raise ValueError('frames-per-folder must be > 0')
    if args.roi_size <= 0:
        raise ValueError('roi-size must be > 0')
    if args.save_step <= 0:
        raise ValueError('save-step must be > 0')
    if args.write_batch_size <= 0:
        raise ValueError('write-batch-size must be > 0')
    if args.write_queue_size <= 0:
        raise ValueError('write-queue-size must be > 0')
    if args.num_workers <= 0:
        raise ValueError('num-workers must be > 0')

    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    use_cuda = cfg.CUDA
    if use_cuda:
        gpu_count = torch.cuda.device_count()
    else:
        gpu_count = 0

    torch.set_num_threads(1)

    if use_cuda:
        print('NanoTrack device: cuda ({} GPUs)'.format(gpu_count))
    else:
        print('NanoTrack device: cpu')

    if args.show and args.num_workers > 1:
        print('Warning: 多进程不支持显示，已自动关闭 --show')
        args.show = False

    mp_context = multiprocessing.get_context('spawn')
    write_queue = mp_context.Queue(maxsize=args.write_queue_size)
    writer_proc = mp_context.Process(target=writer_loop, args=(write_queue,))
    writer_proc.daemon = True
    writer_proc.start()

    video_list = list(list_videos(args.video_root))
    total_videos = len(video_list)
    if not video_list:
        print('No videos found under {}'.format(args.video_root))
        write_queue.put(None)
        writer_proc.join()
        return

    print_lock = mp_context.Lock()
    progress_lock = mp_context.Lock()
    progress_started = mp_context.Value('i', 0)

    video_queue = mp_context.Queue()
    for video_path in video_list:
        video_queue.put(video_path)
    for _ in range(args.num_workers):
        video_queue.put(None)

    args_dict = vars(args)

    workers = []
    for i in range(args.num_workers):
        p = mp_context.Process(
            target=worker_loop,
            args=(i, args_dict, use_cuda, gpu_count, video_queue, write_queue, print_lock,
                  progress_lock, progress_started, total_videos),
        )
        p.daemon = True
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    write_queue.put(None)
    writer_proc.join()


if __name__ == '__main__':
    main()
