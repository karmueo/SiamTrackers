from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import time

import cv2
import torch

sys.path.append(os.getcwd())

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.model_load import load_pretrain


ARROW_LEFT_KEYS = (81, 2424832)  # 左方向键键值
ARROW_RIGHT_KEYS = (83, 2555904)  # 右方向键键值


def parse_args():
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description='interactive tracking for a single video')  # 参数解析器
    parser.add_argument('--config', default='./models/config/configv3.yaml', type=str, help='config file')
    parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth', type=str, help='model path')
    parser.add_argument('--video', required=True, type=str, help='input video file')
    parser.add_argument('--output-dir', default='./results/interactive_track', type=str, help='output directory')
    parser.add_argument('--output-name', default='', type=str, help='output file name (.mp4)')
    parser.add_argument('--track-conf', default=0.3, type=float, help='tracking confidence threshold')
    parser.add_argument(
        '--low-conf-action',
        default='pause',
        choices=['pause', 'exit'],
        type=str,
        help='action when score < track-conf',
    )
    parser.add_argument('--max-cache', default=300, type=int, help='max cached frames for left arrow')
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


def draw_overlay(frame, bbox, score, status_text):
    """绘制跟踪框、置信度与状态信息。

    Args:
        frame (np.ndarray): 当前帧图像。
        bbox (list|None): 边界框 [x, y, w, h]。
        score (float|None): 置信度分数。
        status_text (str): 状态提示文本。

    Returns:
        np.ndarray: 绘制后的图像。
    """
    output = frame.copy()  # 输出图像
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]  # 盒子坐标
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if score is not None:
            cv2.putText(
                output,
                'score: {:.3f}'.format(score),
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    if status_text:
        cv2.putText(
            output,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    return output


def get_output_path(output_dir, video_path, output_name):
    """生成输出视频路径。

    Args:
        output_dir (str): 输出目录。
        video_path (str): 输入视频路径。
        output_name (str): 输出文件名。

    Returns:
        str: 输出文件完整路径。
    """
    if output_name:
        filename = output_name  # 输出文件名
    else:
        base_name = os.path.splitext(os.path.basename(video_path))[0]  # 输入文件名
        filename = '{}_tracking.mp4'.format(base_name)  # 默认输出文件名
    return os.path.join(output_dir, filename)


def main():
    """交互式视频跟踪入口函数。"""
    args = parse_args()  # 参数对象

    if args.max_cache <= 0:
        raise ValueError('max-cache must be > 0')

    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    use_cuda = cfg.CUDA  # CUDA开关
    device = torch.device('cuda' if use_cuda else 'cpu')  # 运行设备
    torch.set_num_threads(1)

    tracker = build_nano_tracker(args.snapshot, device)  # 跟踪器实例

    video_path = args.video  # 视频路径
    cap = cv2.VideoCapture(video_path)  # 视频读取器
    if not cap.isOpened():
        print('无法打开视频: {}'.format(video_path))
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
    if fps <= 1e-3:
        fps = 25.0  # 默认帧率

    ret, first_frame = cap.read()  # 首帧读取结果
    if not ret:
        print('视频为空: {}'.format(video_path))
        cap.release()
        return

    output_dir = args.output_dir  # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = get_output_path(output_dir, video_path, args.output_name)  # 输出路径
    frame_size = (first_frame.shape[1], first_frame.shape[0])  # 视频尺寸
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)  # 视频写入器
    if not writer.isOpened():
        print('无法创建输出文件: {}'.format(output_path))
        cap.release()
        return

    window_name = os.path.basename(video_path)  # 窗口名称
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_cache = [first_frame]  # 帧缓存
    cache_start_idx = 0  # 缓存起始帧索引
    results = {}  # 跟踪结果缓存
    current_idx = 0  # 当前帧索引
    last_processed_idx = -1  # 已处理的最大帧索引
    last_written_idx = -1  # 已写入的最大帧索引
    tracking_active = False  # 跟踪状态
    playing = False  # 播放状态
    last_bbox = None  # 最近一次边界框
    last_score = None  # 最近一次置信度
    status_text = '暂停: 空格播放/暂停, 右键单步, 左键回退, R框选'  # 状态提示

    while True:
        if current_idx < cache_start_idx:
            current_idx = cache_start_idx  # 修正索引

        if current_idx <= cache_start_idx + len(frame_cache) - 1:
            frame = frame_cache[current_idx - cache_start_idx]  # 当前帧
        else:
            ret, frame = cap.read()  # 读取新帧结果
            if not ret:
                break
            frame_cache.append(frame)
            if len(frame_cache) > args.max_cache:
                frame_cache.pop(0)
                cache_start_idx += 1
            if current_idx < cache_start_idx:
                current_idx = cache_start_idx

        if tracking_active and current_idx == last_processed_idx + 1:
            outputs = tracker.track(frame)  # 跟踪输出
            score = outputs.get('best_score', 1.0)  # 置信度分数
            bbox = outputs.get('bbox', None)  # 预测框
            if bbox is not None:
                bbox = clamp_bbox(bbox, frame.shape)
            results[current_idx] = {'bbox': bbox, 'score': score}
            last_processed_idx = current_idx
            last_bbox = bbox
            last_score = score
            if score < args.track_conf:
                if args.low_conf_action == 'exit':
                    status_text = '置信度低于阈值，退出'
                    tracking_active = False
                    playing = False
                    frame_to_show = draw_overlay(frame, last_bbox, last_score, status_text)
                    cv2.imshow(window_name, frame_to_show)
                    cv2.waitKey(500)
                    break
                status_text = '置信度低于阈值，暂停等待重新框选'
                tracking_active = False
                playing = False

        frame_result = results.get(current_idx, None)  # 当前帧结果
        if frame_result is not None:
            last_bbox = frame_result.get('bbox', None)
            last_score = frame_result.get('score', None)

        if tracking_active:
            play_text = '播放中' if playing else '已暂停'  # 播放状态文本
            status_text = '{} | 跟踪中 | 空格暂停, R重选'.format(play_text)
        else:
            play_text = '播放中' if playing else '已暂停'  # 播放状态文本
            status_text = '{} | 未跟踪 | R框选目标'.format(play_text)
            if current_idx < last_written_idx:
                status_text = '{} | 历史帧无法重新初始化'.format(play_text)

        frame_to_show = draw_overlay(frame, last_bbox, last_score, status_text)  # 显示画面
        cv2.imshow(window_name, frame_to_show)

        if current_idx == last_written_idx + 1:
            writer.write(frame_to_show)
            last_written_idx = current_idx

        delay = 1 if playing else 0  # 等待时间
        key = cv2.waitKeyEx(delay)  # 按键值

        if key == 27:
            break
        if key == ord(' '):
            playing = not playing
        elif key in (ord('r'), ord('R')):
            if current_idx < last_written_idx:
                status_text = '当前为历史帧，无法重新初始化'
            else:
                playing = False
                roi = cv2.selectROI(window_name, frame, False, False)  # ROI选择框
                init_rect = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]  # 初始化框
                if init_rect[2] > 0 and init_rect[3] > 0:
                    init_rect = clamp_bbox(init_rect, frame.shape)
                    tracker.init(frame, init_rect)
                    tracking_active = True
                    last_processed_idx = current_idx
                    last_bbox = init_rect
                    last_score = 1.0
                    results[current_idx] = {'bbox': init_rect, 'score': 1.0}
        elif key in ARROW_RIGHT_KEYS or key == ord('d'):
            if not playing:
                current_idx += 1
        elif key in ARROW_LEFT_KEYS or key == ord('a'):
            if not playing:
                if current_idx > cache_start_idx:
                    current_idx -= 1
                else:
                    status_text = '缓存不足，无法继续回退'
        else:
            if playing:
                current_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print('结果已保存: {}'.format(output_path))


if __name__ == '__main__':
    main()
