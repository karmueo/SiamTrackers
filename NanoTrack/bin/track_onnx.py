# Copyright (c) 2026. ONNX视频目标跟踪脚本
# 功能: 使用ONNX Runtime进行NanoTrack目标跟踪
# 创建时间: 2026-03-04
# 依赖: onnxruntime, opencv-python, numpy

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.getcwd())


# 默认跟踪参数（从configv3.yaml移植）
DEFAULT_TRACK_PARAMS = {
    'WINDOW_INFLUENCE': 0.455,
    'PENALTY_K': 0.138,
    'LR': 0.348,
    'EXEMPLAR_SIZE': 127,
    'INSTANCE_SIZE': 255,
    'CONTEXT_AMOUNT': 0.5,
    'OUTPUT_SIZE': 15,
    'STRIDE': 16,
    'CLS_OUT_CHANNELS': 2,
}


def parse_args():
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象，包含backbone路径、head路径、
            视频路径、输出路径、CUDA开关和置信度阈值等。
    """
    parser = argparse.ArgumentParser(description='ONNX视频目标跟踪')
    parser.add_argument('--backbone', required=True, type=str,
                        help='backbone ONNX模型路径')
    parser.add_argument('--head', required=True, type=str,
                        help='head ONNX模型路径')
    parser.add_argument('--video', required=True, type=str,
                        help='输入视频文件路径')
    parser.add_argument('--output', default='', type=str,
                        help='输出视频文件路径（默认保存到results/onnx_track/）')
    parser.add_argument('--use-cuda', action='store_true',
                        help='使用CUDA加速推理')
    parser.add_argument('--track-conf', default=0.3, type=float,
                        help='跟踪置信度阈值')
    return parser.parse_args()


def get_subwindow_np(im, pos, model_sz, original_sz, avg_chans):
    """提取图像子窗口并进行预处理（纯numpy实现）。

    Args:
        im (np.ndarray): BGR格式输入图像。
        pos (list): 目标中心位置 [cx, cy]。
        model_sz (int): 模型输入尺寸。
        original_sz (int): 原始裁剪尺寸。
        avg_chans (np.ndarray): 通道均值用于填充。

    Returns:
        np.ndarray: 预处理后的图像块，形状为(1, 3, H, W)。
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                         int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                      int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    # 转换为(1, 3, H, W)格式，float32类型
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    return im_patch


def corner2center_np(corner):
    """将角点坐标转换为中心坐标（纯numpy实现）。

    Args:
        corner (np.ndarray): 角点坐标数组，形状为(4, N)。

    Returns:
        tuple: (cx, cy, w, h) 中心坐标和宽高。
    """
    x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
    x = (x1 + x2) * 0.5
    y = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


def draw_tracking_result(frame, bbox, fps, score):
    """绘制跟踪结果可视化信息。

    Args:
        frame (np.ndarray): 输入图像帧。
        bbox (list): 边界框 [x, y, w, h]。
        fps (float): 当前帧率。
        score (float): 置信度分数。

    Returns:
        np.ndarray: 绘制后的图像。
    """
    output = frame.copy()
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 绘制置信度
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
    # 绘制FPS
    cv2.putText(
        output,
        'FPS: {:.1f}'.format(fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return output


class ONNXTracker:
    """ONNX版本的目标跟踪器。

    使用分离的backbone和head ONNX模型进行目标跟踪推理。
    """

    def __init__(self, backbone_path, head_path, use_cuda=False, params=None):
        """初始化ONNX跟踪器。

        Args:
            backbone_path (str): backbone ONNX模型路径。
            head_path (str): head ONNX模型路径。
            use_cuda (bool): 是否使用CUDA加速，默认False。
            params (dict): 跟踪参数字典，默认使用DEFAULT_TRACK_PARAMS。
        """
        self.params = params if params is not None else DEFAULT_TRACK_PARAMS
        self.use_cuda = use_cuda

        # 创建ONNX推理会话
        self.backbone_session = self._create_session(backbone_path)
        self.head_session = self._create_session(head_path)

        # 初始化跟踪状态
        self.score_size = self.params['OUTPUT_SIZE']
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        self.cls_out_channels = self.params['CLS_OUT_CHANNELS']

        # 生成锚点
        self.points = self._generate_points(
            self.params['STRIDE'], self.score_size
        )

        # 存储模板特征
        self.zf = None
        self.center_pos = None
        self.size = None
        self.channel_average = None

    def _create_session(self, model_path):
        """创建ONNXRuntime推理会话。

        Args:
            model_path (str): ONNX模型文件路径。

        Returns:
            ort.InferenceSession: ONNX推理会话。
        """
        import onnxruntime as ort
        if self.use_cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        return session

    def _run_backbone(self, image_np):
        """运行backbone推理。

        Args:
            image_np (np.ndarray): 输入图像，形状为(1, 3, H, W)。

        Returns:
            np.ndarray: 提取的特征图。
        """
        input_name = self.backbone_session.get_inputs()[0].name
        output = self.backbone_session.run(None, {input_name: image_np})
        return output[0]

    def _run_head(self, z_feat, x_feat):
        """运行head推理。

        Args:
            z_feat (np.ndarray): 模板特征。
            x_feat (np.ndarray): 搜索区域特征。

        Returns:
            tuple: (cls, loc) 分类分数和边界框定位偏移。
        """
        input_names = [inp.name for inp in self.head_session.get_inputs()]
        feed = {input_names[0]: z_feat, input_names[1]: x_feat}
        outputs = self.head_session.run(None, feed)
        return outputs[0], outputs[1]

    def _generate_points(self, stride, size):
        """生成锚点网格坐标。

        Args:
            stride (int): 特征图步长。
            size (int): 特征图尺寸。

        Returns:
            np.ndarray: 锚点坐标数组，形状为(size*size, 2)。
        """
        ori = - (size // 2) * stride
        x, y = np.meshgrid(
            [ori + stride * dx for dx in np.arange(0, size)],
            [ori + stride * dy for dy in np.arange(0, size)]
        )
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0] = x.astype(np.float32).flatten()
        points[:, 1] = y.astype(np.float32).flatten()
        return points

    def _convert_bbox(self, delta, point):
        """将模型输出转换为边界框坐标（纯numpy实现）。

        Args:
            delta (np.ndarray): 模型输出的定位偏移。
            point (np.ndarray): 锚点坐标。

        Returns:
            np.ndarray: 转换后的边界框，形状为(4, N)。
        """
        # 转换维度: (1, 4, H, W) -> (4, N)
        delta = delta.transpose(1, 2, 3, 0).reshape(4, -1)

        # 计算角点坐标
        delta[0, :] = point[:, 0] - delta[0, :]  # x1
        delta[1, :] = point[:, 1] - delta[1, :]  # y1
        delta[2, :] = point[:, 0] + delta[2, :]  # x2
        delta[3, :] = point[:, 1] + delta[3, :]  # y2

        # 转换为中心坐标格式
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center_np(delta)
        return delta

    def _convert_score(self, score):
        """将分类输出转换为置信度分数（纯numpy实现）。

        Args:
            score (np.ndarray): 模型输出的分类分数。

        Returns:
            np.ndarray: 置信度分数数组。
        """
        if self.cls_out_channels == 1:
            # 单通道输出使用sigmoid
            score = score.transpose(1, 2, 3, 0).reshape(-1)
            score = 1.0 / (1.0 + np.exp(-score))  # sigmoid
        else:
            # 双通道输出使用softmax
            score = score.transpose(1, 2, 3, 0).reshape(self.cls_out_channels, -1)
            score = score.transpose(1, 0)
            # softmax计算
            score_max = np.max(score, axis=1, keepdims=True)
            exp_score = np.exp(score - score_max)
            score = exp_score / np.sum(exp_score, axis=1, keepdims=True)
            score = score[:, 1]
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        """裁剪边界框到图像范围内。

        Args:
            cx (float): 中心x坐标。
            cy (float): 中心y坐标。
            width (float): 边界框宽度。
            height (float): 边界框高度。
            boundary (tuple): 图像边界 (H, W)。

        Returns:
            tuple: 裁剪后的(cx, cy, width, height)。
        """
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, frame, bbox):
        """初始化跟踪器。

        Args:
            frame (np.ndarray): BGR格式图像。
            bbox (list): 初始边界框 [x, y, w, h]。
        """
        self.center_pos = np.array([
            bbox[0] + (bbox[2] - 1) / 2,
            bbox[1] + (bbox[3] - 1) / 2
        ])
        self.size = np.array([bbox[2], bbox[3]])

        # 计算模板裁剪尺寸
        w_z = self.size[0] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        h_z = self.size[1] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # 计算通道均值
        self.channel_average = np.mean(frame, axis=(0, 1))

        # 提取模板区域并推理
        z_crop = get_subwindow_np(
            frame, self.center_pos,
            self.params['EXEMPLAR_SIZE'],
            s_z, self.channel_average
        )
        self.zf = self._run_backbone(z_crop)

    def track(self, frame):
        """执行单帧跟踪。

        Args:
            frame (np.ndarray): BGR格式图像。

        Returns:
            dict: 包含'bbox'和'score'的跟踪结果。
        """
        # 计算搜索区域尺寸
        w_z = self.size[0] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        h_z = self.size[1] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.params['EXEMPLAR_SIZE'] / s_z
        s_x = s_z * (self.params['INSTANCE_SIZE'] / self.params['EXEMPLAR_SIZE'])

        # 提取搜索区域并推理
        x_crop = get_subwindow_np(
            frame, self.center_pos,
            self.params['INSTANCE_SIZE'],
            round(s_x), self.channel_average
        )
        xf = self._run_backbone(x_crop)

        # 运行head推理
        cls, loc = self._run_head(self.zf, xf)

        # 后处理
        score = self._convert_score(cls)
        pred_bbox = self._convert_bbox(loc, self.points)

        # 尺度和长宽比惩罚
        def change(r):
            return np.maximum(r, 1.0 / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :]) /
            sz(self.size[0] * scale_z, self.size[1] * scale_z)
        )
        r_c = change(
            (self.size[0] / self.size[1]) /
            (pred_bbox[2, :] / pred_bbox[3, :])
        )
        penalty = np.exp(-(r_c * s_c - 1) * self.params['PENALTY_K'])

        # 计算最终得分
        pscore = penalty * score
        pscore = pscore * (1 - self.params['WINDOW_INFLUENCE']) + \
                 self.window * self.params['WINDOW_INFLUENCE']

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z

        # 学习率
        lr = penalty[best_idx] * score[best_idx] * self.params['LR']

        # 更新位置
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # 平滑更新尺寸
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # 裁剪到图像边界
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, frame.shape[:2])

        # 更新状态
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        # 转换为[x, y, w, h]格式
        final_bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = float(score[best_idx])

        return {'bbox': final_bbox, 'score': best_score}


def get_output_path(video_path, output_arg):
    """生成输出视频路径。

    Args:
        video_path (str): 输入视频路径。
        output_arg (str): 命令行指定的输出路径（可为空）。

    Returns:
        str: 输出文件完整路径。
    """
    if output_arg:
        return output_arg
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = './results/onnx_track'
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, '{}_tracking.mp4'.format(base_name))


def clamp_bbox(bbox, frame_shape):
    """裁剪边界框到图像范围内。

    Args:
        bbox (list): 边界框 [x, y, w, h]。
        frame_shape (tuple): 图像形状 (H, W, C)。

    Returns:
        list: 裁剪后的边界框。
    """
    x, y, w, h = bbox
    x = max(0, min(x, frame_shape[1] - 1))
    y = max(0, min(y, frame_shape[0] - 1))
    w = max(1, min(w, frame_shape[1] - x))
    h = max(1, min(h, frame_shape[0] - y))
    return [x, y, w, h]


def main():
    """ONNX视频目标跟踪入口函数。

    流程：
    1. 解析命令行参数
    2. 创建ONNX跟踪器实例
    3. 打开视频文件
    4. 交互式框选初始目标
    5. 实时跟踪并显示结果
    6. 保存结果视频
    """
    args = parse_args()

    # 检查ONNX Runtime
    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print('错误: 未安装onnxruntime，请执行 pip install onnxruntime 或 onnxruntime-gpu')
        return

    # 检查模型文件
    if not os.path.exists(args.backbone):
        print('错误: backbone模型不存在: {}'.format(args.backbone))
        return
    if not os.path.exists(args.head):
        print('错误: head模型不存在: {}'.format(args.head))
        return

    # 创建跟踪器
    print('加载ONNX模型...')
    print('  Backbone: {}'.format(args.backbone))
    print('  Head: {}'.format(args.head))
    print('  CUDA: {}'.format('启用' if args.use_cuda else '禁用'))
    tracker = ONNXTracker(args.backbone, args.head, use_cuda=args.use_cuda)

    # 打开视频
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('错误: 无法打开视频: {}'.format(video_path))
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0

    ret, first_frame = cap.read()
    if not ret:
        print('错误: 视频为空: {}'.format(video_path))
        cap.release()
        return

    # 准备输出视频
    output_path = get_output_path(video_path, args.output)
    frame_size = (first_frame.shape[1], first_frame.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        print('错误: 无法创建输出文件: {}'.format(output_path))
        cap.release()
        return

    # 使用安全的窗口名称
    window_name = 'ONNX_Tracker'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 显示首帧并等待用户框选
    cv2.putText(
        first_frame,
        'Press R to select target, ESC to exit',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imshow(window_name, first_frame)

    tracking_active = False
    last_bbox = None
    last_score = None
    frame_count = 0
    total_inference_time = 0.0

    while True:
        if not tracking_active:
            key = cv2.waitKeyEx(0)
            if key == 27:  # ESC
                break
            elif key in (ord('r'), ord('R')):
                # 框选目标
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
                roi = cv2.selectROI(window_name, frame, False, False)
                if roi[2] > 0 and roi[3] > 0:
                    init_rect = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
                    init_rect = clamp_bbox(init_rect, frame.shape)
                    tracker.init(frame, init_rect)
                    tracking_active = True
                    last_bbox = init_rect
                    last_score = 1.0
                    frame_count = 0
                    total_inference_time = 0.0
                    print('目标已初始化，开始跟踪...')
        else:
            ret, frame = cap.read()
            if not ret:
                print('视频结束')
                break

            # 执行跟踪
            start_time = time.time()
            outputs = tracker.track(frame)
            inference_time = time.time() - start_time

            frame_count += 1
            total_inference_time += inference_time

            bbox = outputs.get('bbox', None)
            score = outputs.get('score', None)
            if bbox is not None:
                bbox = clamp_bbox(bbox, frame.shape)
            last_bbox = bbox
            last_score = score

            # 计算平均FPS
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0

            # 绘制结果
            frame_display = draw_tracking_result(frame, last_bbox, avg_fps, last_score)
            cv2.imshow(window_name, frame_display)

            # 写入输出视频
            writer.write(frame_display)

            # 检查置信度
            if score is not None and score < args.track_conf:
                print('警告: 置信度低于阈值 ({:.3f} < {:.3f})'.format(score, args.track_conf))

            key = cv2.waitKeyEx(1)
            if key == 27:  # ESC
                break
            elif key in (ord('r'), ord('R')):
                # 重新框选
                tracking_active = False

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 打印统计信息
    if frame_count > 0:
        avg_fps = frame_count / total_inference_time
        print('跟踪完成:')
        print('  总帧数: {}'.format(frame_count))
        print('  平均FPS: {:.1f}'.format(avg_fps))
        print('  结果保存: {}'.format(output_path))


if __name__ == '__main__':
    main()
