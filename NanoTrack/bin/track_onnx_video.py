# Copyright (c) 2026. ONNX视频目标跟踪脚本
# 功能: 使用ONNX Runtime进行NanoTrack目标跟踪，支持分离模式和合并模式
# 创建时间: 2026-03-04
# 依赖: onnxruntime, opencv-python, numpy
# 规格说明书: bin/SPEC_track_onnx_video.md

"""
ONNX视频目标跟踪脚本

本脚本提供基于ONNX Runtime的目标跟踪功能，支持两种部署模式：
1. 分离模式(separated): 使用独立的backbone和head ONNX模型
2. 合并模式(merged): 使用单个合并的ONNX模型

功能特性:
- 纯NumPy实现的图像预处理，无PyTorch依赖
- 支持CPU和CUDA推理后端
- 实时跟踪结果可视化
- JSON格式轨迹数据导出
- 交互式ROI目标框选

使用示例:
    # 分离模式
    python bin/track_onnx_video.py --mode separated \\
        --backbone models/onnx/nanotrack_backbone.onnx \\
        --head models/onnx/nanotrack_head.onnx \\
        --video test.mp4

    # 合并模式
    python bin/track_onnx_video.py --mode merged \\
        --merged models/onnx/nanotrack_merged.onnx \\
        --video test.mp4
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# 不依赖PyTorch，仅使用ONNX Runtime
# sys.path.append(os.getcwd())  # 不需要项目内部依赖


# ============================================================================
# 默认跟踪参数（从configv3.yaml移植）
# ============================================================================
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


# ============================================================================
# 辅助函数
# ============================================================================

def get_subwindow_np(im, pos, model_sz, original_sz, avg_chans):
    """提取图像子窗口并进行预处理（纯NumPy实现）。

    从输入图像中提取指定位置的子窗口，进行填充和缩放处理，
    最终输出符合模型输入要求的张量格式。

    Args:
        im (np.ndarray): BGR格式输入图像，形状为(H, W, C)。
        pos (list): 目标中心位置 [cx, cy]。
        model_sz (int): 模型输入尺寸（正方形边长）。
        original_sz (int): 原始裁剪尺寸。
        avg_chans (np.ndarray): 通道均值用于边界填充，形状为(C,)。

    Returns:
        np.ndarray: 预处理后的图像块，形状为(1, 3, H, W)，float32类型。
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
    """将角点坐标转换为中心坐标格式（纯NumPy实现）。

    将边界框从(x1, y1, x2, y2)角点格式转换为(cx, cy, w, h)中心格式。

    Args:
        corner (np.ndarray): 角点坐标数组，形状为(4, N)，格式为[x1, y1, x2, y2]。

    Returns:
        tuple: (cx, cy, w, h) 每个元素为np.ndarray，形状为(N,)。
    """
    x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
    x = (x1 + x2) * 0.5
    y = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


def draw_tracking_result(frame, bbox, fps, score):
    """绘制跟踪结果可视化信息。

    在输入帧上绘制目标边界框、置信度分数和实时FPS信息。

    Args:
        frame (np.ndarray): BGR格式输入图像帧。
        bbox (list or None): 边界框 [x, y, w, h]，为None时不绘制边界框。
        fps (float): 当前帧率。
        score (float or None): 置信度分数，为None时不绘制置信度。

    Returns:
        np.ndarray: 绘制后的图像（输入帧的副本）。
    """
    output = frame.copy()
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        # 绘制绿色边界框，线宽2px (AC-19)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 绘制置信度分数 (AC-20)
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
    # 绘制FPS (AC-21)
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


def clamp_bbox(bbox, frame_shape):
    """裁剪边界框到图像范围内。

    确保边界框坐标不超出图像边界，防止后续处理中的越界错误。

    Args:
        bbox (list): 边界框 [x, y, w, h]。
        frame_shape (tuple): 图像形状 (H, W, C) 或 (H, W)。

    Returns:
        list: 裁剪后的边界框 [x, y, w, h]。
    """
    x, y, w, h = bbox
    height, width = frame_shape[0], frame_shape[1]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return [x, y, w, h]


def export_trajectory_json(trajectory_path, video_info, tracking_params,
                           trajectory, statistics):
    """导出跟踪轨迹数据到JSON文件。

    将完整的跟踪轨迹数据序列化为JSON格式并写入文件。
    数据结构包含视频信息、跟踪参数、轨迹列表和统计摘要。

    Args:
        trajectory_path (str): 输出JSON文件路径。
        video_info (dict): 视频元数据信息。
        tracking_params (dict): 跟踪参数配置。
        trajectory (list): 轨迹数据列表，每个元素为单帧跟踪结果。
        statistics (dict): 统计摘要信息。

    Returns:
        bool: 导出成功返回True，失败返回False。
    """
    try:
        data = {
            'video_info': video_info,
            'tracking_params': tracking_params,
            'trajectory': trajectory,
            'statistics': statistics,
        }
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print('错误: JSON写入失败: {}'.format(str(e)))
        return False


# ============================================================================
# 核心跟踪器类
# ============================================================================

class ONNXTracker:
    """ONNX版本的目标跟踪器。

    支持分离模式（backbone+head）和合并模式（merged）两种部署方案。
    使用ONNX Runtime进行推理，无PyTorch依赖。

    Attributes:
        params (dict): 跟踪参数配置。
        use_cuda (bool): 是否使用CUDA加速。
        mode (str): 模型模式，'separated'或'merged'。
        backbone_session: 分离模式下的backbone ONNX会话。
        head_session: 分离模式下的head ONNX会话。
        merged_session: 合并模式下的ONNX会话。
        zf (np.ndarray): 模板特征缓存。
        center_pos (np.ndarray): 目标中心位置。
        size (np.ndarray): 目标尺寸。
        channel_average (np.ndarray): 通道均值。
    """

    def __init__(self, mode='separated', backbone_path=None, head_path=None,
                 merged_path=None, use_cuda=False, params=None):
        """初始化ONNX跟踪器。

        根据指定的模式加载相应的ONNX模型。分离模式需要backbone和head
        两个模型文件，合并模式只需要一个合并的模型文件。

        Args:
            mode (str): 模型模式，'separated'或'merged'，默认'separated'。
            backbone_path (str): 分离模式下backbone ONNX模型路径。
            head_path (str): 分离模式下head ONNX模型路径。
            merged_path (str): 合并模式下合并ONNX模型路径。
            use_cuda (bool): 是否使用CUDA加速，默认False。
            params (dict): 跟踪参数字典，默认使用DEFAULT_TRACK_PARAMS。

        Raises:
            ValueError: 模式参数无效或必需的模型路径缺失时抛出。
            RuntimeError: ONNX Runtime未安装时抛出。
        """
        self.params = params if params is not None else DEFAULT_TRACK_PARAMS.copy()
        self.use_cuda = use_cuda
        self.mode = mode

        # 验证模式参数
        if mode not in ('separated', 'merged'):
            raise ValueError('mode必须为separated或merged')

        # 检查ONNX Runtime
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            raise RuntimeError(
                '未安装onnxruntime，请执行 pip install onnxruntime 或 onnxruntime-gpu'
            )

        # 根据模式创建推理会话
        if mode == 'separated':
            if not backbone_path or not head_path:
                raise ValueError('分离模式需要指定backbone_path和head_path')
            self.backbone_session = self._create_session(backbone_path)
            self.head_session = self._create_session(head_path)
            self.merged_session = None
        else:  # merged
            if not merged_path:
                raise ValueError('合并模式需要指定merged_path')
            self.backbone_session = None
            self.head_session = None
            self.merged_session = self._create_session(merged_path)

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

        根据CUDA可用性选择合适的执行提供程序。如果指定使用CUDA但不可用，
        会回退到CPU并打印警告信息。

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

        # 检查CUDA是否成功启用 (AC-05)
        if self.use_cuda:
            actual_providers = session.get_providers()
            if 'CUDAExecutionProvider' not in actual_providers:
                print('警告: CUDA不可用，回退到CPU推理')

        return session

    def _run_backbone(self, image_np):
        """运行backbone推理（分离模式）。

        将输入图像通过backbone网络提取特征。

        Args:
            image_np (np.ndarray): 输入图像，形状为(1, 3, H, W)。

        Returns:
            np.ndarray: 提取的特征图。
        """
        input_name = self.backbone_session.get_inputs()[0].name
        output = self.backbone_session.run(None, {input_name: image_np})
        return output[0]

    def _run_head(self, z_feat, x_feat):
        """运行head推理（分离模式）。

        将模板特征和搜索区域特征通过head网络进行分类和回归。

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

    def _run_merged(self, z_crop, x_crop):
        """运行合并模型推理（合并模式）。

        将模板图像和搜索图像同时输入合并模型，直接输出分类和回归结果。

        Args:
            z_crop (np.ndarray): 模板图像，形状为(1, 3, H, W)。
            x_crop (np.ndarray): 搜索图像，形状为(1, 3, H, W)。

        Returns:
            tuple: (cls, loc) 分类分数和边界框定位偏移。
        """
        input_names = [inp.name for inp in self.merged_session.get_inputs()]
        feed = {input_names[0]: z_crop, input_names[1]: x_crop}
        outputs = self.merged_session.run(None, feed)
        return outputs[0], outputs[1]

    def _generate_points(self, stride, size):
        """生成锚点网格坐标。

        在特征图上生成均匀分布的锚点坐标，用于后续的边界框解码。

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
        """将模型输出转换为边界框坐标（纯NumPy实现）。

        将head网络输出的定位偏移解码为实际边界框坐标。

        Args:
            delta (np.ndarray): 模型输出的定位偏移，形状为(1, 4, H, W)。
            point (np.ndarray): 锚点坐标，形状为(N, 2)。

        Returns:
            np.ndarray: 转换后的边界框，形状为(4, N)，格式为[cx, cy, w, h]。
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
        """将分类输出转换为置信度分数（纯NumPy实现）。

        根据输出通道数选择sigmoid或softmax进行分数转换。

        Args:
            score (np.ndarray): 模型输出的分类分数。

        Returns:
            np.ndarray: 置信度分数数组，形状为(N,)。
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

        确保目标位置和尺寸不超出图像边界。

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

        提取模板区域特征并存储，为后续跟踪做准备。

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

        # 提取模板区域
        self.z_crop = get_subwindow_np(
            frame, self.center_pos,
            self.params['EXEMPLAR_SIZE'],
            s_z, self.channel_average
        )

        # 根据模式进行推理
        if self.mode == 'separated':
            self.zf = self._run_backbone(self.z_crop)
        # 合并模式下不需要缓存特征，每帧同时输入z和x

    def track(self, frame):
        """执行单帧跟踪。

        对输入帧执行完整的跟踪推理流程，包括搜索区域提取、
        模型推理、后处理和状态更新。

        Args:
            frame (np.ndarray): BGR格式图像。

        Returns:
            dict: 包含以下字段的跟踪结果：
                - 'bbox' (list): 边界框 [x, y, w, h]
                - 'score' (float): 置信度分数
        """
        # 计算搜索区域尺寸
        w_z = self.size[0] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        h_z = self.size[1] + self.params['CONTEXT_AMOUNT'] * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.params['EXEMPLAR_SIZE'] / s_z
        s_x = s_z * (self.params['INSTANCE_SIZE'] / self.params['EXEMPLAR_SIZE'])

        # 提取搜索区域
        x_crop = get_subwindow_np(
            frame, self.center_pos,
            self.params['INSTANCE_SIZE'],
            round(s_x), self.channel_average
        )

        # 根据模式进行推理
        if self.mode == 'separated':
            # 分离模式: 分别运行backbone和head
            xf = self._run_backbone(x_crop)
            cls, loc = self._run_head(self.zf, xf)
        else:
            # 合并模式: 一次推理
            cls, loc = self._run_merged(self.z_crop, x_crop)

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


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    """解析命令行参数。

    支持的参数包括模型模式选择、模型路径、视频路径、输出配置、
    CUDA加速和置信度阈值等。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        description='ONNX视频目标跟踪 - 支持分离模式和合并模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 分离模式
  python %(prog)s --mode separated --backbone backbone.onnx --head head.onnx --video test.mp4

  # 合并模式
  python %(prog)s --mode merged --merged nanotrack_merged.onnx --video test.mp4

  # 使用CUDA加速
  python %(prog)s --mode merged --merged model.onnx --video test.mp4 --use-cuda
        '''
    )

    # 模式选择
    parser.add_argument('--mode', type=str, required=True,
                        choices=['separated', 'merged'],
                        help='模型模式: separated(分离) 或 merged(合并)')

    # 分离模式参数
    parser.add_argument('--backbone', type=str, default='',
                        help='分离模式: backbone ONNX模型路径')
    parser.add_argument('--head', type=str, default='',
                        help='分离模式: head ONNX模型路径')

    # 合并模式参数
    parser.add_argument('--merged', type=str, default='',
                        help='合并模式: 合并ONNX模型路径')

    # 视频参数
    parser.add_argument('--video', type=str, required=True,
                        help='输入视频文件路径')

    # 输出参数
    parser.add_argument('--output', type=str, default='',
                        help='输出视频文件路径（默认: ./results/onnx_track/{name}_tracking.mp4）')
    parser.add_argument('--trajectory', type=str, default='',
                        help='输出轨迹JSON文件路径（默认: 与输出视频同名的.json文件）')

    # 推理参数
    parser.add_argument('--use-cuda', action='store_true',
                        help='使用CUDA加速推理')
    parser.add_argument('--track-conf', type=float, default=0.3,
                        help='跟踪置信度告警阈值（默认: 0.3）')

    return parser.parse_args()


def validate_args(args):
    """验证命令行参数的有效性。

    检查模式与模型路径的组合约束，以及文件存在性。

    Args:
        args (argparse.Namespace): 解析后的参数对象。

    Returns:
        tuple: (is_valid, error_message)
            is_valid为True表示验证通过，error_message包含错误信息。
    """
    # 验证模式与模型路径的组合
    if args.mode == 'separated':
        if not args.backbone:
            return False, '分离模式需要指定 --backbone 参数'
        if not args.head:
            return False, '分离模式需要指定 --head 参数'
        if not os.path.exists(args.backbone):
            return False, '错误: backbone模型不存在: {}'.format(args.backbone)
        if not os.path.exists(args.head):
            return False, '错误: head模型不存在: {}'.format(args.head)
    else:  # merged
        if not args.merged:
            return False, '合并模式需要指定 --merged 参数'
        if not os.path.exists(args.merged):
            return False, '错误: merged模型不存在: {}'.format(args.merged)

    # 验证视频文件
    if not os.path.exists(args.video):
        return False, '错误: 视频文件不存在: {}'.format(args.video)

    return True, ''


def get_output_paths(video_path, output_arg, trajectory_arg):
    """生成输出文件路径。

    根据输入视频路径和命令行参数生成默认的输出视频和轨迹文件路径。

    Args:
        video_path (str): 输入视频文件路径。
        output_arg (str): 命令行指定的输出视频路径（可为空）。
        trajectory_arg (str): 命令行指定的轨迹文件路径（可为空）。

    Returns:
        tuple: (output_video_path, trajectory_path)
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = './results/onnx_track'
    os.makedirs(output_dir, exist_ok=True)

    if output_arg:
        output_path = output_arg
    else:
        output_path = os.path.join(output_dir, '{}_tracking.mp4'.format(base_name))

    if trajectory_arg:
        trajectory_path = trajectory_arg
    else:
        trajectory_path = os.path.join(output_dir, '{}_trajectory.json'.format(base_name))

    return output_path, trajectory_path


# ============================================================================
# 主函数
# ============================================================================

def main():
    """ONNX视频目标跟踪入口函数。

    完整流程：
    1. 解析和验证命令行参数
    2. 创建ONNX跟踪器实例
    3. 打开视频文件并提取元数据
    4. 交互式框选初始目标
    5. 实时跟踪并显示结果
    6. 保存结果视频和轨迹JSON

    Returns:
        int: 退出码，0表示成功，1表示运行时错误，2表示参数错误。
    """
    args = parse_args()

    # 验证参数 (AC-28)
    is_valid, error_msg = validate_args(args)
    if not is_valid:
        print(error_msg)
        return 2 if '需要指定' in error_msg else 1

    # 检查ONNX Runtime
    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print('错误: 未安装onnxruntime，请执行 pip install onnxruntime 或 onnxruntime-gpu')
        return 1

    # 创建跟踪器
    print('加载ONNX模型...')
    print('  模式: {}'.format(args.mode))
    if args.mode == 'separated':
        print('  Backbone: {}'.format(args.backbone))
        print('  Head: {}'.format(args.head))
        tracker = ONNXTracker(
            mode='separated',
            backbone_path=args.backbone,
            head_path=args.head,
            use_cuda=args.use_cuda
        )
    else:
        print('  Merged: {}'.format(args.merged))
        tracker = ONNXTracker(
            mode='merged',
            merged_path=args.merged,
            use_cuda=args.use_cuda
        )
    print('  CUDA: {}'.format('启用' if args.use_cuda else '禁用'))

    # 打开视频 (AC-06)
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('错误: 无法打开视频: {}'.format(video_path))
        return 1

    # 获取视频元数据
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('视频信息:')
    print('  文件: {}'.format(video_path))
    print('  分辨率: {}x{}'.format(width, height))
    print('  帧率: {:.2f}'.format(fps))
    print('  总帧数: {}'.format(frame_count_total))

    # 读取首帧 (AC-07)
    ret, first_frame = cap.read()
    if not ret:
        print('错误: 视频为空或无法读取首帧: {}'.format(video_path))
        cap.release()
        return 1

    # 准备输出路径
    output_path, trajectory_path = get_output_paths(
        video_path, args.output, args.trajectory
    )

    # 创建视频写入器 (AC-22)
    frame_size = (first_frame.shape[1], first_frame.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        print('错误: 无法创建输出文件: {}'.format(output_path))
        cap.release()
        return 1

    # 视频信息字典（用于JSON导出）
    video_info = {
        'source': os.path.basename(video_path),
        'fps': float(fps),
        'resolution': {'width': width, 'height': height},
        'total_frames': frame_count_total
    }

    # 跟踪参数字典（用于JSON导出）
    tracking_params = {
        'mode': args.mode,
        'window_influence': DEFAULT_TRACK_PARAMS['WINDOW_INFLUENCE'],
        'penalty_k': DEFAULT_TRACK_PARAMS['PENALTY_K'],
        'lr': DEFAULT_TRACK_PARAMS['LR']
    }

    # 使用安全的窗口名称
    window_name = 'ONNX_Tracker_Video'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 显示首帧提示 (AC-09)
    prompt_frame = first_frame.copy()
    cv2.putText(
        prompt_frame,
        'Press R to select target, ESC to exit',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imshow(window_name, prompt_frame)

    # 跟踪状态
    tracking_active = False
    last_bbox = None
    last_score = None
    frame_count = 0
    total_inference_time = 0.0

    # 轨迹数据列表
    trajectory_data = []
    all_scores = []

    while True:
        if not tracking_active:
            # 等待用户操作 (AC-09)
            key = cv2.waitKeyEx(0)
            if key == 27:  # ESC
                print('用户取消')
                break
            elif key in (ord('r'), ord('R')):
                # ROI框选 (AC-10)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
                roi = cv2.selectROI(window_name, frame, False, False)

                # 验证ROI有效性 (AC-11, AC-12)
                if roi[2] > 0 and roi[3] > 0:
                    init_rect = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
                    init_rect = clamp_bbox(init_rect, frame.shape)

                    # 初始化跟踪器 (AC-15)
                    tracker.init(frame, init_rect)
                    tracking_active = True
                    last_bbox = init_rect
                    last_score = 1.0
                    frame_count = 0
                    total_inference_time = 0.0
                    trajectory_data = []
                    all_scores = []
                    print('目标已初始化，开始跟踪...')
                else:
                    # 无效框选，返回等待状态 (AC-12)
                    print('框选无效，请重新框选')
                    cv2.imshow(window_name, prompt_frame)
        else:
            # 读取下一帧
            ret, frame = cap.read()
            if not ret:
                print('视频结束')  # (AC-08)
                break

            # 执行跟踪 (AC-16)
            start_time = time.time()
            outputs = tracker.track(frame)
            inference_time = time.time() - start_time
            inference_time_ms = inference_time * 1000

            frame_count += 1
            total_inference_time += inference_time

            bbox = outputs.get('bbox', None)
            score = outputs.get('score', None)

            if bbox is not None:
                bbox = clamp_bbox(bbox, frame.shape)
            last_bbox = bbox
            last_score = score

            # 记录轨迹数据 (AC-24)
            if bbox is not None:
                timestamp_ms = (frame_count - 1) * 1000.0 / fps
                trajectory_data.append({
                    'frame_id': frame_count - 1,
                    'timestamp_ms': round(timestamp_ms, 2),
                    'bbox': [round(x, 2) for x in bbox],
                    'score': round(score, 5) if score else None,
                    'inference_time_ms': round(inference_time_ms, 2)
                })
                if score is not None:
                    all_scores.append(score)

            # 计算平均FPS
            avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0

            # 绘制结果 (AC-19, AC-20, AC-21)
            frame_display = draw_tracking_result(frame, last_bbox, avg_fps, last_score)
            cv2.imshow(window_name, frame_display)

            # 写入输出视频 (AC-22)
            writer.write(frame_display)

            # 检查置信度 (AC-18)
            if score is not None and score < args.track_conf:
                print('警告: 置信度低于阈值 ({:.3f} < {:.3f})'.format(score, args.track_conf))

            # 键盘事件处理 (AC-13, AC-14)
            key = cv2.waitKeyEx(1)
            if key == 27:  # ESC
                print('用户中断')
                break
            elif key in (ord('r'), ord('R')):
                # 重新框选
                tracking_active = False
                cv2.imshow(window_name, prompt_frame)

    # 释放资源
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 计算统计信息 (AC-24)
    if frame_count > 0:
        avg_fps = frame_count / total_inference_time
        statistics = {
            'total_inference_time_ms': round(total_inference_time * 1000, 2),
            'average_fps': round(avg_fps, 1),
            'average_score': round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
            'min_score': round(min(all_scores), 3) if all_scores else 0,
            'max_score': round(max(all_scores), 3) if all_scores else 0
        }

        # 导出轨迹JSON (AC-23)
        json_success = export_trajectory_json(
            trajectory_path, video_info, tracking_params, trajectory_data, statistics
        )

        # 打印统计信息
        print('跟踪完成:')
        print('  总帧数: {}'.format(frame_count))
        print('  平均FPS: {:.1f}'.format(avg_fps))
        print('  平均置信度: {:.3f}'.format(statistics['average_score']))
        print('  结果视频: {}'.format(output_path))
        if json_success:
            print('  轨迹文件: {}'.format(trajectory_path))
        else:
            print('  轨迹文件: 导出失败')  # (AC-25)
    else:
        print('未进行跟踪')

    return 0


if __name__ == '__main__':
    sys.exit(main())
