from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

import sys 
sys.path.append(os.getcwd())  

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.model_load import load_pretrain
from nanotrack.utils.bbox import corner2center

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo') 

parser.add_argument('--config', default='./models/config/configv3.yaml', type=str, help='config file')

parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth', type=str, help='model name')

parser.add_argument('--video_name', default='./bin/girl_dance.mp4', type=str, help='videos or image files')

parser.add_argument('--save', action='store_true', help='whether visualzie result') 

parser.add_argument('--manual_roi', action='store_true', help='select ROI on first frame manually')

parser.add_argument('--use_onnx', action='store_true', help='use ONNXRuntime for inference (requires exported backbone/head)')
parser.add_argument('--onnx_backbone', default='./models/onnx/nanotrack_backbone.onnx', type=str, help='onnx file for backbone (template)')
parser.add_argument('--onnx_backbone_search', default='', type=str, help='onnx file for backbone (search); defaults to onnx_backbone when empty')
parser.add_argument('--onnx_head', default='./models/onnx/nanotrack_head.onnx', type=str, help='onnx file for head')
parser.add_argument('--onnx_provider', default='auto', choices=['auto', 'cpu', 'cuda'], help='ONNXRuntime provider preference')

args = parser.parse_args()

class OnnxNanoTracker:
    """ONNXRuntime tracker using exported backbone/head."""
    def __init__(self, backbone_path, head_path, provider_pref='auto', search_backbone_path=''):
        import onnxruntime as ort

        def resolve_providers(pref):
            available = ort.get_available_providers()
            if pref == 'cpu':
                return ['CPUExecutionProvider']
            if pref == 'cuda':
                if 'CUDAExecutionProvider' in available:
                    return ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print('CUDAExecutionProvider 不可用，回退 CPUExecutionProvider')
                return ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            return ['CPUExecutionProvider']

        providers = resolve_providers(provider_pref)
        self.backbone_sess = ort.InferenceSession(backbone_path, providers=providers)
        self.search_backbone_sess = None
        if search_backbone_path and os.path.exists(search_backbone_path):
            self.search_backbone_sess = ort.InferenceSession(search_backbone_path, providers=providers)
        self.head_sess = ort.InferenceSession(head_path, providers=providers)
        self.backbone_input = self.backbone_sess.get_inputs()[0].name
        self.backbone_input_shape = self.backbone_sess.get_inputs()[0].shape
        # prefer explicit H/W from ONNX input shape, otherwise fall back to cfg
        if len(self.backbone_input_shape) == 4 and isinstance(self.backbone_input_shape[2], int):
            self.template_size = int(self.backbone_input_shape[2])
        else:
            self.template_size = cfg.TRACK.EXEMPLAR_SIZE
        head_inputs = self.head_sess.get_inputs()
        self.head_inputs = [head_inputs[0].name, head_inputs[1].name]
        self.head_template_hw = (head_inputs[0].shape[2], head_inputs[0].shape[3]) if len(head_inputs[0].shape) >= 4 else (None, None)
        self.head_search_hw = (head_inputs[1].shape[2], head_inputs[1].shape[3]) if len(head_inputs[1].shape) >= 4 else (None, None)
        self.score_size = cfg.TRACK.OUTPUT_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.channel_average = None

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()
        delta[0, :] = point[:, 0] - delta[0, :]  # x1
        delta[1, :] = point[:, 1] - delta[1, :]  # y1
        delta[2, :] = point[:, 0] + delta[2, :]  # x2
        delta[3, :] = point[:, 1] + delta[3, :]  # y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _align_feature(self, feat, target_hw):
        """Align backbone output to head expected spatial size by center crop if larger."""
        t_h, t_w = target_hw
        if t_h is None or t_w is None:
            return feat
        _, _, h, w = feat.shape
        if h == t_h and w == t_w:
            return feat
        if h < t_h or w < t_w:
            raise ValueError(f'Backbone feature {h}x{w} smaller than head expects {t_h}x{t_w}')
        h_start = (h - t_h) // 2
        w_start = (w - t_w) // 2
        return feat[:, :, h_start:h_start + t_h, w_start:w_start + t_w]

    def template(self, img, bbox):
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos,
                                    self.template_size,
                                    s_z, self.channel_average)
        z_np = z_crop.cpu().numpy()
        zf = self.backbone_sess.run(None, {self.backbone_input: z_np})[0]
        self.zf = self._align_feature(zf, self.head_template_hw)

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
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
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        import torch
        im_patch = torch.from_numpy(im_patch)
        return im_patch

    def track(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        x_np = x_crop.cpu().numpy()
        if self.search_backbone_sess:
            search_input = self.search_backbone_sess.get_inputs()[0].name
            xf = self.search_backbone_sess.run(None, {search_input: x_np})[0]
        else:
            xf = self.backbone_sess.run(None, {self.backbone_input: x_np})[0]
        xf = self._align_feature(xf, self.head_search_hw)
        cls, loc = self.head_sess.run(None, {self.head_inputs[0]: self.zf,
                                             self.head_inputs[1]: xf})
        import torch
        cls = torch.from_numpy(cls)
        loc = torch.from_numpy(loc)

        score = self._convert_score(cls)
        pred_bbox = self._convert_bbox(loc, self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
        
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)

        pscore = penalty * score 

        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]  

        width = self.size[0] * (1 - lr) + bbox[2] * lr 
        height = self.size[1] * (1 - lr) + bbox[3] * lr 

        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height] 

        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

def get_frames(video_name): 
    if not video_name:
        cap = cv2.VideoCapture(0) 
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break 

    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video_name)
        
        # warmup
        for i in range(50):
            cap.read()

        while True:
            ret, frame = cap.read()
            if ret:
                yield frame 
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def main():
    # load config
    cfg.merge_from_file(args.config)
    if args.use_onnx:
        cfg.CUDA = False
    else:
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    if args.use_onnx:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            print('缺少 onnxruntime，请先安装（CPU 或 GPU 版本）。')
            return
        if not os.path.exists(args.onnx_backbone):
            print('找不到 ONNX backbone: {}，请先运行 pytorch2onnx.py 导出。'.format(args.onnx_backbone))
            return
        search_backbone_path = args.onnx_backbone_search or args.onnx_backbone
        if not os.path.exists(search_backbone_path):
            print('找不到 ONNX search backbone: {}，请先导出。'.format(search_backbone_path))
            return
        if not os.path.exists(args.onnx_head):
            print('找不到 ONNX head: {}，请先运行 pytorch2onnx.py 导出。'.format(args.onnx_head))
            return
        tracker = OnnxNanoTracker(args.onnx_backbone, args.onnx_head, args.onnx_provider, search_backbone_path=search_backbone_path)
    else:
        # create model
        model = ModelBuilder()
        # load model
        model = load_pretrain(model, args.snapshot).to(device).eval()
        # build tracker
        tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        
    frame_idx = 0
    for frame in get_frames(args.video_name):
        if first_frame: 
            # build video writer 
            if args.save: 
                if args.video_name.endswith('avi') or \
                    args.video_name.endswith('mp4') or \
                    args.video_name.endswith('mov'):
                    cap = cv2.VideoCapture(args.video_name)
                    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
                else:
                    fps = 30 
                
                save_video_path = args.video_name.split(video_name)[0] + video_name + '_tracking.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_size = (frame.shape[1], frame.shape[0]) # (w, h)
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size)
            try:
                if args.manual_roi:
                    # allow manual ROI selection on first frame
                    roi = cv2.selectROI(video_name, frame, False, False)
                    init_rect = [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]
                    if init_rect[2] == 0 or init_rect[3] == 0:
                        print('ROI not selected, exit.')
                        exit()
                else:
                    init_rect = [975, 517, 35, 21]
                print('Init ROI (x, y, w, h): {}'.format(init_rect))
            except Exception as e:
                print('Failed to get ROI:', e)
                exit()
            if args.use_onnx:
                tracker.template(frame, init_rect)
            else:
                tracker.init(frame, init_rect) 
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'best_score' in outputs:
                print(f'Frame {frame_idx}: score={outputs["best_score"]:.4f}')
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(1)

        if args.save:
            video_writer.write(frame)
        frame_idx += 1
    
    if args.save:
        video_writer.release()

if __name__ == '__main__':
    main()
