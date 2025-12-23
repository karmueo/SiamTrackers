from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.utils.model_load import load_pretrain

torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Export NanoTrack v3 to ONNX and eval diff')
    parser.add_argument('--config', type=str, default='./models/config/configv3.yaml', help='config file')
    parser.add_argument('--snapshot', type=str, default='./models/pretrained/nanotrackv3.pth', help='pytorch checkpoint')
    parser.add_argument('--backbone_out', type=str, default='./models/onnx/nanotrack_backbone.onnx', help='onnx path for backbone')
    parser.add_argument('--backbone_search_out', type=str, default='./models/onnx/nanotrack_backbone_search.onnx', help='onnx path for backbone (search branch, static shapes)')
    parser.add_argument('--head_out', type=str, default='./models/onnx/nanotrack_head.onnx', help='onnx path for head')
    parser.add_argument('--opset', type=int, default=14, help='onnx opset version')
    parser.add_argument('--static_shapes', action='store_true', help='export with static batch/H/W (no dynamic axes)')
    # I/O 保存/加载（仅保存/读取 backbone 原始输入 z.npy/x.npy）
    parser.add_argument('--save_io', action='store_true', help='保存推理输入 z.npy/x.npy 到 --io_dir')
    parser.add_argument('--load_io', action='store_true', help='从 --io_dir 读取已保存的 z.npy/x.npy 作为输入')
    parser.add_argument('--io_dir', type=str, default='./models/onnx/io', help='保存/读取推理输入数据的目录')
    parser.add_argument('--print_outputs', action='store_true', help='打印 backbone/head 的 torch 与 ORT 输出摘要')
    return parser.parse_args()


def ensure_dir(path):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def export_backbone(model, exemplar_size, out_path, opset, static_shapes=False, verbose=True):
    ensure_dir(out_path)
    dummy = torch.randn(1, 3, exemplar_size, exemplar_size)
    dynamic_axes = None if static_shapes else {'input': {0: 'batch', 2: 'h', 3: 'w'},
                    'output': {0: 'batch', 2: 'h', 3: 'w'}}
    torch.onnx.export(model.backbone, dummy, out_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes=dynamic_axes, opset_version=opset)
    if verbose:
        print('Backbone ONNX saved to {}'.format(out_path))


def export_head(model, z_feat, x_feat, out_path, opset, static_shapes=False, verbose=True):
    ensure_dir(out_path)
    dynamic_axes = None if static_shapes else {
        'input1': {0: 'batch', 2: 'zh', 3: 'zw'},
        'input2': {0: 'batch', 2: 'xh', 3: 'xw'},
        'output1': {0: 'batch', 2: 'oh', 3: 'ow'},
        'output2': {0: 'batch', 2: 'oh', 3: 'ow'}
    }
    # 静态默认可避免 kernel 形状未知报错；如需动态可关闭 static_shapes
    torch.onnx.export(model.ban_head, (z_feat, x_feat), out_path,
                      input_names=['input1', 'input2'],
                      output_names=['output1', 'output2'],
                      dynamic_axes=dynamic_axes,
                      opset_version=opset)
    if verbose:
        print('Head ONNX saved to {}'.format(out_path))


def run_ort(path, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    feed = {sess.get_inputs()[i].name: inp for i, inp in enumerate(inputs)}
    outputs = sess.run(None, feed)
    return outputs


def diff_metric(a, b):
    diff = np.abs(a - b)
    return diff.mean(), diff.max()


def print_sample(name, arr, max_elems=10):
    flat = arr.flatten()
    preview = flat[:max_elems]
    print('{} shape {}, dtype {}, sample {}'.format(name, arr.shape, arr.dtype, preview))


def main():
    args = parse_args()
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print('未安装 onnxruntime，请先安装: pip install onnxruntime 或 onnxruntime-gpu')
        return

    cfg.merge_from_file(args.config)
    cfg.CUDA = False
    torch.manual_seed(0)

    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).cpu().eval()

    exemplar_size = cfg.TRACK.EXEMPLAR_SIZE
    instance_size = cfg.TRACK.INSTANCE_SIZE

    # 生成或读取推理输入 z/x
    with torch.no_grad():
        if args.load_io:
            z_path = os.path.join(args.io_dir, 'z.npy')
            x_path = os.path.join(args.io_dir, 'x.npy')
            if not (os.path.exists(z_path) and os.path.exists(x_path)):
                if not args.print_outputs:
                    print('启用 --load_io 但未找到保存的 z.npy/x.npy，改用随机输入。')
                z = torch.randn(1, 3, exemplar_size, exemplar_size)
                x = torch.randn(1, 3, instance_size, instance_size)
            else:
                z_np_loaded = np.load(z_path)
                x_np_loaded = np.load(x_path)
                z = torch.from_numpy(z_np_loaded).float()
                x = torch.from_numpy(x_np_loaded).float()
                if not args.print_outputs:
                    print('已从 {} 读取 z/x 作为输入'.format(args.io_dir))
        else:
            z = torch.randn(1, 3, exemplar_size, exemplar_size)
            x = torch.randn(1, 3, instance_size, instance_size)

        # 保存 z/x
        if args.save_io:
            os.makedirs(args.io_dir, exist_ok=True)
            np.save(os.path.join(args.io_dir, 'z.npy'), z.detach().numpy())
            np.save(os.path.join(args.io_dir, 'x.npy'), x.detach().numpy())
            if not args.print_outputs:
                print('已保存 z/x 到 {}'.format(args.io_dir))

        # PyTorch 特征（用于与 ORT diff 对比）
        z_feat = model.backbone(z)
        x_feat = model.backbone(x)

    # backbone export: if静态且模板/搜索尺寸不同，导出两份
    if args.static_shapes and exemplar_size != instance_size:
        export_backbone(model, exemplar_size, args.backbone_out, args.opset, static_shapes=True, verbose=not args.print_outputs)
        export_backbone(model, instance_size, args.backbone_search_out, args.opset, static_shapes=True, verbose=not args.print_outputs)
        backbone_search_path = args.backbone_search_out
    else:
        export_backbone(model, exemplar_size, args.backbone_out, args.opset, static_shapes=args.static_shapes, verbose=not args.print_outputs)
        backbone_search_path = args.backbone_out

    export_head(model, z_feat, x_feat, args.head_out, args.opset, static_shapes=args.static_shapes, verbose=not args.print_outputs)

    # ORT inference
    z_np = z.numpy()
    x_np = x.numpy()
    z_feat_torch = z_feat.detach().numpy()
    x_feat_torch = x_feat.detach().numpy()

    # 计算 ORT 的 backbone 输出，作为 head 的输入（不再保存/读取中间特征）
    z_feat_ort = run_ort(args.backbone_out, [z_np])[0]
    x_feat_ort = run_ort(backbone_search_path, [x_np])[0]

    cls_torch, loc_torch = model.ban_head(torch.from_numpy(z_feat_torch),
                                          torch.from_numpy(x_feat_torch))
    cls_ort, loc_ort = run_ort(args.head_out, [z_feat_ort, x_feat_ort])

    if args.print_outputs:
        print_sample('z_feat_torch', z_feat_torch)
        print_sample('x_feat_torch', x_feat_torch)
        print_sample('z_feat_ort', z_feat_ort)
        print_sample('x_feat_ort', x_feat_ort)
        print_sample('cls_torch', cls_torch.detach().numpy())
        print_sample('loc_torch', loc_torch.detach().numpy())
        print_sample('cls_ort', cls_ort)
        print_sample('loc_ort', loc_ort)

    if not args.print_outputs:
        backbone_z_mean, backbone_z_max = diff_metric(z_feat_torch, z_feat_ort)
        backbone_x_mean, backbone_x_max = diff_metric(x_feat_torch, x_feat_ort)
        cls_mean, cls_max = diff_metric(cls_torch.detach().numpy(), cls_ort)
        loc_mean, loc_max = diff_metric(loc_torch.detach().numpy(), loc_ort)

        print('Backbone (template) diff mean {:.6f}, max {:.6f}'.format(backbone_z_mean, backbone_z_max))
        print('Backbone (search)   diff mean {:.6f}, max {:.6f}'.format(backbone_x_mean, backbone_x_max))
        print('Head cls diff       mean {:.6f}, max {:.6f}'.format(cls_mean, cls_max))
        print('Head loc diff       mean {:.6f}, max {:.6f}'.format(loc_mean, loc_max))


if __name__ == '__main__':
    main()
