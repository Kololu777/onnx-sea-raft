import sys

sys.path.append("code/SEA-RAFT")
sys.path.append("code/SEA-RAFT/core")
import argparse
import json

import numpy as np
import torch
from core.raft import RAFT
from core.utils.utils import load_ckpt

def parse_args(parser: argparse.ArgumentParser):
    entry = parser.parse_args()
    json_path = entry.cfg
    with open(json_path, "r") as f:
        cfg = json.load(f)
    for key, value in cfg.items():
        setattr(entry, key, value)
    return entry


class RAFTWrapper(RAFT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_forward_args(self, iters=12, test_mode=True):
        self.iters = iters
        self.test_mode = test_mode

    def forward(self, image1, image2):
        output = super().forward(
            image1, image2, iters=self.iters, test_mode=self.test_mode
        )
        flow_final = output["flow"][-1]
        info_final = output["info"][-1]
        return flow_final, info_final


def load_model(args):
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFTWrapper.from_pretrained(args.url, args=args).eval()
    model.set_forward_args(args.iters, True)
    return model


def test_onnx(onnx_path, args):
    # dummy input
    session = ort.InferenceSession(onnx_path)
    input_shape = session.get_inputs()[0].shape
    dummy_image1 = torch.randn(input_shape)
    dummy_image2 = torch.randn(input_shape)

    # pytorch model.
    model = load_model(args)
    flow_torch, info_torch = model(dummy_image1, dummy_image2)
    flow_np = flow_torch.detach().numpy()
    info_np = info_torch.detach().numpy()

    # onnx model.
    flow_onnx, info_onnx = session.run(
        ["flow_final", "info_final"],
        {"image1": dummy_image1.numpy(), "image2": dummy_image2.numpy()},
    )
    assert np.allclose(flow_np, flow_onnx, rtol=0.01, atol=0.01)
    assert np.allclose(info_np, info_onnx, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument("--path", help="checkpoint path", type=str, default=None)
    parser.add_argument("--url", help="checkpoint url", type=str, default=None)
    parser.add_argument(
        "--output_onnx_file",
        help="output onnx file name",
        type=str,
        default="raft.onnx",
    )
    parser.add_argument("--opset", help="onnx opset version", type=int, default=16)
    parser.add_argument(
        "--check_numerical",
        help="numerical comparison between torch model and converted onnx model",
        type=bool,
        default=True,
    )
    args = parse_args(parser)
    model = load_model(args)
    model.set_forward_args(args.iters, True)

    dummy_input = (
        torch.randn(1, 3, args.image_size[0], args.image_size[1]),
        torch.randn(1, 3, args.image_size[0], args.image_size[1]),
    )

    torch.onnx.export(
        model,
        dummy_input,
        args.output_onnx_file,
        opset_version=args.opset,
        input_names=["image1", "image2"],
        output_names=["flow_final", "info_final"],
    )

    if args.check_numerical:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ValueError("onnxruntime is not installed, pip install onnxruntime")
        test_onnx(args.output_onnx_file, args)
        print("The numerical error test passed.")
