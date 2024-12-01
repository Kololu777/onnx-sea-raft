import argparse
import json

import cv2
import numpy as np
import onnxruntime
from visualizer.flow_vis import flow_to_image


def parse_args(parser: argparse.ArgumentParser):
    entry = parser.parse_args()
    json_path = entry.cfg
    with open(json_path, "r") as f:
        cfg = json.load(f)
    for key, value in cfg.items():
        setattr(entry, key, value)
    return entry


def read_data(
    image_path_1: str,
    image_path_2: str,
    size: tuple = (960, 432),  # (width, height)
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    return image1, image2


class ONNXRAFTImageProcessor:
    def __init__(self, image_size):
        self.image_size = image_size
        
    def preprocess(
        self,
        image_path_1: str,
        image_path_2: str,
        dtype: np.dtype = np.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        image1, image2 = read_data(image_path_1, image_path_2, self.image_size, dtype)
       
        self.image1 = image1
        self.image2 = image2
        self.ori_image_size = (image1.shape[1], image1.shape[0]) #(W, H)
        image1 = cv2.resize(image1, self.image_size, interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, self.image_size, interpolation=cv2.INTER_LINEAR)
        image1 = image1.transpose(2, 0, 1).astype(dtype)
        image2 = image2.transpose(2, 0, 1).astype(dtype)

        return image1, image2

    def postprocess(
        self, flow: np.ndarray, info: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        flow = flow.transpose(1, 2, 0) 
        info = info.transpose(1, 2, 0)
        flow = cv2.resize(flow, self.ori_image_size, interpolation=cv2.INTER_LINEAR)
        info = cv2.resize(info, self.ori_image_size, interpolation=cv2.INTER_AREA)
        flow = flow.transpose(2, 0, 1) # (2, H, W)
        info = info.transpose(2, 0, 1) # (4, H, W)
        return flow, info


class Heatmap:
    def __init__(self, var_min: float=0, var_max: float=10):
        self.var_min = var_min
        self.var_max = var_max

    def softmax(self, x: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

    def heatmap(self, info: np.ndarray) -> np.ndarray:
        raw_b = info[:, 2:]
        log_b = np.zeros_like(raw_b)
        weight = self.softmax(info[:, :2], axis=1)
        log_b[:, 0] = np.clip(raw_b[:, 0], 0, self.var_max)
        log_b[:, 1] = np.clip(raw_b[:, 1], self.var_min, 0)
        heatmap = (log_b * weight).sum(axis=1, keepdims=True)
        return heatmap

    def vis_heatmap(
        self, image: np.ndarray, heatmap: np.ndarray, output_path: str
    ) -> None:
        heatmap = heatmap[:, :, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = image * 0.3 + colored_heatmap * 0.7
        _, width = image.shape[:2]
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.repeat(gradient[np.newaxis, :], 50, axis=0)
        color_bar = cv2.applyColorMap(gradient, colormap=cv2.COLORMAP_JET)
        overlay = overlay.astype(np.uint8)
        combined_image = cv2.vconcat([overlay, color_bar])
        output_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image)


class ONNXRAFT:
    def __init__(self, model_path: str):
        providers =  onnxruntime.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        h, w = self.session.get_inputs()[0].shape[2:]
        self.image_processor = ONNXRAFTImageProcessor(image_size=(w, h))
    def __call__(
        self, image1: str, image2: str, scale: int
    ) -> tuple[np.ndarray, np.ndarray]:
        image1, image2 = self.image_processor.preprocess(image1, image2)
        flow, info = self.session.run(
            ["flow_final", "info_final"],
            {"image1": image1[None, ...], "image2": image2[None, ...]},
        )
        flow, info = self.image_processor.postprocess(flow[0], info[0])
        return flow, info

    @staticmethod
    def viz_flow(
        flow: np.ndarray, output_path: str | None = None
    ) -> np.ndarray:
        flow_vis = flow_to_image(flow, convert_to_bgr=True)
        if output_path is not None:
            cv2.imwrite(output_path, flow_vis)
        return flow_vis

    @staticmethod
    def vis_heatmap(image: np.ndarray, heatmap: np.ndarray, output_path: str) -> None:
        heatmap_vis = Heatmap(var_min=0, var_max=10)
        heatmap = heatmap_vis.heatmap(heatmap[None, ...])
        
        heatmap_vis.vis_heatmap(
            image, heatmap[0].transpose(1, 2, 0), output_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-im1", "--image1", type=str, required=True)
    parser.add_argument("-im2", "--image2", type=str, required=True)
    parser.add_argument("-fl", "--output_flow", type=str, default="flow.jpg")
    parser.add_argument("-he", "--output_heatmap", type=str, default="heatmap.jpg")
    args = parser.parse_args()
    onnx_raft = ONNXRAFT(model_path=args.model_path)

    flow, info = onnx_raft(
        image1=args.image1,
        image2=args.image2,
        scale=0,
    )
    onnx_raft.viz_flow(flow.transpose(1, 2, 0), output_path=args.output_flow)

    onnx_raft.vis_heatmap(onnx_raft.image_processor.image1, info, output_path=args.output_heatmap)
