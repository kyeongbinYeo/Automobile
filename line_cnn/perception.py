"""Swappable segmentation backends. All return (p_line, p_mat), float arrays of shape (IN_H, IN_W).

New perception models (quantized CNN, CIM, TwinLiteNet with real weights, ...) only need
predict(input_bgr) with this contract.
"""
from abc import ABC, abstractmethod

import numpy as np

from common import IN_H, IN_W, adaptive_gamma, to_tensor_np


def check_onnx_kind(sess, kind, path):
    """Refuse an ONNX file that was exported for the other backend (see onnx_tag.py)."""
    got = sess.get_modelmeta().custom_metadata_map.get("backend")
    if got != kind:
        raise ValueError(f"{path} is tagged backend={got!r} but perception.backend needs {kind!r}; "
                         f"use the matching weights, or tag a new export with onnx_tag.py")


class SegmentationBackend(ABC):
    name = "base"

    @abstractmethod
    def predict(self, input_bgr):
        ...


class OnnxBackend(SegmentationBackend):
    name = "onnx"

    def __init__(self, path, use_gamma=True, threads=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(path, opts, providers=providers)
        check_onnx_kind(self.sess, "linenet", path)
        self.use_gamma = use_gamma

    def predict(self, input_bgr):
        p = self.sess.run(None, {"image": to_tensor_np(input_bgr, self.use_gamma)})[0][0]
        return p[0], p[1]


class TorchBackend(SegmentationBackend):
    name = "torch"

    def __init__(self, path, device=None):
        import torch
        from model import LineNet
        self.torch = torch
        ck = torch.load(path, map_location="cpu")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LineNet(ck["base"])
        self.net.load_state_dict(ck["state"])
        self.net.to(self.device).eval()
        self.use_gamma = ck["gamma"]

    def predict(self, input_bgr):
        with self.torch.inference_mode():
            x = self.torch.from_numpy(to_tensor_np(input_bgr, self.use_gamma)).to(self.device)
            p = self.torch.sigmoid(self.net(x))[0].cpu().numpy()
        return p[0], p[1]


class TwinLiteNetBackend(SegmentationBackend):
    """Adapter for TwinLiteNet (two heads: drivable area and lane line).

    Contract checked against the official repo (chequanghuy/TwinLiteNet, test_image.py and
    model/TwinLite.py): resize to 640x360, BGR to RGB, divide by 255 with no mean/std, and the
    model returns (drivable_logits, lane_logits), each (N, 2, H, W) with channel 1 as foreground.
    Here the input is the 320x120 crop, upscaled to 640x240 (a multiple of 8). The weights were
    trained on road footage (BDD100K), so the model sees this track out of its training domain.
    """
    name = "twinlitenet"

    def __init__(self, model, device="cpu", size=(640, 240), use_gamma=False):
        import torch
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.size = size   # (width, height); must match the size the weights were tuned at
        self.use_gamma = use_gamma

    @classmethod
    def from_repo(cls, repo_dir, weights_path, device="cpu", **kw):
        """repo_dir: a TwinLiteNet checkout (contains model/TwinLite.py).

        The module is loaded by path: this project has its own model.py, and a namespace
        package named `model` would lose to it. Weights load with weights_only=True so a
        pickled file cannot run code.
        """
        import importlib.util
        import os
        import torch
        path = os.path.join(repo_dir, "model", "TwinLite.py")
        spec = importlib.util.spec_from_file_location("twinlite_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        net = mod.TwinLiteNet()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        net.load_state_dict(state)
        return cls(net, device, **kw)

    def predict(self, input_bgr):
        import cv2
        torch = self.torch
        w, h = self.size
        if self.use_gamma:
            input_bgr = adaptive_gamma(input_bgr)
        rgb = cv2.resize(input_bgr, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div(255.0)
        with torch.inference_mode():
            da, ll = self.model(x.unsqueeze(0).to(self.device))
            da = torch.softmax(da, 1)[0, 1].cpu().numpy()
            ll = torch.softmax(ll, 1)[0, 1].cpu().numpy()
        out = input_bgr.shape[:2][::-1]
        return (cv2.resize(ll, out, interpolation=cv2.INTER_AREA),
                cv2.resize(da, out, interpolation=cv2.INTER_AREA))


class TwinLiteOnnxBackend(SegmentationBackend):
    """Exported TwinLiteNet (export_twinlite_onnx.py): no PyTorch needed at run time.

    Input (1,3,120,320) RGB/255, output (1,2,120,320): channel 0 lane, channel 1 drivable area.
    """
    name = "twinlite_onnx"

    def __init__(self, path, use_gamma=True, threads=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(path, opts, providers=providers)
        check_onnx_kind(self.sess, "twinlite", path)
        self.use_gamma = use_gamma

    def predict(self, input_bgr):
        if self.use_gamma:
            input_bgr = adaptive_gamma(input_bgr)
        x = np.ascontiguousarray(input_bgr[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        p = self.sess.run(None, {"image": x})[0][0]
        return p[0], p[1]


def create_backend(perception_cfg, path, threads=None):
    """Pick the segmentation backend from config.perception.backend: linenet or twinlite.

    twinlite with a .onnx file needs no checkout; with a .pt file it loads the model code from
    perception.twinlite_repo (relative paths resolve from the project folder).
    """
    import os
    kind = perception_cfg.get("backend", "linenet")
    gamma = perception_cfg["use_gamma"]
    if kind == "twinlite":
        if path.endswith(".onnx"):
            return TwinLiteOnnxBackend(path, gamma, threads)
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.normpath(os.path.join(here, perception_cfg["twinlite_repo"]))
        return TwinLiteNetBackend.from_repo(repo, path, size=(IN_W, IN_H), use_gamma=gamma)
    if kind != "linenet":
        raise ValueError(f"unknown perception.backend: {kind}")
    return OnnxBackend(path, gamma, threads) if path.endswith(".onnx") else TorchBackend(path)


class MockBackend(SegmentationBackend):
    """TEST ONLY. Replays given probability maps; its output is not a lane prediction."""
    name = "mock"

    def __init__(self, maps):
        self.maps = iter(maps)

    def predict(self, input_bgr):
        return next(self.maps)


def validate_prediction(p_line, p_mat):
    return (p_line.ndim == 2 and p_line.shape == p_mat.shape
            and bool(np.isfinite(p_line).all()) and bool(np.isfinite(p_mat).all()))
