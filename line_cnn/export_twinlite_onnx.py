#!/usr/bin/env python3
"""TwinLiteNet weights -> ONNX with probability outputs, plus a speed and equivalence check.

python export_twinlite_onnx.py ../third_party/TwinLiteNet weights/twinlite_ft.pt weights/twinlite_ft.onnx
Input (1,3,120,320) RGB/255. Output (1,2,120,320): channel 0 lane, channel 1 drivable area.
"""
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

from common import IN_H, IN_W
from onnx_tag import tag
from perception import TwinLiteNetBackend


class Export(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        da, ll = self.net(x)
        return torch.stack([torch.softmax(ll, 1)[:, 1], torch.softmax(da, 1)[:, 1]], 1)


def main():
    repo, weights, out = sys.argv[1:4]
    net = TwinLiteNetBackend.from_repo(repo, weights, size=(IN_W, IN_H)).model.eval()
    m = Export(net).eval()
    x = torch.rand(1, 3, IN_H, IN_W)
    torch.onnx.export(m, x, out, input_names=["image"], output_names=["prob"],
                      opset_version=17, dynamo=False)
    tag(out, "twinlite")
    with torch.no_grad():
        ref = m(x).numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
    got = s.run(None, {"image": x.numpy()})[0]
    print("max |torch - onnx| =", float(np.abs(ref - got).max()))
    ts = []
    for _ in range(10):
        t = time.perf_counter()
        s.run(None, {"image": x.numpy()})
        ts.append((time.perf_counter() - t) * 1000)
    print("onnxruntime CPU 4 threads, batch 1: median %.0f ms  min %.0f ms (not a Jetson number)"
          % (np.median(ts), np.min(ts)))


if __name__ == "__main__":
    main()
