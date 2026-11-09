#!/usr/bin/env python3
"""linenet.pt -> linenet.onnx. Output: sigmoid probabilities (1,2,120,320)."""
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

from common import IN_H, IN_W
from model import LineNet
from onnx_tag import tag


class Prob(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def main():
    pt, out = sys.argv[1], sys.argv[2]
    ck = torch.load(pt, map_location="cpu")
    net = LineNet(ck["base"])
    net.load_state_dict(ck["state"])
    m = Prob(net).eval()
    x = torch.rand(1, 3, IN_H, IN_W) * 2 - 1
    torch.onnx.export(m, x, out, input_names=["image"], output_names=["prob"],
                      opset_version=17, dynamo=False)
    tag(out, "linenet")
    with torch.no_grad():
        a = m(x).numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
    print("max |torch-onnx| =", float(np.abs(a - s.run(None, {"image": x.numpy()})[0]).max()))
    ts = []
    for _ in range(100):
        t = time.perf_counter()
        s.run(None, {"image": x.numpy()})
        ts.append((time.perf_counter() - t) * 1000)
    print("host CPU, 1 thread, ms: median %.1f p99 %.1f (not a Jetson number)"
          % (np.median(ts), np.percentile(ts, 99)))


if __name__ == "__main__":
    main()
