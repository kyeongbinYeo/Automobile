#!/usr/bin/env python3
"""Line F1 on the hand-labeled frames under simulated exposure changes.

python lighting_eval.py real.npz weights/linenet.onnx
Exposure is applied digitally, so this approximates a real lighting change.
"""
import sys

import numpy as np

from evaluate import evaluate
from perception import OnnxBackend


def main():
    z = np.load(sys.argv[1])
    imgs = z["img"]   # each z[...] access decompresses the whole array
    frames = {int(i): imgs[int(i)] for i in z["idx"]}
    backend = OnnxBackend(sys.argv[2], use_gamma=True, threads=1)
    for gain in (0.25, 0.5, 1.0, 1.5, 2.0):
        def predict(_, bgr):
            x = np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            return (backend.predict(x)[0] > 0.5).astype(np.uint8)
        m = evaluate(predict, frames)
        print(f"exposure x{gain}: lit {m['lit']}  dark {m['dark']}")


if __name__ == "__main__":
    main()
