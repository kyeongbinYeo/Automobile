#!/usr/bin/env python3
"""Score the earlier classical pipeline (../line_tracker.py) with the same metric as the CNN."""
import os
import sys

import cv2
import numpy as np

from evaluate import evaluate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from line_tracker import LineTracker, Params  # noqa: E402


def main():
    z = np.load(sys.argv[1])
    imgs = z["img"]
    tr = LineTracker(Params(width=320, roi_top=0.0))
    frames = {int(i): imgs[int(i)] for i in z["idx"]}

    def predict(i, bgr):
        return (tr.detect(bgr)[3] > 0).astype(np.uint8)

    m = evaluate(predict, frames)
    print("classic line_tracker.py  lit", m["lit"], "dark", m["dark"])
    print("per_frame", m["per_frame"])
    lit = [i for i in frames if z["lit"][i]]
    frac = np.mean([(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) < 90).mean() for i in lit])
    print("fraction of gray<90 in lit frames:", round(float(frac), 3))


if __name__ == "__main__":
    main()
