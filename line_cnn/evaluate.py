#!/usr/bin/env python3
"""Score line masks against hand_labels.json. Row tolerance tol = 3 + 8*(y/H) px: lines are
wider near the camera and the labels mark the line center."""
import json
import os

import cv2
import numpy as np

from common import IN_H, IN_W

HERE = os.path.dirname(os.path.abspath(__file__))
_ys = np.arange(IN_H, dtype=np.float32)[:, None]
TOL = np.repeat(3.0 + 8.0 * _ys / IN_H, IN_W, axis=1)


def load_labels():
    with open(os.path.join(HERE, "hand_labels.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v for k, v in d["frames"].items()}, set(d["dark_frames"])


def raster(polys):
    gt = np.zeros((IN_H, IN_W), np.uint8)
    for p in polys:
        cv2.polylines(gt, [np.array(p, np.int32)], False, 1, 1)
    return gt


def score(pred, polys):
    gt = raster(polys)
    if pred.sum() == 0:
        return 0.0, 0.0
    d_to_gt = cv2.distanceTransform(1 - gt, cv2.DIST_L2, 3)
    d_to_pred = cv2.distanceTransform(1 - pred.astype(np.uint8), cv2.DIST_L2, 3)
    p = float(np.mean(d_to_gt[pred > 0] <= TOL[pred > 0]))
    r = float(np.mean(d_to_pred[gt > 0] <= TOL[gt > 0]))
    return p, r


def evaluate(predict, frames):
    """predict(idx, bgr) -> uint8 mask. frames: idx -> BGR. F1 uses the mean P and R per group."""
    labels, dark = load_labels()
    res = {"lit": [], "dark": []}
    per = {}
    for i, polys in labels.items():
        p, r = score(predict(i, frames[i]), polys)
        res["dark" if i in dark else "lit"].append((p, r))
        per[i] = (round(p, 3), round(r, 3))
    out = {"per_frame": per}
    for k, v in res.items():
        P, R = np.mean(v, axis=0)
        out[k] = {"P": round(float(P), 3), "R": round(float(R), 3),
                  "F1": round(float(2 * P * R / max(P + R, 1e-6)), 3)}
    return out
