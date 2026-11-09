#!/usr/bin/env python3
"""Does the official pretrained TwinLiteNet see the tape? Same labels and metric as LineNet.

python eval_twinlite.py REAL.npz ../third_party/TwinLiteNet [overlay_sheet.jpg]
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

from common import adaptive_gamma
from evaluate import evaluate
from perception import TwinLiteNetBackend


def main():
    data, repo = sys.argv[1], sys.argv[2]
    sheet = sys.argv[3] if len(sys.argv) > 3 else None
    z = np.load(data)
    imgs = z["img"]
    frames = {int(i): imgs[int(i)] for i in z["idx"]}
    backend = TwinLiteNetBackend.from_repo(repo, os.path.join(repo, "pretrained", "best.pth"))
    n_params = sum(p.numel() for p in backend.model.parameters())
    print("TwinLiteNet parameters:", n_params)

    cache = {}

    def maps(i, bgr):
        if i not in cache:
            cache[i] = backend.predict(bgr)
        return cache[i]

    out = {}
    for thr in (0.3, 0.5, 0.7):
        res = evaluate(lambda i, b: (maps(i, b)[0] > thr).astype(np.uint8), frames)
        out[f"lane_head_thr{thr}"] = {k: res[k] for k in ("lit", "dark")}
        print(f"lane head threshold {thr}: lit {res['lit']}  dark {res['dark']}")
    print("per-frame (P, R) at 0.5:", evaluate(lambda i, b: (maps(i, b)[0] > 0.5).astype(np.uint8), frames)["per_frame"])

    # How much of the mat does the drivable head cover, and do the lane head's pixels sit on lines?
    cover = []
    for i in (0, 130, 390, 455, 520, 715):
        ll, da = maps(i, frames[i])
        cover.append((i, round(float((da > 0.5).mean()), 3), round(float((ll > 0.5).mean()), 4)))
    print("(frame, drivable fraction, lane fraction):", cover)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "twinlite_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"params": n_params, **out}, f, indent=1)

    if sheet:
        tiles = []
        for i in (0, 65, 130, 395, 425, 460, 700, 255, 620):
            im = cv2.resize(adaptive_gamma(frames[i]), None, fx=2, fy=2)
            ll, da = maps(i, frames[i])
            d = cv2.resize((da > 0.5).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            l = cv2.resize((ll > 0.5).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            im[d > 0] = (im[d > 0] * 0.6 + np.array([120, 0, 0]) * 0.4).astype(np.uint8)
            im[l > 0] = (0, 255, 0)
            cv2.putText(im, f"#{i}", (4, 16), 0, 0.5, (0, 255, 255), 1)
            tiles.append(im)
        cv2.imwrite(sheet, np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, 9, 3)]))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
