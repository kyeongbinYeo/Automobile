#!/usr/bin/env python3
"""Open-loop centerline checks on the labeled frames, running the pipeline over the whole video.

python center_eval.py real.npz weights/linenet.onnx CACHE.npz [contact_sheet.jpg]

No centerline ground truth exists, so the hand-labeled lines give two proxies:
  crossing: the path crosses a labeled line (a lane center must not).
  centering: |d(A)-d(B)| / (d(A)+d(B)) to the two labeled boundaries, 0 = centered, 1 = on a
             line. Only for frames in TWO_LINE, where both lane boundaries are labeled.
"""
import os
import sys

import cv2
import numpy as np

from common import IN_H, IN_W, adaptive_gamma
from evaluate import load_labels
from perception import OnnxBackend, SegmentationBackend
from pipeline import Pipeline, load_config

HERE = os.path.dirname(os.path.abspath(__file__))
TWO_LINE = {255: (0, 1), 270: (0, 1), 280: (0, 1), 700: (0, 1), 710: (0, 1)}
FRAME = np.zeros((720, 1280, 3), np.uint8)


class CacheBackend(SegmentationBackend):
    name = "cache"

    def __init__(self, line, mat):
        self.line, self.mat, self.i = line, mat, 0

    def predict(self, _):
        i = self.i
        self.i += 1
        return self.line[i].astype(np.float32) / 255.0, self.mat[i].astype(np.float32) / 255.0


def poly_mask(poly):
    m = np.zeros((IN_H, IN_W), np.uint8)
    cv2.polylines(m, [np.array(poly, np.int32)], False, 1, 1)
    return m


def crosses(path, polys):
    m = np.zeros((IN_H, IN_W), np.uint8)
    cv2.polylines(m, [np.round(path).astype(np.int32)], False, 1, 1)
    gt = np.zeros((IN_H, IN_W), np.uint8)
    for p in polys:
        gt |= cv2.dilate(poly_mask(p), np.ones((3, 3), np.uint8))
    return bool((m & gt).any())


def centering(path, a, b):
    da = cv2.distanceTransform(1 - poly_mask(a), cv2.DIST_L2, 3)
    db = cv2.distanceTransform(1 - poly_mask(b), cv2.DIST_L2, 3)
    p = np.round(path).astype(int)
    p = p[(p[:, 0] >= 0) & (p[:, 0] < IN_W) & (p[:, 1] >= 0) & (p[:, 1] < IN_H)]
    if p.size == 0:
        return None
    x, y = p[:, 0], p[:, 1]
    return float(np.mean(np.abs(da[y, x] - db[y, x]) / (da[y, x] + db[y, x] + 1e-6)))


def main():
    data, model, cache = sys.argv[1:4]
    sheet = sys.argv[4] if len(sys.argv) > 4 else None
    z = np.load(data)
    imgs = z["img"]
    if os.path.exists(cache):
        c = np.load(cache)
        line, mat = c["line"], c["mat"]
    else:
        b = OnnxBackend(model, True, 1)
        ps = [b.predict(im) for im in imgs]
        line = np.stack([(p[0] * 255).astype(np.uint8) for p in ps])
        mat = np.stack([(p[1] * 255).astype(np.uint8) for p in ps])
        np.savez_compressed(cache, line=line, mat=mat)

    labels, _ = load_labels()
    pipe = Pipeline(load_config(os.path.join(HERE, "configs", "default.yaml")), CacheBackend(line, mat))
    counts, reasons, res = {}, {}, {}
    targets, steers = [], []
    tiles = []
    for i in range(len(imgs)):
        r = pipe.step(FRAME, i / 24.0)
        counts[r.state.value] = counts.get(r.state.value, 0) + 1
        reasons[r.reject_reason] = reasons.get(r.reject_reason, 0) + 1
        steers.append(r.steer_norm)
        targets.append(r.target_px)
        if i in labels:
            res[i] = r
            if sheet:
                im = cv2.resize(adaptive_gamma(imgs[i]), None, fx=2, fy=2)
                im[cv2.resize(line[i], None, fx=2, fy=2) > 127] = (0, 200, 0)
                if r.path_xy is not None:
                    cv2.polylines(im, [np.round(r.path_xy * 2).astype(np.int32)], False, (0, 255, 255), 2)
                if r.target_px is not None:
                    cv2.circle(im, (int(r.target_px[0] * 2), int(r.target_px[1] * 2)), 5, (0, 0, 255), -1)
                cv2.putText(im, f"#{i} {r.state.value} s={r.steer_norm:+.2f}", (4, 16), 0, 0.5, (0, 255, 255), 1)
                tiles.append(im)

    with_path = [i for i, r in res.items() if r.path_xy is not None]
    cross = [i for i in with_path if crosses(res[i].path_xy, labels[i])]
    cent = {i: centering(res[i].path_xy, labels[i][a], labels[i][b])
            for i, (a, b) in TWO_LINE.items() if res[i].path_xy is not None}
    cent = {k: round(v, 3) for k, v in cent.items() if v is not None}
    t = [x for x in targets if x is not None]
    jumps = [np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(targets, targets[1:])
             if a is not None and b is not None]
    print("states", counts)
    print("reject", reasons)
    print(f"labeled frames with a path: {len(with_path)}/{len(labels)}  crossing a labeled line: {cross}")
    print("centering (0=centered)", cent, "mean", round(float(np.mean(list(cent.values()))), 3) if cent else None)
    print("target jump px p50/p99", np.percentile(jumps, [50, 99]).round(1).tolist() if jumps else None,
          " frames with target", len(t))
    print("|steer|>0.8 frames", int(np.sum(np.abs(steers) > 0.8)))
    if sheet and tiles:
        cv2.imwrite(sheet, np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]))


if __name__ == "__main__":
    main()
