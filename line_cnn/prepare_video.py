#!/usr/bin/env python3
"""TwinLiteNet overlay video -> real training frames and weak labels.

The green mask is blended at alpha ~0.25, so the original pixels can be recovered:
    obs = (1-a)*orig + a*(0,255,0)  ->  orig = (obs - a*(0,255,0)) / (1-a)
The overlay itself is TwinLiteNet's drivable-area output and is reused as the weak mat label.

Usage: python prepare_video.py VIDEO.mp4 OUT_DIR
Output: OUT_DIR/real.npz (img[N,120,320,3] BGR, line, mat, lit[N], idx[N])
"""
import argparse

import cv2
import numpy as np

from common import CROP_TOP, IN_H, IN_W, adaptive_gamma

ALPHA = 0.25


def unoverlay(frame):
    """Returns (restored BGR, overlay mask)."""
    f = frame.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    excess = g - np.maximum(r, b)
    mask = (excess > 18).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    out = f.copy()
    m = mask > 0
    out[m] = (f[m] - np.array([0.0, 255 * ALPHA, 0.0])) / (1 - ALPHA)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # The pure-green contour holds no original signal; fill it from its surroundings.
    contour = (excess > 90).astype(np.uint8)
    contour = cv2.dilate(contour, np.ones((3, 3), np.uint8))
    if contour.any():
        out = cv2.inpaint(out, contour, 3, cv2.INPAINT_TELEA)
    return out, m.astype(np.uint8)


def pseudo_line(img, mat):
    """Weak white-line label. Top-hat removes the large bright floor tiles."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_TOPHAT, k)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cand = ((tophat > 38) & (hsv[..., 1] < 70) & (gray > 130)).astype(np.uint8)
    # Lines are holes in the overlay: close them, but barely grow upward onto the bright floor.
    near = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cand &= near
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _keep_thin_long(cand)


def _keep_thin_long(cand, min_len=22, min_area=25):
    """Drop short fragments and blobs thicker than a line can be at their image row."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    dist = cv2.distanceTransform(cand, cv2.DIST_L2, 3)
    out = np.zeros_like(cand)
    for i in range(1, n):
        w, h, area = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        if area < min_area or max(w, h) < min_len:
            continue
        ys, xs = np.nonzero(lab == i)
        allowed = 2.0 + 8.0 * (ys / float(IN_H)) ** 1.5   # half-width grows toward the camera
        if np.mean(dist[ys, xs] <= allowed) > 0.9:
            out[lab == i] = 1
    return out


def pseudo_line_dark(img):
    """Tunnel frames have no overlay, so no mat: keep long, thin, near-neutral strokes only."""
    boosted = adaptive_gamma(img)
    gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_TOPHAT, k)
    sat = cv2.cvtColor(boosted, cv2.COLOR_BGR2HSV)[..., 1]
    cand = ((tophat > 45) & (sat < 110)).astype(np.uint8)
    cand[:IN_H // 3] = 0   # tunnel ceiling lights
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _keep_thin_long(cand, min_len=45, min_area=40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    imgs, lines, mats, lit, idx = [], [], [], [], []
    i = -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        rec, m = unoverlay(f)
        y0 = int(f.shape[0] * CROP_TOP)
        im = cv2.resize(rec[y0:], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        ma = cv2.resize(m[y0:], (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        imgs.append(im)
        mats.append(ma)
        is_lit = float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) > 45
        lines.append(pseudo_line(im, ma) if is_lit else pseudo_line_dark(im))
        lit.append(is_lit)
        idx.append(i)
    np.savez_compressed(a.out + "/real.npz", img=np.stack(imgs), line=np.stack(lines),
                        mat=np.stack(mats), lit=np.array(lit), idx=np.array(idx))
    print("frames", len(imgs), "lit", int(np.sum(lit)), "dark", int(len(lit) - np.sum(lit)))


if __name__ == "__main__":
    main()
