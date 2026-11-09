"""Preprocessing shared by training and inference. Frames stay BGR (OpenCV order) throughout."""
import cv2
import numpy as np

CROP_TOP = 0.40   # fraction of the frame removed from the top; depends on camera pitch
IN_W, IN_H = 320, 120


def adaptive_gamma(bgr, target=0.42, min_gamma=0.35):
    """Brighten dark (tunnel) frames only; lit frames pass through unchanged."""
    m = max(float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()) / 255.0, 1e-3)
    if m >= target:
        return bgr
    g = float(np.clip(np.log(target) / np.log(m), min_gamma, 1.0))
    lut = (255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def crop_resize(frame_bgr):
    y0 = int(frame_bgr.shape[0] * CROP_TOP)
    return cv2.resize(frame_bgr[y0:], (IN_W, IN_H), interpolation=cv2.INTER_AREA)


def to_tensor_np(bgr, use_gamma=True):
    """BGR uint8 (IN_H, IN_W, 3) -> float32 (1,3,H,W) in [-1,1]."""
    if use_gamma:
        bgr = adaptive_gamma(bgr)
    return (bgr.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]


def input_to_frame_xy(x, y, frame_shape):
    """Map model-input pixel coordinates to original frame coordinates."""
    h, w = frame_shape[:2]
    y0 = int(h * CROP_TOP)
    return x * w / IN_W, y0 + y * (h - y0) / IN_H


def restore_mask(mask, frame_shape):
    """Paste a model-input mask back into a full-size frame mask."""
    h, w = frame_shape[:2]
    y0 = int(h * CROP_TOP)
    out = np.zeros((h, w), mask.dtype)
    out[y0:] = cv2.resize(mask, (w, h - y0), interpolation=cv2.INTER_NEAREST)
    return out
