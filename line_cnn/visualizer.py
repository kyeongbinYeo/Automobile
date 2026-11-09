"""Annotated debug frames. Blue = mat, green = line, yellow = centerline, red = lookahead."""
import cv2
import numpy as np

from common import input_to_frame_xy, restore_mask
from tracking import PathState

STATE_COLOR = {PathState.VALID: (0, 220, 0), PathState.LOW_CONFIDENCE: (0, 220, 220),
               PathState.TEMP_MISSING: (0, 140, 255), PathState.LOST: (0, 0, 255),
               PathState.FAILURE: (255, 0, 255)}


def fmt(v, spec="+.3f"):
    return "n/a" if v is None else format(v, spec)


def draw(frame, r, fps=None, projector=None, bev_cfg=None):
    out = frame.copy() if frame is not None else np.zeros((360, 640, 3), np.uint8)
    if r.p_line is not None:
        mat = restore_mask((r.p_mat > 0.5).astype(np.uint8), out.shape) > 0
        line = restore_mask((r.p_line > 0.5).astype(np.uint8), out.shape) > 0
        out[mat] = (out[mat] * 0.65 + np.array([120, 40, 0]) * 0.35).astype(np.uint8)
        out[line] = (0, 255, 0)
    if r.path_xy is not None:
        pts = [input_to_frame_xy(x, y, out.shape) for x, y in r.path_xy]
        cv2.polylines(out, [np.array(pts, np.int32)], False, (0, 255, 255), 3)
    if r.stop_line_mask is not None:
        sl = restore_mask(r.stop_line_mask, out.shape) > 0
        out[sl] = (255, 0, 255)
    if r.target_px is not None:
        x, y = input_to_frame_xy(r.target_px[0], r.target_px[1], out.shape)
        cv2.circle(out, (int(x), int(y)), 9, (0, 0, 255), -1)

    lines = [
        f"{r.state.value} conf={r.confidence:.2f} [{r.mode}]",
        f"steer={r.steer_norm:+.2f} (+left) speed={r.target_speed:.2f} m/s (open-loop target)",
        f"lat_err={fmt(r.lateral_error, '+.2f')} {r.lateral_unit} curv={fmt(r.curvature, '+.4f')} {r.curvature_unit}",
        f"fps={fmt(fps, '.1f')} total={r.timing_ms.get('total', 0.0):.1f}ms cnn={r.timing_ms.get('inference', 0.0):.1f}ms",
    ]
    if r.stop_line_visible or r.steer_saturated:
        d = "n/a" if r.stop_line_bumper_m is None else f"{r.stop_line_bumper_m:.2f} m"
        lines.append(f"STOP LINE bumper dist={d}" if r.stop_line_visible else "")
        if r.steer_saturated:
            lines.append("STEER SATURATED (curve tighter than steering lock)")
    lines = [s for s in lines if s]
    if r.stop or r.reject_reason:
        lines.append(f"stop={r.stop_reason or '-'} reject={r.reject_reason or '-'}")
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 760), 26 * len(lines) + 8), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        col = STATE_COLOR[r.state] if i == 0 else (255, 255, 255)
        cv2.putText(out, s, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1, cv2.LINE_AA)

    if projector is not None and bev_cfg is not None and frame is not None:
        from common import crop_resize
        bev = projector.warp_bev(crop_resize(frame), bev_cfg["x_range_m"], bev_cfg["y_half_m"],
                                 bev_cfg["pixels_per_m"])
        h = out.shape[0] // 2
        bev = cv2.resize(bev, (h * bev.shape[1] // bev.shape[0], h))
        out[-h:, -bev.shape[1]:] = bev
    return out
