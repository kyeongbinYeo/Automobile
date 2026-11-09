"""Stop line detection. The stop line is the same white tape as the lane lines, laid across the
lane between the two lane lines (mat 1), so it is a long, nearly horizontal stroke whose length
matches the lane width at its image row.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class StopLine:
    row: float                 # image row of the near edge (largest y)
    center_x: float
    length_px: float
    tilt_deg: float
    mask: np.ndarray           # pixels of the stroke, slightly dilated, to cut it from the lane mask
    bumper_dist_m: Optional[float] = None   # set when a ground projector is available


def detect_stop_line(line, lane_width_px_at, min_run_px=25, max_tilt_deg=20.0,
                     min_ratio=0.6, max_ratio=1.3, min_aspect=4.0, border_px=3, end_reach_px=10):
    """line: binary mask. lane_width_px_at(y): expected horizontal lane width in px, or None.

    A stop line lies between the two lane lines, so the stroke must be fully inside the image
    and have lane line pixels at both ends. That rejects lane edges that run off the image or
    across the far end of a corner. With no width estimate nothing is reported.
    Heuristic: checked on synthetic scenes and against false hits on the sample video, but not
    validated on real stop line footage.
    """
    h, w = line.shape
    # A horizontal opening removes the near-vertical lane lines even where the stop line
    # touches them and joins everything into one component.
    horiz = cv2.morphologyEx(line, cv2.MORPH_OPEN, np.ones((1, min_run_px), np.uint8))
    n, comp = cv2.connectedComponents(horiz)
    best = None
    for c in range(1, n):
        ys, xs = np.nonzero(comp == c)
        if xs.size < min_run_px:
            continue
        pts = np.column_stack([xs, ys]).astype(float)
        mean = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        tilt = abs(np.degrees(np.arctan2(vt[0][1], vt[0][0])))
        tilt = min(tilt, 180.0 - tilt)
        if tilt > max_tilt_deg:
            continue
        length = float(np.ptp((pts - mean) @ vt[0]) + 1)
        thick = float(np.ptp((pts - mean) @ vt[1]) + 1)
        if length < min_aspect * thick:
            continue
        width = lane_width_px_at(mean[1])
        if width is None or not (min_ratio * width <= length <= max_ratio * width):
            continue
        if (xs.min() < border_px or xs.max() >= w - border_px
                or ys.min() < border_px or ys.max() >= h - border_px):
            continue   # runs off the image: a lane edge, not a whole stop line
        stroke = (comp == c).astype(np.uint8)
        others = line.astype(bool) & ~cv2.dilate(stroke, np.ones((3, 3), np.uint8)).astype(bool)
        i_left, i_right = int(np.argmin(xs)), int(np.argmax(xs))
        if not all(_touches(others, xs[i], ys[i], end_reach_px) for i in (i_left, i_right)):
            continue   # a stop line ends on a lane line at both sides
        row = float(ys.max())
        if best is None or row > best.row:
            mask = cv2.dilate(stroke, np.ones((5, 5), np.uint8))
            best = StopLine(row, float(mean[0]), length, float(tilt), mask)
    return best


def _touches(others, x, y, reach):
    h, w = others.shape
    return bool(others[max(y - reach, 0):min(y + reach + 1, h), max(x - reach, 0):min(x + reach + 1, w)].any())


def bumper_distance(projector, stop, bumper_x_m):
    """Ground distance from the front bumper to the near edge of the stop line, in meters."""
    g = projector.image_to_ground([[stop.center_x, stop.row]])[0]
    return float(g[0] - bumper_x_m)
