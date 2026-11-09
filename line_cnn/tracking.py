"""Temporal smoothing, continuity validation and the perception state machine."""
from enum import Enum

import numpy as np


class PathState(str, Enum):
    VALID = "VALID"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    TEMP_MISSING = "TEMP_MISSING"
    LOST = "LOST"
    FAILURE = "FAILURE"


class PathFilter:
    """EMA over the sampled path. Held output expires after missing_timeout_s."""

    def __init__(self, alpha=0.4, conf_ok=0.45, conf_min=0.2, jump_px=40.0,
                 jump_reject_frames=3, missing_timeout_s=0.3):
        self.alpha = alpha
        self.conf_ok = conf_ok
        self.conf_min = conf_min
        self.jump_px = jump_px
        self.jump_reject_frames = jump_reject_frames
        self.timeout = missing_timeout_s
        self.path = None
        self.t_valid = None
        self.jumps = 0
        self.state = PathState.LOST

    def _missing(self, t):
        if self.path is not None and self.t_valid is not None and t - self.t_valid <= self.timeout:
            self.state = PathState.TEMP_MISSING
        else:
            self.state = PathState.LOST
            self.path = None     # never steer from an expired path
        return self.state

    def update(self, centerline, t):
        if not centerline.valid or centerline.confidence < self.conf_min:
            return self._missing(t)
        new = centerline.path
        if self.path is not None and np.max(np.linalg.norm(new - self.path, axis=1)) > self.jump_px \
                and self.jumps < self.jump_reject_frames:
            self.jumps += 1
            return self._missing(t)
        fresh = self.path is None or self.jumps >= self.jump_reject_frames
        self.jumps = 0
        self.path = new.copy() if fresh else self.path + self.alpha * (new - self.path)
        self.t_valid = t
        self.state = PathState.VALID if centerline.confidence >= self.conf_ok \
            else PathState.LOW_CONFIDENCE
        return self.state

    def fail(self):
        """Camera or inference failure: stop immediately and forget the path."""
        self.state = PathState.FAILURE
        self.path = None
        self.t_valid = None
        return self.state
