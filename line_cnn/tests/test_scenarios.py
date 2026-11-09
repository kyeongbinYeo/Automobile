"""Required scenarios on synthetic masks through the whole pipeline (MockBackend).

These test geometry, tracking and control logic only. They are open-loop and say nothing about
CNN accuracy; there is no vehicle dynamics in the loop.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import IN_H, IN_W  # noqa: E402
from perception import MockBackend  # noqa: E402
from pipeline import Pipeline, load_config  # noqa: E402
from tracking import PathState  # noqa: E402

CFG = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "default.yaml"))
# Tests start from an unsurveyed camera; each test sets the pose it needs.
CFG["camera"]["mount"].update(height_m=None, pitch_deg=None, x_m=None)
FRAME = np.zeros((720, 1280, 3), np.uint8)
DT = 1 / 30.0


def road(curve=0.0, gap=None, conf=0.95):
    """Two perspective lines. curve > 0 bends the road to the image right with distance."""
    line = np.zeros((IN_H, IN_W), np.float32)
    for y in range(30, IN_H):
        s = (y - 20) / 100.0
        c = 160 + curve * ((IN_H - y) / 80.0) ** 2
        if gap and gap[0] <= y <= gap[1]:
            continue
        for x in (c - 140 * s, c + 140 * s):
            x = int(x)
            if 0 <= x < IN_W:
                line[y, max(0, x - 2):x + 2] = conf
    mat = np.zeros((IN_H, IN_W), np.float32)
    mat[28:] = conf
    return line, mat


EMPTY = (np.zeros((IN_H, IN_W), np.float32), np.zeros((IN_H, IN_W), np.float32))


def run(seq, cfg=CFG):
    p = Pipeline(cfg, MockBackend(seq))
    out = []
    for i, m in enumerate(seq):
        out.append(p.step(None if m is None else FRAME, i * DT))
    return out


class Scenarios(unittest.TestCase):
    def test_1_straight(self):
        r = run([road()] * 30)
        self.assertEqual(r[-1].state, PathState.VALID)
        self.assertLess(abs(r[-1].steer_norm), 0.05)
        self.assertGreater(r[-1].target_speed, 0.3)

    def _curve(self, curve, sign):
        r = run([road(curve)] * 45)
        self.assertEqual(r[-1].state, PathState.VALID, r[-1].reject_reason)
        self.assertGreater(sign * r[-1].steer_norm, 0.02)
        return r[-1]

    def test_2_gentle_left(self):
        self._curve(-40, +1)

    def test_3_gentle_right(self):
        self._curve(40, -1)

    def test_4_sharp_left(self):
        sharp, gentle = self._curve(-120, +1), self._curve(-40, +1)
        self.assertGreater(abs(sharp.steer_norm), abs(gentle.steer_norm))
        self.assertLess(sharp.target_speed, gentle.target_speed)

    def test_5_sharp_right(self):
        self._curve(120, -1)

    def test_6_partial_occlusion(self):
        r = run([road(gap=(60, 80))] * 20)
        self.assertIn(r[-1].state, (PathState.VALID, PathState.LOW_CONFIDENCE))

    def test_7_temporary_missing_recovers(self):
        r = run([road()] * 15 + [EMPTY] * 4 + [road()] * 5)
        self.assertEqual(r[16].state, PathState.TEMP_MISSING)
        self.assertLess(r[18].target_speed, r[14].target_speed)
        self.assertIn(r[-1].state, (PathState.VALID, PathState.LOW_CONFIDENCE))

    def test_8_low_confidence_slows(self):
        # Lighting change as seen by this layer: probabilities near the threshold.
        r = run([road()] * 30 + [road(conf=0.6)] * 30)
        self.assertLessEqual(r[-1].target_speed, r[29].target_speed)

    def test_9_camera_dropout_stops_at_once(self):
        r = run([road()] * 10 + [None] * 2 + [road()] * 2)
        self.assertEqual(r[10].state, PathState.FAILURE)
        self.assertTrue(r[10].stop)
        self.assertEqual(r[10].target_speed, 0.0)

    def test_10_no_region_stops_after_timeout(self):
        r = run([road()] * 10 + [EMPTY] * 15)
        self.assertEqual(r[-1].state, PathState.LOST)
        self.assertTrue(r[-1].stop)
        self.assertEqual(r[-1].steer_norm, 0.0)

    def test_never_moves_without_path(self):
        self.assertTrue(all(x.stop and x.target_speed == 0 for x in run([EMPTY] * 10)))

    def test_inference_exception_is_safe(self):
        class Bad(MockBackend):
            def predict(self, _):
                raise RuntimeError("boom")
        r = Pipeline(CFG, Bad([])).step(FRAME, 0.0)
        self.assertEqual(r.state, PathState.FAILURE)
        self.assertTrue(r.stop)

    def test_nan_prediction_is_safe(self):
        nan = (np.full((IN_H, IN_W), np.nan, np.float32),) * 2
        self.assertEqual(run([nan])[0].state, PathState.FAILURE)

    def test_metric_mode_with_calibration(self):
        cfg = dict(CFG)
        cfg["calibration"] = {"image_points": [[100, 110], [220, 110], [130, 50], [190, 50]],
                              "ground_points": [[0.5, 0.2], [0.5, -0.2], [1.5, 0.2], [1.5, -0.2]]}
        r = run([road(-60)] * 40, cfg)[-1]
        self.assertEqual(r.mode, "metric_pure_pursuit")
        self.assertEqual(r.curvature_unit, "1/m")
        self.assertGreater(r.steer_norm, 0)


if __name__ == "__main__":
    unittest.main()
