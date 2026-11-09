import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import IN_H, IN_W, input_to_frame_xy, restore_mask  # noqa: E402
from control import (PID, MockVehicle, PurePursuit, SafetySupervisor, SpeedPlanner,  # noqa: E402
                     SteeringLimiter, create_vehicle, image_baseline_steer, select_target)
from geometry import (N_PATH, Centerline, CenterlineExtractor, GroundProjector,  # noqa: E402
                      clean_mask, ground_curvature, turning_angle)
from tracking import PathFilter, PathState  # noqa: E402


def maps(left=None, right=None, mat=True, width=4):
    """left/right: callables y -> x. Mat covers the frame unless disabled."""
    line = np.zeros((IN_H, IN_W), np.float32)
    for f in (left, right):
        if f is None:
            continue
        for y in range(IN_H):
            x = int(f(y))
            if 0 <= x < IN_W:
                line[y, max(0, x - width // 2):x + width // 2] = 0.95
    return line, np.full((IN_H, IN_W), 0.95 if mat else 0.02, np.float32)


class MaskTests(unittest.TestCase):
    def test_clean_removes_small(self):
        m = np.zeros((20, 20), np.uint8)
        m[1, 1] = 1
        m[5:15, 5:15] = 1
        out = clean_mask(m, 25)
        self.assertEqual(out[1, 1], 0)
        self.assertEqual(out[10, 10], 1)

    def test_restore_mask_coordinates(self):
        m = np.zeros((IN_H, IN_W), np.uint8)
        m[60, 160] = 1
        full = restore_mask(m, (720, 1280, 3))
        ys, xs = np.nonzero(full)
        fx, fy = input_to_frame_xy(160, 60, (720, 1280))
        self.assertLessEqual(abs(xs.mean() - fx), 4)
        self.assertLessEqual(abs(ys.mean() - fy), 4)


class ProjectionTests(unittest.TestCase):
    IMG = [[100, 100], [220, 100], [130, 50], [190, 50]]
    GND = [[0.5, 0.2], [0.5, -0.2], [1.5, 0.2], [1.5, -0.2]]

    def test_roundtrip(self):
        p = GroundProjector.from_points(self.IMG, self.GND)
        np.testing.assert_allclose(p.image_to_ground(self.IMG), self.GND, atol=1e-6)
        np.testing.assert_allclose(p.ground_to_image(self.GND), self.IMG, atol=1e-6)

    def test_needs_four_points(self):
        with self.assertRaises(ValueError):
            GroundProjector.from_points(self.IMG[:3], self.GND[:3])


class CenterlineTests(unittest.TestCase):
    def test_two_lines_straight(self):
        cl = CenterlineExtractor(IN_W, IN_H).extract(*maps(lambda y: 100, lambda y: 220))
        self.assertTrue(cl.valid, cl.reject_reason)
        self.assertEqual(cl.source, "two_lines")
        self.assertLess(np.abs(cl.path[:, 0] - 160).max(), 3)

    def test_nearly_horizontal_lines(self):
        # A corner seen by the camera: two lines running across the image.
        line = np.zeros((IN_H, IN_W), np.float32)
        line[58:62, :] = 0.95
        line[98:102, :] = 0.95
        mat = np.full((IN_H, IN_W), 0.95, np.float32)
        cl = CenterlineExtractor(IN_W, IN_H).extract(line, mat)
        self.assertTrue(cl.valid, cl.reject_reason)
        self.assertLess(np.abs(cl.path[:, 1] - 80).max(), 3)

    def test_single_line_never_assumes_width(self):
        ex = CenterlineExtractor(IN_W, IN_H)
        cl = ex.extract(*maps(lambda y: 100, None))
        self.assertIsNone(ex.hw)
        self.assertFalse(cl.valid)
        self.assertEqual(cl.reject_reason, "one_line_no_width")

    def test_single_line_uses_measured_width_on_vehicle_side(self):
        ex = CenterlineExtractor(IN_W, IN_H)
        for _ in range(4):
            ex.extract(*maps(lambda y: 100, lambda y: 220))
        self.assertIsNotNone(ex.hw)
        left = ex.extract(*maps(lambda y: 100, None))
        self.assertEqual(left.source, "one_line")
        self.assertLess(np.abs(left.path[:, 0] - 160).max(), 4)
        right = ex.extract(*maps(None, lambda y: 220))
        self.assertLess(np.abs(right.path[:, 0] - 160).max(), 4)

    def test_path_across_a_line_is_rejected(self):
        ex = CenterlineExtractor(IN_W, IN_H)
        ex.hw = (0.0, 200.0)   # stale width: the medial with the blob fails the width check
        line, mat = maps(lambda y: 100, None)
        line[58:72, 296:304] = 0.95
        cl = ex.extract(line, mat)
        self.assertFalse(cl.valid)
        self.assertEqual(cl.reject_reason, "crosses_line")

    def test_line_off_the_mat_is_ignored(self):
        line, mat = maps(lambda y: 100, lambda y: 220)
        mat[:] = 0.02
        mat[:, :170] = 0.95
        ex = CenterlineExtractor(IN_W, IN_H)
        cl = ex.extract(line, mat)
        self.assertFalse(cl.valid)

    def test_no_evidence(self):
        cl = CenterlineExtractor(IN_W, IN_H).extract(*maps(mat=False))
        self.assertFalse(cl.valid)
        self.assertEqual(cl.confidence, 0.0)

    def test_turning_angle(self):
        straight = np.column_stack([np.zeros(12), np.arange(12.0)])
        corner = np.vstack([straight[:6], np.column_stack([np.arange(1.0, 7.0), np.full(6, 5.0)])])
        self.assertAlmostEqual(turning_angle(straight), 0.0)
        self.assertGreater(turning_angle(corner), 1.2)

    def test_curvature_of_circle(self):
        R = 2.0
        xs = np.linspace(-0.5, 0.5, 201)
        coeffs = np.polyfit(xs, R - np.sqrt(R ** 2 - xs ** 2), 2)
        self.assertAlmostEqual(ground_curvature(coeffs, 0.0), 1 / R, delta=0.01)


class SteeringTests(unittest.TestCase):
    def setUp(self):
        self.pp = PurePursuit(0.32, math.radians(30))

    def test_formula(self):
        x, y = 1.0, 0.25
        ld = math.hypot(x, y)
        expected = math.atan(0.32 * 2 * math.sin(math.atan2(y, x)) / ld)
        self.assertAlmostEqual(self.pp.steer(x, y)[0], expected, places=9)

    def test_sign_left_positive_and_straight_zero(self):
        self.assertGreater(self.pp.steer(1.0, 0.3)[0], 0)
        self.assertLess(self.pp.steer(1.0, -0.3)[0], 0)
        self.assertEqual(self.pp.steer(1.0, 0.0)[0], 0)

    def test_saturation(self):
        self.assertAlmostEqual(self.pp.steer(0.05, 0.5)[0], math.radians(30))

    def test_rejects_unknown_geometry(self):
        with self.assertRaises(ValueError):
            PurePursuit(0.0, 0.5)

    def test_rate_limit(self):
        lim = SteeringLimiter(6.0)
        self.assertAlmostEqual(lim.step(1.0, 0.02), 0.12)

    def test_image_baseline_sign(self):
        self.assertGreater(image_baseline_steer(100, 160, 1.5), 0)   # path left -> steer left
        self.assertEqual(image_baseline_steer(1000, 160, 1.5), -1.0)

    def test_select_target(self):
        x, y, reached = select_target(np.array([0.0, 0.0]), 1.0)
        self.assertTrue(reached)
        self.assertAlmostEqual(x, 1.0, delta=0.03)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.f = PathFilter(missing_timeout_s=0.3)

    @staticmethod
    def cl(x=160.0, conf=0.8):
        path = np.column_stack([np.full(N_PATH, x), np.linspace(115, 40, N_PATH)])
        return Centerline(path=path, confidence=conf, reject_reason="")

    def test_confidence_levels(self):
        self.assertEqual(self.f.update(self.cl(conf=0.8), 0.0), PathState.VALID)
        self.assertEqual(self.f.update(self.cl(conf=0.3), 0.1), PathState.LOW_CONFIDENCE)
        self.assertEqual(self.f.update(self.cl(conf=0.05), 0.15), PathState.TEMP_MISSING)

    def test_timeout_forgets_path(self):
        self.f.update(self.cl(), 0.0)
        self.assertEqual(self.f.update(Centerline(), 0.2), PathState.TEMP_MISSING)
        self.assertEqual(self.f.update(Centerline(), 0.4), PathState.LOST)
        self.assertIsNone(self.f.path)

    def test_jump_rejected_then_accepted(self):
        self.f.update(self.cl(160.0), 0.0)
        for i in range(3):
            self.assertEqual(self.f.update(self.cl(260.0), 0.01 * (i + 1)), PathState.TEMP_MISSING)
        self.assertEqual(self.f.update(self.cl(260.0), 0.04), PathState.VALID)
        self.assertAlmostEqual(self.f.path[0, 0], 260.0)

    def test_failure_clears(self):
        self.f.update(self.cl(), 0.0)
        self.assertEqual(self.f.fail(), PathState.FAILURE)
        self.assertIsNone(self.f.path)


class SafetyTests(unittest.TestCase):
    def test_estop_latches(self):
        s = SafetySupervisor(1.0)
        s.estop("test")
        self.assertEqual(s.filter(PathState.VALID, 0.5, 0.5)[:3], (0.0, 0.0, True))

    def test_lost_and_failure_stop(self):
        s = SafetySupervisor(1.0)
        for st in (PathState.LOST, PathState.FAILURE):
            self.assertTrue(s.filter(st, 0.5, 0.5)[2])

    def test_frame_timeout(self):
        self.assertEqual(SafetySupervisor(1.0, 0.3).filter(PathState.VALID, 0.5, 0.5, 0.5)[3], "frame_timeout")

    def test_saturation(self):
        self.assertEqual(SafetySupervisor(1.0).filter(PathState.VALID, 2.0, 9.0)[:2], (1.0, 1.0))

    def test_hardware_gate(self):
        self.assertIsInstance(create_vehicle(False), MockVehicle)
        with self.assertRaises(NotImplementedError):
            create_vehicle(True)


class SpeedTests(unittest.TestCase):
    def setUp(self):
        self.p = SpeedPlanner(0.8, 0.3, 1.15)

    def test_curve_slower_than_straight(self):
        self.assertLess(self.p.raw(PathState.VALID, 1.0, 1.0, 1.0, 0, 1),
                        self.p.raw(PathState.VALID, 0.0, 0.0, 1.0, 0, 1))

    def test_low_conf_slower(self):
        self.assertLess(self.p.raw(PathState.LOW_CONFIDENCE, 0, 0, 0.3, 0, 1),
                        self.p.raw(PathState.VALID, 0, 0, 0.9, 0, 1))

    def test_lost_zero_without_ramp(self):
        self.p.value = 0.8
        self.assertEqual(self.p.step(PathState.LOST, 0, 0, 0, 0, 1, 0.02), 0.0)

    def test_pid_anti_windup(self):
        pid = PID(0.1, 1.0, 0.0, -0.1, 0.1, 5.0)
        for _ in range(1000):
            pid.step(10.0, 0.0, 0.02)
        self.assertLess(abs(pid.i), 1.0)


if __name__ == "__main__":
    unittest.main()
