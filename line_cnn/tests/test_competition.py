"""Tests for the competition-specific parts: lane spec prior, stop line, corner feasibility,
TwinLiteNet adapter. A synthetic camera is built from four ground correspondences, and lanes
are drawn from ground coordinates at the track's real dimensions. Synthetic only: this checks
the geometry code, not the CNN or real footage.
"""
import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import IN_H, IN_W  # noqa: E402
from control import corner_window, min_turn_radius, required_steer  # noqa: E402
from geometry import CenterlineExtractor, GroundProjector, ground_path  # noqa: E402
from perception import MockBackend, TwinLiteNetBackend  # noqa: E402
from pipeline import Pipeline, load_config  # noqa: E402
from stopline import bumper_distance, detect_stop_line  # noqa: E402

IMG = [[100, 100], [220, 100], [130, 50], [190, 50]]
GND = [[0.5, 0.2], [0.5, -0.2], [1.5, 0.2], [1.5, -0.2]]
PROJ = GroundProjector.from_points(IMG, GND)
CFG = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "default.yaml"))
# Tests start from an unsurveyed camera; each test sets the pose it needs.
CFG["camera"]["mount"].update(height_m=None, pitch_deg=None, x_m=None)
SPACING = CFG["track"]["lane_center_spacing_m"]
BUMPER_X = CFG["vehicle"]["wheelbase_m"] + CFG["vehicle"]["front_overhang_m"]
FRAME = np.zeros((720, 1280, 3), np.uint8)


def draw(line, pts_g, thick=3):
    px = PROJ.ground_to_image(np.array(pts_g, float))
    cv2.polylines(line, [np.round(px).astype(np.int32)], False, 1, thick)


def scene(left=True, right=True, spacing=SPACING, stop_x=None, xs=(0.45, 1.7)):
    line = np.zeros((IN_H, IN_W), np.uint8)
    x = np.linspace(xs[0], xs[1], 30)
    for on, sign in ((left, 1), (right, -1)):
        if on:
            draw(line, np.column_stack([x, np.full_like(x, sign * spacing / 2)]))
    if stop_x is not None:   # spans the inner width, so it touches both lane lines
        half = CFG["track"]["lane_inner_width_m"] / 2
        draw(line, [[stop_x, -half - 0.01], [stop_x, half + 0.01]])
    mat = np.full((IN_H, IN_W), 0.95, np.float32)
    return line.astype(np.float32) * 0.95, mat


def extractor():
    ex = CenterlineExtractor(IN_W, IN_H)
    ex.set_lane_prior(PROJ, SPACING)
    return ex


def lateral(path):
    g = PROJ.image_to_ground(path)
    return g[(g[:, 0] > 0.3) & (g[:, 0] < 1.6)][:, 1]


class LanePriorTests(unittest.TestCase):
    def test_spacing_is_the_competition_value(self):
        self.assertAlmostEqual(SPACING, CFG["track"]["lane_inner_width_m"] + CFG["track"]["tape_width_m"])

    def test_prior_width_in_pixels(self):
        ex = extractor()
        self.assertAlmostEqual(ex.lane_width_px_at(100), SPACING * 300.0, delta=3.0)   # 300 px/m at x=0.5
        self.assertIsNone(ex.lane_width_px_at(0))   # above the calibrated ground range

    def test_two_lines_center_on_the_floor(self):
        cl = extractor().extract(*scene())
        self.assertEqual(cl.source, "two_lines", cl.reject_reason)
        self.assertLess(np.abs(lateral(cl.path)).max(), 0.04)

    def test_one_line_uses_spec_width_with_no_measurement(self):
        for kw in ({"right": False}, {"left": False}):
            ex = extractor()
            cl = ex.extract(*scene(**kw))
            self.assertIsNone(ex.hw)
            self.assertEqual(cl.source, "one_line", cl.reject_reason)
            self.assertLess(np.abs(lateral(cl.path)).max(), 0.06, kw)

    def test_two_lines_that_are_not_one_lane_apart(self):
        cl = extractor().extract(*scene(spacing=1.8))
        self.assertNotEqual(cl.source, "two_lines")


class StopLineTests(unittest.TestCase):
    def test_found_with_bumper_distance(self):
        ex = extractor()
        p_line, _ = scene(stop_x=1.0)
        stop = detect_stop_line(ex.lines(p_line), ex.lane_width_px_at)
        self.assertIsNotNone(stop)
        self.assertLess(stop.tilt_deg, 5.0)
        # the near edge sits about half a tape width before the center line at x=1.0
        self.assertAlmostEqual(bumper_distance(PROJ, stop, BUMPER_X), 1.0 - BUMPER_X, delta=0.06)

    def test_lane_still_found_after_cutting_the_stop_line(self):
        ex = extractor()
        p_line, mat = scene(stop_x=1.0)
        stop = detect_stop_line(ex.lines(p_line), ex.lane_width_px_at)
        cl = ex.extract(p_line, mat, exclude=stop.mask)
        self.assertEqual(cl.source, "two_lines", cl.reject_reason)
        self.assertLess(np.abs(lateral(cl.path)).max(), 0.05)

    def test_no_stop_line_on_a_plain_lane(self):
        ex = extractor()
        self.assertIsNone(detect_stop_line(ex.lines(scene()[0]), ex.lane_width_px_at))

    def test_stroke_much_longer_than_a_lane_is_rejected(self):
        ex = extractor()
        line = np.zeros((IN_H, IN_W), np.float32)
        line[60:63, 5:315] = 0.95
        self.assertIsNone(detect_stop_line(ex.lines(line), ex.lane_width_px_at))

    def test_tilted_stroke_is_rejected(self):
        ex = extractor()
        line = np.zeros((IN_H, IN_W), np.uint8)
        w = int(ex.lane_width_px_at(80))
        cv2.line(line, (160 - w // 3, 80 + w // 3), (160 + w // 3, 80 - w // 3), 1, 3)
        self.assertIsNone(detect_stop_line(ex.lines(line.astype(np.float32)), ex.lane_width_px_at))

    def test_isolated_stroke_of_the_right_length_is_rejected(self):
        ex = extractor()
        line = np.zeros((IN_H, IN_W), np.uint8)
        w = int(ex.lane_width_px_at(80))
        line[80:83, 160 - w // 2:160 + w // 2] = 1   # no lane lines at its ends
        self.assertIsNone(detect_stop_line(ex.lines(line.astype(np.float32)), ex.lane_width_px_at))

    def test_stroke_running_off_the_image_is_rejected(self):
        ex = extractor()
        line = np.zeros((IN_H, IN_W), np.uint8)
        w = int(ex.lane_width_px_at(80))
        line[80:83, 0:w] = 1                           # starts at the left border
        line[40:120, w:w + 3] = 1                      # a line meeting its right end
        self.assertIsNone(detect_stop_line(ex.lines(line.astype(np.float32)), ex.lane_width_px_at))

    def test_stroke_at_the_bottom_edge_is_rejected(self):
        ex = extractor()
        line = np.zeros((IN_H, IN_W), np.uint8)
        w = int(ex.lane_width_px_at(110))
        x0 = 160 - w // 2
        line[117:120, x0:x0 + w] = 1
        line[60:120, x0 - 3:x0] = 1
        line[60:120, x0 + w:x0 + w + 3] = 1
        self.assertIsNone(detect_stop_line(ex.lines(line.astype(np.float32)), ex.lane_width_px_at))

    def test_nothing_reported_without_a_width_estimate(self):
        line = scene(stop_x=1.0)[0]
        ex = CenterlineExtractor(IN_W, IN_H)   # no prior, nothing measured
        self.assertIsNone(detect_stop_line(ex.lines(line), ex.lane_width_px_at))

    def test_pipeline_reports_it(self):
        cfg = dict(CFG)
        cfg["calibration"] = {"image_points": IMG, "ground_points": GND}
        maps = [scene(stop_x=1.0)] * 6
        pipe = Pipeline(cfg, MockBackend(maps))
        r = [pipe.step(FRAME, i / 24.0) for i in range(6)][-1]
        self.assertTrue(r.stop_line_visible)
        self.assertAlmostEqual(r.stop_line_bumper_m, 1.0 - BUMPER_X, delta=0.06)
        self.assertEqual(r.mode, "metric_pure_pursuit")
        self.assertEqual(r.reject_reason, "")


class UncalibratedStopLineTest(unittest.TestCase):
    def test_not_reported_without_calibration(self):
        maps = [scene()] * 6 + [scene(stop_x=1.0)] * 4
        pipe = Pipeline(CFG, MockBackend(maps))
        results = [pipe.step(FRAME, i / 24.0) for i in range(len(maps))]
        self.assertEqual(pipe.mode, "image_baseline")
        self.assertFalse(any(r.stop_line_visible for r in results))
        self.assertTrue(all(r.stop_line_bumper_m is None for r in results))


class CornerTests(unittest.TestCase):
    """Corner radii from the PPT with the vehicle numbers of the hybrid_drive README."""

    def setUp(self):
        self.info = Pipeline.corner_check(CFG["vehicle"], CFG["track"])

    def test_steering_lock_limit(self):
        self.assertAlmostEqual(min_turn_radius(0.32, math.radians(30)), 0.5543, places=3)

    def test_lane_center_is_tighter_than_the_lock_allows(self):
        self.assertFalse(self.info["lane_center_reachable"])
        self.assertAlmostEqual(self.info["required_steer_deg_at_lane_center"], 32.6, delta=0.1)
        self.assertGreater(self.info["required_steer_deg_at_lane_center"], CFG["vehicle"]["max_steer_deg"])

    def test_a_narrow_window_exists_outside_the_lane_center(self):
        self.assertTrue(self.info["window_exists"])
        self.assertAlmostEqual(self.info["min_radius_m"], 0.554, delta=0.002)
        self.assertAlmostEqual(self.info["max_radius_m"], 0.594, delta=0.003)

    def test_window_closes_for_a_longer_car(self):
        r_min, r_max = corner_window(0.50, math.radians(30), 0.14, 0.55, 0.825)
        self.assertLess(r_max, r_min)

    def test_required_steer_matches_radius(self):
        self.assertAlmostEqual(math.tan(required_steer(0.32, 0.554)), 0.32 / 0.554)


class TwinLiteNetAdapterTests(unittest.TestCase):
    def test_contract_with_a_stand_in_model(self):
        import torch

        class Fake(torch.nn.Module):
            def forward(self, x):
                self.seen = x
                n, _, h, w = x.shape
                da = torch.zeros(n, 2, h, w)
                da[:, 1] = 5.0
                ll = torch.zeros(n, 2, h, w)
                ll[:, 1, :, : w // 2] = 5.0
                ll[:, 0, :, w // 2:] = 5.0
                return da, ll

        fake = Fake()
        backend = TwinLiteNetBackend(fake)
        img = np.zeros((IN_H, IN_W, 3), np.uint8)
        img[...] = (10, 20, 30)   # BGR
        p_line, p_mat = backend.predict(img)
        np.testing.assert_allclose(fake.seen[0, :, 0, 0].numpy(), np.array([30, 20, 10]) / 255.0, atol=1e-6)
        self.assertEqual(tuple(fake.seen.shape), (1, 3, 240, 640))
        self.assertEqual(p_line.shape, (IN_H, IN_W))
        self.assertEqual(p_mat.shape, (IN_H, IN_W))
        self.assertGreater(p_mat.min(), 0.9)                      # drivable head -> mat
        self.assertGreater(p_line[:, : IN_W // 2 - 4].min(), 0.9)  # lane head -> line
        self.assertLess(p_line[:, IN_W // 2 + 4:].max(), 0.1)


REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "TwinLiteNet")


@unittest.skipUnless(os.path.exists(os.path.join(REPO, "pretrained", "best.pth")),
                     "official TwinLiteNet checkout not present")
class TwinLiteNetRealWeightsTests(unittest.TestCase):
    """Loads the official weights through the adapter. Skipped without the checkout."""

    @classmethod
    def setUpClass(cls):
        cls.backend = TwinLiteNetBackend.from_repo(REPO, os.path.join(REPO, "pretrained", "best.pth"))

    def test_parameter_count_matches_the_published_model(self):
        n = sum(p.numel() for p in self.backend.model.parameters())
        self.assertEqual(n, 439633)   # 0.44 M, as reported in the paper

    def test_outputs_are_probability_maps_at_the_input_size(self):
        img = np.random.default_rng(0).integers(0, 255, (IN_H, IN_W, 3), dtype=np.uint8)
        ll, da = self.backend.predict(img)
        for m in (ll, da):
            self.assertEqual(m.shape, (IN_H, IN_W))
            self.assertTrue(np.isfinite(m).all())
            self.assertGreaterEqual(float(m.min()), 0.0)
            self.assertLessEqual(float(m.max()), 1.0)

    def test_input_order_matters(self):
        # The model expects RGB; feeding the same pixels as BGR must give a different answer.
        img = np.random.default_rng(1).integers(0, 255, (IN_H, IN_W, 3), dtype=np.uint8)
        a = self.backend.predict(img)[1]
        b = self.backend.predict(np.ascontiguousarray(img[:, :, ::-1]))[1]
        self.assertGreater(float(np.abs(a - b).mean()), 1e-3)


if __name__ == "__main__":
    unittest.main()
