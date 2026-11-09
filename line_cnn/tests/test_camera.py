"""Camera model and pose-based ground projection, checked against OpenCV's own projection."""
import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from camera import CameraGroundProjector, CameraModel  # noqa: E402
from common import IN_H, IN_W  # noqa: E402
from geometry import CenterlineExtractor  # noqa: E402
from perception import MockBackend  # noqa: E402
from pipeline import Pipeline, load_config  # noqa: E402

CFG = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "default.yaml"))
# Tests start from an unsurveyed camera; each test sets the pose it needs.
CFG["camera"]["mount"].update(height_m=None, pitch_deg=None, x_m=None)
CAM = CFG["camera"]
POSE = dict(height_m=0.11, pitch_deg=18.0, x_m=0.30, y_m=0.02, roll_deg=0.0, yaw_deg=1.5)
GRID = np.array([[u, v] for u in (5, 80, 160, 240, 315) for v in (3, 30, 60, 90, 117)], float)


def model(rotate180):
    return CameraModel(CAM["K"], CAM["D"], CAM["width"], CAM["height"], rotate180)


class CameraModelTests(unittest.TestCase):
    def test_pixel_roundtrip_through_undistortion(self):
        for flip in (False, True):
            m = model(flip)
            back = m.normalized_to_input(m.input_to_normalized(GRID))
            self.assertLess(np.abs(back - GRID).max(), 0.05, flip)

    def test_principal_point_is_the_optical_axis(self):
        for flip in (False, True):
            m = model(flip)
            n = m.input_to_normalized(m.principal_input()[None])[0]
            self.assertLess(np.abs(n).max(), 1e-4, flip)

    def test_flip_moves_the_principal_point_as_the_team_did(self):
        # camera_alignment.py sets cx' = W-1-cx for the upright image
        up = model(True).principal_input()[0]
        expect = ((CAM["width"] - 1 - CAM["K"][2]) + 0.5) * IN_W / CAM["width"] - 0.5
        self.assertAlmostEqual(up, expect, places=6)
        self.assertNotAlmostEqual(up, model(False).principal_input()[0], places=1)

    def test_left_of_the_axis_is_positive(self):
        m = model(True)
        p = m.principal_input()
        left, right = p - [30, 0], p + [30, 0]
        b = m.bearing_rad(np.array([left, right]))
        self.assertGreater(b[0], 0)
        self.assertLess(b[1], 0)
        self.assertAlmostEqual(b[0], -b[1], delta=0.02)

    def test_bearing_matches_the_focal_length(self):
        m = model(False)
        p = m.principal_input()
        du = 30 / m.sx / 4.0 * 4.0   # 30 model pixels in full-frame pixels
        expected = math.atan(30 / m.sx / CAM["K"][0])
        self.assertAlmostEqual(-m.bearing_rad((p + [30, 0])[None])[0], expected, delta=0.005)
        self.assertGreater(du, 0)


class GroundProjectorTests(unittest.TestCase):
    def projector(self, flip=True, **over):
        return CameraGroundProjector(model(flip), **{**POSE, **over})

    def opencv_reference(self, proj, ground_xy):
        """Independent projection: OpenCV projectPoints with the sensor-frame rotation."""
        cam = proj.camera
        R = proj.R.copy()
        if cam.rotate180:
            R = np.diag([-1.0, -1.0, 1.0]) @ R      # sensor frame is the upright frame turned 180 deg
        rvec, _ = cv2.Rodrigues(R)
        tvec = -R @ proj.C
        world = np.column_stack([ground_xy, np.zeros(len(ground_xy))])
        raw, _ = cv2.projectPoints(world, rvec, tvec, cam.K, cam.D)
        raw = raw.reshape(-1, 2)
        up = np.column_stack([cam.W - 1 - raw[:, 0], cam.H - 1 - raw[:, 1]]) if cam.rotate180 else raw
        return cam.upright_to_input(up)

    def test_matches_opencv_projection(self):
        ground = np.array([[x, y] for x in (0.5, 0.8, 1.2, 2.0) for y in (-0.4, -0.1, 0.0, 0.2, 0.4)])
        for flip in (False, True):
            p = self.projector(flip)
            ours = p.ground_to_image(ground)
            ref = self.opencv_reference(p, ground)
            ok = np.isfinite(ours).all(1) & (ours[:, 0] > -50) & (ours[:, 0] < IN_W + 50)
            self.assertGreater(ok.sum(), 10)
            self.assertLess(np.abs(ours[ok] - ref[ok]).max(), 0.05, flip)

    def test_roundtrip(self):
        p = self.projector()
        ground = np.array([[0.6, 0.0], [0.9, 0.3], [1.5, -0.3], [2.5, 0.1]])
        px = p.ground_to_image(ground)
        np.testing.assert_allclose(p.image_to_ground(px), ground, atol=2e-3)

    def test_optical_axis_hits_the_floor_where_geometry_says(self):
        p = self.projector(yaw_deg=0.0, y_m=0.0)
        hit = p.image_to_ground(p.camera.principal_input()[None])[0]
        self.assertAlmostEqual(hit[0], POSE["x_m"] + POSE["height_m"] / math.tan(math.radians(POSE["pitch_deg"])), places=3)
        self.assertAlmostEqual(hit[1], 0.0, places=4)

    def test_left_in_the_image_is_left_on_the_floor(self):
        p = self.projector(yaw_deg=0.0, y_m=0.0)
        c = p.camera.principal_input()
        g = p.image_to_ground(np.array([c - [40, 0], c + [40, 0]]))
        self.assertGreater(g[0, 1], 0)
        self.assertLess(g[1, 1], 0)

    def test_rays_above_the_horizon_are_nan(self):
        p = self.projector(pitch_deg=0.0)
        g = p.image_to_ground(np.array([[160.0, 2.0], [160.0, 115.0]]))
        self.assertTrue(np.isnan(g[0]).all())
        self.assertTrue(np.isfinite(g[1]).all())

    def test_nearer_floor_is_lower_in_the_image(self):
        p = self.projector()
        v = p.ground_to_image(np.array([[0.6, 0.0], [1.0, 0.0], [2.0, 0.0]]))[:, 1]
        self.assertTrue(v[0] > v[1] > v[2])

    def test_rejects_a_bad_height(self):
        with self.assertRaises(ValueError):
            self.projector(height_m=0.0)


class PipelineUsesTheCameraTests(unittest.TestCase):
    def cfg(self, **mount):
        cfg = dict(CFG)
        cfg["camera"] = {**CAM, "mount": {**CAM["mount"], **mount}}
        return cfg

    def test_image_mode_until_the_pose_is_measured(self):
        pipe = Pipeline(self.cfg(), MockBackend([]))
        self.assertIsNone(pipe.projector)
        self.assertEqual(pipe.mode, "image_baseline")

    def test_metric_mode_once_the_pose_is_given(self):
        pipe = Pipeline(self.cfg(height_m=0.11, pitch_deg=18.0, x_m=0.30), MockBackend([]))
        self.assertIsInstance(pipe.projector, CameraGroundProjector)
        self.assertEqual(pipe.mode, "metric_pure_pursuit")

    def test_partial_pose_does_not_enable_metric_mode(self):
        pipe = Pipeline(self.cfg(height_m=0.11, pitch_deg=18.0), MockBackend([]))
        self.assertIsNone(pipe.projector)

    def test_lanes_drawn_from_the_floor_give_a_centered_path(self):
        # Nearly level and 15 cm up, like the sample footage. A camera pitched far down sees the
        # lanes only outside the field of view or above the crop, which is a mounting problem.
        cfg = self.cfg(height_m=0.15, pitch_deg=5.0, x_m=0.30, y_m=0.0, yaw_deg=0.0)
        pipe = Pipeline(cfg, MockBackend([]))
        proj, spacing = pipe.projector, cfg["track"]["lane_center_spacing_m"]
        x = np.linspace(0.9, 2.2, 30)
        line = np.zeros((IN_H, IN_W), np.uint8)
        for side in (1, -1):
            px = proj.ground_to_image(np.column_stack([x, np.full_like(x, side * spacing / 2)]))
            cv2.polylines(line, [np.round(px).astype(np.int32)], False, 1, 3)
        mat = np.full((IN_H, IN_W), 0.95, np.float32)
        ex = CenterlineExtractor(IN_W, IN_H)
        ex.set_lane_prior(proj, spacing)
        cl = ex.extract(line.astype(np.float32) * 0.95, mat)
        self.assertEqual(cl.source, "two_lines", cl.reject_reason)
        g = proj.image_to_ground(cl.path)
        g = g[np.isfinite(g).all(1)]
        self.assertLess(np.abs(g[:, 1]).max(), 0.04)


if __name__ == "__main__":
    unittest.main()
