"""Camera intrinsics (including the upside-down mount) and a pose-based ground projector.

Frames: the sensor image is rotated 180 degrees when rotate180 is set. "Upright" is the image the
model sees (after rotating). Model-input pixels are the 320x120 crop of the upright image.
Vehicle frame: x forward, y left, z up; origin on the ground below the rear axle.
"""
import math

import cv2
import numpy as np

from common import CROP_TOP, IN_H, IN_W


class CameraModel:
    """Model-input pixel <-> undistorted, upright normalized ray (x right, y down, z forward)."""

    def __init__(self, K, D, width, height, rotate180=False):
        self.K = np.asarray(K, float).reshape(3, 3)
        self.D = np.asarray(D, float).ravel()
        self.W, self.H = int(width), int(height)
        self.rotate180 = bool(rotate180)
        self.y0 = int(self.H * CROP_TOP)
        self.sx = IN_W / self.W
        self.sy = IN_H / (self.H - self.y0)

    @classmethod
    def from_config(cls, c):
        return cls(c["K"], c["D"], c["width"], c["height"], c.get("rotate180", False))

    def upright(self, frame):
        """The image the model should see."""
        return cv2.rotate(frame, cv2.ROTATE_180) if self.rotate180 else frame

    def input_to_upright(self, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        return np.column_stack([(p[:, 0] + 0.5) / self.sx - 0.5,
                                (p[:, 1] + 0.5) / self.sy - 0.5 + self.y0])

    def upright_to_input(self, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        return np.column_stack([(p[:, 0] + 0.5) * self.sx - 0.5,
                                (p[:, 1] - self.y0 + 0.5) * self.sy - 0.5])

    def _flip(self, p):
        return np.column_stack([self.W - 1 - p[:, 0], self.H - 1 - p[:, 1]]) if self.rotate180 else p

    def principal_input(self):
        """Principal point in model-input pixels: where the optical axis lands."""
        up = self._flip(np.array([[self.K[0, 2], self.K[1, 2]]]))
        return self.upright_to_input(up)[0]

    def input_to_normalized(self, pts):
        """(N,2) normalized coordinates in the upright camera frame. NaN where undistortion fails."""
        raw = self._flip(self.input_to_upright(pts))
        n = cv2.undistortPoints(raw.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        return -n if self.rotate180 else n

    def normalized_to_input(self, xy):
        n = np.asarray(xy, float).reshape(-1, 2)
        n_sensor = -n if self.rotate180 else n
        pts3 = np.column_stack([n_sensor, np.ones(len(n_sensor))]).reshape(-1, 1, 3)
        raw, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), self.K, self.D)
        return self.upright_to_input(self._flip(raw.reshape(-1, 2)))

    def bearing_rad(self, pts):
        """Horizontal angle of the ray, + = left of the optical axis."""
        return -np.arctan(self.input_to_normalized(pts)[:, 0])


class CameraGroundProjector:
    """Ground projection from the camera pose. Same interface as geometry.GroundProjector.

    The pose is measured (height, pitch, mount position), not fitted, so nothing here depends on
    floor markings. Pitch is positive when the camera looks down. Roll is about the optical axis.
    """

    def __init__(self, camera, height_m, pitch_deg, x_m, y_m=0.0, roll_deg=0.0, yaw_deg=0.0):
        if height_m <= 0:
            raise ValueError("camera height must be positive")
        self.camera = camera
        th, ps, ph = (math.radians(v) for v in (pitch_deg, yaw_deg, roll_deg))
        z = np.array([math.cos(th) * math.cos(ps), math.cos(th) * math.sin(ps), -math.sin(th)])
        x0 = np.array([math.sin(ps), -math.cos(ps), 0.0])
        y0 = np.cross(z, x0)
        x = math.cos(ph) * x0 + math.sin(ph) * y0
        y = -math.sin(ph) * x0 + math.cos(ph) * y0
        self.R = np.vstack([x, y, z])                       # vehicle -> camera
        self.C = np.array([x_m, y_m, height_m], float)
        self.params = dict(height_m=height_m, pitch_deg=pitch_deg, x_m=x_m, y_m=y_m,
                           roll_deg=roll_deg, yaw_deg=yaw_deg)
        m = np.column_stack([self.R[:, 0], self.R[:, 1], -self.R @ self.C])   # ground(x,y,1) -> ray
        self._to_ground = np.linalg.inv(m)

    def image_to_ground(self, pts):
        """Model-input pixels -> ground (x, y) in meters. NaN for rays that never reach the floor."""
        n = self.camera.input_to_normalized(pts)
        ray = np.column_stack([n, np.ones(len(n))])
        g = ray @ self._to_ground.T
        out = g[:, :2] / g[:, 2:3]
        up_z = (ray @ self.R)[:, 2]            # z of the ray in the vehicle frame
        out[up_z >= 0.0] = np.nan              # at or above the horizon
        return out

    def ground_to_image(self, pts):
        g = np.asarray(pts, float).reshape(-1, 2)
        pc = (np.column_stack([g, np.zeros(len(g))]) - self.C) @ self.R.T
        front = pc[:, 2] > 1e-6
        n = np.full((len(g), 2), np.nan)
        n[front] = pc[front, :2] / pc[front, 2:3]
        out = np.full((len(g), 2), np.nan)
        if front.any():
            out[front] = self.camera.normalized_to_input(n[front])
        return out

    def warp_bev(self, image, x_range_m, y_half_m, ppm):
        """Top-down view of the model-input image; vehicle ahead is up."""
        w, h = int(2 * y_half_m * ppm), int(x_range_m * ppm)
        cols, rows = np.meshgrid(np.arange(w), np.arange(h))
        ground = np.column_stack([(h - rows.ravel()) / ppm, (w / 2.0 - cols.ravel()) / ppm])
        px = self.ground_to_image(ground).reshape(h, w, 2).astype(np.float32)
        return cv2.remap(image, px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderValue=0)
