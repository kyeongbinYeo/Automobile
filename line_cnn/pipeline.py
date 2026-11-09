"""Camera frame -> segmentation -> centerline -> tracking -> steering/speed -> safety."""
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml

from camera import CameraGroundProjector, CameraModel
from common import IN_H, IN_W, crop_resize
from control import (PurePursuit, SafetySupervisor, SpeedPlanner, SteeringLimiter,
                     corner_window, image_baseline_steer, lookahead_distance,
                     min_turn_radius, required_steer, select_target)
from geometry import (CenterlineExtractor, GroundProjector, ground_curvature, ground_path,
                      turning_angle)
from perception import validate_prediction
from stopline import bumper_distance, detect_stop_line
from tracking import PathFilter, PathState


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class StepResult:
    state: PathState
    mode: str                    # "metric_pure_pursuit" or "image_baseline"
    steer_norm: float = 0.0      # + = left
    target_speed: float = 0.0
    stop: bool = False
    stop_reason: str = ""
    confidence: float = 0.0
    reject_reason: str = ""
    target_px: Optional[tuple] = None
    bearing_rad: Optional[float] = None    # + = left of the optical axis, from K and D
    lateral_error: Optional[float] = None
    lateral_unit: str = ""
    curvature: Optional[float] = None
    curvature_unit: str = ""
    lookahead_m: Optional[float] = None
    path_xy: Optional[np.ndarray] = None
    steer_saturated: bool = False          # required curvature exceeds the steering lock
    stop_line_visible: bool = False
    stop_line_row: Optional[float] = None
    stop_line_bumper_m: Optional[float] = None   # front bumper to near edge, calibrated only
    stop_line_mask: Optional[np.ndarray] = None
    p_line: Optional[np.ndarray] = None
    p_mat: Optional[np.ndarray] = None
    timing_ms: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg, backend):
        self.cfg = cfg
        self.backend = backend
        self.camera = CameraModel.from_config(cfg["camera"])
        self.cx = float(self.camera.principal_input()[0])   # image column of the optical axis
        self.projector = self._build_projector(cfg)
        self.mode = "metric_pure_pursuit" if self.projector else "image_baseline"
        v = cfg["vehicle"]
        self.max_steer = math.radians(v["max_steer_deg"])
        self.pp = PurePursuit(v["wheelbase_m"], self.max_steer)
        self.kappa_max = math.tan(self.max_steer) / v["wheelbase_m"]
        self.limiter = SteeringLimiter(v["steer_rate_norm_per_s"])
        s = cfg["speed"]
        self.speed = SpeedPlanner(s["cruise_mps"], s["min_mps"], s["max_mps"],
                                  rise=s["rise_mps2"], fall=s["fall_mps2"])
        self.safety = SafetySupervisor(s["max_mps"], cfg["safety"]["frame_timeout_s"])
        self.extractor = CenterlineExtractor(IN_W, IN_H)
        tr = cfg["track"]
        if self.projector is not None:
            self.extractor.set_lane_prior(self.projector, tr["lane_center_spacing_m"])
        self.bumper_x_m = v["wheelbase_m"] + v["front_overhang_m"]
        self.corner = self.corner_check(v, tr)
        t = cfg["tracking"]
        self.filter = PathFilter(t["ema_alpha"], t["conf_ok"], t["conf_min"],
                                 t["jump_px"], t["jump_reject_frames"], t["missing_timeout_s"])
        self.t_prev = None

    def _build_projector(self, cfg):
        """Measured camera pose first, then floor correspondences, else none (image mode)."""
        m = cfg["camera"].get("mount") or {}
        if all(m.get(k) is not None for k in ("height_m", "pitch_deg", "x_m")):
            return CameraGroundProjector(self.camera, m["height_m"], m["pitch_deg"], m["x_m"],
                                         m.get("y_m", 0.0), m.get("roll_deg", 0.0), m.get("yaw_deg", 0.0))
        cal = cfg["calibration"]
        if cal.get("image_points") and cal.get("ground_points"):
            return GroundProjector.from_points(cal["image_points"], cal["ground_points"])
        return None

    def upright(self, frame):
        """Raw camera frame -> the image the model sees (rotated if the camera is upside down)."""
        return self.camera.upright(frame)

    @staticmethod
    def corner_check(v, tr):
        """Can the car steer the corner of the track? Numbers from the config only."""
        L, lock = v["wheelbase_m"], math.radians(v["max_steer_deg"])
        r_lane = tr["corner_radius_center_m"]
        outer_edge = tr["corner_radius_outer_tape_m"] - tr["tape_width_m"] / 2.0
        r_min, r_max = corner_window(L, lock, v["width_m"] / 2.0, L + v["front_overhang_m"],
                                     outer_edge)
        return {"lane_center_radius_m": r_lane, "min_radius_m": r_min, "max_radius_m": r_max,
                "required_steer_deg_at_lane_center": math.degrees(required_steer(L, r_lane)),
                "lane_center_reachable": r_lane >= r_min, "window_exists": r_max >= r_min}

    def _fail(self, reason, timing):
        self.filter.fail()
        self.limiter.reset()
        self.speed.value = 0.0
        return StepResult(PathState.FAILURE, self.mode, stop=True, stop_reason=reason,
                          reject_reason=reason, timing_ms=timing)

    def step(self, frame_bgr, t, measured_speed=0.0, frame_age_s=0.0):
        """frame_bgr=None marks a camera dropout."""
        timing = {}
        dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0)
        self.t_prev = t
        if frame_bgr is None:
            return self._fail("camera_failure", timing)

        t0 = time.perf_counter()
        inp = crop_resize(self.upright(frame_bgr))
        t1 = time.perf_counter()
        timing["preprocess"] = (t1 - t0) * 1e3
        try:
            p_line, p_mat = self.backend.predict(inp)
        except Exception as e:   # any backend error must end in a safe stop
            return self._fail(f"inference_failure:{type(e).__name__}", timing)
        t2 = time.perf_counter()
        timing["inference"] = (t2 - t1) * 1e3
        if not validate_prediction(p_line, p_mat):
            return self._fail("invalid_prediction", timing)

        thr = self.cfg["perception"]["threshold"]
        try:
            # Only with a calibrated ground plane: a lane width measured from earlier frames is
            # too unreliable to separate a stop line from a lane edge (77 false hits in 716
            # frames of the sample video without calibration).
            stop = None
            if self.projector is not None:
                stop = detect_stop_line(self.extractor.lines(p_line, thr),
                                        self.extractor.lane_width_px_at)
            cl = self.extractor.extract(p_line, p_mat, thr,
                                        exclude=None if stop is None else stop.mask)
        except Exception as e:   # a geometry bug must also end in a safe stop
            return self._fail(f"geometry_failure:{type(e).__name__}", timing)
        state = self.filter.update(cl, t)
        t3 = time.perf_counter()
        timing["geometry"] = (t3 - t2) * 1e3

        res = StepResult(state, self.mode, confidence=cl.confidence, reject_reason=cl.reject_reason,
                         p_line=p_line, p_mat=p_mat)
        if stop is not None:
            res.stop_line_visible, res.stop_line_row = True, stop.row
            res.stop_line_mask = stop.mask
            if self.projector is not None:
                res.stop_line_bumper_m = bumper_distance(self.projector, stop, self.bumper_x_m)
        steer_target = 0.0
        curv_norm = err_norm = 0.0
        la_ratio = 1.0
        if self.filter.path is not None:
            res.path_xy = self.filter.path
            if self.projector is not None:
                steer_target, curv_norm, err_norm, la_ratio = self._metric(res, measured_speed)
            else:
                steer_target, curv_norm, err_norm = self._image(res)

        steer = self.limiter.step(steer_target, dt)
        spd = self.speed.step(state, curv_norm, abs(steer), cl.confidence, err_norm, la_ratio, dt)
        res.steer_norm, res.target_speed, res.stop, res.stop_reason = self.safety.filter(
            state, steer, spd, frame_age_s)
        if res.stop:
            self.limiter.reset()
            self.speed.value = 0.0
        t4 = time.perf_counter()
        timing["control"] = (t4 - t3) * 1e3
        timing["total"] = (t4 - t0) * 1e3
        res.timing_ms = timing
        return res

    def _image(self, res):
        c = self.cfg["image_baseline"]
        path = self.filter.path
        origin = np.array([self.cx, float(IN_H)])
        d = np.linalg.norm(path - origin, axis=1)
        idx = np.flatnonzero(d >= c["lookahead_px"])
        x, y = path[int(idx[0])] if idx.size else path[-1]
        res.target_px = (float(x), float(y))
        res.lateral_error = float(path[int(np.argmin(d)), 0] - self.cx)
        res.lateral_unit = "px"
        res.curvature = turning_angle(path)
        res.curvature_unit = "rad over path"
        res.bearing_rad = float(self.camera.bearing_rad([[x, y]])[0])
        steer = image_baseline_steer(x, self.cx, c["kp"])
        curv_norm = res.curvature / c["turn_ref_rad"]
        return steer, curv_norm, min(abs(res.lateral_error) / self.cx, 1.0)

    def _metric(self, res, speed):
        c = self.cfg["pure_pursuit"]
        path = self.filter.path
        coeffs = ground_path(self.projector, path[:, 1], path[:, 0], np.ones(len(path)))
        if coeffs is None:
            res.reject_reason = "path_outside_calibrated_area"
            return 0.0, 1.0, 1.0, 0.3
        ld = lookahead_distance(speed, c["lookahead_base_m"], c["lookahead_speed_gain_s"],
                                c["lookahead_min_m"], c["lookahead_max_m"])
        x, y, reached = select_target(coeffs, ld)
        delta, kappa, alpha = self.pp.steer(x, y)
        res.lookahead_m = ld
        res.bearing_rad = alpha
        res.curvature = ground_curvature(coeffs, x)
        res.curvature_unit = "1/m"
        res.lateral_error = float(np.polyval(coeffs, 0.0))
        res.lateral_unit = "m"
        res.target_px = tuple(self.projector.ground_to_image([[x, y]])[0])
        res.steer_saturated = abs(kappa) > self.kappa_max
        ratio = 1.0 if reached else math.hypot(x, y) / ld
        return delta / self.max_steer, abs(kappa) / self.kappa_max, min(abs(res.lateral_error) / 0.5, 1.0), ratio
