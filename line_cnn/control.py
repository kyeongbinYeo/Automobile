"""Steering, speed and safety. Steering sign matches hybrid_drive: + = left."""
import math
from abc import ABC, abstractmethod

import numpy as np

from tracking import PathState


class PurePursuit:
    """Metric controller. Use only with a calibrated GroundProjector."""

    def __init__(self, wheelbase_m, max_steer_rad):
        if wheelbase_m <= 0 or max_steer_rad <= 0:
            raise ValueError("wheelbase and max steering angle must be measured positive values")
        self.L = wheelbase_m
        self.max_steer = max_steer_rad

    def steer(self, target_x, target_y):
        """Returns (delta_rad saturated, curvature 1/m, alpha_rad)."""
        ld = math.hypot(target_x, target_y)
        if ld < 1e-6:
            return 0.0, 0.0, 0.0
        alpha = math.atan2(target_y, target_x)
        kappa = 2.0 * math.sin(alpha) / ld
        delta = math.atan(self.L * kappa)
        return float(np.clip(delta, -self.max_steer, self.max_steer)), kappa, alpha


def min_turn_radius(wheelbase_m, max_steer_rad):
    """Smallest rear-axle turn radius of a bicycle model at full steering lock."""
    return wheelbase_m / math.tan(max_steer_rad)


def required_steer(wheelbase_m, radius_m):
    """Steering angle needed for a steady turn of the given rear-axle radius."""
    return math.atan(wheelbase_m / radius_m)


def corner_window(wheelbase_m, max_steer_rad, half_width_m, front_x_m, outer_edge_radius_m):
    """Rear-axle radii (r_min, r_max) at which a steady corner turn fits.

    r_min: steering lock. r_max: the outer front corner of the body reaches the inner edge of
    the outer lane tape (outer_edge_radius_m). Returns r_max < r_min when no radius fits.
    Bicycle model, no slip, outer front corner at (front_x_m, half_width_m) from the rear axle.
    """
    r_min = min_turn_radius(wheelbase_m, max_steer_rad)
    room = outer_edge_radius_m ** 2 - front_x_m ** 2
    r_max = math.sqrt(room) - half_width_m if room > 0 else float("-inf")
    return r_min, r_max


def lookahead_distance(speed, base_m, gain_s, min_m, max_m):
    return float(np.clip(base_m + gain_s * abs(speed), min_m, max_m))


def select_target(coeffs, ld, step=0.02, x_max=4.0):
    """First point on y(x) at distance >= ld. Returns (x, y, reached)."""
    xs = np.arange(step, x_max, step)
    ys = np.polyval(coeffs, xs)
    d = np.hypot(xs, ys)
    idx = np.flatnonzero(d >= ld)
    if idx.size:
        i = int(idx[0])
        return float(xs[i]), float(ys[i]), True
    return float(xs[-1]), float(ys[-1]), False


def image_baseline_steer(x_target, cx, kp):
    """Uncalibrated pixel-error baseline. Output is normalized steering, not an angle."""
    return float(np.clip(-kp * (x_target - cx) / max(cx, 1.0), -1.0, 1.0))


class SteeringLimiter:
    """Normalized steering rate limit. Stops bypass it."""

    def __init__(self, rate_per_s):
        self.rate = rate_per_s
        self.value = 0.0

    def step(self, target, dt):
        d = self.rate * max(dt, 0.0)
        self.value += float(np.clip(target - self.value, -d, d))
        return self.value

    def reset(self):
        self.value = 0.0


class SpeedPlanner:
    """Target speed from curvature, steering demand, confidence, error and lookahead."""

    STATE_SCALE = {PathState.VALID: 1.0, PathState.LOW_CONFIDENCE: 0.5,
                   PathState.TEMP_MISSING: 0.3, PathState.LOST: 0.0, PathState.FAILURE: 0.0}

    def __init__(self, cruise, min_speed, max_speed, curve_gain=0.6, steer_gain=0.4,
                 error_gain=0.3, rise=1.0, fall=1.8):
        if max_speed < cruise:
            raise ValueError("cruise exceeds the maximum speed")
        self.cruise, self.min_speed, self.max_speed = cruise, min_speed, max_speed
        self.curve_gain, self.steer_gain, self.error_gain = curve_gain, steer_gain, error_gain
        self.rise, self.fall = rise, fall
        self.value = 0.0

    def raw(self, state, curv_norm, steer_abs, conf, err_norm, lookahead_ratio):
        scale = self.STATE_SCALE[state]
        if scale == 0.0:
            return 0.0
        f = (1 - self.curve_gain * min(curv_norm, 1.0)) \
            * (1 - self.steer_gain * min(steer_abs, 1.0)) \
            * (1 - self.error_gain * min(err_norm, 1.0)) \
            * float(np.clip(lookahead_ratio, 0.3, 1.0)) \
            * float(np.clip(0.5 + conf, 0.5, 1.0))
        return float(np.clip(self.cruise * f * scale, self.min_speed * scale, self.max_speed))

    def step(self, state, curv_norm, steer_abs, conf, err_norm, lookahead_ratio, dt):
        target = self.raw(state, curv_norm, steer_abs, conf, err_norm, lookahead_ratio)
        if target == 0.0:
            self.value = 0.0   # LOST/FAILURE ignore the ramp
            return 0.0
        d = self.value + float(np.clip(target - self.value, -self.fall * dt, self.rise * dt))
        self.value = d
        return d


class PID:
    """Closed-loop speed PID with integral clamping and output limits.

    Requires real velocity feedback. Video runs have none, so they report the target speed only.
    """

    def __init__(self, kp, ki, kd, out_min, out_max, i_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_limit = i_limit
        self.reset()

    def reset(self):
        self.i = 0.0
        self.prev = None

    def step(self, target, measured, dt):
        e = target - measured
        d = 0.0 if self.prev is None or dt <= 0 else (e - self.prev) / dt
        self.prev = e
        unsat = self.kp * e + self.ki * self.i + self.kd * d
        out = float(np.clip(unsat, self.out_min, self.out_max))
        if out == unsat or (e > 0) != (unsat > 0):   # anti-windup
            self.i = float(np.clip(self.i + e * dt, -self.i_limit, self.i_limit))
        return out


class VehicleInterface(ABC):
    @abstractmethod
    def command(self, steer_norm, speed_mps): ...

    @abstractmethod
    def stop(self): ...


class MockVehicle(VehicleInterface):
    """Simulation-only sink; never touches hardware."""

    def __init__(self):
        self.last = (0.0, 0.0)
        self.stops = 0

    def command(self, steer_norm, speed_mps):
        self.last = (steer_norm, speed_mps)

    def stop(self):
        self.last = (0.0, 0.0)
        self.stops += 1


def create_vehicle(hardware_enable):
    """Actuation on the real car stays in the existing controller node (hybrid_drive)."""
    if hardware_enable:
        raise NotImplementedError(
            "direct hardware control is not implemented; publish /line_cnn/* and let the "
            "existing controller own /commands/*")
    return MockVehicle()


class SafetySupervisor:
    """E-stop latch, command saturation and frame-age timeout."""

    def __init__(self, max_speed, frame_timeout_s=0.3):
        self.max_speed = max_speed
        self.frame_timeout = frame_timeout_s
        self.estopped = False
        self.reason = ""

    def estop(self, reason="manual"):
        self.estopped = True
        self.reason = reason

    def filter(self, state, steer, speed, frame_age_s=0.0):
        """Returns (steer, speed, stop_flag, reason)."""
        if self.estopped:
            return 0.0, 0.0, True, f"estop:{self.reason}"
        if frame_age_s > self.frame_timeout:
            return 0.0, 0.0, True, "frame_timeout"
        if state in (PathState.LOST, PathState.FAILURE):
            return 0.0, 0.0, True, state.value.lower()
        return float(np.clip(steer, -1.0, 1.0)), float(np.clip(speed, 0.0, self.max_speed)), False, ""
