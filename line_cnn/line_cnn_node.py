#!/usr/bin/env python3
"""ROS2 perception node: 2.OAK -> line_cnn -> 5.controller -> /commands/*.

Subscribes /oak/rgb/image_raw and /oak/rgb/camera_info.
Publishes /line_cnn/target_angle_deg (+ = left), /line_cnn/confidence, /line_cnn/state,
/line_cnn/processing_ms, /line_cnn/stop_line_visible and /line_cnn/stop_line_distance_m
(front bumper to the stop line, NaN until the camera is calibrated).
Subscribes /line_cnn/enable (Bool, default true): false stops inference and reports confidence 0,
so a mission layer can hand a zone (tunnel, cones, guide line) to the LiDAR planner.
It never publishes /commands/*; the controller owns actuation.

python3 line_cnn_node.py --ros-args -p model:=/path/linenet.onnx -p config:=/path/default.yaml
"""
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String

from perception import create_backend
from pipeline import Pipeline, load_config
from tracking import PathState

USABLE = (PathState.VALID, PathState.LOW_CONFIDENCE, PathState.TEMP_MISSING)


def to_bgr(msg):
    """Avoids cv_bridge and its NumPy version conflicts on Jetson."""
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    a = a[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        return a
    if msg.encoding == "rgb8":
        return a[:, :, ::-1]
    raise ValueError(f"unsupported encoding: {msg.encoding}")


class LineCnnNode(Node):
    def __init__(self):
        super().__init__("line_cnn")
        here = os.path.dirname(os.path.abspath(__file__))
        self.declare_parameter("model", os.path.join(here, "weights", "linenet.onnx"))
        self.declare_parameter("config", os.path.join(here, "configs", "default.yaml"))
        self.cfg = load_config(self.get_parameter("config").value)
        self.backend = create_backend(self.cfg["perception"], self.get_parameter("model").value)
        self.get_logger().info(
            f"backend={self.backend.name} providers={self.backend.sess.get_providers()}")
        self.pipe = None
        self.size_warned = False
        self.t_last_image = None   # stays None until the first frame arrives
        self.pub_angle = self.create_publisher(Float32, "/line_cnn/target_angle_deg", 10)
        self.pub_conf = self.create_publisher(Float32, "/line_cnn/confidence", 10)
        self.pub_ms = self.create_publisher(Float32, "/line_cnn/processing_ms", 10)
        self.pub_state = self.create_publisher(String, "/line_cnn/state", 10)
        self.pub_stop_seen = self.create_publisher(Bool, "/line_cnn/stop_line_visible", 10)
        self.pub_stop_dist = self.create_publisher(Float32, "/line_cnn/stop_line_distance_m", 10)
        self.enabled = True
        self.create_subscription(Bool, "/line_cnn/enable", self.on_enable, 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CameraInfo, "/oak/rgb/camera_info", self.on_info, qos)
        self.create_subscription(Image, "/oak/rgb/image_raw", self.on_image, qos)
        self.create_timer(0.1, self.watchdog)

    def on_info(self, m):
        """Build the pipeline from the live intrinsics, so K and D always match the stream."""
        if self.pipe is not None or m.k[0] <= 0.0:
            return
        cam = self.cfg["camera"]
        if (m.width, m.height) != (cam["width"], cam["height"]):
            if not self.size_warned:
                self.get_logger().error(
                    f"camera_info is {m.width}x{m.height} but the config expects "
                    f"{cam['width']}x{cam['height']}; not starting")
                self.size_warned = True
            return
        d = list(m.d) if len(m.d) >= 4 else cam["D"]
        cfg = dict(self.cfg)
        cfg["camera"] = {**cam, "K": list(m.k), "D": d}
        self.pipe = Pipeline(cfg, self.backend)
        self.get_logger().info(
            f"mode={self.pipe.mode} rotate180={cam['rotate180']} fx={m.k[0]:.1f} "
            f"metric_geometry={self.pipe.projector is not None}")

    def on_enable(self, m):
        if self.enabled and not m.data and self.pipe is not None:
            self.pipe.filter.fail()   # drop the old path; re-enabling starts from scratch
        self.enabled = bool(m.data)

    def report_stop_line(self, res):
        self.pub_stop_seen.publish(Bool(data=bool(res.stop_line_visible)))
        d = res.stop_line_bumper_m
        self.pub_stop_dist.publish(Float32(data=float("nan") if d is None else float(d)))

    def report(self, state, angle_deg, conf, ms):
        self.pub_state.publish(String(data=state if isinstance(state, str) else state.value))
        self.pub_conf.publish(Float32(data=float(conf if state in USABLE else 0.0)))   # strings give 0
        self.pub_angle.publish(Float32(data=float(angle_deg)))
        self.pub_ms.publish(Float32(data=float(ms)))

    def watchdog(self):
        """Camera disconnect: tell the controller even though no callback runs."""
        if self.pipe is None:
            return
        limit = self.cfg["safety"]["frame_timeout_s"]
        if not self.enabled:
            return
        if self.t_last_image is None or time.monotonic() - self.t_last_image > limit:
            r = self.pipe.step(None, time.monotonic())
            self.report(r.state, 0.0, 0.0, 0.0)

    def on_image(self, msg):
        if self.pipe is None:   # no bearing without camera_info
            return
        self.t_last_image = time.monotonic()
        if not self.enabled:
            self.report("DISABLED", 0.0, 0.0, 0.0)
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        age = now - stamp if stamp > 0 else 0.0
        r = self.pipe.step(to_bgr(msg), time.monotonic(), frame_age_s=age)
        angle = 0.0 if r.bearing_rad is None or r.stop else math.degrees(r.bearing_rad)
        self.report(r.state, angle, 0.0 if r.stop else r.confidence, r.timing_ms.get("total", 0.0))
        self.report_stop_line(r)


def main():
    rclpy.init()
    node = LineCnnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
