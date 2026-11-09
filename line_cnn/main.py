#!/usr/bin/env python3
"""Offline/live runner. Simulation-only: outputs go to a MockVehicle, never to hardware.

python main.py --source video.mp4 --model weights/linenet.onnx --out out/run1 [--unoverlay]
python main.py --source 0 --model weights/linenet.onnx --show
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from control import create_vehicle
from perception import create_backend
from pipeline import Pipeline, load_config
from visualizer import draw

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["frame", "t", "state", "mode", "confidence", "steer_norm", "target_speed", "stop",
          "stop_reason", "reject_reason", "lateral_error", "lateral_unit", "curvature",
          "curvature_unit", "bearing_deg", "preprocess_ms", "inference_ms", "geometry_ms",
          "control_ms", "total_ms"]


def percentiles(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return None
    return {"mean": round(float(v.mean()), 3), "p50": round(float(np.percentile(v, 50)), 3),
            "p99": round(float(np.percentile(v, 99)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or camera index")
    ap.add_argument("--model", required=True, help=".onnx or .pt")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "default.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", choices=["linenet", "twinlite"],
                    help="overrides perception.backend; --model must be the matching weights")
    ap.add_argument("--upright", action="store_true",
                    help="source frames are already upright (skip the 180 degree rotation)")
    ap.add_argument("--unoverlay", action="store_true",
                    help="source has the TwinLiteNet green overlay burned in; remove it first")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.upright:
        cfg["camera"]["rotate180"] = False
    if a.backend:
        cfg["perception"]["backend"] = a.backend
    backend = create_backend(cfg["perception"], a.model, a.threads)
    pipe = Pipeline(cfg, backend)
    vehicle = create_vehicle(cfg["vehicle"]["hardware_enable"])

    live = a.source.isdigit()
    cap = cv2.VideoCapture(int(a.source) if live else a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source}")
    if live:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # no unbounded frame queue
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    unoverlay = None
    if a.unoverlay:
        from prepare_video import unoverlay

    writer, csv_f, csv_w = None, None, None
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        csv_f = open(os.path.join(a.out, "telemetry.csv"), "w", newline="", encoding="utf-8")
        csv_w = csv.DictWriter(csv_f, FIELDS)
        csv_w.writeheader()

    try:
        import psutil
        proc = psutil.Process()
        proc.cpu_percent(None)
    except ImportError:
        proc = None

    stats = {k: [] for k in ("acquire", "preprocess", "inference", "geometry", "control", "total")}
    states, steers, conf, failed_reads = {}, [], [], 0
    n, peak_rss, wall0 = 0, 0, time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if not live:
                break
            failed_reads += 1
            frame = None
            if failed_reads > 30:
                break
        else:
            failed_reads = 0
            if unoverlay is not None:
                frame = unoverlay(frame)[0]
        acquire_ms = (time.perf_counter() - t0) * 1e3
        t = time.monotonic() if live else n / src_fps

        r = pipe.step(frame, t)
        if r.stop:
            vehicle.stop()
        else:
            vehicle.command(r.steer_norm, r.target_speed)

        stats["acquire"].append(acquire_ms)
        for k, v in r.timing_ms.items():
            stats[k].append(v)
        states[r.state.value] = states.get(r.state.value, 0) + 1
        steers.append(r.steer_norm)
        conf.append(r.confidence)
        rss = proc.memory_info().rss if proc else 0
        peak_rss = max(peak_rss, rss)

        if csv_w:
            csv_w.writerow({
                "frame": n, "t": round(t, 3), "state": r.state.value, "mode": r.mode,
                "confidence": round(r.confidence, 3), "steer_norm": round(r.steer_norm, 4),
                "target_speed": round(r.target_speed, 3), "stop": r.stop,
                "stop_reason": r.stop_reason, "reject_reason": r.reject_reason,
                "lateral_error": r.lateral_error, "lateral_unit": r.lateral_unit,
                "curvature": r.curvature, "curvature_unit": r.curvature_unit,
                "bearing_deg": None if r.bearing_rad is None else round(np.degrees(r.bearing_rad), 2),
                **{f"{k}_ms": round(r.timing_ms.get(k, 0.0), 2) for k in
                   ("preprocess", "inference", "geometry", "control", "total")}})
        if a.out or a.show:
            elapsed = time.perf_counter() - wall0
            shown = None if frame is None else pipe.upright(frame)
            vis = draw(shown, r, (n + 1) / max(elapsed, 1e-6), pipe.projector, cfg["bev_view"])
            if a.out and not a.no_video:
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if a.show:
                cv2.imshow("line_cnn", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    vehicle.stop()
    cap.release()
    if writer:
        writer.release()
    if csv_f:
        csv_f.close()

    wall = time.perf_counter() - wall0
    d = np.abs(np.diff(steers)) if len(steers) > 1 else np.zeros(1)
    summary = {
        "evaluation_type": "open-loop perception and controller output (no vehicle in the loop)",
        "mode": pipe.mode, "backend": backend.name, "frames": n,
        "states": states, "confidence_mean": round(float(np.mean(conf)), 3) if conf else None,
        "steer_abs_step_mean": round(float(d.mean()), 4), "steer_abs_step_p99": round(float(np.percentile(d, 99)), 4),
        "latency_ms": {k: percentiles(v) for k, v in stats.items()},
        "fps_processed_including_debug_output": round(n / max(wall, 1e-6), 1),
        "process_peak_rss_mb": round(peak_rss / 2 ** 20, 1) if proc else None,
        "process_cpu_percent": round(proc.cpu_percent(None), 1) if proc else None,
        "not_available": ["closed-loop tracking error", "track completion", "lane departures",
                          "Jetson latency", "drivable-area IoU (no ground truth)",
                          "centerline error (no ground truth)"],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if a.out:
        with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
