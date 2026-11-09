"""Mask post-processing, ground projection and centerline extraction. No ROS/torch deps."""
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

N_PATH = 12   # path samples, ordered from the vehicle outward


def clean_mask(mask, min_area=25):
    """Drop connected components smaller than min_area."""
    mask = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    keep = st[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab].astype(np.uint8)


class GroundProjector:
    """Planar homography from model-input pixels to the vehicle frame (x forward, y left, meters).

    Only constructed from explicit correspondences; there is no default geometry.
    """

    def __init__(self, H, reprojection_error_m=0.0):
        self.H = np.asarray(H, dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.reprojection_error_m = reprojection_error_m

    @classmethod
    def from_points(cls, image_points, ground_points, max_error_m=0.03):
        img = np.asarray(image_points, dtype=np.float64)
        gnd = np.asarray(ground_points, dtype=np.float64)
        if len(img) < 4 or len(img) != len(gnd):
            raise ValueError("need at least 4 matching image/ground points")
        H, _ = cv2.findHomography(img, gnd, 0)
        if H is None:
            raise ValueError("degenerate calibration points")
        proj = cls(H)
        err = float(np.max(np.linalg.norm(proj.image_to_ground(img) - gnd, axis=1)))
        if err > max_error_m:
            raise ValueError(f"calibration reprojection error {err:.3f} m > {max_error_m} m")
        proj.reprojection_error_m = err
        return proj

    @staticmethod
    def _apply(H, pts):
        p = cv2.perspectiveTransform(np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2), H)
        return p.reshape(-1, 2)

    def image_to_ground(self, pts):
        return self._apply(self.H, pts)

    def ground_to_image(self, pts):
        return self._apply(self.H_inv, pts)

    def warp_bev(self, image, x_range_m, y_half_m, ppm):
        """Top-down view. Returns the image; vehicle at bottom center, forward is up."""
        w, h = int(2 * y_half_m * ppm), int(x_range_m * ppm)
        view = np.array([[0.0, -ppm, w / 2.0], [-ppm, 0.0, float(h)], [0.0, 0.0, 1.0]])
        return cv2.warpPerspective(image, view @ self.H, (w, h))


@dataclass
class Centerline:
    path: Optional[np.ndarray] = None   # (N_PATH, 2) x, y in input pixels, near to far
    source: str = ""                    # "two_lines" or "one_line"
    confidence: float = 0.0
    reject_reason: str = "no_data"

    @property
    def valid(self):
        return self.path is not None


def arc_length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def turning_angle(path):
    """Heading change between the first and last third of the path, in radians."""
    k = max(len(path) // 3, 1)
    a, b = path[k] - path[0], path[-1] - path[-1 - k]
    return float(abs(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])))


class CenterlineExtractor:
    """Lane center as a curve in the image, so paths may run nearly horizontal in corners.

    Two lines: the Voronoi edge between them, kept where both lines lie on opposite sides.
    One line: offset toward the vehicle (a vehicle in its lane is on the lane's side of a line)
    by a half width measured from earlier two-line frames, fitted linearly in the image row as
    on a planar floor. Without a measurement a single line gives no center.
    """

    def __init__(self, width, height=120, min_area=25, min_half_width_px=8.0, min_length_px=25.0,
                 full_length_px=100.0, max_fit_rms_px=8.0, min_mat_frac=0.35, max_line_hits=1,
                 hw_tolerance=0.5, hw_min_frames=3, hw_min_samples=30, hw_min_span_px=15.0,
                 hw_min_path_px=50.0, min_on_mat=0.8):
        self.w, self.h = width, height
        self.min_on_mat = min_on_mat
        self.hw_min_path = hw_min_path_px
        self.origin = np.array([width / 2.0, float(height)])
        self.min_area = min_area
        self.min_half = min_half_width_px
        self.min_len = min_length_px
        self.full_len = full_length_px
        self.max_rms = max_fit_rms_px
        self.min_mat_frac = min_mat_frac
        self.max_line_hits = max_line_hits
        self.hw_tol = hw_tolerance
        self.hw_min_frames = hw_min_frames
        self.hw_min_samples = hw_min_samples
        self.hw_min_span = hw_min_span_px
        self.samples = deque(maxlen=1500)
        self.frames_measured = 0
        self.hw = None   # (slope, intercept) of half width against image row
        self.projector = None
        self.lane_half_m = None          # half the tape-center spacing, from the track spec
        self.lane_tol = 0.4              # allowed relative error of the measured lane half width
        self.ground_max_x = 4.0
        self.prior_width_px = None       # horizontal lane width per image row, from the projector
        self.last_line = np.zeros((height, width), np.uint8)   # cleaned mask of the last call

    def reset(self):
        self.samples.clear()
        self.frames_measured = 0
        self.hw = None

    def set_lane_prior(self, projector, spacing_m):
        """Use the known lane geometry (tape-center spacing) with a calibrated ground plane.

        Lane width then comes from the track spec instead of from earlier frames.
        """
        self.projector = projector
        self.lane_half_m = spacing_m / 2.0
        rows = np.arange(self.h, dtype=float)
        g = projector.image_to_ground(np.column_stack([np.full(self.h, self.w / 2.0), rows]))
        width = np.full(self.h, np.nan)
        ok = (g[:, 0] > 0.05) & (g[:, 0] < self.ground_max_x)
        left = projector.ground_to_image(np.column_stack([g[ok, 0], np.full(ok.sum(), self.lane_half_m)]))
        right = projector.ground_to_image(np.column_stack([g[ok, 0], np.full(ok.sum(), -self.lane_half_m)]))
        width[ok] = np.abs(left[:, 0] - right[:, 0])
        self.prior_width_px = width

    def lane_width_px_at(self, y):
        """Expected horizontal tape-center spacing in pixels at row y, or None if unknown."""
        if self.prior_width_px is not None:
            v = self.prior_width_px[int(np.clip(round(y), 0, self.h - 1))]
            return None if not np.isfinite(v) else float(v)
        hw = self.half_width_at(y)
        return None if hw is None else 2.0 * hw

    def half_width_at(self, y):
        if self.hw is None:
            return None
        v = float(self.hw[0] * y + self.hw[1])
        return v if v >= self.min_half else None

    def _learn_half_width(self, ys, hws):
        self.samples.extend(zip(ys[::2].tolist(), hws[::2].tolist()))
        self.frames_measured += 1
        s = np.array(self.samples)
        if (self.frames_measured >= self.hw_min_frames and len(s) >= self.hw_min_samples
                and s[:, 0].max() - s[:, 0].min() >= self.hw_min_span):
            fit = np.polyfit(s[:, 0], s[:, 1], 1)
            keep = np.abs(np.polyval(fit, s[:, 0]) - s[:, 1]) <= 0.5 * np.median(s[:, 1])
            if keep.sum() >= self.hw_min_samples:
                fit = np.polyfit(s[keep, 0], s[keep, 1], 1)
            self.hw = (float(fit[0]), float(fit[1]))

    def _join_fragments(self, line, max_gap_px=70.0, max_angle_deg=20.0):
        """Bridge pieces of one broken line; otherwise their bisector looks like a lane center."""
        n, comp = cv2.connectedComponents(line)
        ends = []
        for c in range(1, n):
            ys, xs = np.nonzero(comp == c)
            pts = np.column_stack([xs, ys]).astype(float)
            mean = pts.mean(0)
            _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
            t = (pts - mean) @ vt[0]
            k = max(int(len(t) * 0.15), 1)
            order = np.argsort(t)
            ends.append((pts[order[:k]].mean(0), pts[order[-k:]].mean(0), vt[0]))
        cos_lim = np.cos(np.radians(max_angle_deg))
        out = line.copy()
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                pairs = [(p, q) for p in ends[i][:2] for q in ends[j][:2]]
                p, q = min(pairs, key=lambda e: np.linalg.norm(e[0] - e[1]))
                d = q - p
                gap = np.linalg.norm(d)
                if gap < 1.0 or gap > max_gap_px:
                    continue
                u = d / gap
                if abs(u @ ends[i][2]) >= cos_lim and abs(u @ ends[j][2]) >= cos_lim:
                    cv2.line(out, tuple(np.round(p).astype(int).tolist()),
                             tuple(np.round(q).astype(int).tolist()), 1, 2)
        return out

    def _medial_groups(self, line, comp):
        """Voronoi edges per pair of line components: {(a, b): (points, half_widths)}."""
        src = (line == 0).astype(np.uint8)
        dist, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(line)
        n = int(lab.max()) + 1
        lx, ly, lc = np.zeros(n), np.zeros(n), np.zeros(n, int)
        lid = lab[ys, xs]
        lx[lid], ly[lid], lc[lid] = xs, ys, comp[ys, xs]
        near = lc[lab]
        groups = {}
        for dy, dx in ((0, 1), (1, 0)):
            a = near[:self.h - dy, :self.w - dx]
            b = near[dy:, dx:]
            py, px = np.nonzero((a != b) & (a > 0) & (b > 0))
            if py.size == 0:
                continue
            qy, qx = py + dy, px + dx
            la, lb = lab[py, px], lab[qy, qx]
            # Keep points whose two nearest line pixels are well apart in direction. This drops
            # the bisector past the end of the shorter line. The limit is loose because lane
            # lines converge strongly with this low camera: frame 255 has about a 120 degree
            # wedge, which leaves only 60 degrees here.
            v1 = np.stack([lx[la] - px, ly[la] - py], 1).astype(float)
            v2 = np.stack([lx[lb] - qx, ly[lb] - qy], 1).astype(float)
            cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
            hw = (dist[py, px] + dist[qy, qx]) / 2.0
            ok = (cos < 0.55) & (hw >= self.min_half)
            for i in np.flatnonzero(ok):
                key = tuple(sorted((int(lc[la[i]]), int(lc[lb[i]]))))
                # columns: x, y, half width, nearest line pixel (x, y)
                groups.setdefault(key, []).append((px[i], py[i], hw[i], lx[la[i]], ly[la[i]]))
        return {k: np.array(v, float) for k, v in groups.items()}

    def _ground_lane_width(self, comp, a, b, max_points=80):
        """Median nearest distance between two lines on the ground plane, in meters.

        Both lines are projected first; measuring in the image and converting afterwards would
        mix in depth differences, because the nearest pixel is not the nearest point on the floor.
        """
        g = []
        for c in (a, b):
            ys, xs = np.nonzero(comp == c)
            pts = self.projector.image_to_ground(np.column_stack([xs, ys]).astype(float))
            pts = pts[(pts[:, 0] > 0.05) & (pts[:, 0] < self.ground_max_x)]
            if len(pts) < 5:
                return None
            g.append(pts[:: max(len(pts) // max_points, 1)])
        d = np.linalg.norm(g[0][:, None, :] - g[1][None, :, :], axis=2)
        return float(np.median(np.concatenate([d.min(1), d.min(0)])))

    def _fit_path(self, pts):
        """Order along the principal axis, near end first, and fit x(s), y(s).

        Sorting by distance from the vehicle would fold a path that passes across the image.
        Returns (path, rms) or None.
        """
        mean = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        pts = pts[np.argsort((pts - mean) @ vt[0])]
        if np.linalg.norm(pts[-1] - self.origin) < np.linalg.norm(pts[0] - self.origin):
            pts = pts[::-1]
        step = np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1)]
        s = np.cumsum(step)
        if s[-1] < self.min_len:
            return None
        u = s / s[-1]
        deg = min(3, max(1, len(pts) // 4))
        cx, cy = np.polyfit(u, pts[:, 0], deg), np.polyfit(u, pts[:, 1], deg)
        rms = float(np.sqrt(np.mean((np.polyval(cx, u) - pts[:, 0]) ** 2
                                    + (np.polyval(cy, u) - pts[:, 1]) ** 2)))
        t = np.linspace(0, 1, N_PATH)
        return np.column_stack([np.polyval(cx, t), np.polyval(cy, t)]), rms

    def _one_line_points(self, comp, cid):
        """Offset the component's center curve toward the vehicle by half the lane width.

        Calibrated: offset by the specified half spacing on the ground plane, so the result is
        right for any line direction. Uncalibrated: offset in pixels by the measured half width.
        """
        ys, xs = np.nonzero(comp == cid)
        pts = np.column_stack([xs, ys]).astype(float)
        if self.projector is not None:
            space = self.projector.image_to_ground(pts)
            space = space[(space[:, 0] > 0.05) & (space[:, 0] < self.ground_max_x)]
            origin, bin_size = np.zeros(2), 0.04
            if len(space) < 5:
                return None
        else:
            space, origin, bin_size = pts, self.origin, 6.0
        mean = space.mean(0)
        _, _, vt = np.linalg.svd(space - mean, full_matrices=False)
        t = (space - mean) @ vt[0]
        bins = np.floor((t - t.min()) / bin_size).astype(int)
        curve = np.array([space[bins == b].mean(0) for b in np.unique(bins)])
        if len(curve) < 3:
            return None
        tangent = np.gradient(curve, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-6
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        i = int(np.argmin(np.linalg.norm(curve - origin, axis=1)))
        if normal[i] @ (origin - curve[i]) < 0:
            normal = -normal
        if self.projector is not None:
            return self.projector.ground_to_image(curve + normal * self.lane_half_m)
        out = []
        for c, n in zip(curve, normal):
            hw = self.half_width_at(c[1])
            if hw is not None:
                out.append(c + n * hw)
        return np.array(out) if len(out) >= 3 else None

    def _crosses(self, path, line):
        m = np.zeros_like(line)
        cv2.polylines(m, [np.round(path).astype(np.int32)], False, 1, 2)   # no diagonal leaks
        return int((m & line).sum()) >= self.max_line_hits

    def _off_mat(self, path, closed_mat):
        p = np.round(path).astype(int)
        p = p[(p[:, 0] >= 0) & (p[:, 0] < self.w) & (p[:, 1] >= 0) & (p[:, 1] < self.h)]
        return p.size == 0 or closed_mat[p[:, 1], p[:, 0]].mean() < self.min_on_mat

    def lines(self, p_line, thr=0.5):
        """Cleaned binary line mask, as used by extract()."""
        return clean_mask(p_line > thr, self.min_area)

    def extract(self, p_line, p_mat, thr=0.5, exclude=None):
        """exclude: optional mask of pixels to ignore, e.g. a stop line that joins both lanes."""
        line = self.lines(p_line, thr)
        if exclude is not None:
            line[exclude > 0] = 0
            line = clean_mask(line, self.min_area)
        self.last_line = line
        if not line.any():
            return Centerline(reject_reason="no_evidence")
        mat = (p_mat > thr).astype(np.uint8)
        mat_ok = mat[self.h // 2:].mean() >= self.min_mat_frac
        n, comp, st, _ = cv2.connectedComponentsWithStats(line)
        closed_mat = None
        if mat_ok:   # lit scene: a line must lie on or next to the mat
            closed_mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            near_mat = cv2.dilate(mat, np.ones((15, 15), np.uint8))
            for c in range(1, n):
                if near_mat[comp == c].mean() < 0.5:
                    line[comp == c] = 0
            if not line.any():
                return Centerline(reject_reason="no_line_on_mat")
        line = self._join_fragments(line)
        n, comp, st, _ = cv2.connectedComponentsWithStats(line)

        reason = "one_line_no_width"
        best = None
        for (comp_a, comp_b), pts3 in self._medial_groups(line, comp).items():
            if closed_mat is not None:   # the bisector may continue past the mat; keep what is on it
                pts3 = pts3[closed_mat[pts3[:, 1].astype(int), pts3[:, 0].astype(int)] > 0]
                if len(pts3) < 5:
                    reason = "off_mat"
                    continue
            pts = pts3[:, :2]
            if self.projector is not None:
                # The two lines must be one lane apart on the floor, not just any two lines.
                gw = self._ground_lane_width(comp, comp_a, comp_b)
                spacing = 2.0 * self.lane_half_m
                if gw is not None and abs(gw - spacing) > self.lane_tol * spacing:
                    reason = "lane_width_mismatch"
                    continue
            elif self.half_width_at(np.median(pts[:, 1])) is not None:
                expected = self.half_width_at(np.median(pts[:, 1]))
                if abs(np.median(pts3[:, 2]) - expected) > self.hw_tol * expected:
                    reason = "width_mismatch"
                    continue
            fit = self._fit_path(pts)
            if fit is None:
                reason = "too_short"
                continue
            path, rms = fit
            if rms > self.max_rms:
                reason = "fit_residual"
                continue
            if self._crosses(path, line):
                reason = "crosses_line"
                continue
            if closed_mat is not None and self._off_mat(path, closed_mat):
                reason = "off_mat"
                continue
            d = float(np.linalg.norm(path[0] - self.origin))
            if best is None or d < best[0]:
                best = (d, path, pts3)

        if best is not None:
            path, source, quality = best[1], "two_lines", 1.0
            if self.projector is None and arc_length(path) >= self.hw_min_path:
                self._learn_half_width(best[2][:, 1], best[2][:, 2])   # short medials measure poorly
        else:
            order = np.argsort([-st[c, cv2.CC_STAT_AREA] for c in range(1, n)]) + 1
            dists = {int(c): np.min(np.hypot(*(np.nonzero(comp == c)[::-1] - self.origin[:, None])))
                     for c in order[:3]}
            cid = min(dists, key=dists.get)
            known = self.hw is not None or self.projector is not None
            off = self._one_line_points(comp, cid) if known else None
            fit = self._fit_path(off) if off is not None else None
            if fit is None or fit[1] > self.max_rms:
                return Centerline(reject_reason=reason)
            path = fit[0]
            if self._crosses(path, line):
                return Centerline(reject_reason="crosses_line")
            if closed_mat is not None and self._off_mat(path, closed_mat):
                return Centerline(reject_reason="off_mat")
            source, quality = "one_line", 0.6

        if np.any(np.abs(path[:, 0] - self.w / 2.0) > 0.75 * self.w):
            return Centerline(reject_reason="implausible_center")
        p = float(p_line[line > 0].mean())
        conf = min(arc_length(path) / self.full_len, 1.0) * quality * p
        return Centerline(path=path, source=source, confidence=conf, reject_reason="")


def ground_path(projector, rows, xs, weights, max_x_m=4.0):
    """Project centerline samples and fit y(x). Returns coeffs or None."""
    g = projector.image_to_ground(np.column_stack([xs, rows]))
    keep = (g[:, 0] > 0.0) & (g[:, 0] < max_x_m)
    if keep.sum() < 3:
        return None
    return np.polyfit(g[keep, 0], g[keep, 1], 2 if keep.sum() >= 5 else 1, w=np.asarray(weights)[keep])


def ground_curvature(coeffs, x):
    """Signed curvature of y(x) in 1/m (+ = left)."""
    d1 = np.polyval(np.polyder(coeffs), x)
    d2 = np.polyval(np.polyder(coeffs, 2), x) if len(coeffs) > 2 else 0.0
    return float(d2 / (1 + d1 ** 2) ** 1.5)
