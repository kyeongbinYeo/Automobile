#!/usr/bin/env python3
"""Generate the two diagrams as SVG: track map and code flow.

python make_diagrams.py OUT_DIR   # writes track_map.svg and pipeline.svg (self-contained)

Mat numbering is derived from the competition PPT: 30 mats counterclockwise, corners 2, 5, 17, 20,
stop line on mat 1, start on mat 30. It matches every slide (tunnel 15-19 = straight 2, corner 1,
straight 2; cones 7-8; guide 11-13; dynamic obstacle 23-28), but it is a derivation, not a drawing
taken from the PPT.
"""
import math
import os
import sys

STYLE = """<style>
text{font-family:system-ui,'Malgun Gothic',sans-serif}
.t{font-size:14px;fill:#2C2C2A}.ts{font-size:12px;fill:#5F5E5A}.th{font-size:14px;font-weight:500;fill:#2C2C2A}
.box{fill:#F1EFE8;stroke:#B4B2A9}
.arr{fill:none;stroke:#5F5E5A;stroke-width:1.5}
.c-gray>rect{fill:#F1EFE8;stroke:#5F5E5A}.c-gray>.th{fill:#444441}.c-gray>.ts{fill:#5F5E5A}
.c-purple>rect{fill:#EEEDFE;stroke:#534AB7}.c-purple>.th{fill:#3C3489}.c-purple>.ts{fill:#534AB7}
.c-coral>rect{fill:#FAECE7;stroke:#993C1D}.c-coral>.th{fill:#712B13}.c-coral>.ts{fill:#993C1D}
.c-amber>rect{fill:#FAEEDA;stroke:#854F0B}.c-amber>.th{fill:#633806}.c-amber>.ts{fill:#854F0B}
.c-teal>rect{fill:#E1F5EE;stroke:#0F6E56}.c-teal>.th{fill:#085041}.c-teal>.ts{fill:#0F6E56}
</style>"""
MARKER = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
          'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
          'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')

X0, Y0, S = 80, 56, 40   # track origin and mat size in px (1 mat = 1 m)


def mat_cell(m):
    """(col, row) of a mat on the 13 x 4 loop; row 0 is the top (long side B)."""
    if m == 20 or 21 <= m <= 30:
        return (m - 20, 3)
    if m == 1:
        return (11, 3)
    if m == 2:
        return (12, 3)
    if m in (3, 4, 5):
        return (12, {3: 2, 4: 1, 5: 0}[m])
    if 6 <= m <= 16:
        return (17 - m, 0)
    if m == 17:
        return (0, 0)
    return (0, {18: 1, 19: 2}[m])


ZONES = {}
for m in (15, 16, 17, 18, 19):
    ZONES[m] = "purple"
for m in (7, 8, 11, 12, 13):
    ZONES[m] = "coral"
for m in range(23, 29):
    ZONES[m] = "amber"
for m in (30, 1):
    ZONES[m] = "teal"

texts = []   # (x, y, anchor, string) for the overlap check


def text(x, y, s, cls="ts", anchor="middle", central=True):
    texts.append((x, y, anchor, s, cls))
    d = ' dominant-baseline="central"' if central else ""
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}"{d}>{s}</text>'


def track_map():
    o = [f'<svg width="100%" viewBox="0 0 680 330" role="img"><title>대회 트랙 배치도</title>'
         f'<desc>1 m 매트 30장이 13 m x 4 m 직사각형 루프를 반시계로 이룬다. 터널, 라바콘, 유도선, '
         f'동적 장애물, 정지선 구간을 색으로 표시했다.</desc>', MARKER]
    for m in range(1, 31):
        c, r = mat_cell(m)
        x, y = X0 + c * S, Y0 + r * S
        z = ZONES.get(m)
        cls = f' class="c-{z}"' if z else ""
        rect = f'<rect x="{x}" y="{y}" width="{S}" height="{S}" stroke-width="0.5"' + ("" if z else ' class="box"') + "/>"
        # cones sit on top of these mats, so their numbers are left to the zone callouts
        label = "" if m in (7, 8, 11, 12, 13) else text(x + 5, y + 14, m, "ts", "start")
        o.append(f"<g{cls}>{rect}{label}</g>")
    # lane centerline (rounded rectangle, corner radius 0.5 m)
    o.append(f'<rect x="{X0 + 20}" y="{Y0 + 20}" width="{12 * S}" height="{3 * S}" rx="20" fill="none" '
             f'stroke="#888780" stroke-width="1" stroke-dasharray="4 4"/>')
    # auxiliary mats A-C inside the loop, under mats 13, 12, 11
    for i, name in enumerate("ABC"):
        x, y = X0 + (4 + i) * S, Y0 + S
        o.append(f'<g class="c-coral"><rect x="{x}" y="{y}" width="{S}" height="{S}" stroke-width="0.5" '
                 f'stroke-dasharray="3 3"/>{text(x + S // 2, y + 32, name, "ts")}</g>')
    # construction guide: bell curve y = 34 (1 - cos(2 pi x / 300)) cm, 19 cones per side
    def bell(t):
        return 360 - 120 * t, Y0 + 20 + 13.6 * (1 - math.cos(2 * math.pi * t))
    pts = " ".join("%.1f,%.1f" % bell(i / 40) for i in range(41))
    o.append(f'<polyline points="{pts}" fill="none" stroke="#D85A30" stroke-width="1.5"/>')
    for k in range(19):
        x, y = bell(k / 18)
        for dy in (-13.6, 13.6):
            o.append(f'<circle cx="{x:.1f}" cy="{y + dy:.1f}" r="2.2" fill="#D85A30"/>')
    # avoidance cones on mats 7 and 8 (two cones per point, alternating left/right of travel)
    east = {7: X0 + 11 * S, 8: X0 + 10 * S}
    for m, (d25, d75, side25, side75) in {7: (10, 30, +1, -1), 8: (10, 30, +1, -1)}.items():
        for d, side in ((d25, side25), (d75, side75)):
            o.append(f'<circle cx="{east[m] - d}" cy="{Y0 + 20 + side * 11}" r="4" fill="#D85A30"/>')
    # stop line on mat 1 and traffic light below it
    cx1 = X0 + 11 * S + S // 2
    o.append(f'<line x1="{cx1}" y1="{Y0 + 3 * S + 7}" x2="{cx1}" y2="{Y0 + 4 * S - 7}" stroke="#0F6E56" stroke-width="3"/>')
    o.append(f'<rect x="{cx1 - 18}" y="{Y0 + 4 * S + 14}" width="36" height="16" rx="4" fill="#2C2C2A"/>'
             f'<circle cx="{cx1 - 8}" cy="{Y0 + 4 * S + 22}" r="5" fill="#E24B4A"/>'
             f'<circle cx="{cx1 + 8}" cy="{Y0 + 4 * S + 22}" r="5" fill="#639922"/>')
    o.append(text(cx1, Y0 + 4 * S + 44, "신호등", "ts"))
    # zone callouts
    o.append(text(X0 + 2 * S, 42, "꺾이는 터널 15–19", "th"))
    o.append(text(X0 + 5 * S + 20, 42, "유도선 11–13", "th"))
    o.append(text(X0 + 9 * S + 20, 42, "라바콘 7–8", "th"))
    o.append(text(X0 + 6 * S, Y0 + 4 * S + 22, "동적 장애물 RC카 23–28", "th"))
    o.append(text(X0 + 12 * S - 4, Y0 + 2 * S + 20, "출발 30 · 정지선 1", "ts", "end"))
    o.append(text(X0 + 5 * S + 20, Y0 + 3 * S - 14, "반시계 방향", "ts"))
    # legend
    items = [("box", "일반 라인 구간"), ("c-purple", "터널"), ("c-coral", "콘 · 유도선"),
             ("c-amber", "동적 장애물"), ("c-teal", "출발 · 정지선")]
    lx = 40
    for cls, label in items:
        sw = (f'<rect x="{lx}" y="288" width="16" height="16" rx="3" stroke-width="0.5" class="box"/>'
              if cls == "box" else f'<g class="{cls}"><rect x="{lx}" y="288" width="16" height="16" rx="3" stroke-width="0.5"/></g>')
        o.append(sw + text(lx + 22, 296, label, "ts", "start"))
        lx += 22 + len(label) * 11 + 24
    o.append("</svg>")
    return "".join(o)


def pipeline():
    o = ['<svg width="100%" viewBox="0 0 680 480" role="img"><title>코드 흐름과 기존 시스템 연결</title>'
         '<desc>카메라 프레임이 전처리, CNN, 중심선 추출, 상태 추적, 조향 계산을 거쳐 토픽으로 나가고, '
         '점선으로 기존 경로 계획기와 주행 제어기에 연결하는 제안을 나타낸다.</desc>', MARKER]
    left = [("카메라 프레임", "/oak/rgb/image_raw"), ("전처리", "위 40% 자르고 320×120"),
            ("CNN (LineNet)", "라인과 바닥 확률맵"), ("중심선 추출", "두 라인의 가운데 곡선"),
            ("상태 추적", "VALID에서 LOST까지"), ("조향·속도 계산", "각도, 신뢰도, 정지선")]
    ys = [30 + 72 * i for i in range(6)]
    for (title, sub), y in zip(left, ys):
        o.append(f'<g class="c-purple"><rect x="40" y="{y}" width="240" height="48" rx="8" stroke-width="0.5"/>'
                 f'{text(160, y + 17, title, "th")}{text(160, y + 35, sub, "ts")}</g>')
    for y in ys[:-1]:
        o.append(f'<line x1="160" y1="{y + 48}" x2="160" y2="{y + 70}" class="arr" marker-end="url(#arrow)"/>')
    right = [("① 경로 계획기", "LiDAR 스캔으로 계획", 150), ("② 주행 제어기", "50 Hz 경로 실행", 234),
             ("③ 안전 게이트", "AEB와 최종 승인", 318), ("VESC · 서보", "실제 구동", 402)]
    for title, sub, y in right:
        o.append(f'<g class="c-gray"><rect x="400" y="{y}" width="240" height="48" rx="8" stroke-width="0.5"/>'
                 f'{text(520, y + 17, title, "th")}{text(520, y + 35, sub, "ts")}</g>')
    for y in (150, 234, 318):
        o.append(f'<line x1="520" y1="{y + 48}" x2="520" y2="{y + 84 - 14}" class="arr" marker-end="url(#arrow)"/>')
    o.append('<path d="M280 414 L340 414 L340 174 L398 174" fill="none" stroke="#7F77DD" stroke-width="1.5" '
             'stroke-dasharray="5 4" marker-end="url(#arrow)"/>')
    o.append('<path d="M340 258 L398 258" fill="none" stroke="#7F77DD" stroke-width="1.5" '
             'stroke-dasharray="5 4" marker-end="url(#arrow)"/>')
    o.append('<line x1="40" y1="462" x2="70" y2="462" stroke="#7F77DD" stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(text(78, 462, "점선: /line_cnn 토픽을 쓰는 제안 연결 (아직 적용 안 됨)", "ts", "start"))
    o.append("</svg>")
    return "".join(o)


def check_overlaps():
    """Rough text extent check: Korean glyphs ~12 px, Latin ~6.5 px at 12 px (x1.17 at 14 px)."""
    boxes = []
    for x, y, anchor, s, cls in texts:
        k = 14 / 12 if cls in ("t", "th") else 1.0
        w = sum(12 * k if ord(c) > 127 else 6.8 * k for c in str(s))
        x0 = x - w / 2 if anchor == "middle" else (x if anchor == "start" else x - w)
        boxes.append((x0, y - 8, x0 + w, y + 8, s))
    bad = []
    for i, a in enumerate(boxes):
        if a[0] < 0 or a[2] > 680:
            bad.append(("outside", a[4]))
        for b in boxes[i + 1:]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                bad.append((a[4], b[4]))
    return bad


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    for name, fn in (("track_map", track_map), ("pipeline", pipeline)):
        texts.clear()
        svg = fn()
        bad = check_overlaps()
        print(name, "possible text overlaps:", bad or "none")
        standalone = svg.replace('role="img">', 'role="img" xmlns="http://www.w3.org/2000/svg">' + STYLE, 1)
        open(os.path.join(out, name + ".svg"), "w", encoding="utf-8").write(standalone)
        open(os.path.join(out, name + ".widget.svg"), "w", encoding="utf-8").write(svg)
