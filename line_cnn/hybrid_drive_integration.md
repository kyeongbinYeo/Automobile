# 5번 제어기(hybrid_drive.py V14.7) 연결 방법

> 이 문서는 받은 zip의 V14.7 단일 파일 기준입니다. 현재 선택본은 경로 계획기, 주행 제어기, 안전 게이트의 3개 프로세스 (`/hybrid_drive/plan`, `state`, `command`, `applied`)이고 그 코드는 받지 못했습니다. 그 구조에서는 제안 위치가 다음과 같습니다.
> - `/line_cnn/target_angle_deg`, `confidence`: ① 경로 계획기가 후보 평가에 차선 힌트로 넣습니다. 기존 안전 검증을 통과한 후보만 채택하는 구조를 그대로 둡니다.
> - `/line_cnn/stop_line_*`: ② 주행 제어기가 감속과 정지 요청으로 받고, ③ 안전 게이트의 최종 승인을 거칩니다.
> - `/line_cnn/enable`: 구간별로 카메라를 끄고 LiDAR에 맡길 때 씁니다.
> 현재 선택본 코드를 받으면 이 문서를 다시 쓰겠습니다. 아래 내용은 V14.7 기준이며 적용하지 않은 제안입니다.

아직 적용하지 않은 제안입니다. `hybrid_drive.py`를 수정하지 않았고, 이 코드는 실행해 보지 않았습니다.
기본값이 꺼짐(`line_follow_enable: false`)이므로 켜기 전에는 기존 동작이 그대로입니다.

원칙:
- LiDAR 정지 판단(`selected_path_blocked`)은 조향 계산보다 앞에 있으므로 그대로 유지됩니다.
- 라인 방향에 LiDAR 여유가 없거나, 라인 정보가 오래됐거나, 신뢰도가 낮으면 LiDAR 경로로 돌아갑니다.
- `last_target_angle`은 덮어쓰지 않습니다. LiDAR planner의 내부 상태가 라인 값으로 오염되지 않게 하기 위해서입니다.

## 1. import, 구독 (`__init__`)

```python
from std_msgs.msg import Float32   # top of file

# declare with the existing gp pattern and add to the YAML
self.line_follow_enable = bool(gp('line_follow_enable'))   # default false
self.line_conf_min = float(gp('line_conf_min'))            # start at 0.40
self.line_timeout = float(gp('line_timeout_sec'))          # 0.20
self.line_angle_rad = 0.0
self.line_conf = 0.0
self.last_line_wall = -1.0
self._line_clear_version = -1
self._line_clearances = None
self.create_subscription(Float32, '/line_cnn/target_angle_deg', self.line_angle_callback, sensor_qos)
self.create_subscription(Float32, '/line_cnn/confidence', self.line_conf_callback, sensor_qos)
```

## 2. 메서드 추가

```python
def line_angle_callback(self, msg):
    self.line_angle_rad = math.radians(float(msg.data))
    self.last_line_wall = time.monotonic()

def line_conf_callback(self, msg):
    self.line_conf = float(msg.data)

def select_target_angle(self):
    lidar = self.last_target_angle
    if (not self.line_follow_enable
            or time.monotonic() - self.last_line_wall > self.line_timeout
            or self.line_conf < self.line_conf_min):
        return lidar
    lookahead, half_width, stop, _ = self.dynamic_safety_values()
    if self._line_clear_version != self.scan_version:   # once per new scan
        self._line_clearances = self.all_candidate_clearances(lookahead, half_width)
        self._line_clear_version = self.scan_version
    idx = int(np.argmin(np.abs(self.candidate_angles - self.line_angle_rad)))
    # take the worse of the straight corridor and the curved body sweep
    # (curved_candidate_clearance passes candidates under 5 deg unchecked)
    clearance = min(float(self._line_clearances[idx]),
                    float(self.curved_candidate_clearance(idx, lookahead, half_width)))
    if clearance <= stop + self.selected_path_stop_margin:
        return lidar   # line direction blocked: obstacle avoidance wins
    return self.line_angle_rad
```

## 3. `compute_auto_steering` 첫 줄 변경

```python
target_angle = self.select_target_angle()   # was: self.last_target_angle
```

## 첫 시험 순서
1. `line_follow_enable: false`로 주행하며 `/line_cnn/*` 값만 기록합니다.
2. 바퀴를 띄우고 켜서 라인이 왼쪽으로 치우칠 때 왼쪽으로 조향되는지 부호를 확인합니다.
3. 넓은 곳에서 저속으로, 버튼 8+9 dead-man을 잡은 채 시험합니다.
