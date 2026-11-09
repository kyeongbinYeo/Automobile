# line_cnn: CNN 라인 트래킹

**현재 상태: 실차 주행 불가.** 라인 분할은 쓸 만하지만, 중심선은 영상에서 라인을 가로지르는 경우가 남아 있습니다.
모든 결과는 영상을 입력으로 한 개방 루프 평가이며, 차량이 포함된 폐루프 검증은 하지 않았습니다.

## 구조

| 파일 | 역할 |
|---|---|
| `model.py`, `train.py`, `export_onnx.py` | LineNet(라인+판 2-헤드) 학습과 변환 |
| `perception.py` | 교체 가능한 추론 백엔드 (ONNX, PyTorch, 테스트 전용 Mock) |
| `geometry.py` | 마스크 정리, 지면 투영(BEV), 중심선 추출 |
| `tracking.py` | 시간 필터와 5단계 상태 |
| `control.py` | Pure Pursuit, 픽셀 기준 조향, 속도 계획, PID, 안전 |
| `pipeline.py`, `visualizer.py`, `main.py` | 전체 연결, 디버그 화면, 실행 |
| `camera.py` | 카메라 K·D, 180도 장착, 높이·기울기 기반 지면 투영 |
| `stopline.py` | 정지선 인식 (지면 투영이 있을 때만) |
| `train_twinlite.py`, `export_twinlite_onnx.py`, `eval_twinlite.py`, `onnx_tag.py` | TwinLiteNet 디코더 미세조정, ONNX 내보내기, 평가, 가중치 종류 표시 |
| `docs/` | 트랙 배치도와 코드 흐름 다이어그램 (`make_diagrams.py`로 재생성) |
| `line_cnn_node.py` | ROS2 노드 (`/commands/*`는 발행하지 않음) |
| `hybrid_drive_integration.md` | 5번 제어기 연결 제안 (미적용) |

조향 부호는 `hybrid_drive`와 같이 왼쪽이 +입니다.

## 실행

Windows에서는 PATH의 `python`이 스토어 스텁이므로 설치된 파이썬으로 가상환경을 만듭니다.

```bash
C:\Users\ejrgu\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

테스트:

```bash
.venv\Scripts\python.exe -m unittest discover -s tests
```

녹색 오버레이가 입혀진 영상을 평가할 때는 `--unoverlay`를 쓰고, 이미 똑바로 저장된 영상에는 `--upright`를 함께 씁니다(그렇지 않으면 카메라가 뒤집혀 달렸다고 보고 180도 돌립니다). 원본 영상이나 카메라에는 `--unoverlay`를 쓰지 않습니다.

```bash
.venv\Scripts\python.exe main.py --source VIDEO.mp4 --model weights\linenet.onnx --unoverlay --upright --out out\run1
```

```bash
.venv\Scripts\python.exe main.py --source 0 --model weights\linenet.onnx --show
```

ROS2:

```bash
python3 line_cnn_node.py --ros-args -p model:=weights/linenet.onnx -p config:=configs/default.yaml
```

학습 데이터 생성과 재학습:

```bash
.venv\Scripts\python.exe prepare_video.py VIDEO.mp4 DATA_DIR
```

```bash
.venv\Scripts\python.exe train.py --data DATA_DIR\real.npz --out runs\x --loss focal_dice --gamma --dark-aug 0.25 --dark-weight 0.5
```

## 대회 규격 반영 (`2026 자율주행 경연대회 트랙 구성.pptx`)

- **트랙:** 검은 매트 위 흰 테이프(폭 3 cm), 테이프 중심 간격 0.68 m(안쪽 0.65 m), 코너 차로 중심 반지름 0.50 m. `configs/default.yaml`의 `track:`에 넣었습니다.
- **차로 폭:** 카메라 보정이 있으면 영상에서 추정하지 않고 규격값을 지면 평면에서 직접 씁니다. 라인이 하나만 보여도 규격 폭의 절반만큼 차 쪽으로 옮겨 중심선을 만들고, 두 라인이 한 차로 간격(±40%)이 아니면 짝으로 인정하지 않습니다.
- **정지선(1번 매트):** 두 차선 사이를 가로지르는 같은 테이프입니다. 보정된 카메라에서만 인식하며 `/line_cnn/stop_line_visible`, `/line_cnn/stop_line_distance_m`(앞범퍼 기준)로 내보냅니다. 보정 없이는 영상 716프레임 중 77프레임에서 차선 가장자리를 정지선으로 잘못 잡아서 꺼 두었습니다. 실제 정지선 영상으로는 검증하지 못했습니다.
- **코너 한계:** 축거 0.32 m, 최대 조향 약 30°이면 최소 회전 반지름이 0.554 m인데 차로 중심 반지름은 0.50 m라서, 차로 중심은 그대로 따라갈 수 없습니다(필요 조향 약 32.6°). 차체 앞 바깥 모서리가 바깥 테이프 안쪽 가장자리(0.825 m)에 닿지 않는 후방 차축 반지름은 약 0.554~0.594 m, 즉 4 cm 폭입니다. 30° 값은 끝단 근사 실측이고 슬립은 무시한 계산이라 실차로 확인해야 합니다. 이 상황은 `steer_saturated`로 표시됩니다.
- **구간별 카메라 끄기:** `/line_cnn/enable`(Bool)로 상위 로직이 카메라 라인을 끄고 LiDAR에 맡길 수 있습니다. 유도선 구간(11~13번)은 콘이 차로 밖으로 이끌기 때문에 "라인을 가로지르는 경로 거부" 규칙과 충돌하고, 터널(15~19번)은 흰 A4 벽이 라인처럼 보일 수 있습니다. 어느 구간에서 끌지는 위치 정보가 있는 상위 로직이 정해야 하며, 아직 연결하지 않았습니다.
- **미구현:** 신호등(빨강/초록) 인식, 콘 인식. 영상에 신호등이 없어 검증할 수 없었습니다.
- **TwinLiteNet:** 아래 "TwinLiteNet 사용" 절을 보세요.
- **다이어그램:** `docs/track_map.svg`, `docs/pipeline.svg`. 매트 번호 배치는 PPT 설명으로 역산한 것이며 모든 슬라이드와 맞지만 PPT 도면 그대로는 아닙니다.

## 카메라와 Drive 데이터 (`자율주행` 폴더 조사)

- **내부 파라미터:** 팀의 `sensors_20260922_181049_analysis/camera_info.json`에서 가져왔습니다 (1280x720, fx 908.9, fy 908.7, cx 619.7, cy 368.8, 왜곡 8계수 rational). 노드는 실행 중 `/oak/rgb/camera_info`로 덮어씁니다.
- **180도 장착:** 팀 코드(`camera_alignment.py`)가 이미 "카메라가 뒤집혀 있다"고 보고 영상을 돌리고 주점도 뒤집습니다. `image_raw`가 원본이면 모델에 똑바로 넣어야 하므로 `camera.rotate180: true`가 기본입니다. 실차에서 `main.py --show`로 확인하세요.
- **장착 자세:** 높이, 기울기, 후방 차축으로부터의 위치는 실측되어 있지 않습니다(팀의 11/02 보고서도 같은 결론). `configs/default.yaml`의 `camera.mount`에 `height_m`, `pitch_deg`, `x_m` 세 값을 넣으면 렌즈 왜곡을 보정한 미터 단위 지오메트리가 자동으로 켜집니다. 바닥 점 4개를 재는 방식보다 간단하고 왜곡도 반영합니다.
- **자세 가이드:** 높이 11 cm에 18° 아래로 향하면 양쪽 차선(±0.34 m)이 화면 밖으로 나가고 위쪽은 잘려서 라인이 보이지 않습니다. 수평에 가까운(0~8°) 자세가 필요합니다.
- **영상:** Drive에는 별도 MP4 주행 영상이 없습니다. 카메라는 rosbag(`.db3`)에만 있고, 팀이 뽑은 `camera_preview.mp4`가 `가공/sensors_*_analysis/` 4곳(8/24, 8/27, 9/15, 9/22, 각 18.7~37.7 MB)에 있습니다. 10월 `drive_debug_*` 기록에는 카메라 이미지가 없습니다(현재 주행 코드는 카메라를 쓰지 않음).
- **Drive 연결 한계:** 파일당 10 MB까지만 받을 수 있어 위 영상과 rosbag은 가져오지 못했습니다. 사용자가 직접 내려받아야 합니다.
- **TwinLiteNet:** Drive에는 소스가 없어 공식 GitHub 저장소(`chequanghuy/TwinLiteNet`, MIT)에서 `third_party/`로 받았습니다.

## TwinLiteNet 사용

공식 TwinLiteNet(0.44 M 파라미터, 주행 가능 영역과 차선 두 헤드)을 `third_party/TwinLiteNet/`에 받았고, 가중치 `weights/twinlite_ft.onnx`는 이 영상의 약한 라벨로 디코더를 미세조정한 것입니다.

- **왜 필요했나:** 영상 속 TwinLiteNet 오버레이는 "주행 가능 영역" 헤드뿐이었습니다. 이 헤드는 매트를 채우고 테이프 위는 비웁니다. 차선 헤드는 쓰이지 않았습니다.
- **선택:** `configs/default.yaml`의 `perception.backend`(`linenet` 또는 `twinlite`) 또는 `main.py --backend twinlite --model weights/twinlite_ft.onnx`. 기본값은 LineNet입니다.
- **비교(12프레임, 직접 찍은 라벨):** LineNet은 밝은 구간 0.78, 터널 0.74이고 이 PC에서 1 ms입니다. TwinLiteNet 미세조정은 밝은 구간 0.89, 터널 0.48이고 54 ms입니다. 영상 1개의 12프레임이라 차이를 확정적으로 볼 수는 없습니다.
- **잘못된 가중치 방지:** 두 ONNX는 입출력 모양이 같지만 정규화가 달라서, 다른 쪽 파일을 넣으면 오류 없이 틀린 결과가 나옵니다. ONNX에 종류 표시(`onnx_tag.py`)를 넣었고 불러올 때 확인합니다.
- **CPU 주의:** PyTorch CPU에서는 TwinLiteNet이 장당 약 0.84초(팽창 합성곱이 느림)이고 ONNX Runtime에서는 약 54 ms입니다. 학습이 느려서 인코더는 고정하고 디코더만 학습했습니다.
- **다시 만들기:**
  `python train_twinlite.py --data DATA_DIR\real.npz --repo ..\third_party\TwinLiteNet --out runs\tw`, 그다음 `python export_twinlite_onnx.py ..\third_party\TwinLiteNet weights\twinlite_ft.pt weights\twinlite_ft.onnx`.
- **모델 규약:** 공식 코드 확인 결과 640x360 입력, BGR을 RGB로 바꿔 255로 나누기, 평균·표준편차 정규화 없음입니다. 이 프로젝트는 320x120 크롭을 그대로 넣을 때 사전학습 상태에서도 점수가 높았습니다(0.67 대 0.50).
- 공식 저장소 README는 후속 모델 TwinLiteNetPlus가 공개됐다고 안내합니다. 아직 써 보지 않았습니다.

## 설정

`configs/default.yaml`을 수정합니다. 차량 값은 `hybrid_drive` V14.7에서 가져왔고, `safe default`와 `uncalibrated`로 표시된 값은 실측값이 아닙니다.

BEV와 미터 단위 Pure Pursuit를 켜려면 `calibration`에 대응점을 4개 이상 넣습니다. 바닥에 표시한 점의 입력 영상 좌표(320x120)와 뒤차축 기준 실측 좌표(x 전방, y 왼쪽, m)입니다. 재투영 오차가 3 cm를 넘으면 시작 시 거부됩니다. 비워 두면 픽셀 기준 조향으로 동작하고 미터 값은 내지 않습니다.

## 측정 결과

입력은 제공된 영상 1개입니다. 이 PC의 CPU 1스레드 기준이며 Jetson 수치가 아닙니다.

라인 검출 (직접 라벨을 찍은 12프레임, 학습 제외 구간):

| 모델 | 밝은 구간 F1 | 터널 F1 |
|---|---|---|
| `line_tracker.py` (이전 고전 방식) | 0.03 | 0.07 |
| LineNet 첫 학습 (BCE) | 0.72 | 0.37 |
| LineNet 개선판 (현재 가중치) | 0.78 | 0.74 |
| TwinLiteNet 사전학습 그대로 (320x120) | 0.67 | 0.00 |
| TwinLiteNet 디코더 미세조정 | 0.89 | 0.48 |

노출 변화 (디지털로 밝기를 조정해 흉내 낸 값, 현재 가중치):

| 노출 배율 | 밝은 구간 F1 | 터널 F1 |
|---|---|---|
| 0.25 | 0.73 | 0.68 |
| 0.5 | 0.78 | 0.72 |
| 1.0 | 0.78 | 0.74 |
| 1.5 | 0.79 | 0.73 |
| 2.0 | 0.47 | 0.72 |

영상 716프레임 상태 분포: VALID 352, LOW_CONFIDENCE 97, TEMP_MISSING 243, LOST 24, FAILURE 0.

중심선 전후 비교 (`center_eval.py`, 직접 찍은 라인 라벨로 만든 대리 지표):

| 항목 | 이전 (행별 x(y)) | 현재 (보로노이 경계) |
|---|---|---|
| 경로가 있는 라벨 프레임 | 8/12 | 12/12 |
| 그중 라벨 라인을 가로지른 프레임 | 4 | 3 |
| 가운데 정렬 (0이 정중앙, 두 라인 라벨 프레임) | 0.61 (3프레임) | 0.41 (5프레임) |
| VALID 프레임 | 182 | 354 |
| LOST 프레임 | 155 | 24 |
| 목표점 이동 p99 | 47 px | 158 px (악화) |
| 조향 절댓값 0.8 초과 프레임 | 86 | 109 (악화) |

이 12프레임을 보면서 알고리즘과 임계값을 조정했으므로 독립 평가가 아닙니다. VALID 증가도 정확도 향상을 뜻하지 않습니다.

지연 (ms):

| 단계 | p50 | p99 |
|---|---|---|
| 전처리 | 2.6 | 5.5 |
| CNN 추론 | 5.6 | 11.1 |
| 중심선 | 8.9 | 21.7 |
| 제어 | 0.2 | 0.5 |
| 합계 | 17.4 | 38.0 |

오버레이 제거(약 110 ms)는 이 영상에만 필요한 단계라 합계에서 뺐습니다. 디버그 영상 저장을 포함한 처리 속도는 5.6 FPS, 최대 메모리는 182 MB입니다. 파라미터는 13,042개입니다.

자동 테스트 94개가 통과합니다 (`center_eval.py`는 별도 실행). 시나리오 테스트는 합성 마스크로 기하, 추적, 제어 로직만 확인하며 CNN 정확도나 주행 성능과는 무관합니다.

측정하지 못한 항목: 폐루프 추종 오차, 완주율, 차선 이탈 횟수, Jetson 지연과 GPU 사용률, 판 IoU와 중심선 오차(정답 없음).

## 알려진 한계

- **중심선:** 밝은 구간에서는 대체로 차로 안쪽에 놓이지만, 틀린 프레임이 남아 있습니다. 700번은 판 밖 바닥을 지나고, 터널의 595번은 먼 라인을 가로지르며, 255번은 한쪽 라인에 치우칩니다. 터널은 라인 한쪽을 자주 놓쳐 특히 불안정합니다.
- **조향 안정성:** 이전 방식보다 목표점이 크게 튀고, 프레임 간 조향 변화 평균도 커졌습니다(0.038 → 0.074). 손으로 움직인 영상 탓도 있지만 분리해 측정하지 못했습니다. TEMP_MISSING 프레임의 상당수는 추출 실패가 아니라 경로 점프 거부나 낮은 신뢰도에서 나옵니다.
- **차로 폭:** 영상에서 잰 반폭을 재사용합니다. 평평한 바닥을 가정한 근사이고 측정 전에는 라인 하나로 경로를 만들지 않습니다.
- **판 헤드:** 700번처럼 밝은 바닥을 판으로 잡는 경우가 있어, 판 위 검사만으로 걸러지지 않습니다.
- **데이터:** 영상 1개, 학습 351프레임입니다. 라벨 대부분이 약한 라벨이고, 평가 프레임도 같은 영상의 다른 구간입니다. 다른 조명이나 트랙은 확인하지 않았습니다.
- **과노출:** 노출 2배에서 밝은 구간 F1이 0.47로 떨어집니다.
- **미실행 코드:** BEV와 Pure Pursuit는 합성 데이터로만 시험했고, 실제 카메라 보정값이 없습니다. PID는 단위 테스트만 했고 파이프라인에는 연결되어 있지 않습니다. ROS2 노드와 제어기 연결은 실행해 보지 않았습니다.
- **TwinLiteNet:** 미세조정은 디코더만 했고(인코더 고정), 학습 프레임이 351개이며 라벨은 약한 라벨입니다. 터널에서는 LineNet보다 약합니다(0.48 대 0.74). Jetson 속도는 측정하지 못했습니다.
- **차로 구분:** 라인이 차로 경계만 있다고 가정합니다. 395번의 코너 박스처럼 다른 흰 선이 있으면 어느 두 라인이 차로인지 구분하지 못합니다.
- **`../line_tracker.py`:** 비교용으로만 남겨 둔 이전 코드입니다. 이 트랙에서 동작하지 않고 조향 부호도 반대(오른쪽이 +)이므로 쓰지 마세요.
