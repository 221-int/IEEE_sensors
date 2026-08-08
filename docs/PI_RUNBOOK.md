# PI_RUNBOOK — Raspberry Pi 5 실측 절차

> **작성 2026-08-06.** `src/v2/deploy/run_video.py` 실제 CLI 기준.
> 환경 구축 상세는 [`Pi_실행_가이드.md`](Pi_실행_가이드.md), 측정 조건 근거는 [`PI5_BENCHMARK.md`](PI5_BENCHMARK.md).

> 🔵 **2026-08-08 최신 상태:** image_cnn 두 변형의 ONNX export·실행 경로와 실제 Raspberry
> Pi 5 측정이 모두 완료되었다. 아래 절차는 재현용으로 유지하며, 4모드·2해상도 측정과
> G-E1 판정 결과는 논문 문서의 Table II-b에 기록되어 있다.

---

## 0. 🔴 지금 잴 수 있는 것과 없는 것

`run_video.py` 의 현재 CLI:

```
--mode {ours, ear, image_cnn_max, image_cnn_head}
```

| 모드 | 상태 |
|---|---|
| **ours** (encoder + 시간 헤드) | ✅ export·실행 경로 준비 |
| **ear** (규칙 기반 drop_ratio) | ✅ 실행 경로 준비 |
| **image_cnn_max** | ✅ export·실행 경로 준비 |
| **image_cnn_head** | ✅ export·실행 경로 준비 |

**→ 완료된 측정: 네 모드 × (640×480, 1280×720) = 8런.**
학습된 정확도 결과와 분리하기 위해 image_cnn 지연 측정용 export의 가중치와 경고 필드를
런 JSON에서 확인한다.

> 🔵 **측정 완료.** 게이트 **G-E1**(e2e p99 ≤ 33.3 ms)은 실제 Pi 5 측정으로 판정되었다.
> 이 문서는 동일 조건의 재현·결과 회수 절차로 유지한다.

---

## 1. 사전 확인 (PC 에서)

### 1-1. 동치 게이트

`run_video.py` 는 게이트를 통과하지 않으면 **`return 2` 로 실행을 거부한다.**

```bash
python -c "import json;d=json.load(open('results/v2/check_equivalence.json'));print(d['pi_measurement_authorized'])"
# -> True 여야 한다
```

`False` 면 `python -m src.v2.deploy.check_equivalence` 를 먼저 통과시킨다.
**이 파일을 Pi 로 함께 복사해야 한다** (Pi 에서도 같은 검사를 한다).

### 1-2. 720p 측정용 클립 만들기 — **원본은 4.2 GB 라 그대로 못 옮긴다**

`data/raw/mEBAL2/User 1/RealSense/Color_Webcam/color.mp4` 는 **4,235,710,924 바이트**다.
**재인코딩 없이 앞 3분만 잘라낸다** (약 100~200 MB).

```bash
ffmpeg -ss 0 -t 180 -i "data/raw/mEBAL2/User 1/RealSense/Color_Webcam/color.mp4" \
       -c copy -avoid_negative_ts make_zero clips/mebal2_u1_720p_180s.mp4
```

> ⚠️ **`-c copy` 로 스트림 복사한다.** 재인코딩하면 화질이 바뀌어 검출 성능이 달라진다.
> ⚠️ 5분 측정인데 클립이 3분이면 **중간에 끝난다** — `run_video.py` 는 클립을 반복 재생하지
> 않고 `if not ok: break` 로 종료한다. **클립 길이 ≥ 측정 시간**이어야 한다.
> → **`-t 320` 으로 5분 20초를 자르는 것을 권한다.**

640×480 클립은 그대로 쓴다 (9.1 MB):
`data/_legacy_public/eyeblink8/eyeblink8/9/27122013_152435_cam.avi` — 640×480, 30 fps, 5,183프레임(약 2분 53초)

> 🔴 **이 클립도 3분이라 5분 측정에 모자란다.** 두 가지 선택:
> 1. `--duration 170` 으로 맞춘다 (v1 은 5분이었으므로 조건이 달라진다 — **JSON 에 기록**)
> 2. ffmpeg 로 이어붙여 5분 이상으로 만든다 ← **권장**
>
> ```bash
> for i in 1 2; do echo "file '$PWD/clips/eb9.avi'"; done > clips/list.txt
> ffmpeg -f concat -safe 0 -i clips/list.txt -c copy clips/eb9_x2.avi
> ```

---

## 2. Pi 로 옮길 것

```
models/v2/onnx/            encoder.onnx (311 KB) + head.onnx (23 KB)  ← 폴더째
src/v2/                    deploy/, dataset/, model/, common/, __init__.py
results/v2/check_equivalence.json    ← 게이트. 없으면 실행 거부됨
requirements-pi.txt
clips/                     640×480 클립, 720p 클립
```

> ✅ **이번엔 `.onnx.data` 분리가 없다** (`external_data: false`). v1 의 "폴더째 복사" 함정은 사라졌지만
> 그래도 폴더째 옮기는 게 안전하다.

```bash
# PowerShell 에서 (PowerShell 은 외부 명령에 * 를 확장하지 않으니 폴더 단위로)
scp -r models\v2\onnx        hanool@192.168.0.20:~/IEEE_sensors/models/v2/
scp -r src\v2                hanool@192.168.0.20:~/IEEE_sensors/src/
scp results\v2\check_equivalence.json hanool@192.168.0.20:~/IEEE_sensors/results/v2/
scp requirements-pi.txt      hanool@192.168.0.20:~/IEEE_sensors/
scp clips\*                  hanool@192.168.0.20:~/IEEE_sensors/clips/
```

---

## 3. Pi 환경 (T5-6)

```bash
ssh hanool@192.168.0.20        # cat5.local 은 Windows 에서 mDNS 해석 안 됨. IP 사용

conda activate eyeblink        # Python 3.11 필수
python -V                      # -> 3.11.x

pip install -r requirements-pi.txt
python -c "import onnxruntime,mediapipe,cv2,numpy; \
print(onnxruntime.__version__, mediapipe.__version__, cv2.__version__, numpy.__version__)"
# -> 1.19.2  0.10.18  4.11.0  1.26.4
```

> 🔴 **시스템 파이썬이 3.13 이면 mediapipe 가 설치되지 않는다.** aarch64/3.13 휠이 존재하지 않는다.
> conda(Miniforge)로 3.11 환경을 만드는 것이 **선택이 아니다.**
> onnxruntime 은 **1.18.0 이상** 이어야 한다 (우리 ONNX 가 IR version 10).

### 측정 전 1회 — 재부팅하면 초기화된다

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor    # -> performance
vcgencmd measure_clock arm                                   # -> frequency(0)=2400020480
vcgencmd get_throttled                                       # -> throttled=0x0 이어야 시작 가능
```

---

## 4. 측정 명령 — 4런

**리포지토리 루트에서** 실행한다 (`python -m` 모듈 실행, 파일 직접 실행 금지).

```bash
cd ~/IEEE_sensors
conda activate eyeblink
```

### 4-1. 640×480 (주 측정)

```bash
# ours
python -m src.v2.deploy.run_video \
    --mode ours --source clips/eb9_x2.avi \
    --width 640 --height 480 --fps 30 \
    --duration 300 --intra-threads 2 --no-spin \
    --out results/v2/pi_ours_480p.json

# ear (규칙 기반)
python -m src.v2.deploy.run_video \
    --mode ear --source clips/eb9_x2.avi \
    --width 640 --height 480 --fps 30 \
    --duration 300 --intra-threads 2 --no-spin \
    --out results/v2/pi_ear_480p.json
```

### 4-2. 1280×720 (보조 측정)

```bash
python -m src.v2.deploy.run_video \
    --mode ours --source clips/mebal2_u1_720p_320s.mp4 \
    --width 1280 --height 720 --fps 30 \
    --duration 300 --intra-threads 2 --no-spin \
    --out results/v2/pi_ours_720p.json

python -m src.v2.deploy.run_video \
    --mode ear --source clips/mebal2_u1_720p_320s.mp4 \
    --width 1280 --height 720 --fps 30 \
    --duration 300 --intra-threads 2 --no-spin \
    --out results/v2/pi_ear_720p.json
```

### 🔴 플래그 주의

| 플래그 | 왜 |
|---|---|
| `--intra-threads 2` | 기본값이 **0** 이다. 안 주면 결과 JSON 에 **경고가 박힌다** |
| `--no-spin` | **필수.** 빼면 ONNX 스레드가 spin-wait 하며 디코딩·검출을 **27% 느리게** 만들어 비교가 무효 |
| `--head-stride 1` | 기본값 1. 건드리지 않는다 (매 프레임 판정) |
| `--refine-landmarks` | **주지 않는다.** iris 서브모델은 크롭에 안 쓰이는데 CPU 를 먹는다 |
| `--width/--height` | ⚠️ **영상 파일 입력에는 적용되지 않는다** (코드상 카메라일 때만 `cap.set`). 클립 자체 해상도가 기준이다. 인자는 기록용 |

> 각 런 5분 × 4 = **약 20분**. 런 사이에 온도가 식도록 1~2분 쉬는 것을 권한다.

---

## 5. 실행 중·후 확인

### 게이트 G-E1 (논문의 핵심 주장)

```bash
python -c "
import json
d=json.load(open('results/v2/pi_ours_480p.json'))
e=d['e2e_ms']; print('e2e p50/p95/p99:', e['p50'], e['p95'], e['p99'])
print('budget 33.3ms 통과:', e['p99'] <= 33.3)
print('sustained fps:', d.get('sustained_fps'))
print('throttled:', d.get('throttled'))
print('warnings:', d.get('warnings'))
"
```

**통과 조건**

- [ ] `e2e p99 ≤ 33.3 ms`
- [ ] `throttled` 플래그가 전부 비어 있음 (`0x0`)
- [ ] `warnings` 가 비어 있음 (intra-threads / no-spin 누락 경고 없음)
- [ ] `frames_processed` 가 예상치(300초 × 30fps ≈ 9,000)에 근접

### 🔴 단계별 분해를 반드시 본다

```bash
python -c "
import json
d=json.load(open('results/v2/pi_ours_480p.json'))
s=d['stages_ms']; tot=d['e2e_ms']['p50']
for k,v in s.items(): print(f'{k:8s} p50 {v[\"p50\"]:6.2f} ms  ({100*v[\"p50\"]/tot:5.1f}%)')
"
```

**데스크탑에서는 detect 가 59% 였다.** Pi 에서 이 비중이 어떻게 변하는지가
논문 Table II 의 핵심이다 — **e2e 만 보면 인코더 비용 차이가 통째로 묻힌다.**

---

## 6. 결과 회수

```bash
scp hanool@192.168.0.20:~/IEEE_sensors/results/v2/pi_*.json results\v2\
scp hanool@192.168.0.20:~/IEEE_sensors/results/v2/pi_*.csv  results\v2\
```

---

## 7. 문제가 생기면

| 증상 | 원인 / 조치 |
|---|---|
| `🔴 train/serve 동치 게이트를 통과하지 않았습니다` → 종료코드 2 | `check_equivalence.json` 이 없거나 `pi_measurement_authorized: false`. PC 에서 통과시킨 뒤 파일을 복사 |
| `영상원을 열 수 없습니다` | 클립 경로 오타, 또는 OpenCV 가 해당 코덱을 못 읽음. `.avi`(MJPG)가 가장 안전 |
| mediapipe import 실패 | Python 이 3.11 이 아니다. `conda activate eyeblink` 확인 |
| onnxruntime 로드 실패 | 1.18.0 미만. `pip install onnxruntime==1.19.2` |
| 측정이 5분 못 채우고 끝남 | **클립이 짧다.** `run_video.py` 는 반복 재생하지 않는다 (§1-2) |
| e2e 가 예상보다 느림 | governor 가 `ondemand` 로 돌아갔거나 `--no-spin` 을 빠뜨렸다 |
| 온도 상승 / throttled 플래그 | 액티브 쿨러 확인. 런 사이 휴식 |

---

## 8. 이 측정으로 답하는 것 / 못 답하는 것

**답한다**

- G-E1: 에지 실시간 성립 여부 (**논문의 D10**)
- 단계별 지연 분해 — 얼굴 검출이 지배적인가
- ours vs 규칙 기반 EAR 의 실측 비용 차이
- 해상도(480p/720p)가 지연에 미치는 영향
- 메모리(RSS peak) · CPU% · 온도 · 스로틀

**측정 완료 — 결과 확인 위치**

- **image_cnn 대비 비용** — Table II-b의 `image_cnn_max`·`image_cnn_head` 행
- **G-E1** — Table II-b의 e2e p99 열
- 양자화 효과 ← G-E1 통과하면 불필요
