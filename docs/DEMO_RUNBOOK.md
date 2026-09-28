# DEMO_RUNBOOK — 변리사 미팅용 라즈베리파이 실시간 동작 화면

> 발표자료 9쪽 `[라즈베리파이 실시간 동작 화면 캡처 삽입 예정]` 자리를 채우기 위한 절차.
> 구성: **맥북 = 카메라(MJPEG 서버)**, **라즈베리파이 5 = 처리·표시**.
> 지연 측정(논문 Table II)은 이 화면이 아니라 `PI_RUNBOOK.md` 절차로 따로 잰다.

```
   맥북 웹캠 ──▶ mac_camera_server.py ──HTTP MJPEG──▶ Pi 5 ──▶ pi_demo.py 창
   (카메라)        :8000/stream.mjpg        같은 WiFi      (MediaPipe → 크롭 →
                                                            encoder.onnx → head.onnx)
```

| 항목 | 값 |
|---|---|
| Pi 접속 | `ssh hanool@192.168.0.20` (`cat5.local` 은 Windows 에서 해석 안 됨) |
| Pi 환경 | Miniforge `eyeblink` (Python 3.11) — `conda activate eyeblink` |
| 저장소 | `~/IEEE_sensors` |
| 모델 | `models/v2/onnx/encoder.onnx` (D=16) + `head.onnx` (창 19, 매 프레임) |
| 화면 | RealVNC 로 Pi 데스크탑을 띄우고 그 안에서 실행 |

---

## 0. 파일 두 개

| 파일 | 어디서 도나 | 무엇 |
|---|---|---|
| `src/tools/mac_camera_server.py` | **맥북** | 웹캠 → MJPEG 서버. 의존성 `opencv-python` 하나 |
| `src/v2/deploy/pi_demo.py` | **Pi** | 실시간 처리 + 시연 화면(라이트 테마, 1280×720) |

---

## 1. 맥북 — 카메라 서버

```bash
python3 -m pip install opencv-python          # 처음 한 번
python3 src/tools/mac_camera_server.py        # 640x480 / 30fps / :8000
```

첫 실행 때:

- **카메라 권한** 팝업 → 허용 (시스템 설정 → 개인정보 보호 및 보안 → 카메라 → 터미널)
- **방화벽** "들어오는 연결을 허용하시겠습니까?" → 허용

실행하면 터미널이 파이에서 쓸 명령을 그대로 찍어 준다:

```
  맥북 카메라 서버   http://192.168.0.31:8000/
  라즈베리파이에서:
    python -m src.v2.deploy.pi_demo --source http://192.168.0.31:8000/stream.mjpg
```

브라우저로 `http://<맥IP>:8000/` 를 열어 **영상이 보이면 성공**이다. 여기서 안 보이면
파이에서도 안 보인다 — 파이로 넘어가기 전에 여기서 끝내라.

| 옵션 | 쓸 때 |
|---|---|
| `--list` | 카메라 인덱스 훑기 (외장 캠이 붙어 있으면 0 이 아닐 수 있다) |
| `--camera 1` | 그 인덱스로 |
| `--width 1280 --height 720` | 720p 로 (프레임이 커지면 Pi 의 detect 가 느려진다) |
| `--quality 70` | WiFi 가 약할 때 JPEG 품질을 낮춘다 |

> 💡 시연 중 맥이 절전으로 들어가면 스트림이 끊긴다. 다른 탭에서 `caffeinate -d` 를 띄워 둘 것.

---

## 2. Pi — 파일 복사 (Windows PowerShell 에서 한 번)

```powershell
scp C:\Users\sch\PycharmProjects\IEEE_sensors\src\v2\deploy\pi_demo.py `
    hanool@192.168.0.20:~/IEEE_sensors/src/v2/deploy/
```

`models/v2/onnx/`, `src/v2/` 는 `PI_RUNBOOK.md` §2 로 이미 올라가 있다. 안 올라가 있으면
그 절에 따라 먼저 올린다. **`results/v2/check_equivalence.json` 은 이 데모에 필요 없다**
(동치 게이트는 측정용이고, 데모는 게이트를 걸지 않는다).

한글 폰트가 없으면 화면 글자가 물음표로 나온다. Pi 에서 한 번만:

```bash
sudo apt install -y fonts-nanum
python -c "import PIL; print(PIL.__version__)" || pip install pillow
```

---

## 3. Pi — 실행

```bash
ssh hanool@192.168.0.20            # 창을 띄울 것이므로 실제로는 VNC 데스크탑에서
cd ~/IEEE_sensors
conda activate eyeblink

python -m src.v2.deploy.pi_demo --source http://192.168.0.31:8000/stream.mjpg
```

자주 쓰는 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--fullscreen` | off | 전체화면으로 시작 (`f` 로도 토글) |
| `--canvas 1280x720` | 1280x720 | 창 크기. VNC 해상도보다 작게 |
| `--no-mirror` | 미러 on | 좌우 반전 끄기 |
| `--blink-th 0.5` | 0.5 | 깜빡임 판정 임계값 (`-` / `=` 키로 실시간 조정) |
| `--id-th 0.85` | 0.85 | 사용자 식별 코사인 임계값 (`[` / `]` 키) |
| `--baseline-min-sec 60` | 60 | 이 시간 이상 누적돼야 **개인 기준선**으로 전환 |
| `--mode ear` | ours | 모델 없이 규칙 기반으로 화면만 확인 |
| `--headless --shot-every 5 --duration 60` | — | 창 없이 SSH 에서 PNG 만 뽑기 |

### 조작키

| 키 | 동작 |
|---|---|
| `1`~`6` | 지금 화면의 사람을 그 번호 사용자로 **등록**(재등록도 같은 키) |
| `r` | 식별 버퍼 초기화 — **사용자 교체 시연**은 이 키로 |
| `c` | 이번 세션 카운터 초기화 (누적 이력은 유지) |
| `s` | **화면 캡처 저장** → `results/v2/demo_shots/demo_YYYYmmdd_HHMMSS.png` |
| `x` | 등록 프로필 전체 삭제(파일까지) |
| `f` / `q` | 전체화면 / 종료 |

---

## 4. 발표자료 9쪽에 넣을 장면 만들기

화면의 네 블록이 발표자료의 청구 구성과 1:1 로 대응한다. 캡처 한 장에 넷이 다 살아
있어야 의미가 있으므로, 아래 순서로 만들어 두고 `s` 를 누른다.

1. **A 가 앉아 `1` 로 등록** → 30~60초 그대로 두고 깜빡인다.
   → ②에 `사용자 A / cos 0.9x`, ③에 누적 깜빡임·누적 관측이 쌓인다.
2. **B 가 앉아 `2` 로 등록** → 30초.
3. **A 가 다시 앉는다**(`r` 로 버퍼를 비우면 전환이 빠르다).
   → ②가 `사용자 A` 로 돌아오고 ③의 누적값이 **1번에서 이어진다**
   (= 세션을 넘겨 보존·연속 누적. `results/v2/demo_profiles.json` 이 그 증거다).
4. 카메라 앞에서 **10초 이상 안 깜빡인다** → ③에 `20-20-20 휴식 권고` 가 뜬다.
5. 등록 안 된 사람이 앉으면 ②가 `미등록 사용자`, ③이 `이력을 귀속하지 않습니다` 가 된다.
   → 공용 단말 이력 혼입 방지 구성의 화면 증거.
6. 원하는 장면에서 `s`. 파일을 Windows 로 회수:

```powershell
scp hanool@192.168.0.20:~/IEEE_sensors/results/v2/demo_shots/*.png `
    C:\Users\sch\PycharmProjects\IEEE_sensors\results\v2\demo_shots\
```

> 캡처는 1280×720(16:9) 흰 바탕이다. 발표자료 9쪽에 여백 없이 그대로 들어간다.
> 하단의 장치·지연 줄이 잘리지 않게 원본 비율로 넣을 것 — 그 줄이 "별도 가속기 없는
> 저사양 단말" 주장을 받쳐 준다.

---

## 5. 화면이 무엇을 보여주는가 (발표자료 대응)

| 화면 블록 | 발표자료 |
|---|---|
| 카메라 영상(눈 랜드마크·크롭 상자) + "원본 프레임·크롭 영상은 임베딩 산출 후 폐기 · 외부 전송 없음" | 6쪽 전체 구조 |
| ① 깜빡임 검출 (blink_prob, 세션 횟수, 분당, 무깜빡임 경과) | 7쪽 — 같은 벡터가 검출부로 |
| ② 사용자 식별 (코사인 유사도, 미등록 판정) | 7쪽 — 같은 벡터가 식별부로 |
| ③ 사용자별 누적 이력 · 개인 기준선 · 권고 | 8쪽 핵심 구성 ② |
| 하단 단계별 지연 (read/detect/crop/encode/head, e2e p99 ≤ 33.3 ms) | 9쪽 상시 동작 |

> 크롭 썸네일(64×160)·임베딩 벡터 막대·처리 경로 목록은 화면이 번잡해져 **뺐다.**
> 파이프라인은 그대로 돌고(크롭 → encoder → head) 화면에만 안 띄운다. 그 구성은
> 발표자료 6·7쪽 도면으로 설명하고, 이 화면은 **동작한다는 증거** 역할만 한다.

### 🔴 미리 알고 있어야 할 것

- **지연 숫자는 시연용이다.** 이 스크립트는 렌더링과 함께 돌고 클립이 아니라 네트워크
  스트림을 받는다. 논문에 쓰는 값은 `run_video.py` 로 잰 `results/v2/pi_*.json` 이다.
  화면의 `e2e` 는 파이프라인 단계(detect+crop+encode+head)만 합산하고 렌더링은 뺐지만,
  그래도 측정 조건(governor, `--no-spin`, 5분 지속)이 다르다.
- **사용자 식별은 현재 인코더 임베딩의 코사인 유사도로 구현했다.** v2 인코더는 깜빡임
  검출로 학습된 D=16 벡터이므로 개인 분리도가 보장되지 않는다. 시연 전에 두 사람을
  등록해 화면의 `cos` 값을 보고 `[` / `]` 로 임계값을 맞춰라(두 사람의 cos 가 벌어지는
  중간값). 발표에서는 "같은 벡터를 식별부가 공유하는 **구성**"을 보이는 것이 목적이고,
  식별 정확도 수치는 이 화면의 주장이 아니다.
- **개인 기준선**은 누적 관측이 `--baseline-min-sec`(기본 60초) 미만이면 공통 초기값
  15 bpm 을 잠정 적용하고 화면에 그렇게 표시한다(발표자료 8쪽의 마지막 항목 그대로).

---

## 6. 안 될 때

| 증상 | 조치 |
|---|---|
| 파이 화면이 "카메라 입력 대기 중" 에서 안 넘어감 | 맥 브라우저에서 `http://<맥IP>:8000/` 먼저 확인. 맥 방화벽·같은 WiFi 대역 확인 |
| 맥 서버가 "카메라를 열 수 없습니다" | macOS 카메라 권한. `--list` 로 인덱스 확인. FaceTime/Zoom 이 캠을 잡고 있으면 종료 |
| 글자가 전부 `?` | `sudo apt install fonts-nanum` (또는 `--font /path/to/NanumGothic.ttf`) |
| `ImportError: mediapipe` | `conda activate eyeblink` 안 함. 시스템 파이썬 3.13 은 mediapipe 휠이 없다 |
| `Unsupported model IR version: 10` | `pip install "onnxruntime==1.19.2"` |
| 창이 안 뜸 (`cvNamedWindow ... not implemented`) | SSH 로만 접속한 상태. VNC 데스크탑에서 실행하거나 `--headless --shot-every 5` |
| 영상이 몇 초씩 밀림 | 맥 서버 `--fps 20 --quality 70` 으로 낮춘다. WiFi 2.4GHz 면 5GHz 로 |
| fps 가 10 아래 | 720p 로 받고 있지 않은지 확인(`--width 640 --height 480`). governor 를 `performance` 로 |
| 깜빡임이 과하게/적게 잡힘 | `-` / `=` 로 `--blink-th` 조정. 조명이 어두우면 얼굴 쪽 조명을 올린다 |

시연 전 Pi 에서 한 번:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
vcgencmd get_throttled       # throttled=0x0 이어야 함
```
