> ▶ **2026-08-08 아카이브로 이동.** v1 시절 내용과 **폐기된 프라이버시 목표**가
> 섞여 있다. 현재 상태는 [`docs/STATUS_2026-08-08.md`](../STATUS_2026-08-08.md) 가 기준이다.
> 이 문서는 **연구 기록 보존용**이며 논문·결정의 근거로 인용하지 않는다.

# 라즈베리파이 5 실행 가이드

> 2026-07-29 전면 개정. 이전 판은 삭제된 `src/eyeblink` · `src/scripts` 패키지와
> `face_landmarker.task` 기준이라 전부 무효였다. 아래는 **실제로 통한 절차**다.
> 측정 결과와 조건은 `docs/PI5_BENCHMARK.md` 참조.

---

## 0. 접속 정보

| 항목 | 값 |
|---|---|
| 호스트 | `cat5.local` / `192.168.0.20` |
| 사용자 | `hanool` |
| SSH | `ssh hanool@192.168.0.20` |
| 데스크탑 | RealVNC (창 띄울 때만 필요) |
| 종료 | `sudo poweroff` — 전원을 그냥 뽑지 말 것 |

`cat5.local` 은 mDNS 이름이라 **Windows에서는 해석이 안 될 수 있다**(Bonjour 필요).
`Could not resolve hostname` 이 뜨면 IP를 쓴다. IP 확인은 파이에서 `hostname -I`.

---

## 1. ★ 파이썬 버전 — 이 프로젝트 최대의 함정

이 파이는 **Debian 13 (trixie)** 이고 시스템 파이썬이 **3.13** 이다.
**mediapipe는 aarch64용 Python 3.13 휠이 존재하지 않는다.** 어떤 버전을 시도해도 없다.

PyPI aarch64 휠 조사 결과 (2026-07 기준):

| 패키지 | 3.9 | 3.10 | 3.11 | 3.12 | 3.13 |
|---|:--:|:--:|:--:|:--:|:--:|
| mediapipe 0.10.14 / 0.10.18 | ✅ | ✅ | ✅ | ✅ | **없음** |
| mediapipe 0.10.20 이상 | — | — | — | — | — |
| numpy 1.26.4 | ✅ | ✅ | ✅ | ✅ | **없음** |
| onnxruntime 1.19.2 | ✅ | ✅ | ✅ | ✅ | — |
| opencv-contrib-python 4.11 | ✅ | ✅ | ✅ | ✅ | ✅ |

→ **mediapipe가 상한을 정한다. Python 3.11 또는 3.12를 써야 한다.**
→ mediapipe 0.10.18은 `numpy<2` 를 요구하므로 **numpy 1.x 고정**이다.
→ 시스템 파이썬으로는 불가능하므로 **conda로 3.11 환경을 따로 만든다.**

---

## 2. 환경 (이미 구축돼 있음)

파이에 Miniforge가 설치돼 있고 `eyeblink` 환경이 이미 만들어져 있다.

```bash
conda activate eyeblink
python -V      # 3.11.15
```

### 새로 만들어야 할 때

```bash
conda create -y -n eyeblink -c conda-forge python=3.11
conda activate eyeblink
pip install --upgrade pip
pip install -r requirements-pi.txt
```

**콘다는 파이썬 버전 고정 용도로만 쓴다. 패키지는 전부 pip.**
mediapipe는 conda-forge aarch64에 없어서 conda로 받으려 하면 막힌다.

Anaconda/Miniconda가 아니라 **Miniforge**를 쓸 것 (aarch64는 conda-forge 채널이 사실상 필수).

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
exec $SHELL -l          # 쉘 재시작해야 conda 명령이 먹는다
```

> 이전 판에 있던 `libatlas-base-dev` 는 불필요하다. numpy·onnxruntime 휠이 각자
> OpenBLAS를 번들로 들고 온다. `ImportError: libGL.so.1` 이 날 때만
> `sudo apt install -y libgl1 libglib2.0-0`.

---

## 3. 파일 옮기기

`models/` 와 `data/` 는 git에 없다. **`git clone` 만으로는 모델이 안 따라온다.**

### 【PowerShell】

```powershell
ssh hanool@192.168.0.20 "mkdir -p ~/IEEE_sensors/models ~/IEEE_sensors/clips ~/IEEE_sensors/results"

scp -r C:\Users\sch\PycharmProjects\IEEE_sensors\src         hanool@192.168.0.20:~/IEEE_sensors/
scp -r C:\Users\sch\PycharmProjects\IEEE_sensors\models\onnx hanool@192.168.0.20:~/IEEE_sensors/models/
scp C:\Users\sch\PycharmProjects\IEEE_sensors\data\_legacy_public\eyeblink8\eyeblink8\9\27122013_152435_cam.avi hanool@192.168.0.20:~/IEEE_sensors/clips/
```

**주의**

- `encoder.onnx` 의 가중치는 별도 파일 `encoder.onnx.data`(6.5 MB)에 있다. **폴더째** 복사할 것.
- PowerShell은 외부 명령에 대해 `*` 를 확장하지 않는다. `cam.*` 같은 와일드카드는 실패한다. 파일명을 정확히 쓸 것.
- 붙여넣기 시 명령이 중복되지 않았는지 확인할 것 (`scp -r ... scp -r ...` 로 두 번 들어가면 뒤쪽이 에러난다).

### 비밀번호 없이 (선택, 권장)

```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh hanool@192.168.0.20 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

---

## 4. 설치 확인

### 【파이】

```bash
conda activate eyeblink
cd ~/IEEE_sensors

python -c "import cv2,mediapipe,onnxruntime as o,numpy; print('cv2',cv2.__version__,'| mp',mediapipe.__version__,'| ort',o.__version__,'| np',numpy.__version__)"

python -m src.deploy.frontend          # 크롭+텐서 경로 자체 점검

python -c "
import onnxruntime as ort, numpy as np
s=ort.InferenceSession('models/onnx/pipeline.onnx', providers=['CPUExecutionProvider'])
print('ONNX OK ->', s.run(None,{'input':np.random.rand(1,1,64,160).astype(np.float32)})[0])
"
```

기대 출력: `cv2 4.11.0 | mp 0.10.18 | ort 1.19.2 | np 1.26.4`,
`OK crop (64, 160) uint8 -> tensor (1, 1, 64, 160) float32`.

> `ONNX OK -> [-0.0000000]` 처럼 음의 0이 나와도 정상이다. 균등 난수는 학습 분포
> 밖이라 sigmoid가 포화하며, ARM의 NEON 근사가 −2⁻²⁴ 를 반환한다. 실제 눈 크롭에서는
> 발생하지 않는다.

**onnxruntime은 1.18.0 이상이어야 한다.** 우리 ONNX가 IR version 10이라 1.17 이하에서는
`Unsupported model IR version: 10, max supported IR version: 9` 로 로드 자체가 실패한다.

---

## 5. 벤치마크 실행

### 【파이】 CPU governor 고정 (측정 전 1회)

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
vcgencmd measure_clock arm        # frequency(0)=2400020480 확인
```

재부팅하면 `ondemand` 로 돌아간다.

### 【파이】 짧은 점검

```bash
python -m src.deploy.run_video --video clips/27122013_152435_cam.avi --mode ours \
    --intra-threads 2 --no-spin --max-frames 300
```

### 【파이】 본 측정 (각 5분)

```bash
python -m src.deploy.run_video --video clips/27122013_152435_cam.avi --mode ours \
    --intra-threads 2 --no-spin --minutes 5 --csv results/pi_ours.csv --json results/pi_ours.json

python -m src.deploy.run_video --video clips/27122013_152435_cam.avi --mode ear \
    --minutes 5 --csv results/pi_ear.csv --json results/pi_ear.json
```

**`--intra-threads 2 --no-spin` 은 선택이 아니다.** 빼면 ONNX 스레드가 spin-wait 하며
디코딩·검출을 27% 느리게 만들어 EAR 비교가 무효가 된다. 근거는 `PI5_BENCHMARK.md` §3-A.

### 【파이】 예산 초과 프레임 비율

```bash
python -c "
import csv
for f in ['results/pi_ours.csv','results/pi_ear.csv']:
    v=[float(r['e2e_ms']) for r in csv.DictReader(open(f))]
    o=[x for x in v if x>33.3]
    print(f'{f}: {len(v)} frames | over 33.3ms: {len(o)} ({100*len(o)/len(v):.4f}%) | max {max(v):.2f} ms')
"
```

### 【PowerShell】 결과 회수

```powershell
scp hanool@192.168.0.20:~/IEEE_sensors/results/* C:\Users\sch\PycharmProjects\IEEE_sensors\results\
```

---

## 6. `run_video.py` 옵션

| 옵션 | 뜻 |
|---|---|
| `--mode ours` / `ear` | 벡터 파이프라인 / EAR 베이스라인. 같은 하네스라 비교 가능 |
| `--minutes N` | N분 될 때까지 클립을 반복 재생 |
| `--max-frames N` | N프레임에서 중단 |
| `--warmup N` | 통계에서 제외할 앞쪽 프레임 (기본 30) |
| `--intra-threads N` | ORT intra-op 스레드 (0=전 코어) |
| `--no-spin` | ORT 스레드 spin-wait 비활성 ← **본 측정에 필수** |
| `--refine-landmarks` | MediaPipe 홍채 서브모델 활성 (느려짐, 크롭엔 불필요) |
| `--encoder` / `--mlp` | 기본은 융합된 `pipeline.onnx` 1단계. `mlp.onnx` 를 따로 주면 encode/mlp 2단계로 분리 계측 |
| `--csv` / `--json` | 프레임별 원자료 / 요약+환경정보 |
| `--budget-ms` | 프레임 예산 (기본 33.3 = 30 fps) |

---

## 7. 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| `ModuleNotFoundError: mediapipe` | `conda activate eyeblink` 안 함. 또는 Python 3.13 환경 (§1) |
| `Unsupported model IR version: 10` | onnxruntime < 1.18. `pip install "onnxruntime==1.19.2"` |
| `video not found` | 클립이 안 옮겨졌다. PowerShell 와일드카드 문제일 가능성 (§3) |
| `mediapipe is required for EyeFrontend` | 위와 동일 |
| `Error in cpuinfo: prctl(PR_SVE_GET_VL) failed` | 무해. Pi 5에 SVE가 없어서 나는 경고 |
| `Feedback manager requires...` / `NORM_RECT without IMAGE_DIMENSIONS` | 무해. MediaPipe 내부 경고 |
| `detect` 가 9~10 ms로 나옴 | `--no-spin` 을 빼먹었다 (§5) |
| `ssh: Could not resolve hostname cat5.local` | Windows mDNS. IP(`192.168.0.20`) 사용 |
| `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0` |
