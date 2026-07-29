# 프로젝트 상세 기술 문서 — 엣지 프라이버시 보존 눈 깜빡임 표현

> 목적: 이 문서 하나로 프로젝트 전체(배경·방법·데이터·모델·결과·코드·로드맵)를 이해할 수 있게
> 자체 완결적으로 정리한다. 학습·공유용.

---

## 0. 한눈 요약

- **무엇**: 눈 영상을 저장하지 않고 **벡터(숫자 코드)**만으로 깜빡임(→눈 건강 지표)을 검출.
- **어떻게**: `카메라 → 양눈 크롭 → 인코더(CNN) → 128차원 벡터 → 판정기(MLP) → 깜빡임 O/X`.
- **왜 프라이버시**: 벡터만 남기고, 벡터를 이미지로 되돌리는 **디코더를 배포하지 않음** → 원본 복원 불가.
- **어디서**: 라즈베리파이5 엣지, ONNX 경량 모델. **목표 30fps / 프레임당 33.3ms**
  (이전 문서의 "15fps / 66ms"는 데스크탑 추정 기반이었고, 실측 후 상향했다).
- **현재 상태 (2026-07-29)**:
  - **엣지 실증 완료** — Pi 5에서 e2e p99 **12.06 ms**, 86 fps, 5분 연속 스로틀 없음.
    EAR 베이스라인 대비 **+2.67 ms (1.30×)**. → `docs/PI5_BENCHMARK.md`
  - **유용성** — EAR과 대등하나 **8명 데이터로는 우열 판정 불가**(시드에 따라 70–73/74,
    EAR 70/74).
  - **프라이버시** — 잠재 차원을 16배 좁혀도 선형 재식별 99%+ →
    **차원 축소는 프라이버시 메커니즘이 아니며 disentanglement가 필수**.
  - **다음** — mEBAL2(180명) 도착 예정(약 1주). 이후 전 실험 재실행.

---

## 1. 배경과 동기

기존 눈 깜빡임 검출은 **원본 픽셀 → 얼굴 랜드마크(MediaPipe) → EAR(Eye Aspect Ratio) 임계값**으로 판정한다.
문제는 두 가지:
1. **사용자 수용성**: 눈을 계속 촬영하는 방식은 기능이 정상이어도 거부감을 준다. 눈 영상은 생체정보이고,
   촬영본이 어디 남는지 사용자가 알 수 없다.
2. **연구 확장성**: 단일 기능(깜빡임 검출)에 머물러 확장 여지가 좁았다.

→ 질문을 바꿈: **"눈 영상을 보존하지 않고도 눈 상태를 다룰 수 있는가?"**

## 2. 핵심 아이디어와 프라이버시 논리

원본 픽셀 대신 **학습된 벡터 표현(latent representation)**을 사용한다. 저장·전송되는 것은 벡터뿐이고,
벡터를 눈 이미지로 복원하는 **디코더를 배포하지 않기 때문에 복원 수단 자체가 부재**하다.
→ 프라이버시가 **정책·약속이 아니라 구조(structure)에서** 나온다.

**중요한 경계 구분 (심사·발표 시 방어 포인트):**
- 프라이버시는 **배포된 제품(Pi5)의 성질**이다. 저장·전송되는 것이 벡터뿐이라는 뜻.
- **개발·학습 단계에서는 원본을 사용한다** (데이터를 만들려면 당연히 필요). "연구자가 원본을 안 만진다"가
  아니라 "배포 시스템이 원본을 안 남긴다"가 주장이다.
- 랜드마크(MediaPipe)는 **눈 위치를 잡아 크롭하는 휘발성 용도**로만 쓰고 좌표는 버린다. EAR 계산이나
  저장에 쓰지 않는다. (그래서 "랜드마크에서 벗어난다더니 왜 MediaPipe를 쓰나"라는 지적을 차단)

## 3. 전체 파이프라인

```
[운용 경로 - 배포]
카메라 프레임 → (MediaPipe로 눈 위치 검출) → 양눈 크롭(64x160 그레이)
             → 인코더(CNN) → 128차원 벡터 → 판정기(MLP) → 깜빡임 확률 → O/X

[비배포 - 개발 검증 도구]
벡터 → 디코더 → 눈 이미지 복원 (벡터가 눈을 담았는지 육안/정량 확인)
```

- 도구 역할 분담: **MediaPipe** = 눈 위치 찾기, **OpenCV** = 자르기·흑백·리사이즈, **PyTorch 인코더** = 벡터 추출.
- 운용 경로에 **디코더 없음** = 프라이버시 주장의 근거.

## 4. 데이터셋

### 4.1 현재 사용: Eyeblink8 (공개)
- 8개 클립(폴더 1,2,3,4,8,9,10,11), 각 ~8분, 640×480, 30fps.
- 파일 3종:
  - `.avi` = 영상
  - `.tag` = **주석(정답)**. 프레임별: `frameID:blinkID:NF:LE_FC:LE_NV:RE_FC:RE_NV:F_X:F_Y: ... :눈꼬리 좌표`
    - `blinkID = -1` → 깜빡임 아님, 그 외 숫자 → 해당 깜빡임 이벤트에 속함
    - `C` 플래그(LE_FC/RE_FC) → 눈 완전 폐안(fully closed)
    - 줄의 **마지막 8개 숫자 = 4개 눈꼬리 좌표** (좌안 2점 + 우안 2점)
  - `.txt` = 프레임번호 + **타임스탬프(초)** — 라벨 아님(주의)
- **라벨은 수동 주석**이라 EAR과 독립적 → 정직한 평가 기준으로 사용.

### 4.2 파생 라벨 (우리가 뽑은 것)
- `blink_event` = 1 if `blinkID != -1` (깜빡임 구간 프레임)
- `eye_closed` = 1 if 우안 `C` (완전 폐안 프레임)
- `subject` = 클립 id (eb01…eb11), 피험자 분리 분할용

### 4.3 소비 데이터 (전처리 산출물)
- `data/processed/eyeblink8_eyes.npz`: `images`(uint8 [N,64,160]), `subject`, `frame_id`,
  `blink_id`, `blink_event`, `eye_closed`
- 규모: **35,496 프레임**(stride 2로 30→15fps 다운샘플), 깜빡임 5.25%, 완전폐안 2.49% (극심한 불균형)

## 5. 전처리 규격 (canonical crop — 단일 소스, 학습=운용 동일)

`src/dataset/eye_preprocess.py`의 `crop_both_eyes_from_corners()`가 유일한 크롭 함수. 학습·운용 모두 이걸 import.

- **입력**: 양쪽 눈의 눈꼬리 4점 (Eyeblink8는 tag에서, 웹캠은 MediaPipe에서)
- **처리**:
  1. 두 눈 중심의 중점을 기준점으로
  2. 두 눈 중심을 잇는 선이 수평이 되도록 회전 정렬 (뒤집힘 방지 위해 tilt를 −90~90도로 정규화)
  3. 크롭 폭 = 눈 사이 거리 × `BOTH_MARGIN(2.2)`, 크롭 높이 = 폭 × (64/160) → 종횡비 유지
  4. 그레이스케일 변환, **64×160**으로 리사이즈 (업스케일이면 INTER_CUBIC, 다운스케일이면 INTER_AREA)
- **다운샘플**: stride 2 (30fps→15fps), 운용 프레임레이트와 평가 조건 일치

> 참고: 이전에는 **한쪽 눈 64×64**였다가, "눈이 둘인데 하나만 볼 이유가 없다"는 지적으로 **양눈 64×160**으로 전환.

## 6. 모델 구조

### 6.1 인코더 (배포됨) — `src/encoder/train_autoencoder.py`
```
입력 1×64×160
Conv(1→32, 3x3, s2) + BN + ReLU   → 32×80
Conv(32→64, 3x3, s2) + BN + ReLU  → 16×40
Conv(64→128, 3x3, s2) + BN + ReLU → 8×20
Conv(128→256, 3x3, s2) + BN + ReLU→ 4×10
Flatten(256×4×10=10240) → Linear → 128차원 벡터
```
- 입력 크기(in_hw)에 따라 flatten 차원을 자동 계산 (64×160 → 4×10).

### 6.2 디코더 (비배포, 검증용)
- 인코더의 역구조 (Linear → ConvTranspose ×4 → Sigmoid) → 64×160 복원.
- 벡터가 눈을 담았는지 육안 확인(`recon_val.png`) + 향후 프라이버시(복원 화질) 지표.

### 6.3 판정기 (배포됨) — `src/classifier/train_classifier.py`
```
128차원 벡터 → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → (sigmoid) → 깜빡임 확률
```
- EAR 규칙(고정 임계값)을 대체한 **학습된 분류기**.

## 7. 학습 방법

### 7.1 인코더 (비지도, 오토인코더)
- 손실 = **복원 MSE** (라벨 불필요). Adam lr=1e-3, 30 epoch.
- **피험자 분리** val (eb04, eb11 홀드아웃), val MSE 최소 모델 저장.

### 7.2 판정기 (지도학습)
- 인코더 **고정** → 이미지들을 벡터로 변환 → 벡터로 MLP 학습.
- 손실 = `BCEWithLogitsLoss(pos_weight)` — 깜빡임 희소성 보정(pos_weight = 음성/양성 비율).
- 임계값 0.5 및 best-F1 임계값 보고.

## 8. 평가 방법

### 8.1 지표 (Phase 0에서 확정)
- **1순위: 이벤트 단위 Recall** — 깜빡임 이벤트(같은 blink_id)를 프레임 1개라도 잡으면 성공.
  (프레임 단위는 눈 감기는 전환 프레임 때문에 과소평가됨)
- 보조: 프레임 false-alarm(비깜빡임 프레임 오탐 비율), precision, F1.
- **구속 기준**: 절대 수치가 아니라 **"동일 피험자 분리 테스트셋에서 EAR 규칙 기반과 동급 이상"**.

### 8.2 스크립트
- `src/eval/eval_events.py` — 이벤트 단위 recall / frame false-alarm 임계값 스윕
- `src/eval/ear_baseline.py` — 같은 클립에 MediaPipe EAR 돌려 head-to-head.
  legacy / left / right / **mean** / min 5개 변형을 한 번에 계산하며, 양눈 모델과
  같은 조건의 비교 행은 **mean** 이다
- `src/eval/privacy_metrics.py` — (추후 연구) 재식별 정확도 + 복원 PSNR
- `src/experiments/dim_sweep.py` — 잠재 차원 스윕(유용성 + 선형 신원 프로브 동시 측정)
- `src/deploy/run_video.py` — Pi5 단계별 지연 측정 하네스 (`--mode ours|ear`)

> **임계값 격자 주의.** 두 스크립트 모두 기본이 조밀 격자다. 예전의 성긴 격자
> (`eval_events` 0.3–0.8, `ear_baseline` 7점)는 저FA 구간을 놓쳐 비교를 왜곡했다.
> 한쪽만 조밀하게 하면 그쪽이 유리해지므로 **양쪽을 같이** 조밀하게 둘 것.

## 9. 결과 (현재까지 — 정직하게 예비 수준)

### 9.1 유용성 — 양눈, 5시드 (2026-07-29)

판정기를 시드만 바꿔 5회 재학습, 이벤트 recall @ frame false-alarm ≤ 3%:

| 시드 | 0 | 1 | 2 | 3 | 4 | 범위 |
|---|---|---|---|---|---|---|
| recall | 0.973 | 0.946 | 0.973 | 0.973 | 0.986 | **70–73 / 74** |

**EAR 양눈(mean) = 0.946 (70/74)** — 우리 최악 시드와 동점, 나머지는 1~3 이벤트 우세.
**차이가 시드 흔들림 폭 안이므로 우열 판정 불가.** 인코더 복원 val MSE ≈ 0.0045.

⚠️ 이 숫자는 **에폭·임계값을 모두 val에서 고르고 val에서 보고**한 값이다. EAR은 자유
파라미터가 임계값 하나뿐이므로 우리 쪽이 더 부풀려져 있다. → ④(정식 분할) 선행 필요.

### 9.2 ~~양눈 전환 후 false-alarm 상승~~ — 상당 부분이 측정 아티팩트였다

이전 판의 *"128차원 병목은 그대로인데 이미지가 커져 신호가 희석됨"* 은 **기각되었다**(9.5).
"상승"으로 보이던 것의 실제 원인은 셋이었다:

1. **배포된 판정기가 이상치** — 69/74로, 재학습 범위(70–73/74) 아래
2. **임계값 격자가 0.3–0.8 에서 끊김** — `metrics.json` 은 최적 F1을 thr 0.9로 기록하는데
   저FA 구간을 아예 측정하지 않았다
3. **이전 한쪽눈 비교가 서로 다른 눈을 씀** — 벡터=이미지-우안, EAR=이미지-좌안.
   `capture_eye_dataset.EYE` 의 "image-right eye" 주석이 틀렸다

셋을 바로잡으면 양눈 성능은 EAR과 대등하다. **"벡터 차원 확대"라는 개선안은 효과가
없음이 확인되었다**(9.5-a).

### 9.3 엣지 — Pi 5 실측 (2026-07-29)

데스크탑 추정치를 실측으로 대체. 상세·측정 조건은 `docs/PI5_BENCHMARK.md`.

| 단계 (ms, p50) | EAR 베이스라인 | Ours |
|---|---:|---:|
| frame read | 1.01 | 0.96 |
| detect (MediaPipe) | 7.74 | 7.76 |
| crop + 전처리 | — | 0.56 |
| EAR / 인코더+MLP | 0.09 | 2.25 |
| **e2e p50 / p95 / p99** | **8.86 / 9.09 / 9.36** | **11.53 / 11.78 / 12.06** |
| 지속 fps | 112.3 | 86.1 |

- **p99가 30fps 예산(33.3ms)의 36%.** 5분 연속에서 스로틀링 없음(최고 60.4°C, 액티브 쿨러).
- 예산 초과 프레임 2/25,734(0.008%) — EAR도 동일하게 2건이므로 원인은 공유 검출기.
- ~~"MediaPipe Pi5 기준 ~25ms"~~ → **실측 7.76ms.** 예상의 1/3이었다.
- ~~"66ms 예산(15fps)"~~ → **33.3ms(30fps)** 로 상향.

**측정 시 반드시 통제할 것 두 가지** (안 하면 표가 무너진다):
1. **ONNX Runtime spin-wait** — 기본 설정에서 ONNX 스레드가 추론 후에도 회전 대기하며
   디코딩·검출을 27% 느리게 만든다(`detect` 가 ours 9.84 / ear 7.73 으로 갈림).
   `--intra-threads 2 --no-spin` 으로 통제 후 7.69 vs 7.73 으로 일치.
2. **CPU governor = performance** (기본값 ondemand). 재현성 목적이며 지연시간 자체에는
   영향이 없었다.

**전처리 ROI 최적화**: 눈 영역만 필요한데 640×480 전체를 회전시키고 있었다. 크롭 상자만
warp 하도록 바꿔 **4.35 → 0.56 ms (7.8×)**, EAR 대비 오버헤드 6.6 → 2.67 ms.
bit 단위로 동일하지는 않다(판정 뒤집힘 1/2,400) — 근거는 `PI5_BENCHMARK.md` §3-C.

### 9.4 (추후 연구) 신원 disentanglement PoC
- gradient-reversal(DANN) + λ 워밍업으로, **재식별 0.94→0.77 하락 시 깜빡임 AUC ~0.98 유지** 확인.
- 단 8명뿐이라 약한 개념증명. **추후 연구로 이동** (다신원 데이터 필요).
- 그림: `docs/figures/pareto_disentangle.png`
- **9.5-b가 이 방향의 필요성을 정량적으로 뒷받침한다** — 차원을 줄이는 것으로는
  대체할 수 없음이 확인되었다.

### 9.5 잠재 차원 스윕 32/64/128/256/512 × 3시드 (2026-07-29)

`src/experiments/dim_sweep.py`, 원자료 `results/dim_sweep.json`.
각 폭마다 오토인코더를 새로 학습하고 인코더를 동결한 뒤 판정기를 학습해,
**유용성과 신원 정보량을 같은 인코더에서** 측정한다.

**(a) 유용성 — "128 병목" 가설 기각**

| latent | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| recall @FA3% 평균 | **0.986** | 0.973 | 0.910 | — | 0.982 |
| 시드 std | **0.000** | 0.011 | 0.109 | 0.465 | 0.013 |

차원 간 평균의 std 0.125 vs 차원 내 시드 std 0.120 → **신호 대 잡음 비가 1.**
차원 효과를 분리할 수 없다. 다만 **32차원이 가장 안정적이면서 최상위권**이라는 점은
"병목이 좁아 성능이 나쁘다"와 배치된다. 재구성 MSE는 차원에 따라 단조 개선
(0.0053→0.0045)되므로 인코더는 정상적으로 더 담고 있으나, 그 여유가 깜빡임 성능으로
이어지지 않는다. → **차원 확대는 해법이 아니다.**

**(b) 프라이버시 — "좁은 병목이면 신원도 못 담는다" 가설 기각 ★**

선형 프로브 재식별 (chance 0.125):

| latent | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| 선형 re-ID | **0.995** | 0.998 | 0.999 | 0.999 | 0.999 |

**512 → 32 로 16배 좁혀도 누출이 0.45%p 줄어든다.** 전 구간 99% 이상.
→ **차원 축소는 프라이버시 메커니즘이 아니다. 명시적 disentanglement가 필수.**

단서: 8명 식별은 3비트면 충분해 32차원 실수 벡터는 애초에 과잉이다. mEBAL2 180명도
7.5비트라 크게 다르지 않다. **반증으로는 유효하나 강한 주장으로 쓰기는 어렵다.**

## 10. 코드 구조 & 실행

리포지토리는 `src/` 패키지. **리포지토리 루트에서 `python -m src.<하위>.<모듈>`로 실행** (파일 직접 실행 금지 — 상호 임포트 깨짐).

> `src/deploy/frontend.py` 는 **추론 경로의 필수 부품**이다. 이전에는 크롭 좌표를
> Eyeblink8 `.tag` 정답에서만 얻을 수 있어 실제 카메라/영상 추론이 불가능했다.
> `eye_preprocess.eye_corners_from_landmarks` 가 랜드마크를 canonical 크롭으로 잇는다.
> 랜드마크 인덱스를 인자로 받으므로 **mEBAL2의 68점 규격도 인덱스 교체만으로 대응**된다.

```
src/
├─ dataset/    eye_preprocess.py(크롭 규격)  capture_eye_dataset.py(웹캠 수집)
│              prepare_eyeblink8.py(단일 클립)  build_eyeblink8_npz.py(전체 배치→npz)
├─ encoder/    train_autoencoder.py(인코더/디코더 학습)
├─ classifier/ train_classifier.py(판정기 학습)
├─ eval/       eval_events.py  ear_baseline.py  privacy_metrics.py
├─ experiments/ dim_sweep.py(잠재 차원 스윕)
├─ deploy/     export_onnx.py(ONNX 변환)  frontend.py(MediaPipe→canonical 크롭)
│              run_video.py(Pi5 단계별 지연 측정)  bench_latency.py(그래프 지연만)
└─ tools/      find_cam.py(카메라 인덱스 탐색)
data/processed/eyeblink8_eyes.npz   models/{autoencoder,classifier,onnx}
results/{pi_ours,pi_ear,dim_sweep,ear_dense}.*   docs/figures/
```

주요 명령:
```
# 데이터셋(양눈 재생성)
python -m src.dataset.build_eyeblink8_npz --root data/_legacy_public/eyeblink8/eyeblink8 --out data/processed/eyeblink8_eyes.npz --stride 2
# 학습
python -m src.encoder.train_autoencoder   --data data/processed/eyeblink8_eyes.npz --out models/autoencoder
python -m src.classifier.train_classifier --data data/processed/eyeblink8_eyes.npz --encoder models/autoencoder/encoder.pt --out models/classifier
# 평가
python -m src.eval.eval_events            --data data/processed/eyeblink8_eyes.npz --encoder models/autoencoder/encoder.pt --classifier models/classifier/classifier.pt
# 엣지
python -m src.deploy.export_onnx   --encoder models/autoencoder/encoder.pt --classifier models/classifier/classifier.pt --out models/onnx
python -m src.deploy.bench_latency --model models/onnx/pipeline.onnx --iters 500
```

## 11. 현재 스코프 vs 추후 연구

**현재 스코프**: 양눈 벡터 깜빡임 검출 + 구조적 프라이버시(벡터만 저장) + 엣지 실시간.
데이터는 Eyeblink8 + (예정) 자체 웹캠 소량.

**추후 연구 (현재 주장 아님)**:
- **신원 재식별 / disentanglement** — 다신원 데이터 필요. 지금은 PoC만.
- **데모 UI** — 눈 감지 on/off 표시, 5초 단위 캡처 (특허 지향)
- **노이즈 주입** anti-model-inversion (학습데이터 무단사용·복원공격 방어)

## 12. 남은 작업 (로드맵)

- **A. 양눈 코어 갱신** (완료: 재생성·재학습 기능 확인) — EAR 베이스라인 양눈 평균 업데이트만 남음
- **C. 엣지** (진행: ONNX·데스크탑 지연 완료) — **Pi5 실측** 남음
- **B. 엄밀성** (새 데이터 후): 정식 train/val/**test** 분리, 여러 시드 반복(평균±표준편차)
- **D. 자체 데이터**: 웹캠 수집(피험자·조명·거리·안경 다양화), 수동라벨 test 일부, 크로스도메인 평가
- **E. 눈 건강 지표**: 깜빡임 → 빈도·불완전 깜빡임·PERCLOS (문헌 기준 위험 지표, 진단 아님)
- **F. 초기 등록 구간 실험 → 논문화 (IEEE Sensors)**

## 13. 알려진 한계 / 주의

- **작은 검증셋**: val이 2명·74이벤트 → "동급"은 정직한 표현이지 "우월"은 아님. 통계적 유의성 없음(단일 시드).
- **단일 데이터셋**: Eyeblink8만, 정면·통제·실내. 실제 웹캠·다양한 사람에서 미검증.
- **양눈 false-alarm 상승**: 128차원 병목 대비 이미지가 커서. 개선 여지 존재.
- **평가 프로토콜 혼재 주의**: 유용성=피험자분리·이벤트단위 vs disentanglement PoC=프레임분할·프레임AUC —
  직접 비교 불가. 서로 다른 숫자.
- **엣지 미확정**: 데스크탑 추정만. Pi5 실측 필요.

## 14. 용어 정리

- **벡터(embedding/latent)**: 눈 이미지를 요약한 128개 숫자. 사람이 읽어도 눈인지 모름.
- **인코더/디코더**: 이미지→벡터 / 벡터→이미지. 디코더는 배포 안 함.
- **EAR (Eye Aspect Ratio)**: 눈 랜드마크로 눈 벌어진 정도를 계산하는 고전적 규칙 지표. 우리의 베이스라인.
- **MLP (다층 퍼셉트론)**: 가장 단순한 신경망. 여기선 벡터→깜빡임 판정.
- **오토인코더**: 입력을 압축(인코더)했다 복원(디코더)하도록 학습하는 비지도 모델.
- **이벤트 단위 recall**: 깜빡임 사건을 몇 %나 잡았나(프레임 1개라도). 우리의 1순위 지표.
- **피험자 분리(subject-separated)**: 같은 사람이 학습·평가에 안 겹치게 분할. 겹치면 성능 과대평가.
- **gradient reversal (GRL/DANN)**: 신원 분류기의 그래디언트를 뒤집어 인코더가 신원을 못 맞히게 하는 적대 학습.
- **PERCLOS**: 눈이 감겨 있는 시간 비율. 피로·졸음 지표.
- **ONNX**: 프레임워크 독립 모델 포맷. 경량 배포·엣지 실행에 사용.
