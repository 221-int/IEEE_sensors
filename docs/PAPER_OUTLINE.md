# PAPER_OUTLINE — IEEE Sensors Letters 초안 구성

> **작성 2026-08-05.** 범위는 `PROJECT_DIRECTION.md` §4 를 따른다.
> **분량 제약 — 약 4페이지.** 미팅에서 나온 현실적 추정 (전사문 09:07):
> Introduction + Related Work 로 **1페이지 이상**, Method 는 **Overall Architecture 그림 필수**,
> Experiments 는 표 + 그래프 몇 개. **여유가 거의 없다.**
>
> 🔴 **신원 식별 · 디코더 복원 방지 내용을 본문 중심에 넣지 않는다.**
> 허용/금지 문장은 `PROJECT_DIRECTION.md` §5 를 그대로 따른다.

---

## 지면 배분 (목표)

| 섹션 | 목표 분량 | 필수 그림/표 |
|---|---|---|
| I. Introduction | 0.6 p | — |
| II. Related Work | 0.4 p | — |
| III. Proposed Method | 1.0 p | **Fig. 1 Overall Architecture** |
| IV. Experimental Setup | 0.6 p | — |
| V. Results and Discussion | 1.2 p | **Table I**, **Table II**, Fig. 2 (, Fig. 3) |
| VI. Conclusion | 0.2 p | — |

> Fig. 3(실패 사례)은 **지면이 남을 때만.** Fig. 2 와 Table II 가 우선이다.

---

## I. Introduction

**들어갈 내용**

1. 눈 깜빡임 검출의 응용 — 눈 건강(장시간 화면 응시 시 깜빡임 감소), 졸음·피로 모니터링.
   화상회의처럼 카메라가 상시 켜진 환경이 늘고 있다는 맥락.
2. 기존 접근의 한계
   - **EAR 기반**: 랜드마크 좌표에서 계산한 기하 비율 **스칼라 하나**. 조도·자세 등
     주변 조건 정보를 담지 못한다.
   - **원본 이미지 기반 CNN**: 성능은 좋지만 판정 경로에 원본 픽셀이 그대로 흐른다.
     카메라 영상을 직접 다룬다는 점이 사용자 거부감으로 이어질 수 있다.
3. 제안 — 프레임을 encoder 로 **저차원 embedding vector** 로 바꾸고,
   **판정 단계는 벡터만** 본다. 경량 구조라 **edge device 에서 실시간**으로 돈다.
4. Contributions (3~4개, `RESEARCH_PLAN.md` §5)

**주의**

- ⚠️ Contribution 문장에 **아직 확보되지 않은 것**을 쓰지 않는다. 현재 미확보:
  image_cnn 비교(#3), 조건별 강건성 전체(#4), v2 edge 실측(#5).
- ⚠️ "privacy-preserving", "identity-removed", "irreversible" 같은 단어를 쓰지 않는다.
  쓸 수 있는 것은 *"the classification stage operates on embeddings rather than raw eye images"*
  수준의 **구조 서술**까지다.

---

## II. Related Work

**들어갈 내용 (3문단, 각 3~4문장)**

1. **눈 깜빡임 검출** — EAR 기반 규칙, 프레임 단위 CNN, 시퀀스 모델(ConvLSTM 등).
   mEBAL2 원논문(Daza et al., PRL 2024)을 데이터셋 출처로 인용.
2. **경량 / 온디바이스 깜빡임 검출** — edge device 에서 동작하는 선행 연구.
   → ⚠️ **현재 `docs/v2/RELATED_WORK.md` 에 1건뿐이다. T6-1 조사 필요.**
3. **표현 학습 기반 접근** — encoder 로 뽑은 표현 위에서 판정하는 방식이
   깜빡임 검출에서는 거의 다뤄지지 않았다는 위치 설정.
   → ⚠️ **T6-2 로 재확인 필요.** "없다"를 주장하려면 검색 근거가 있어야 한다.

**주의**

- 🔴 **mEBAL2 원논문의 accuracy 99% 와 우리 PR-AUC 를 나란히 놓지 않는다.**
  입력 스펙트럼(NIR+RGB vs RGB 단독) · 인원(180 vs 57) · 이벤트 정의 · 분할 · 지표가 전부 다르다.
  "우리가 99% 에 근접했다"는 틀린 문장이다.
- 눈 주변 영역의 식별성 / 표현의 신원 누출 문헌(§B·§C·§E)은 **이 논문에 넣지 않는다.**
  넣으면 "그래서 너희는 신원을 어떻게 했느냐"는 질문을 자초한다.

---

## III. Proposed Method

**A. Overview — Fig. 1**

```
Camera frame
  → Eye region detection & tracking (MediaPipe landmarks, coordinates discarded)
  → Both-eye crop, rotation-aligned, grayscale 64×160 (margin 2.2 × inter-eye distance)
  → Encoder (asymmetric CNN, "vpres")
  → Embedding vector  z ∈ R^D
  → 19-frame causal ring buffer
  → Temporal head (1D TCN, dilation 1/2/4) + classification head (MLP)
  → blink / unblink
```

Fig. 1 에서 **판정 헤드로 들어가는 화살표가 embedding vector 뿐**임을 시각적으로 드러낸다.
디코더는 그림에 넣지 않는다 (배포되지 않으므로).

**B. Eye Crop Normalization**

- 두 눈 중심의 중점을 기준, 두 눈을 잇는 선을 수평으로 회전 정렬.
- 크롭 폭 = 눈 사이 거리 × 2.2 → 고정 64×160 리사이즈.
  **카메라 거리가 변해도 크롭 스케일이 정규화된다.**
- 입력 정규화는 프레임 단위 표준화 `(x − μ) / σ`.
  근거: 광학 교란(밝기·대비·선명도)의 재식별 프로브 정확도가 0.329 → 0.080 으로 떨어진다.
  → 논문에서는 **"지름길(shortcut) 제거"** 로 서술한다. 프라이버시 기전으로 쓰지 않는다.

**C. Encoder Design — 핵심 설계 논거**

이 논문에서 **가장 독자적인 설계 근거**다. 반드시 넣는다.

- 눈꺼풀 상하 간격은 크롭 세로 64 px 중 **8.7 px** (실측: 눈꺼풀간격/눈사이거리 = 0.1201).
- 흔히 쓰는 대칭 stride-2 conv 4단을 쓰면 세로가 1/16 이 되어 **0.55 px** 가 되고,
  3×3 커널이 변화를 볼 수 없어 **깜빡임 신호가 특징맵에서 사라진다.**
- → **세로 총 stride 를 2 로 제한**하고(8.73/2 = 4.37 px), 가로만 32배 줄인다.
  깜빡임은 세로 운동이고 가로 160 px 은 두 눈을 나란히 담기 위한 것이지 해상도가
  필요해서가 아니다. **비대칭 stride 가 설계의 핵심이다.**

> 🔴 **이 주장에는 절제 실험이 필요하다.** 현재 근거는 **계산 논거뿐**이고
> 대칭 stride(sym16)를 실제로 학습해 본 적이 없다 (`grep -rl sym16 results/` → 0건).
> 심사자가 반드시 묻는다. → `TASKS.md` **T3-8** 로 신설. 비용은 `--arch sym16` 한 줄.
>
> | 구조 | 세로 stride | 병목에 남는 눈꺼풀 | conv MMAC | PR-AUC |
> |---|---:|---:|---:|---|
> | sym16 (대칭, 흔한 설계) | 16 | **0.55 px** | 9.22 | ⚠️ **미측정** |
> | **vpres (채택)** | 2 | **4.37 px** | **12.41** | 0.9886 ± 0.0038 |
> | vfull (세로 안 줄임) | 1 | 8.73 px | 24.21 | ⚠️ 미측정 |
>
> **vpres 는 sym16 보다 conv 연산이 34% 많다.** 그 비용을 정당화하려면
> sym16 이 실제로 더 나쁘다는 숫자가 있어야 한다. 결과가 비슷하게 나오면
> **설계 주장을 바꿔야 한다** (sym16 이 오히려 edge 에 유리하므로).

**D. Causal Temporal Modeling**

- 프레임마다 19장을 다시 인코딩하면 비용이 19배 → edge 예산 초과.
- → **프레임당 인코딩 1회 + 벡터 19개 링버퍼 + 시간 헤드.**
  시간 헤드는 이벤트당 0.2 MMAC 수준으로 사실상 무시 가능.
- 얼굴 해소 실패 프레임은 크롭이 없다 → 0 으로 채우면 "변화 없음"과 "모름"이 구분되지 않는다.
  **마스크를 함께 넣고 pooling 에서 제외**한다.

**E. Deployment**

- ONNX 로 컴파일한 encoder + 시간 헤드 + 판정 헤드.
- 랜드마크 좌표는 크롭 좌표를 얻는 휘발성 용도로만 쓰고 버린다.
- 디코더는 배포하지 않는다. (**한 문장. 여기서 멈춘다.**)

---

## IV. Experimental Setup

**A. Dataset**

- **a 58-subject subset of mEBAL2** (전체 mEBAL2 는 180명임을 명시).
  RGB webcam 스트림만 사용 (원논문은 NIR 2대 + RGB).
- 배포본 구조: mEBAL1 38명(2020) + mEBAL2 신규 20명(2022).
- **U18 제외 → 57명.** 사유: 다중 인원 동시 녹화 환경에서 주 피험자 미검출 시 옆자리
  인물이 크롭된 비율이 이벤트의 31%. **각주로 명시한다.**
- 유효 이벤트 **27,758** (blink 13,820 / unblink 13,938), 크롭 532,109장.
- 이벤트 단위: 19프레임 고정창.

**B. Protocol**

- **피험자 분리 5-fold**, 이벤트 수 + 수집 배치 **이중 층화**.
- **3 seed** (0,1,2). 15런.
- 임계값 격자·시드·분할·부트스트랩은 단일 구현을 공유.
- val 에서 고르고 val 에서 보고하지 않는다.

**C. Baselines**

| 이름 | 설명 |
|---|---|
| EAR-rule | drop_ratio (창 안 상대 하강) + 임계값 |
| EAR-head | EAR 스칼라 → 동일 시간·판정 헤드 |
| Image-CNN | 크롭 → 소형 CNN 직접 판정 (embedding 병목 없음) ⚠️ **미측정** |
| Ours | 크롭 → encoder → embedding vector → 동일 시간·판정 헤드 |

- EAR 은 **mEBAL2 제공 랜드마크**로 계산한다. 같은 검출기·같은 얼굴 선택 규칙을 써야
  "다른 검출기라서 졌다"는 반론이 봉쇄된다.
- EAR 변형은 **강한 쪽(drop_ratio)** 을 쓴다. min_ear 보다 AUC 가 0.17 높다.

**D. Metrics**

- 주 지표 **PR-AUC**. 함께 보고: ROC-AUC, Accuracy, Precision, Recall, F1.
- 판정: (ours − baseline) 차이의 **짝지은 피험자 클러스터 부트스트랩 95% CI**,
  비열등 마진 **δ = 0.02** (결과를 보기 전에 고정).
- Edge 지표: 파라미터 수, 모델 크기, MMAC, 프레임당 추론 시간, e2e 지연 p50/p95/p99,
  sustained FPS, RSS peak, CPU %.

**E. Edge Platform**

- Raspberry Pi 5 (16 GB, active cooling), Debian 13, Python 3.11,
  onnxruntime (intra-op 2 threads, spinning disabled), MediaPipe, OpenCV (NEON FP16).
- CPU governor `performance`. 각 모드 5분 지속 측정. 스로틀 플래그 기록.

---

## V. Results and Discussion

**A. Detection Accuracy — Table I**

| Method | PR-AUC | ROC-AUC | Acc | Prec | Rec | F1 |
|---|---|---|---|---|---|---|
| EAR-rule | 0.8931 ± 0.0136 | | | | | |
| EAR-head | 0.9724 ± 0.0096 | | | | | |
| Image-CNN (max, 문헌 원형) | (T3-6 확정 후) | | | | | |
| Image-CNN (+ our temporal head) | (T3-6 확정 후) | | | | | |
| **Ours** | **0.9886 ± 0.0038** | | | | | |

### 🔵 2026-08-06 T3-1 종결 — **주 비교가 예상과 반대로 나왔다**

탐색 fold 0·1 × 3seed, 21명 (`results/v2/imgcnn_{max,head}_pilot.json`,
`cmp_ours_vs_imgcnn_{max,head}.json`, `cmp_imgcnn_head_vs_max.json`).
🔴 **탐색값이다. 확정 값으로 인용하지 않는다.**

| 방법 | PR-AUC (런평균) | 총 MMAC | params | 시간 처리 |
|---|---:|---:|---:|---|
| **ours** (vpres D16) | 0.9850 ± 0.0022 | **12.44** | **84,049** | TCN 헤드 |
| image_cnn_head | **0.9866 ± 0.0021** | 31.81 | 476,161 | TCN 헤드 (동일) |
| image_cnn_max | 0.9066 ± 0.0182 | 31.81 | 470,561 | max pooling (원문 §5.1) |

```
ours − image_cnn_max      +0.0735  [+0.0517, +0.0945]
image_cnn_head − max      +0.0783  [+0.0546, +0.0994]   ← 백본 동일, 시간처리만 다름
ours − image_cnn_head     −0.0048  [−0.0083, −0.0017]   ← 우리가 더 낮다
```

**서술 규칙 (반드시 지킬 것)**

1. ❌ **"임베딩이 이미지보다 정확하다"는 쓸 수 없다.** ours 는 `image_cnn_head` 보다
   근소하게 **낮다**.
2. ✅ 쓸 수 있는 것: **"동등한 정확도를 연산 2.6배·파라미터 5.7배 적게 얻는다"**
   — δ=0.02 기준 판정 **non_inferior** (CI 하한 −0.0083 > −0.02).
   PROTOCOL §9-1 이 원래 정해 둔 프레이밍(비열등, 우월 아님)과 일치한다.
3. 🔴 **`ours − image_cnn_max = +0.0735` 를 단독으로 쓰지 않는다.** 그 격차는
   **표현이 아니라 시간 모델링**이다 — 같은 백본에 우리 헤드만 붙이면 +0.0783 이 나온다.
   `ear_rule` vs `ear_head` 와 **정확히 같은 함정**이고 PROTOCOL §9 대조군 1 이
   막으려던 것이다.
4. 🔴 **각주 필수**: 원문은 프레임별 open/closed 라벨로 학습했고 우리는 창당 라벨
   하나(max-pooling MIL)뿐이다. **문헌 방법을 과소평가했을 수 있다.**
5. ⚠️ `ours − image_cnn_head` 는 두 추정량이 갈린다 — 풀링 −0.0048(CI 상한 <0) vs
   런평균 −0.0016 ± 0.0026 (시드 std 0.0031 미만). **"미측정"으로 적고 어느 쪽으로도
   단정하지 않는다.**

- ⚠️ **Recall · F1 열은 현재 계산되지 않는다.** 저장된 지표는 accuracy · precision ·
  pr_auc · roc_auc 뿐이다 (`results/v2/train_encoder.json`). → `TASKS.md` T3-6
- (Ours − EAR-head) = **+0.0151, 95% CI [+0.0106, +0.0203]** → 비열등 충족 (δ=0.02).
- ⚠️ **우월(superior)을 강하게 밀지 않는다.** 안경 서브그룹에서 이득이 사실상 사라지고
  (+0.0084 이벤트 가중), 그 결론은 한 명(U1)에 좌우된다. 우월을 밀면 그 표가 반론이 된다.

**B. Robustness across Conditions — Fig. 2**

- 안경 착용 / 미착용, 수집 배치 2020 / 2022, 광학 3분위(밝기·대비·선명도), 합성 저조도.
- **전체 + 조건별 숫자를 항상 함께** 낸다. 이벤트 가중을 주 숫자로, 피험자 평균을 병기.
- ⚠️ **안경군에서 이득이 거의 없다는 실측을 숨기지 않는다.** PROTOCOL §9 는 반대 방향을
  예상했으나 실측에서 뒤집혔다. 예상이 아니라 실측을 쓴다.
- 평가할 수 없는 조건(실제 저조도, 거리, 고개 각도, 부분 깜빡임, 빠른/느린 깜빡임)은
  **Limitations 에 명시**하고 표에 추정치로 채우지 않는다.

**C. Model Complexity and Edge Performance — Table II** 🔵 **2026-08-06 재설계**

### 왜 다시 짰나 — e2e 만 내면 기여가 사라진다

하네스 실측(`results/v2/`, 데스크톱 1,666프레임, **인용 금지**)에서 단계 비중이 나왔다:

| stage | p50 (ms) | e2e 비중 |
|---|---:|---:|
| read | 2.01 | 29% |
| **detect (MediaPipe)** | **4.12** | **59%** |
| crop | 0.36 | 5% |
| **encode (우리 인코더)** | **0.26** | **4%** |
| decide (head, stride 1) | 0.14 | 2% |

🔴 **read 와 detect 는 세 방법이 똑같이 치르는 비용이다.** 같은 프레임을 읽고 같은
MediaPipe 로 같은 메시를 뽑는다. 즉 e2e 의 **88% 가 공통 항**이고, 방법이 달라서
생기는 차이는 **crop+encode+decide 11%** 안에서만 일어난다.

→ e2e 한 열로 세 방법을 비교하면 **공통 비용이 분모를 채워 차이가 묻힌다.**
   인코더 MMAC 을 72% 줄여도(vpres→vdrop) e2e 는 3% 밖에 안 움직인다.
   그 표는 "세 방법이 다 비슷하다"는 잘못된 인상을 준다.

### 재설계 — 공통 항과 방법 항을 분리한다

**Table II-a. 정적 비용** (Pi 불필요, 이미 측정됨 — `results/v2/export_onnx.json`)

| Method | Params | Model size | MMAC/frame |
|---|---:|---:|---:|
| EAR (rule) | 0 | — | ~0 |
| EAR-head | (T3-6 재실행 후) | | |
| Image-CNN (mEBAL) | 471,536 | | 31.81 |
| **Ours (vpres, D=16)** | **84,049** | **335 KB** | **12.49** |

- Ours 내역: encoder 79,424 + head 4,625 / encoder.onnx 311.4 KB + head.onnx 23.3 KB
- MMAC 내역: encoder 12.44 + head **0.0459** (stride 1, 매 프레임 판정). head 비중 0.37%
- ⚠️ **head 를 stride 19 로 세지 않는다.** 창 경계에서만 판정하면 깜빡임을 최대
  18프레임(600 ms @30 fps) 늦게 잡는다. 연속 검출의 정직한 값은 stride 1 이다.

**Table II-b. Pi 5 런타임** (640×480 @ 30 fps, 5분 지속, `--intra-threads 2 --no-spin`)

| Method | crop | encode | decide | **method 소계** | e2e p99 | FPS | RSS MB | CPU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EAR | | — | | | | | | |
| Image-CNN | | | | | | | | |
| **Ours** | | | | | | | | |

+ 공통 항(세 방법 동일): read ___ ms, detect ___ ms → 표 아래 한 줄로 **한 번만** 적는다.

- **판정 열은 `method 소계`** 다. e2e p99 는 **예산 게이트(G-E1 ≤ 33.3 ms)** 확인용이지
  방법 비교용이 아니다. 둘의 역할을 섞지 않는다.
- 해상도를 720p 로 올리면 detect 비중이 더 커져 공통 항이 더 지배적이 된다
  (EXPERIMENT_PLAN §6-3). **보조 측정에서도 분해를 그대로 낸다.**

### 논지 (프레이밍)

- ❌ "엣지 실시간이 어렵다 → 시스템 기여" — v1 에서 예산의 36% 밖에 안 나와 이 프레이밍이
  스스로 약해진 전례가 있다. v2 도 여유가 클 것으로 보인다.
- ✅ **"EAR 대비 벡터 표현의 추가 비용을 동일 조건에서 정량화했고, 그 비용이 파이프라인의
  몇 % 인지 분해해 보였다."** encoder 가 e2e 의 4% 라는 것은 약점이 아니라 **결과**다 —
  표현을 학습으로 얻는 대가가 검출기 비용에 묻힐 만큼 작다는 뜻이다.
- ✅ head 0.0459 MMAC/frame 은 **병목 설계의 직접적 이득**이다. 병목이 없었다면
  (F=512) 시간 헤드가 2.36 MMAC/frame 으로 **51배** 커진다. 이 대비를 한 줄 넣는다.

⚠️ v1 수치(e2e p99 12.06 ms / 86 fps)는 인코더 구조와 해상도가 달라 **인용 금지**
(PROTOCOL §13).

**D. Failure Case Analysis — Fig. 3 (지면이 남으면)**

- U1: ours 0.8200 / EAR-head 0.9802. 오류 36건 중 **33건이 blink 미검출**.
- 원인은 **고유한 전이 실패** — train−test 격차 0.1798 로 57명 중 최대(중앙값 0.0063).
  좌석 오염·크롭 기하·결측·시드 잡음은 전부 배제됨.
- 한계로 정직하게 서술하면 신뢰도가 올라간다.

**E. Discussion / Limitations**

1. 안경 착용 조건에서 이득이 작다. 원인 미규명.
2. 조도·거리·고개 각도·부분 깜빡임은 **현 데이터셋에 라벨이 없어 평가하지 못했다.**
3. 이벤트가 19프레임 고정창이라 **연속 영상 event recall / frame false-alarm 은 측정하지 않았다.**
4. 57명 단일 데이터셋. 크로스 데이터셋 일반화 미검증.
5. **판정 단계가 원본 픽셀을 쓰지 않는다는 것은 구조적 성질이며, 표현으로부터 신원이
   복원되지 않음을 의미하지 않는다.** 그 평가는 이 논문의 범위 밖이다.
   → 🔴 이 한 문장은 **반드시 넣는다.** 심사자가 물을 것을 미리 선을 긋는다.

---

## VI. Conclusion

- 원본 눈 이미지 대신 encoder 의 embedding vector 만으로 눈 깜빡임을 검출하는 파이프라인을 제시.
- 피험자 분리 5-fold × 3 seed 에서 EAR 기반 및 이미지 기반 방법과 동일 조건 비교.
- Raspberry Pi 5 에서 실시간 동작 확인.
- Future work: 조도·자세·거리 조건이 라벨된 데이터셋에서의 강건성 검증,
  표현 수준의 정보 노출 분석. (**한 줄. 구체적 기법은 쓰지 않는다 — 특허 출원 전이다.**)

> 🔴 **Future work 서술 주의.** 특허 출원 전이라면 GRL·VIB 같은 구체적 억제 기법을
> 논문에 적지 않는다. 논문 공개가 특허 심사에서 선행 기술이 될 수 있다.

---

## 작성 순서 제안

1. **III. Proposed Method** — 이미 다 확정돼 있다. Fig. 1 과 함께 먼저 쓴다.
2. **IV. Experimental Setup** — 프로토콜이 확정돼 있다.
3. **V-A Table I** — 3개 행은 이미 있다. Image-CNN 행만 비어 있다.
4. (실험) T3-1 → T5-x → V-C Table II
5. (실험) T4-1, T4-4 → V-B Fig. 2
6. **II. Related Work** — T6-1~T6-3 조사 후
7. **I. Introduction** — Contribution 이 확정된 뒤 마지막에
8. **VI. Conclusion**

---

## 논문 서술 금지 목록 (체크리스트)

투고 전 본문을 검색해 다음이 없는지 확인한다.

- [ ] "privacy-preserving" / "프라이버시 보호"
- [ ] "identity-disentangled" / "신원 제거" / "de-identification"
- [ ] "irreversible" / "cannot be reconstructed" / "복원 불가"
- [ ] "anonymized" / "익명화"
- [ ] mEBAL2 의 "180 subjects" 를 우리 실험 규모처럼 쓴 문장
- [ ] mEBAL2 원논문 99% 와 우리 지표를 같은 표/문장에 놓은 것
- [ ] v1(Eyeblink8 8명) 수치
- [ ] v1 Pi 5 지연 수치 (12.06 ms / 86 fps)
- [ ] 측정하지 않은 조건을 측정한 것처럼 쓴 문장
