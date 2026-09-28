# PAPER_OUTLINE — IEEE Sensors Letters 초안 구성

> **작성 2026-08-05 · 최종 갱신 2026-08-11.** 범위는 `PROJECT_DIRECTION.md` §4 를 따른다.
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
  → Embedding vector  z ∈ R^16
  → 19-frame ring buffer + validity mask
  → Temporal head (1D TCN, dilation 1/2/4) + classification head (MLP)
  → blink / unblink
```

Fig. 1 에서 **판정 헤드로 들어가는 화살표가 embedding vector 뿐**임을 시각적으로 드러낸다.
디코더는 그림에 넣지 않는다 (배포되지 않으므로).

**B. Eye Crop Normalization**

- 두 눈 중심의 중점을 기준, 두 눈을 잇는 선을 수평으로 회전 정렬.
- 크롭 폭 = 눈 사이 거리 × 2.2 → 고정 64×160 리사이즈.
  **카메라 거리가 변해도 크롭 스케일이 정규화된다.**
- 입력 정규화는 현재 프레임 안에서만 하는 표준화
  `(x − μ) / max(σ, 0.255)`다. `0.255 = 10⁻³ × 255` 하한은 거의 균일한 크롭에서
  0으로 나누는 것을 막는다. **광학값의 전역 차이를 그대로 쓰는 지름길을 줄인다**고만 쓴다.

**C. Encoder Design**

- 실측 눈꺼풀간격/눈사이거리 = 0.1201. 크롭 원본 높이는 `0.88 × 눈사이거리`이므로,
  64 px 크롭에서 눈꺼풀 간격은 **8.73 px**다. 이는 `0.1201 × 160 / 2.2`로 계산하며,
  `0.1201 × 64`가 아니다.
- encoder 는 Conv-BN-ReLU 4블록(채널 16/32/48/64, 커널 3×5/3×3/3×3/3×3,
  stride (세로,가로) = (1,4)/(2,2)/(1,2)/(1,2))이다. 총 stride 는 세로 2, 가로 32,
  가로 평균 pooling 뒤 선형층으로 D=16을 만든다. **79,424 params, 12.44 MMAC/frame.**

> 🔴 **2026-08-06 탐색 절제 — 성능으로 설계 주장을 지지하지 않는다.**
>
> 교락 없는 대조군 `vdrop`(vpres 와 커널·채널·가로 stride 가 전부 같고 **세로 stride 만** 16)로 검정했다.
>
> | 구조 | 세로 stride | 병목 눈꺼풀 | conv MMAC | PR-AUC (탐색 fold 0·1) |
> |---|---:|---:|---:|---|
> | sym16 (대칭, 교락됨) | 16 | 0.55 px | 9.22 | 0.9832 ± 0.0036 |
> | **vdrop** (교락 없음) | 16 | 0.55 px | **3.44** | 0.9817 ± 0.0034 |
> | **vpres** (채택) | 2 | 4.37 px | 12.41 | 0.9850 ± 0.0022 |
> | vfull | 1 | 8.73 px | 24.21 | 0.9825 ± 0.0024 |
>
> **vpres − vdrop = +0.0010 [−0.0032, +0.0069]** (`results/v2/cmp_vpres_vs_vdrop.json`,
> fold 0·1 × 3 seed = 6런, 21명 탐색).
> "세로를 1/16 로 줄이면 신호가 사라진다"는 **파국적 열화를 예측했지만 일어나지 않았다.**
> 세로 해상도와 성능은 단조도 아니다.
>
> → 이 절제는 다섯 fold를 모두 덮지 않으므로 **서술적 탐색 결과**로만 쓴다.
> "세로 해상도가 중요하지 않다" 또는 "더 싼 인코더가 동등하다"고 결론 내리지 않는다.
> `vpres` 는 나머지 확정 산출물과의 일관성을 위해 유지한다.

**D. Temporal Modeling on Embeddings**

- 프레임마다 19장을 다시 인코딩하면 비용이 19배 → edge 예산 초과.
- → **프레임당 인코딩 1회 + 벡터 19개 링버퍼 + 시간 헤드.**
  시간 헤드는 dilation 1/2/4의 residual 1-D Conv-BN-ReLU 3층과 MLP(32→64→1,
  dropout 0.3)이며, stride 1 판정 비용은 **0.0459 MMAC/frame (전체의 0.37%)**이다.
- 얼굴 해소 실패 프레임은 크롭이 없다 → 0 으로 채우면 "변화 없음"과 "모름"이 구분되지 않는다.
  **마스크를 각 convolution 앞에 적용하고, 두 pooling에서 제외**한다. 6프레임 이상
  미해소인 이벤트는 데이터셋에서 제외한다.
- 대칭 padding이라 창 안의 뒤쪽 프레임을 쓸 수 있어 **causal 이 아니다.** 다만 판정 시점에
  현재 19프레임 버퍼 밖의 프레임은 기다리지 않으므로 추가 버퍼링 지연은 없다.

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
- 유효 이벤트 **27,758** (blink 13,820 / unblink 13,938). 추출 파이프라인은 고유 크롭
  **532,109장**을 저장하며, 겹치는 이벤트는 이 프레임을 재사용한다.
- 이벤트 단위: 19프레임 고정창. 6프레임 이상 얼굴 미해소 이벤트는 제외한다.

**B. Protocol**

- **피험자 분리 5-fold**, 이벤트 수 + 수집 배치 **이중 층화**, seed 0/1/2의 15런.
- 회전 `i`에서 test = fold `i`, validation = fold `(i+1) mod 5`, 나머지는 train이다.
- 임계값은 validation의 모든 고유 점수와 그 중점에서 Accuracy를 최대화해 한 번 고르고,
  test에 바꾸지 않고 적용한다. 분할·시드·임계값 선택·부트스트랩은 단일 구현을 공유한다.
- EAR 두 변형은 확률적 프론트엔드가 없으므로 세 seed에서 비트 동일해야 하며, 실제로 그렇다.

**C. Training**

- 학습 방법은 BCE-with-logits + Adam(1e−3), batch 32, 최대 80 epoch, validation PR-AUC
  기준 patience 10 early stopping이다. 매 런 deterministic 설정을 켠다.
- Image-CNN은 mEBAL 구조를 64×160 그레이스케일 두 눈 크롭과 이벤트 라벨에 맞게 이식했다.
  원문 프레임별 open/closed 학습과 같지 않다는 점은 베이스라인 해석에 계속 남긴다.

**D. Baselines**

| 이름 | 설명 |
|---|---|
| EAR-rule | drop_ratio (창 안 상대 하강) + 임계값 |
| EAR-head | EAR 스칼라 → 동일 시간·판정 헤드 |
| Image-CNN (max) | mEBAL 원논문 CNN + 원문 §5.1 의 프레임 점수 **max pooling** |
| Image-CNN (+head) | 같은 백본 + **우리와 동일한 시간·판정 헤드** (프론트엔드만 교체한 통제 비교) |
| Ours | 크롭 → encoder → embedding vector → 동일 시간·판정 헤드 |

- EAR 은 **mEBAL2 제공 랜드마크**로 계산한다. 같은 검출기·같은 얼굴 선택 규칙을 써야
  "다른 검출기라서 졌다"는 반론이 봉쇄된다.
- EAR 변형은 **강한 쪽(drop_ratio)** 을 쓴다. min_ear 보다 AUC 가 0.17 높다.
- Image-CNN 은 **우리가 설계하지 않았다.** mEBAL2 데이터셋 저자들이 제안·벤치마크한
  구조를 이식했다(Daza et al., ICMI '20 Companion §4, 원문 확인). 그래야
  "약한 베이스라인을 골랐다"는 반론이 막힌다.
  ⚠️ 원문은 **프레임별 open/closed 라벨**로 학습했고 우리는 창당 라벨 하나뿐이라
  (max-pooling MIL) **문헌 방법을 과소평가했을 수 있다. 각주 필수.**

**E. Metrics**

- 주 지표 **PR-AUC**(blink를 positive class). 함께 보고: ROC-AUC, Accuracy, Precision,
  Recall, F1. PR-AUC·ROC-AUC은 15런 평균±표준편차, 임계값 지표는 15런 평균으로 보고한다.
- 판정: 시드 3개의 held-out 예측을 이어 붙인 뒤 피험자 57명을 복원추출해 PR-AUC 차이를
  다시 계산하는 **짝지은 피험자 클러스터 부트스트랩 2,000회, 95% CI**. 비열등 마진
  **δ = 0.02**는 최종 비교 전에 고정했다.
- Edge 지표: 파라미터 수, 모델 크기, MMAC, 프레임당 추론 시간, e2e 지연 p50/p95/p99,
  sustained FPS, RSS peak, CPU %.

**F. Edge Platform**

- Raspberry Pi 5 (16 GB, active cooling), Debian 13, Python 3.11,
  onnxruntime (intra-op 2 threads, spinning disabled). 지연 측정 frontend는 MediaPipe
  FaceMesh(한 얼굴, `refine_landmarks=False`)이고, 정확도 실험은 mEBAL2 제공 mesh를 쓴다.
- CPU governor `performance`. 각 모드 5분 지속 측정. 스로틀 플래그 기록.

---

## V. Results and Discussion

**A. Detection Accuracy — Table I**

| Method | PR-AUC | ROC-AUC | Acc | Prec | Rec | F1 | MMAC | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|
| EAR-rule | 0.8931 ± 0.0136 | 0.9169 ± 0.0146 | 0.8505 | 0.8268 | 0.8878 | 0.8550 | ~0 | 0 |
| EAR-head | 0.9724 ± 0.0096 | 0.9738 ± 0.0113 | 0.9276 | 0.9312 | 0.9228 | 0.9268 | 0.0462 | 5,009 |
| Image-CNN (max, 문헌 원형) | 0.9114 ± **0.0291** | 0.9404 ± 0.0207 | 0.8917 | 0.8526 | 0.9480 | 0.8971 | 31.81 | 470,561 |
| Image-CNN (+ our temporal head) | **0.9906 ± 0.0037** | **0.9930 ± 0.0036** | 0.9651 | 0.9713 | 0.9584 | 0.9644 | 31.85 | 476,161 |
| **Ours** | 0.9886 ± 0.0038 | 0.9914 ± 0.0037 | 0.9620 | 0.9605 | 0.9636 | 0.9618 | **12.49** | **84,049** |

🔵 **전부 확정 정밀도** (5 fold × 3 seed, 57명, 27,758 이벤트).
출처: `results/v2/train_encoder_final.json`, `train_image_cnn_{head,max}_final.json`.
세 런의 config 는 `front`/`out`/`models_dir` 외 **전 항목 동일**이고, `ear_head`·`ear_rule`
값이 세 런에서 **비트 동일**하다(front 와 무관하므로 그래야 한다) — 조건 동일성의 독립 검증.

**짝지은 피험자 클러스터 부트스트랩** (풀링 / 런평균 병기)

| 비교 | 풀링 | 95% CI | δ=0.02 판정 | 런평균 | 시드 std |
|---|---:|---|---|---:|---:|
| **ours − Image-CNN(+head)** ★ | **−0.0040** | [−0.0063, −0.0020] | **non_inferior** | −0.0019 ± 0.0024 | 0.0031 |
| ours − Image-CNN(max) | +0.0766 | [+0.0556, +0.0986] | superior | +0.0772 ± 0.0291 | 0.0460 |
| Image-CNN(+head) − Image-CNN(max) | +0.0806 | [+0.0592, +0.1028] | superior | +0.0791 ± 0.0289 | 0.0460 |

### 🔵 2026-08-11 — 두 요약값은 서로 다른 변동을 보여 준다

두 요약값은 **서로 다른 변동을 보여 준다.** 하나를 고르는 것이 아니라 둘 다 보고한다.

| 추정량 | 답하는 질문 | 값 |
|---|---|---|
| 풀링 피험자 클러스터 부트스트랩 | *"다른 57명을 뽑았다면?"* | −0.0040 [−0.0063, −0.0020] |
| 런평균 vs fold 내 시드 std | *"다른 시드로 재학습했다면?"* | −0.0019 ± 0.0024 vs 0.0031 |

피험자 단위 클러스터 부트스트랩은 57명을 다시 뽑았을 때의 변동을, 런평균은 fold·seed
반복의 변동을 요약한다. 두 값 모두 관측된 차이가 δ=0.02 안에 있음을 보인다.

> **본문 서술 (이대로 쓴다)**
>
> *"ours 는 image_cnn_head 보다 0.004 낮다. 피험자 재표집 부트스트랩과 15런 평균은
> 서로 다른 변동을 요약하지만, 어느 쪽에서도 이 차이는 사전 지정한 δ=0.02 안에 있다."*

**심사 대응은 회피가 아니라 인정으로 간다:**
*"0.99 척도에서 0.004(상대 0.4%) 낮다. 그 대가로 파라미터 5.7배·연산 2.6배가 적다.
δ 는 결과를 보기 전에 고정했다."*

**Image-CNN(+head) 의 0.9906 이 표 최고값인 것도 그대로 싣는다.**
우리 주장은 정확도가 아니라 **비용**이다.

⚠️ **Image-CNN(max) 의 시드 std 가 0.0291 로 ours(0.0038)의 7.7배다.** 좁히려 하지 않고
그대로 보고한다. Discussion 에 **한 줄만** — *"학습된 시간 모델 없이 프레임 점수를 max 로
합치는 방식은 시드에 민감했다"*. 주장으로 밀지 않는다(n=15, 원인 미규명).

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
5. ⚠️ `ours − image_cnn_head` 는 두 추정량이 갈린다. → **2026-08-08 정정: "미측정"으로
   뭉개지 않고 둘 다 보고한다.** 위 §"두 추정량을 미측정으로 뭉개지 않는다" 참조.

- ✅ **Recall · F1 계산 완료** (T3-6, `results/v2/train_encoder_final.json`).
  혼동행렬(tp/fp/fn/tn)과 fa 도 함께 저장되므로 사후에 어떤 지표든 재구성 가능하다.
- (Ours − EAR-head) = **+0.0151, 95% CI [+0.0106, +0.0203]** → 비열등 충족 (δ=0.02).
- 🔵 **2026-08-06 확정 런에서 Table I 이 완성됐다** (`results/v2/train_encoder_final.json`):

  | Method | PR-AUC | ROC-AUC | Acc | Prec | Rec | F1 |
  |---|---|---|---|---|---|---|
  | EAR-rule | 0.8931 ± 0.0136 | 0.9169 ± 0.0146 | 0.8505 | 0.8268 | 0.8878 | 0.8550 |
  | EAR-head | 0.9724 ± 0.0096 | 0.9738 ± 0.0113 | 0.9276 | 0.9312 | 0.9228 | 0.9268 |
  | **Ours** | **0.9886 ± 0.0038** | **0.9914 ± 0.0037** | **0.9620** | 0.9605 | 0.9636 | **0.9618** |

  15런 전부 `baseline_used = ear_head` (더 강한 쪽과 비교됐다).
- ⚠️ **그래도 EAR 상대 "우월"을 논문의 헤드라인으로 쓰지 않는다.** 헤드라인은
  **image_cnn 대비 "동등 정확도 + 훨씬 싼 비용"** 이다 (§V-A2). EAR 우월은 보조 결과다.

**B. Robustness across Conditions — Fig. 2**

🔵 **2026-08-06 풀링 서브그룹 확정** (`results/v2/posthoc_subgroups_final.json`,
사이드카 점수 기반, ear_head 상대, 부트스트랩 2,000)

| 그룹 | 인원 | 이벤트 | ours | ear_head | 이득 | 95% CI | 판정 |
|---|---:|---:|---:|---:|---:|---|---|
| 전체 | 57 | 27,758 | 0.9861 | 0.9709 | +0.0151 | [+0.0106, +0.0203] | **superior** |
| 2020 | 38 | 19,883 | 0.9901 | 0.9783 | +0.0118 | [+0.0083, +0.0154] | **superior** |
| 2022 | 19 | 7,875 | 0.9722 | 0.9466 | +0.0256 | [+0.0123, +0.0364] | **superior** |
| **안경** | 17 | 9,947 | 0.9845 | 0.9727 | **+0.0118** | **[+0.0061, +0.0165]** | **superior** |
| 미착용 | 40 | 17,811 | 0.9869 | 0.9698 | +0.0171 | [+0.0107, +0.0250] | **superior** |

**다섯 서브그룹 전부 superior.** Image-CNN(+head) 상대에서도 같은 다섯 그룹 전부가
비열등 마진 안에 있다. 밝기·대비·선명도 3분위의 9개 그룹도 모두 ours 쪽을 가리킨다.
이 광학 층화는 Fig. 2에 숫자를 별도 추가하지 않고 본문 한 문장으로만 보고한다.

> 서브그룹은 피험자별 PR-AUC를 평균내지 않고 **이벤트를 풀링**한다. 그렇지 않으면 이벤트
> 수가 적은 피험자도 같은 가중치를 받아, 예를 들어 한 피험자가 안경군 이벤트의 1.1%인데
> 비가중 피험자 평균에서는 5.9%를 차지하는 문제가 생긴다.

- **전체 + 조건별 숫자를 항상 함께** 낸다. 이벤트 가중 풀링을 주 숫자로 쓴다.
- 남은 조건: 합성 저조도(P2).
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
   따라서 e2e 한 열만으로는 세 방법의 모델 비용 차이가 작게 보인다.

### 재설계 — 공통 항과 방법 항을 분리한다

**Table II-a. 정적 비용** (Pi 불필요, 이미 측정됨 — `results/v2/export_onnx.json`)

| Method | frontend params | head params | **총 params** | Model size | frontend MMAC | head MMAC | **총 MMAC/frame** |
|---|---:|---:|---:|---:|---:|---:|---:|
| EAR (rule) | 0 | 0 | **0** | — | ~0 | 0 | **~0** |
| EAR-head | 384 | 4,625 | **5,009** | 25.7 KB | 0.00032 | 0.0459 | **0.0462** |
| Image-CNN (max) | 470,561 | 0 | **470,561** | 1,840.2 KB | 31.806 | 0 | **31.81** |
| Image-CNN (+head) | 471,536 | 4,625 | **476,161** | 1,867.4 KB | 31.807 | 0.0459 | **31.85** |
| **Ours (vpres, D=16)** | 79,424 | 4,625 | **84,049** | **335 KB** | 12.443 | 0.0459 | **12.49** |

🔵 **2026-08-09 — 행을 두 변형으로 쪼개고 frontend/head/총계를 분리했다.**
이전 표의 "Image-CNN (mEBAL) | 471,536" 은 틀린 값은 아니지만 **모호했다** —
471,536 은 `image_cnn_head` 의 **backbone-only** 이고, `max` 의 backbone 은 470,561,
head 포함 총계는 476,161 이다. Table I 과 숫자가 어긋나 보였다.
두 백본의 차이 **975** = `Dense(64→16)` 1,040 − `Dense(64→1)` 65 (출력 차원 차이).

🔴 **MMAC 열의 정의 — "랜드마크를 입력으로 받은 뒤"의 비용이다.**
EAR 은 얼굴 랜드마크 6점에서 계산되므로 랜드마크가 먼저 있어야 한다. 학습에서는
mEBAL2 제공 랜드마크를 썼으므로 추가 비용이 0 이지만, **배포에서는 EAR 모드도 MediaPipe
검출을 똑같이 치른다**(Pi 실측 detect 8.32 ms, `pi_ear_480p.json`). 네 방법이 같은
검출기를 공유하므로 **공통 항을 빼고 비교**하는 것이며, EAR-rule 의 "~0" 은 *"랜드마크에서
나눗셈 몇 번"* 이라는 뜻이지 *"공짜"* 가 아니다. **이 정의를 표 각주에 반드시 넣는다.**

- Ours 내역: encoder 79,424 + head 4,625 / encoder.onnx 311.4 KB + head.onnx 23.3 KB
- EAR-head 내역: EarLift `Linear(4,16)`+BN+ReLU+`Linear(16,16)` = 384 (80+32+272) + head 4,625
- MMAC 내역: encoder 12.44 + head **0.0459** (stride 1, 매 프레임 판정). head 비중 0.37%
- ⚠️ **head 를 stride 19 로 세지 않는다.** 창 경계에서만 판정하면 깜빡임을 최대
  18프레임(600 ms @30 fps) 늦게 잡는다. 연속 검출의 정직한 값은 stride 1 이다.

**Table II-b. Pi 5 런타임** ✅ **2026-08-08 실측 완료**
(640×480 @ 30 fps, **각 모드 300 초 지속**, `--intra-threads 2 --no-spin`,
governor `performance`, `refine_landmarks=False`, **스로틀 없음**,
온도 시작 49.4~51.6 → **p95 58.7~59.3 °C**, 종료 54.9~58.7 °C)
클립: `eyeblink8_c9_480p_x8.avi` (eyeblink8 #9 를 8회 이어붙임, 41,464 프레임 ≈ 1,382 s).
`run_video` 는 반복 재생을 안 하므로 300 초 측정에 그만큼의 길이가 필요했다.
출처: `results/v2/pi_{ear,ours,image_cnn_max,image_cnn_head}_480p.json` (2026-08-08 검증)

| Method | crop | encode | decide | **method 소계** | e2e p50 | e2e p99 | FPS | RSS MB | CPU % | 온도 p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EAR (rule) | 0.08 | — | 0.14 | **0.22** | 9.56 | **10.10** | **104.2** | 184.2 | 139.5 | 59.3 |
| **Ours** | 0.67 | **0.86** | 0.25 | **1.78** | 11.27 | **11.85** | **88.5** | 204.1 | 140.6 | 58.7 |
| Image-CNN (max) | 0.67 | 2.04 | 0.17 | **2.88** | 12.26 | 12.87 | 81.3 | 206.1 | 146.3 | 59.3 |
| Image-CNN (head) | 0.67 | **2.05** | 0.31 | **3.03** | 12.51 | 13.10 | 79.7 | 207.5 | 145.6 | 59.3 |

온도는 **p95 를 쓴다.** 종료 온도는 그 시점의 순간값이라 측정 구간의 열 부하를 대표하지
못한다(ours 는 종료 58.7 = p95 지만 image_cnn_max 는 종료 54.9 / p95 59.3 으로 4.4 °C 차이).
네 모드 전부 `throttled = 0x0`, 스로틀 이력 플래그 없음.

**공통 항 (네 방법 동일)**: `read` **1.00~1.02 ms** · `detect` **8.32~8.46 ms**
→ 표 아래 한 줄로 **한 번만** 적는다. detect 가 모든 모드에서 일정하다는 것이
**같은 프론트엔드를 쓰는 공정 비교임의 실측 증거**다.

`p50` 기준 detect 비중: EAR **87%** / Ours **75%** / Image-CNN **68%**

### 🔵 실측이 확인한 것

| # | 발견 |
|---|---|
| 1 | **네 방법 전부 G-E1 PASS.** e2e p99 가 예산 33.3 ms 의 **30~39%**. 스로틀 없음 |
| 2 | **표현 단계에서 ours 는 image_cnn 의 1/2.4** (0.86 vs 2.04 ms). MMAC 비 1/2.55 와 **거의 일치** — 정적 예측이 실측으로 검증됐다 |
| 3 | **method 소계로는 1/1.7** (1.78 vs 3.03 ms) |
| 4 | 🔴 Image-CNN(+head)의 e2e 중앙값은 ours보다 **11% 느리다** (12.51 vs 11.27 ms). **얼굴 검출이 68~87%를 차지한다** |

> ✅ **논문 문장 (확정)**
> *"On a Raspberry Pi 5, the embedding path costs 0.86 ms per frame at the representation
> stage versus 2.04 ms for the image-based CNN — a 2.4× reduction consistent with the 2.6×
> MMAC ratio. The image-based control has an 11% higher end-to-end median latency (12.51 ms
> vs 11.27 ms), because face and landmark detection accounts for 68–87% of the pipeline
> under these measurement conditions."*

> ⚠️ **image_cnn ONNX 는 무작위 가중치**(지연 전용). 하네스가 런 JSON 에 경고를 기록했다.
> **정확도 수치와 섞지 않는다.** 연산 그래프가 같으므로 지연은 유효하다.
> 확정 런이 끝나면 실제 가중치로 재-export 해 지연이 변하지 않음만 확인한다.

> ⚠️ v1 수치(e2e p99 12.06 ms / 86 fps)와 우연히 비슷하지만 **인용 금지**.
> 인코더·`refine_landmarks` 설정이 다르다.

✅ **RSS peak · CPU% 를 표에 반영했다** (2026-08-08). RSS 는 네 모드가 184~208 MB 로
비슷하고, CPU 는 140~146% (4코어 400% 기준). image_cnn 이 ours 보다 RSS +3 MB,
CPU +5%p 높다 — 차이가 작다.

- **판정 열은 `method 소계`** 다. e2e p99 는 **예산 게이트(G-E1 ≤ 33.3 ms)** 확인용이지
  방법 비교용이 아니다. 둘의 역할을 섞지 않는다.
- 720p 보조 측정은 출처 JSON·클립 매니페스트가 함께 확인될 때만 별도 부록에 넣는다.
  현재 본문 주 측정은 640×480뿐이며, 해상도를 올릴수록 공통 detect 비용이 더 지배적일 수 있다.

### 논지 (프레이밍)

- ❌ "엣지 실시간이 어렵다 → 시스템 기여" — v1 에서 예산의 36% 밖에 안 나와 이 프레이밍이
  스스로 약해진 전례가 있다. v2 도 여유가 클 것으로 보인다.
- ✅ **"EAR 대비 벡터 표현의 추가 비용을 동일 조건에서 정량화했고, 그 비용이 파이프라인의
  어느 단계에 놓이는지 분해해 보였다."** 표현 단계의 비용 차이는 분명하지만, 측정 조건에서는
  공통 landmark frontend가 e2e 지연의 대부분을 차지한다.
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
- 가장 강한 이미지 기반 통제군 대비 δ=0.02에서 비열등, 파라미터 5.7배·연산 2.6배 절감.
- Raspberry Pi 5에서 실시간 동작을 확인했고, 측정 조건에서는 landmark frontend가 남은 e2e 지연의
  대부분을 차지한다.
- Future work: 조도·자세·거리 조건이 라벨된 데이터셋에서의 강건성 검증. (**한 줄만 쓴다.**)

> 🔴 **Future work 서술 주의.** 특허 출원 전이라면 GRL·VIB 같은 구체적 억제 기법을
> 논문에 적지 않는다. 논문 공개가 특허 심사에서 선행 기술이 될 수 있다.

---

## 작성 순서 제안

1. **III. Proposed Method** — Fig. 1과 함께 작성. 수식은 필요한 범위만 남긴다.
2. **IV. Experimental Setup** — 데이터·분할·학습·통계·Pi 조건을 확정값으로 옮긴다.
3. **V-A Table I / V-C Table II** — 5개 방법과 Pi 4모드 수치가 모두 채워져 있다.
4. **V-B Fig. 2** — acquisition batch·안경 5개 서브그룹, 광학 3분위 결과는 본문 한 문장.
5. **II. Related Work** — 인용 원문 확인 상태를 점검한 뒤 압축한다.
6. **I. Introduction** — 정확도 우위가 아닌 비용·파이프라인 병목이라는 기여를 마지막에 정리한다.
7. **VI. Conclusion** — 확정된 비교·비용·edge 결과만 한 단락으로 쓴다.

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
