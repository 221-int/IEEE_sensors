> **[POLARIZED/폐기]** 이 문서의 "EAR과 대등 / 8명으로는 우열 판정 불가",
> "Privacy: Identity re-ID 0.998 / PSNR 32.4 dB", "Privacy is structural ... not recoverable",
> disentanglement λ 스윕 인용은 **틀렸음.** 유효한 값은 Ours 0.803±0.321 vs EAR 0.977±0.035,
> 재식별 0.994(`results/identity_probe.json`)이며 λ 스윕은 한쪽눈 64×64 예비라 인용 금지다.
> **`docs/PROJECT.md` 참조.**

---

# Results so far — On-Device Privacy-Preserving Eye-Blink Representation

> 🚫 **정정 알림 (2026-07-30)** — 이 문서의 "EAR과 대등 / 우열 판정 불가", "복원 불가",
> "rank 보정으로 해결", "프라이버시 우선" 서술은 **철회되었습니다.**
> 조밀 격자로 다시 재면 Ours 0.803 ± 0.321 vs EAR 0.977 ± 0.035 (**−0.174, 우리가 열세**)이며,
> 선형 프로브 재식별은 Ours 0.994 vs EAR 0.170 입니다.
> 반드시 [`docs/정정_2026-07-30.md`](정정_2026-07-30.md) 를 먼저 읽으십시오.


_Status snapshot. One encoder branch (custom autoencoder) taken end-to-end
on a public dataset with subject-separated validation._

## Direction (re-scoped)
Detect blinks (eye-health primitive) from a **privacy-preserving vector**
instead of raw pixels / EAR, running lightweight on the **edge (Pi5, ONNX)**.
Privacy is **structural**: only vectors are stored/transmitted and no decoder is
deployed, so raw eye video is not recoverable in deployment.
Input = **both eyes in one crop → one vector** (blinks are bilateral).
**Identity re-identification / disentanglement is moved to FUTURE WORK**
(needs many subjects); the identity results below are preliminary, kept as a
future-work direction, not a current claim.

> **2026-07-29 갱신.** 로드맵 ①(Pi5 실측) · ②(양눈 EAR) · ③(차원 스윕) 완료.
>
> - **엣지**: Pi 5에서 p99 12.06 ms, 86 fps, 스로틀 없음. EAR 대비 +2.67 ms(1.30×).
>   → [`PI5_BENCHMARK.md`](PI5_BENCHMARK.md). **논쟁 여지 없이 확정된 결과.**
> - **유용성**: 벡터 모델과 EAR이 **대등**하나, 8명 데이터로는 **우열 판정 불가**.
>   시드에 따라 70–73/74로 흔들리고 EAR은 70/74다.
> - **프라이버시**: 잠재 차원을 16배 좁혀도 선형 재식별이 99%+로 유지된다.
>   → **차원 축소는 프라이버시 메커니즘이 아니며 disentanglement가 필수.**
>
> 현재 상태는 **데모 수준**으로 읽어야 한다. 공개셋 1개(8명) · val 2명(74이벤트).
> **mEBAL2 도착 완료(2026-07-31) — 180명이 아니라 58명.** 이후 전부 재실행 대상이다.
> → `docs/mEBAL2_실측_2026-07-31.md`

## Pipeline (all validated)
```
eye crop (both eyes, gray 64x160)  ->  encoder  ->  128-d vector  ->  blink O/X
   [canonical crop, shared]           [autoencoder]                 [MLP classifier]
```
No landmarks / EAR in the judgment path. Decoder = local validation tool only.

## Data
- **Eyeblink8**, 8 clips -> both-eyes 64x160 crops (~35k frames, 30->15 fps).
  (Rebuild the .npz after the both-eyes switch.)
- Labels (manual, EAR-independent): `blink_event` 5.25%, `eye_closed` 2.49%.
- **Subject-separated** val = clips eb04, eb11 (74 blink events).

## Utility — vector model vs EAR (BOTH eyes, 2026-07-29)

동일 클립(eb04, eb11) · 동일 지표 · **양쪽 모두 양눈**. 74 이벤트, 4,776 비깜빡임 프레임.
EAR은 좌우 EAR의 평균, 벡터 모델은 64×160 인코더 + MLP.

### 결론: 대등하되 **8명 데이터로는 우열 판정 불가**

판정기를 시드만 바꿔 5회 재학습한 결과 (선택 기준 = 이벤트 recall, FA ≤ 3%):

| 시드 | 0 | 1 | 2 | 3 | 4 | 범위 |
|---|---|---|---|---|---|---|
| 이벤트 recall | 0.973 | **0.946** | 0.973 | 0.973 | **0.986** | 70–73 / 74 |

**EAR 양눈 = 0.946 (70/74)** — 우리 모델의 **최악 시드와 정확히 동점**이고, 나머지 4개
시드는 1~3 이벤트를 더 잡는다.

**차이(1~3 이벤트)가 시드 흔들림 폭(3 이벤트) 안에 있다. 어느 쪽이 낫다고 말할 수 없다.**
한쪽눈 시절의 "parity in the usable region"이라는 표현은 양눈에서도 유지되며, 다만
판정 불가의 이유가 "차이가 작아서"가 아니라 **"검증셋이 작아서"** 임이 이번에 정량화되었다.

### 이 결론에 도달하기까지 (기록)

같은 오후에 결론이 세 번 바뀌었다. 원인이 전부 방법론이었으므로 남긴다.

| 판단 | 근거 | 무엇이 잘못됐나 |
|---|---|---|
| "EAR 우세" | `eval_events.py` 임계값 0.3–0.8 | **격자가 성겨** 저FA 구간 미측정. `metrics.json` 은 최적 F1을 thr 0.9로 기록하고 있었다 |
| "벡터 우세" | `dim_sweep` 의 R@FA3% 0.986 | **다른 모델의 숫자.** 스윕이 새로 학습한 모델을 배포본과 혼동 |
| "EAR 우세" | 배포본 조밀 격자 재측정 | **단일 시드 1회**. 배포본이 이상치임을 몰랐다 |
| **"판정 불가"** | **5시드 재학습 + 조밀 격자 양측** | — |

교훈: **단일 학습 결과로 베이스라인과 비교하지 말 것.** 이 데이터 규모에서는
시드 하나가 결론을 뒤집는다.

### 배포된 모델은 이상치였다

| | 이벤트 recall @FA3% |
|---|---|
| 배포본 (`models/classifier`, 7/27 학습) | **0.932 (69/74)** |
| 재학습 5회 범위 | 0.946 – 0.986 (70–73/74) |

배포본은 재학습 범위 **아래**에 있다. 선택 기준(프레임 F1)을 이벤트 기준으로 바꾼
효과는 0.959 → 0.973(1 이벤트)으로 작았고, **더 큰 요인은 학습 운**이었다.
→ 파이에 올린 ONNX도 이 이상치 모델이므로 교체 후보: `models/_diag_s4` (73/74).
지연시간은 가중치와 무관하므로 Pi 재측정은 불필요하다.

### 남은 방법론적 문제 (중요)

위 숫자는 **에폭·임계값을 모두 val에서 고르고 val에서 보고**한 값이다. EAR은 자유
파라미터가 임계값 하나뿐인 반면 우리는 에폭(40) × 임계값(97) × 시드 중에서 골랐다.
74 이벤트에 대해 자유도가 훨씬 크므로 **우리 쪽 숫자가 더 부풀려져 있다.**
단일 학습 안에서도 에폭별로 0.919–0.973을 오갔고, 그중 최댓값을 취한 것이다.

→ **로드맵 ④(train/val/test 정식 분할)이 선행되지 않으면 이 비교는 논문에 쓸 수 없다.**

원자료: `results/ear_dense.json`, `results/ear_baseline_botheyes.json`.

### 왜 EAR이 양눈에서 강해졌나

| | 한쪽 눈 (이전) | 양눈 (현재) |
|---|---|---|
| EAR @ recall 0.986 | FA 26.4% (좌안 단독) | FA **5.4%** |

좌우 평균이 단안 노이즈를 상쇄한다. 이전 판의
*"EAR's false-alarm explodes when pushed for recall (13–50%)"* 는 **좌안 단독** 기준이었고,
양눈 mean에서는 5.4–11.2%다. **이 문장은 폐기한다.**

### ⚠️ 이전 표의 결함 (정정)

이전 한쪽눈 비교는 **서로 다른 눈을 비교했다.**

| | 사용한 눈 |
|---|---|
| 한쪽눈 벡터 모델 (`.tag` RE 코너) | 이미지-**오른쪽** |
| 이전 EAR 베이스라인 (MediaPipe 33/133) | 이미지-**왼쪽** |

`ear_baseline.py` 의 `legacy` 와 `left` 변형이 4,776 프레임 전부에서 동일하게 나와
확인되었고, `.tag` 좌표로 교차 검증했다(8개 클립 전부에서 RE 평균 x > LE 평균 x).
`capture_eye_dataset.EYE` 의 "image-right eye" 주석은 **틀렸다** — 실제로는 이미지-좌안이다.

방향은 보수적이었다: 눈을 맞췄다면 EAR(우안)은 recall 0.946에서 FA 4.00%로
기록된 1.51%보다 나빴을 것이므로 벡터 모델에 유리했다. 결론은 뒤집히지 않지만
기록은 정정한다. 양쪽 모두 양눈인 현재 표에서는 이 문제가 해소되었다.

Autoencoder reconstruction sanity: val MSE 0.003 (`models/autoencoder/recon_val.png`).
`docs/figures/vector_vs_ear.png` 는 한쪽눈 시절 그림이라 **재생성 필요**.

## Latent dimension sweep — roadmap (3), 2026-07-29

32 / 64 / 128 / 256 / 512 × 3 시드. 각 폭마다 오토인코더를 새로 학습하고 인코더를
동결한 뒤 판정기를 학습해 **유용성과 신원 정보량을 같은 인코더에서** 측정.
스크립트 `src/experiments/dim_sweep.py`, 원자료 `results/dim_sweep.json`.

### (a) 유용성 — 차원은 원인이 아니다

| latent | 이벤트 recall @FA3% (3시드) | 평균 | 시드 std |
|---|---|---|---|
| 32 | 0.986 / 0.986 / 0.986 | 0.986 | **0.000** |
| 64 | 0.959 / 0.986 / 0.973 | 0.973 | 0.011 |
| 128 | 0.973 / 1.000 / 0.757 | 0.910 | 0.109 |
| 256 | n/a / 1.000 / 0.973 | — | 0.465 |
| 512 | 1.000 / 0.973 / 0.973 | 0.982 | 0.013 |

**차원 간 평균의 std 0.125 vs 차원 내 시드 std 0.120 — 신호 대 잡음 비가 1.**
차원 효과를 분리할 수 없다. 다만 **32차원이 가장 안정적이면서 최상위권**이라는 점은
*"입력이 2.5배 커졌는데 병목이 128로 그대로여서 성능이 나쁘다"* 는 가설과 배치된다.
→ **"128 병목" 가설 기각.** 양눈 false-alarm 문제는 용량 문제가 아니다.

재구성 MSE는 차원에 따라 단조 개선(0.0053 → 0.0045)되므로, 인코더는 정상적으로
더 많은 정보를 담고 있다. 그 여유가 깜빡임 성능으로 이어지지 않을 뿐이다.

### (b) 프라이버시 — 차원 축소는 프라이버시가 아니다 ★

선형 프로브 신원 정확도 (chance 0.125):

| latent | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| 선형 re-ID | **0.995** | 0.998 | 0.999 | 0.999 | 0.999 |

**512 → 32 로 16배 좁혀도 신원 누출은 0.45%p 줄어든다.** 전 구간 99% 이상.

→ 로드맵 §4-(3)의 전제 *"그 병목은 신원 정보가 들어갈 자리도 없다는 뜻이기도 하다"* 는
**틀렸다.** 병목을 좁히는 것으로는 프라이버시를 얻을 수 없으며, **명시적
disentanglement가 선택이 아니라 필수**임이 정량적으로 확인되었다.

단서: 8명 식별은 3비트면 충분하므로 32차원 실수 벡터는 애초에 과잉이다.
mEBAL2 **58명**도 5.9비트라 크게 다르지 않을 수 있다. "좁히면 프라이버시가 생긴다"의
**반증으로는 유효하지만**, 강한 주장으로 쓰기는 어렵다.

## Privacy (current, NOT-yet-disentangled encoder)
| Metric | Value | Meaning |
|---|---|---|
| Identity re-ID (vector -> subject) | **0.998** (chance 0.125) | vector fully carries identity |
| Reconstruction PSNR | 32.4 dB | eye appearance recoverable |

→ The reconstruction-trained vector is **not private yet**. This is the
**left endpoint of the utility–privacy Pareto curve** and the quantitative
motivation for identity-disentanglement.
위 (b)가 이를 보강한다 — 차원을 줄여도 이 값은 내려가지 않는다.

## Disentanglement PoC (FUTURE WORK — preliminary, one-eye pipeline)
Encoder trained with a blink head + gradient-reversal identity adversary, with
lambda warmup (DANN schedule) + gradient clipping. Frame-split over all 8
subjects; utility = blink ROC-AUC, privacy = identity re-ID.

| lambda | blink AUC (utility) | re-ID (leak, chance 0.125) |
|---|---|---|
| 0.0 | 0.984 | 0.942 |
| 0.1 | 0.988 | 0.913 |
| 0.2 | 0.983 | 0.868 |
| 0.5 | 0.975 | 0.819 |
| 1.0 | 0.977 | 0.767 |

**Identity leakage drops (re-ID 0.94 → 0.77) at essentially zero utility cost
(AUC stays ~0.98)** → the core hypothesis holds: identity is removable while
keeping blink utility. See `docs/figures/pareto_disentangle.png`.
Not yet at chance: full removal needs higher lambda / better method /
many-identity data. (Naive fixed-lambda GRL was unstable; warmup fixed it.)

## Edge (Pi 5) — measured 2026-07-29

| | EAR 베이스라인 | Ours (벡터) |
|---|---:|---:|
| e2e p50 / p95 / p99 (ms) | 8.86 / 9.09 / 9.36 | **11.53 / 11.78 / 12.06** |
| 지속 처리량 | 112.3 fps | **86.1 fps** |
| 30 fps 예산(33.3 ms) 대비 p99 | 28% | **36%** |
| 예산 초과 프레임 | 2 / 33,584 | 2 / 25,734 |

벡터 표현의 비용 = **+2.67 ms (1.30×)**, 그중 84%가 인코더+MLP(2.25 ms)이고
전처리는 0.56 ms. 5분 연속에서 스로틀링 없음. 상세·측정 조건은
[`PI5_BENCHMARK.md`](PI5_BENCHMARK.md).

**즉 엣지 실현성은 확보되었고, 현재 부족한 것은 정확도다.**

## Where we are (of the 9-step plan)
Done: 1 preprocess, 2 (train/val, test split pending), 3-B autoencoder,
4 decoder check, 5 single-frame judgment, 6 evaluation (utility + privacy),
disentanglement PoC (mechanism validated), **8 ONNX + Pi latency (완료)**.
Pending: extend lambda sweep to trace the full frontier; 3-A off-the-shelf
encoder, 7 A/B compare, 9 registration window; **양눈 정확도 회복(③ 차원 스윕)**;
many-identity dataset → **mEBAL2 도착 예정**.

## Next dataset — mEBAL2 (도착 완료 2026-07-31)

실측 확인 결과 **180명이 아니라 58명**이다. 현재 8명 → 58명으로 7.25배.
전체 실측은 [`mEBAL2_실측_2026-07-31.md`](mEBAL2_실측_2026-07-31.md).

| | Eyeblink8 (현재) | mEBAL2 (실측) |
|---|---|---|
| 피험자 | 8 | **58** |
| 이벤트 | 74 (val) / ~35k 프레임 | **28,728** (14,364 blink + 14,364 no-blink) |
| 센서 | RGB 1 | RGB 1 + **NIR 2** |
| 구조 | 연속 영상 + 프레임 라벨 | **이벤트당 19프레임 고정창** (지속시간 라벨 없음) |
| 제공물 | .avi + .tag | 영상 + face bbox(xyxy) + **FaceMesh 468 랜드마크** + 크롭된 눈 영역 + EEG |
| 라벨 커버리지 | 전 프레임 | **11.96%** (1.88~37.99%) → 연속 FA 측정 불가 |
| 검출 결측 | 0.23% (Pi 실측) | **8.8%** (제공 랜드마크 기준) |

**8-way 재식별 → 58-way** (chance 0.125 → **0.017**). §6(d)를 완전히는 아니어도 크게 완화한다.
다만 **연속 평가·눈 건강 지표는 mEBAL2로 확장되지 않는다.**

### 도착 전에 준비해야 할 것 (구조가 다르다)

1. **데이터 로더 신규 필요.** 현재 `build_eyeblink8_npz.py` 는 연속 .avi를 프레임 단위로
   읽고 `blink_id` 로 이벤트를 묶는다. mEBAL2는 이미 잘린 19프레임 이벤트 시퀀스다.
2. **68-랜드마크 어댑터.** 우리 `eye_corners_from_landmarks` 는 MediaPipe 인덱스 기준이지만
   `pair_a`/`pair_b` 를 인자로 받도록 만들어 두었으므로 **인덱스만 바꾸면 된다.**
   MediaPipe를 돌릴 필요 없이 제공된 랜드마크로 canonical 크롭을 재현할 수 있다.
3. **NIR 활용 여지.** 우리 전처리는 어차피 grayscale이라 NIR을 그대로 넣을 수 있다.
   저조도 강건성 주장의 재료.
4. **피험자 수를 파라미터로.** §0의 재현성 원칙 — "재실행이지 재설계가 되면 안 된다".

## Caveats
- **현재는 데모 수준이다.** 공개셋 1개(8명) · val 2명(74이벤트).
  이 조건에서 나온 정확도 숫자는 방향 판단용이며 논문 주장이 아니다.
- **에폭·임계값을 val에서 고르고 val에서 보고한다.** EAR은 자유 파라미터가 1개,
  우리는 40×97×시드다. 우리 쪽이 더 부풀려져 있다 → ④(정식 분할)가 선행 조건.
- **단일 학습 결과로 베이스라인과 비교하지 말 것.** 이 규모에서는 시드 하나가
  결론을 뒤집는다(같은 오후에 결론이 세 번 바뀌었다).
- 배포된 classifier는 재학습 범위 밖의 이상치(69/74)다. 교체 필요.
- re-ID with 8 clips is proof-of-concept; mEBAL2(**58명**)가 선행 조건.
  8-way 식별은 3비트면 충분해 차원 실험의 해석력이 제한된다.
- Pi 측정은 클립 1개(eb09, 640×480) 기준. 해상도·피험자가 바뀌면 `detect` 가 달라진다.
- `docs/figures/vector_vs_ear.png` 는 한쪽눈 시절 그림이라 재생성 필요.
