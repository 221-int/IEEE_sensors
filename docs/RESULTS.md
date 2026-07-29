# Results so far — On-Device Privacy-Preserving Eye-Blink Representation

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

> **2026-07-29 갱신.** 양눈 파이프라인으로 유용성(utility)을 재측정했다.
> **결론이 뒤집혔다 — 양눈 조건에서 벡터 모델은 EAR 베이스라인에 밀린다.**
> 이전 판의 "Phase-0 ≥ EAR 충족" 주장을 **철회한다.** 상세는 아래 Utility 절.
> 엣지 실측은 별도로 완료되었다 → [`PI5_BENCHMARK.md`](PI5_BENCHMARK.md).
>
> 현재 상태는 **데모 수준**으로 읽어야 한다. 8명(공개셋 1개) · val 2명(74이벤트) ·
> 단일 시드다. **mEBAL2(180명) 도착 예정(약 1주)** 이후 전부 재실행 대상이다.

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
EAR은 좌우 EAR의 평균(`--mode mean`), 벡터 모델은 현재 64×160 인코더+MLP.

| 동일 지점 | EAR 양눈(mean) | 벡터 모델 (ours) | |
|---|---|---|---|
| false-alarm ≈ 5.4% | recall **0.986** (73/74) | recall **0.932** (69/74) | EAR 우세 |
| false-alarm ≈ 2.6–3.0% | recall 0.946 @ 2.97% | recall 0.932 @ 2.60% | 대등, EAR 소폭 우세 |
| recall ≈ 0.96 | false-alarm ≈ **4%** | false-alarm **9.1%** | EAR 2.3× 우세 |
| recall 0.973 | false-alarm ≈ **5%** | false-alarm **28.6%** | EAR 5× 우세 |
| 최대 recall | **1.000** @ FA 11.2% | **0.973** @ FA 28.6% | EAR만 전건 검출 |

**~~Parity in the usable region~~ → 철회.** 양눈 조건에서 벡터 모델은 전 구간에서
EAR에 밀린다. Phase-0의 "vector model ≥ EAR baseline"은 **현재 성립하지 않는다.**

원자료: `results/ear_baseline_botheyes.json`.
재현: `python -m src.eval.ear_baseline ... --json ...` / `python -m src.eval.eval_events ...`

### 왜 뒤집혔나 — 양쪽이 반대로 움직였다

| | 한쪽 눈 (이전) | 양눈 (현재) | |
|---|---|---|---|
| 벡터 모델 @ recall ≈ 0.96 | FA **3.1%** | FA **9.1%** | 2.9× 악화 |
| 벡터 모델 @ recall 0.932 | FA 1.5% | FA 2.6% | 1.7× 악화 |
| EAR @ recall 0.986 | FA 26.4% (좌안 단독) | FA **5.4%** | 4.9× 개선 |

입력 픽셀이 64×64 → 64×160 으로 2.5배가 되었는데 잠재 차원은 128로 그대로다.
가설: **병목 대비 이미지가 크다.** 로드맵 ③(차원 스윕 64/128/256)이 이 가설을 검증한다.
동시에 EAR은 양눈 평균이 단안 노이즈를 상쇄해 오히려 강해졌다.

이전 판의 *"EAR's false-alarm explodes when pushed for recall (13–50%)"* 는 **좌안 단독**
기준이었다. 양눈 mean에서는 5.4–11.2%이며, 이 문장은 폐기한다.

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

## Privacy (current, NOT-yet-disentangled encoder)
| Metric | Value | Meaning |
|---|---|---|
| Identity re-ID (vector -> subject) | **0.998** (chance 0.125) | vector fully carries identity |
| Reconstruction PSNR | 32.4 dB | eye appearance recoverable |

→ The reconstruction-trained vector is **not private yet**. This is the
**left endpoint of the utility–privacy Pareto curve** and the quantitative
motivation for identity-disentanglement.

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

## Next dataset — mEBAL2 (도착 예정, 약 1주)

라이선스 요청 진행 중. 현재 8명 → **180명**으로 22.5배.

| | Eyeblink8 (현재) | mEBAL2 |
|---|---|---|
| 피험자 | 8 | **180** |
| 이벤트 | 74 (val) / ~35k 프레임 | **21,100** (10,550 blink + 10,550 no-blink) |
| 센서 | RGB 1 | RGB 1 + **NIR 2** |
| 구조 | 연속 영상 + 프레임 라벨 | **이벤트당 19프레임 시퀀스** |
| 제공물 | .avi + .tag | 영상 + face bbox + **68 랜드마크** + 크롭된 눈 영역 + EEG |
| 변이 | 거의 없음 | 조명·자세·거리·**안경 32%**·가림 |

**이것이 §6(d)(신원 통계력 부족)를 해소한다.** 8-way 재식별 → 180-way.

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
- **현재는 데모 수준이다.** 공개셋 1개(8명) · val 2명(74이벤트) · 단일 시드.
  이 조건에서 나온 정확도 숫자는 방향 판단용이며 논문 주장이 아니다.
- 양눈 조건에서 벡터 모델이 EAR에 밀린다. "parity"조차 현재는 주장할 수 없다.
- re-ID with 8 clips is proof-of-concept; mEBAL2(180명)가 선행 조건.
- `eval_events.py` 의 임계값 스윕이 0.8에서 끊긴다. `metrics.json` 은 최적 F1을
  thr 0.9로 기록하므로 **저FA 구간이 측정되지 않았다.** 스윕 확장 필요.
- Pi 측정은 클립 1개(eb09, 640×480) 기준. 해상도·피험자가 바뀌면 `detect` 가 달라진다.
