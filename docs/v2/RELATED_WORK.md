# 관련 연구 — 2026-08-04 조사

> 🔵 **2026-08-05 범위 정정.** 이 조사는 **프라이버시 축으로 수행**되었고,
> 논문 범위가 **눈 깜빡임 검출 + 에지 성능 검증**으로 축소되면서 **절반만 쓸 수 있다.**
>
> | 절 | 새 방향에서의 위치 |
> |---|---|
> | **§A** mEBAL2 · 깜빡임 검출 성능 | ✅ **논문에 사용.** 단 [검색요약] 은 원문 확인 필요. **확장 필요** |
> | §B 눈 영역의 식별성 | ▶ 논문 범위 밖 → 특허 배경기술 / 후속 연구 |
> | §C 학습된 표현의 신원 누출 | ▶ 논문 범위 밖 → 후속 연구 Related Work |
> | §D 눈 데이터 프라이버시 종합 | ▶ 논문 범위 밖 |
> | §E 깜빡임 동역학만으로 식별 | ▶ 논문 범위 밖 → **특허 §1 의 핵심 반례.** 보존 필수 |
>
> **🔴 새 방향에 필요한데 비어 있는 조사** (`docs/TASKS.md` T6-1~T6-3):
> 1. **에지 / 온디바이스 눈 깜빡임 검출** 선행 연구 — 현재 §A 에 1건뿐 (전사문 10:52)
> 2. **encoder 로 벡터화해 깜빡임 검출을 하는** 선행 연구 — 미확인 (전사문 06:11)
> 3. **깜빡임 검출 과제의 일반적 실패 조건**을 정리한 리뷰 — 미조사 (전사문 10:29).
>    이 결과가 `EXPERIMENT_PLAN.md` §4 의 조건 목록을 확정한다

> ~~**우리 위치 한 줄**: 깜빡임 검출 문헌은 유용성만 재고, 눈 프라이버시 문헌은 신원만 잰다.
> **유용성을 동등하게 고정한 채 같은 표현의 신원 노출량을 재는 연구가 안 보인다.** 거기가 우리 자리다.~~
> ⚠️ **2026-08-05 폐기.** 새 위치 문장: *깜빡임 검출을 원본 이미지가 아니라 encoder 의
> embedding vector 위에서 수행하고, 그 비용을 edge device 에서 정량화한 연구가 안 보인다.*
> → 위 조사 1·2 로 확인해야 성립한다.
>
> ⚠️ **확인 상태를 구분해 적었다.** [원문] = PDF/본문을 직접 열어 확인. [검색요약] = 검색
> 결과 요약만 봄, 원문 미확인. **[검색요약] 은 논문에 인용하기 전에 반드시 원문을 열 것**
> (이 프로젝트 부록 #10: 문서가 아니라 파일을 열어 확인한다).

---

## A. mEBAL2 자체와 깜빡임 검출 성능

| 연구 | 내용 | 확인 |
|---|---|---|
| **mEBAL2 원논문** (Daza et al., Pattern Recognition Letters 182, 2024; arXiv 2309.07880) | 180명·21,100 시퀀스·200만+ 라벨 이미지. NIR 2대 + RGB 1대 + EEG. 프레임 단위 CNN 과 **시퀀스 단위 ConvLSTM**, 최대 **99%**. NIR 을 넣으면 추론 시 RGB 만 있어도 개선 | [검색요약] |
| A review of deep learning in blink detection (PMC, 2025) | 깜빡임 검출 딥러닝 개관 | [검색요약] |
| OptFlowBlinkFormer (PRCV 2025) | 광류 + RGB 융합 Transformer, in-the-wild | [검색요약] |
| Optimized deep learning architectures for high precision eye blink detection on consumer grade hardware (Discover AI, 2026) | 소비자급 하드웨어 대상 최적화 | [검색요약] |

---

## A2. 🔵 2026-08-05 조사 — 새 방향(깜빡임 검출 + 에지)에 필요한 축

> 조사 목적: **image_cnn 베이스라인을 우리가 임의로 설계하면 허수아비가 된다.**
> 문헌의 대표 구조를 근거와 함께 채택해야 방어된다 (`docs/T3_WORKORDER.md` §4).
> 확인 상태: 전부 **[검색요약]** 이다. **논문 인용 전 원문 확인 필수.**

### A2-1. ★ mEBAL/mEBAL2 저자들의 자체 CNN — **image_cnn 확정** ✅ [원문 확인]

**Daza, Morales, Fierrez, Tolosana. mEBAL. ICMI '20 Companion.** — **원문 §4 직접 확인 (2026-08-05)**

> *"Inspired in the popular VGG16 architecture, we propose an eye blink detector based on a
> CNN trained from scratch. The proposed network comprises an input layer of 50 × 50 size,
> followed by 3 convolutional layers with ReLU activation (32/32/64 filters of size 3 × 3),
> with 3 max pooling layers between them, a dense layer of 64 units with ReLU activations,
> and a final output layer with one unit (sigmoid activation). Also, we use dropout (0.5)...
> The batch size is set up to 50. Adam optimizer is considered with default parameters
> (0.001 learning rate). The network is trained as a binary classifier (eyes open or closed),
> using the mEBAL subset of RGB cropped eyes."*

| 항목 | 값 |
|---|---|
| 입력 | **50 × 50**, RGB, **한쪽 눈** 크롭 |
| 구조 | Conv(32,3×3)+ReLU → MaxPool → Conv(32,3×3)+ReLU → MaxPool → Conv(64,3×3)+ReLU → MaxPool → **Dense(64)+ReLU** → Dense(1)+sigmoid |
| 정규화 | Dropout 0.5 |
| 학습 | Adam lr=1e-3, batch 50, **프레임 단위** 이진 분류(open/closed) |
| 영감 | VGG16 |

### 🔴 A2-1b. 시퀀스 판정 방식 — **원문에서 새로 확인. 베이스라인 설계에 직접 영향**

mEBAL 원문 §5.1:

> *"All 13 frames are processed with the CNN proposed in Section 4, which generates for each
> input image an eye blink strength score. Among the 13 scores obtained for a sample
> (one per frame), **the maximum is selected to represent the sample score.**"*

**즉 mEBAL 의 시퀀스 판정은 학습된 시간 모델이 아니라 프레임 점수의 max pooling 이다.**

→ `image_cnn` 베이스라인을 **원 논문에 충실하게** 구현할 수 있다:

```
프레임마다 mEBAL CNN → 점수 1개
19프레임 점수의 max → 이벤트 점수
```

**학습된 시간 헤드가 필요 없다.** 이것이 박사님이 03:11 에서 말한
*"얘네는 이미지 자체를 가지고 바로 하니까 한울 씨가 최소 한 단계 정도 더 처리 과정이 길다"* 의
정확한 실체다 — 그들은 프레임 → 점수로 끝나고, 우리는 프레임 → 벡터 → 시간헤드 → 점수다.

> 🔵 **권고: 두 변형을 낸다** (`ear_rule` / `ear_head` 와 정확히 같은 구조의 대비다)
>
> | 변형 | 시간 처리 | 무엇을 답하나 |
> |---|---|---|
> | `image_cnn_max` | **max pooling** (원 논문 그대로) | 문헌 방법과의 정직한 비교 |
> | `image_cnn_head` | **우리와 동일한 시간 헤드** | 프론트엔드만 갈아끼운 통제 비교 |

### A2-1c. mEBAL 데이터셋 사실 (원문 확인) — 우리 설정과의 대비

| 항목 | mEBAL | 우리 (mEBAL2 배포본) |
|---|---|---|
| 피험자 | **38명** | 57명 |
| 샘플 | 3,000 blink + 3,000 no-blink | 13,820 + 13,938 |
| **샘플당 프레임** | **21 (약 600 ms)** — 이벤트 전후 10프레임씩 | **19** |
| 해상도 | 1280×720, 30 Hz | 1280×720 |
| **안경 착용** | **11 / 38 = 28.9%** | **17 / 57 = 29.8%** ← 거의 같다 |
| 깜빡임 길이 | **3~13 프레임** (100~400 ms) | 동일 전제 |

> 💡 **"깜빡임은 3~13 프레임"** 은 원문 인용 가능한 수치다.
> `T3_WORKORDER.md` §5-quinquies 의 **검출 지연** 논의에 직접 쓴다.

> ⚠️ **라벨 잡음의 출처가 원문에 적혀 있다**: *"in some no-blink cases the eyes seem to be
> closed due to the gaze orientation."* → 우리 §3-ter-2 의 **환원 불가 오차 0.59%** 와
> 연결해 Discussion 에 쓸 수 있다.

### A2-1d. 기존 데이터셋 표 (mEBAL 원문 Table 1) — 해상도 관행의 근거

| 데이터셋 | 연도 | Blinks | Users | 해상도 |
|---|---|---:|---:|---|
| Talking Face | — | 61 | 1 | 720×576 |
| Pan et al. | 2007 | 255 | 20 | **320×240** |
| Drutarovsky & Fogelton | 2014 | 353 | 4 | **640×480** |
| Silesian Deception | 2015 | 300 | 5 | **640×480** |
| **HUST-LEBW** | 2019 | 381 | 172 | **1280×720** |
| mEBAL | 2020 | 3,000 | 38 | 1280×720 |

> 🔵 **`EXPERIMENT_PLAN.md` §6-3 의 "640×480 주 / 1280×720 보조" 결정이 원문 표로 뒷받침된다.**

### A2-1e. mEBAL2 확장 (arXiv 2309.07880) — [검색요약, 원문 미확인]

- 같은 구조를 mEBAL2 로 재학습 + **ConvLSTM** 시퀀스 단위 추가 → 최대 **99%**
- NIR 을 학습에 넣으면 추론 시 RGB 만 있어도 개선
- HUST-LEBW 로 일반화 검증

> 🔴 **image_cnn 은 이제 원문 근거로 확정됐다.** *"데이터셋 저자들이 제안한 구조"* 이므로
> "약한 베이스라인을 골랐다"는 반론이 원천 봉쇄된다.

### A2-2. ★★ Nousias, Delibasis & Labiris (J. Imaging 11(27), 2025) — **신규성 주장에 직접 타격**

*"Blink Detection Using 3D Convolutional Neural Architectures and Analysis of Accumulated
Frame Predictions"* — [검색요약, 본문 일부 확인]

**이 논문은 잠재 공간(latent space)에 분류기를 붙여 깜빡임을 검출한다.** 우리와 구조적으로 가깝다.

| 항목 | 값 |
|---|---|
| 눈 검출 | YOLOX |
| 입력 | **48 × 48 × 12** (한쪽 눈, 12프레임 = 300 ms 3D 입력) |
| 비교 구조 | 3D CNN / **3D ResNet** / **3D autoencoder + latent 분류기** |
| 3D AE 인코더 | Conv3D 4블록 (16→32→32→32, 5×5×5), stride [2,2,2]/[2,2,2]/[1,1,1]/[2,2,1] |
| **잠재 차원** | **2048 units** |
| 손실 | **cross-entropy + 재구성 MSE 가중 결합** (분류에 더 큰 가중치) |
| 기타 | 인코더↔디코더 **skip connection** |
| 데이터 | 15명 (train 10 / test 5), 162,400 프레임, 눈당 1,172 blinks |
| F1 | 3D CNN 89.72 / **3D AE 89.63** / **3D ResNet 93.25 (최고)** |
| 프레임 정확도 | 93.97 / 93.74 / 94.24 |
| **3D AE 파라미터** | **35,375,507 (35.4M)** |
| 에지 배포 | **없음** |

#### 🔴 이 발견이 강제하는 서술 변경

**"encoder 로 벡터화해 깜빡임을 검출한 연구는 거의 없다"는 더 이상 쓸 수 없다.**
(`docs/PROJECT_DIRECTION.md` §3-1 D9, 전사문 06:11 의 가정이 부분적으로 틀렸다.)

**그러나 우리 기여가 사라지는 것은 아니다. 차별점이 오히려 선명해진다:**

| | Nousias 2025 | 우리 |
|---|---|---|
| 표현 차원 | **2048** | **16** (128배 작다) |
| 파라미터 | **35.4 M** | **~84 K** (약 **420배** 작다) |
| 학습 목적 | 재구성 + 분류 결합 | **분류 단독** |
| 시간 처리 | 3D conv, 12프레임 **블록 단위** | **프레임당 인코딩 1회 + 링버퍼** (인과적 스트리밍) |
| 입력 | 48×48 **한쪽 눈** | 64×160 **양눈** |
| 에지 실측 | 없음 | **Raspberry Pi 5** |
| 저자들의 결론 | latent 방식이 **3D ResNet 보다 나빴다** | — |

**즉 그들에게 잠재 공간은 "압축"도 "에지"도 아닌 단순한 구조 변형이었고, 성능도 지지 않았다.**
우리 주장은 *"encoder 를 쓴다"* 가 아니라 —

> **"16차원까지 압축한 임베딩만으로 판정하고, 그것이 에지 디바이스에서 실시간으로 돈다"**

로 좁혀야 한다. 이게 새 연구 방향(`PROJECT_DIRECTION.md`)과도 정확히 일치한다.

---

#### ✅ 2026-08-06 **원문 확인 완료** — [검색요약] → **[원문]**

출처: [J. Imaging 11(1):27](https://doi.org/10.3390/jimaging11010027) ·
본문 [PMC11765999](https://pmc.ncbi.nlm.nih.gov/articles/PMC11765999/) (MDPI 직접 접근은 403)

**위 표의 수치는 전부 확인됐다**: 잠재 2048 units, 3D AE **35,375,507** params,
입력 **48×48×12**, AE 81.21% acc / 89.63% F1 vs 3D ResNet 87.36% / 93.25%
(원문 Table 2·3). 35,375,507 / 84,049 = **420.9배** — "420배"도 맞다.

> 🔴 **그런데 "420배 작다"를 논문에 그대로 쓰면 위험하다.**
>
> 원문 Table 3 의 파라미터 수는 세 가지다:
>
> | 구조 | 파라미터 | 정확도 | F1 |
> |---|---:|---:|---:|
> | 3D CNN | 20,590,034 | — | — |
> | **3D ResNet** (그들의 **최고** 모델) | **174,500** | **87.36%** | **93.25%** |
> | 3D autoencoder + latent | 35,375,507 | 81.21% | 89.63% |
>
> **그들의 최고 모델은 174,500 params 로 우리(84,049)의 2.1배밖에 안 된다.**
> 35.4 M 은 그들이 **버린** 변형이다. 420배를 내세우면 *"가장 무거운 실패작과 비교했다"* 는
> 반론을 그대로 맞는다 — 심사자가 Table 3 을 열면 바로 보인다.
> → **파라미터 비교를 신규성의 축으로 삼지 않는다.** 쓸 거면 174,500 과 나란히 적는다.

> 🔵 **대신 원문 Table 3 에 훨씬 강한 것이 있다 — 셋 다 실시간이 아니다.**
> **60초 영상(한쪽 눈) 처리 시간**: 3D ResNet **135 s**, 3D CNN 195 s, 3D AE **225 s**.
> 최고 모델조차 실시간의 **2.25배 느리다**. 원문도 *"3D ResNet was the fastest model,
> as well as the best performing one"* 이라고만 적고 에지 언급이 없다.
> → 우리 기여축은 **파라미터 수가 아니라 "실시간 에지 동작"** 이다. 이건 반박이 어렵다.
>
> ⚠️ **미확인**: 그 135 s/225 s 가 **어느 하드웨어**인지 이번 추출에서 확보하지 못했다
> (§여는 질문 2번과 같은 항목). 우리 Pi 5 수치와 나란히 놓으려면 반드시 확인해야 한다.
> 하드웨어가 다르면 "그들은 느리고 우리는 빠르다"로 쓸 수 없다.

**정정된 대비표 (논문에 쓸 형태)**

| | Nousias 2025 최고(3D ResNet) | Nousias 2025 latent(3D AE) | 우리 |
|---|---:|---:|---:|
| 표현 차원 | (직접 분류) | 2048 | **16** |
| 파라미터 | 174,500 | 35,375,507 | **84,049** |
| 정확도 / F1 | 87.36% / 93.25% | 81.21% / 89.63% | (T3-6 후 기입) |
| 60 s 영상 처리 | 135 s | 225 s | (Pi 5 실측 예정) |
| 에지 실측 | 없음 | 없음 | **Raspberry Pi 5** |

### A2-3. 입력 눈 크롭 해상도 — 문헌 관행

| 연구 | 크롭 |
|---|---|
| mEBAL / mEBAL2 | **50 × 50**, 한쪽 눈 |
| Nousias 2025 | **48 × 48**, 한쪽 눈 (×12프레임) |
| **우리** | **64 × 160**, 양눈 (한쪽당 약 64×80 상당) |

**우리가 문헌보다 해상도가 높고 양눈을 쓴다.** 논문에 명시할 사실이다.

> ⚠️ **단, 절제 실험 해석에 주의가 필요하다.** 우리 눈꺼풀 간격 **8.73 px** 는
> 크롭을 **눈 사이 거리의 2.2배**로 넓게 잡은 결과다. 48×48 로 **한쪽 눈만 타이트하게**
> 자르면 같은 눈꺼풀이 훨씬 많은 픽셀을 차지한다.
> → *"세로 해상도를 지켜야 한다"* 는 우리 논거는 **넓은 양눈 크롭을 쓴다는 선택과 묶여 있다.**
> T3-8(vpres vs vdrop) 결과를 해석할 때 이 결합을 밝혀야 한다.

### A2-4. 에지 / 소비자급 하드웨어

| 연구 | 내용 |
|---|---|
| **Optimized deep learning architectures for high precision eye blink detection on consumer grade hardware** (Discover AI, 2026) | ResNet / VGG-19 / **경량 맞춤 CNN** 비교. 4개 안구 상태(open/closed/left/right), 5MP 웹캠. ResNet 과 맞춤 CNN 모두 평균 정확도 **99.81%**. 50명 3,206 프레임, 층화 10-fold |
| Raspberry Pi 5 졸음 검출 데이터셋 (2025) | **640×480 @ 30 fps**, H.264 |
| PiCamera 기반 EAR 구현들 | 640×480, 최대 32 fps |
| Efficient Eye-Blinking Detection on Smartphones (하이브리드 딥러닝) | 스마트폰 대상 | 

> ⚠️ **"99.81%" 를 우리 PR-AUC 와 나란히 놓지 마라.** 3,206 프레임 / 4-클래스 /
> 층화 10-fold(피험자 분리 아님) 로 우리 프로토콜과 전혀 다르다.

### A2-5. 깜빡임 검출의 일반적 실패 조건 — 전사문 10:29 의 직접 지시

리뷰(*A review of deep learning in blink detection*, PeerJ CS 2594 / PMC11784707)와
최신 연구가 공통으로 드는 조건 — [검색요약]

| # | 조건 | 문헌 서술 |
|---|---|---|
| F1 | **조도 변화** | 실외 자연광은 매우 동적. 너무 밝거나 어두우면 **눈 특징의 변별력이 떨어진다** |
| F2 | **가림(occlusion)** | **안경·머리카락**에 의한 가림 + **조명에 의한 가림** |
| F3 | **고개 자세 / 극단적 눈 각도** | 고전적 특징 기반 알고리즘이 특히 취약 |
| F4 | **머리 움직임에 따른 slippage** | 추적 이탈 |
| F5 | **학습 데이터 불균형** | — |
| F6 | **복잡한 환경 간섭** | — |
| F7 | **실시간 처리 / 디바이스 제약** | 우리 논문의 두 번째 기여축과 직결 |

**해결 시도 (최근)**

- **OptFlowBlinkFormer** (PRCV 2025) — 광류 + RGB 융합 Transformer, in-the-wild
- **BlinkLinMulT** (J. Imaging 9(10):196, 2023) — 선형 복잡도 cross-modal attention.
  **조명 변화와 넓은 범위의 고개 자세**를 명시적으로 다룬다

> 🔵 **`EXPERIMENT_PLAN.md` §4 의 조건 목록이 문헌과 일치한다** — 조도·안경·고개 각도·
> 눈 위치 변화는 전부 F1~F4 에 대응한다. **우리가 임의로 만든 목록이 아니라는 근거가 생겼다.**
> 다만 F4(slippage)는 우리 조건 목록에 없다 → `face_position.json` 의 off_seat_rate 층화가
> 여기에 해당하므로 그렇게 연결해 서술한다.

### A2-6. 아직 확인 못 한 것

| # | 항목 | 왜 필요한가 |
|---|---|---|
| 1 | mEBAL/mEBAL2 CNN 의 **공개 구현·가중치** 유무 | 재현 대신 그대로 쓸 수 있는지 |
| 2 | Nousias 2025 의 **추론 시간 0.15 s 가 어느 하드웨어인지** | 에지 비교 가능 여부 |
| 3 | HUST-LEBW 벤치마크의 표준 프로토콜 | 크로스 데이터셋 검증을 넣을지 |
| 4 | 위 모든 [검색요약] 의 **원문** | 논문 인용 전 필수 |

---

### 🔴 우리 숫자를 이것과 나란히 놓으면 안 되는 이유

| | mEBAL2 공식 | 우리 |
|---|---|---|
| 입력 | **NIR + RGB 멀티스펙트럴** | **RGB 단독** |
| 인원 | 180명 | **57명** (배포본 58명 중 U18 제외) |
| 이벤트 | 21,100 (벤치마크 부분집합) | **27,758** (배포 CSV 전량 기준) |
| 분할 | 공식 벤치마크 프로토콜 | **피험자 분리 5-fold × 3시드** |
| 지표 | accuracy 99% | **PR-AUC 0.9886 ± 0.0038** |

**"우리가 99% 에 근접했다"고 쓰면 틀린 문장이다.** 입력 스펙트럼·인원·이벤트 정의·분할·지표가
전부 다르다. PROTOCOL §3-bis 가 이미 공식 서술과 실측이 어긋난다고 기록해 뒀다.

---

## B. 눈 영역은 원래 강하게 식별적이다 — 우리 문제의 전제

| 연구 | 내용 | 확인 |
|---|---|---|
| **Leveraging Large-Scale Face Datasets for Deep Periocular Recognition via Ocular Cropping** (arXiv 2510.26294, 2025-10) | VGGFace2 에서 **눈 주변 크롭 190만 장**으로 CNN 학습. **EER 9–15%** (VGGFace2-Pose), **1–2%** (UFPR-Periocular) | [검색요약] |
| VRBiom (Electronics 14(9):1835, 2025) | HMD 내향 카메라 눈 주변 영역 생체 데이터셋 | [검색요약] |
| Deep Learning for Iris Recognition: A Survey (ACM CSUR) | 홍채 인식 개관 | [검색요약] |

**함의**: 우리 재식별 0.6345 는 **놀라운 값이 아니라 예상된 값**이다. 눈 크롭이 신원을
담는다는 건 이미 확립돼 있다. 논문에서 "눈 크롭이 신원을 담는다"를 발견처럼 쓰면 안 된다.

**우리 쪽의 새로움은 그게 아니라**: 깜빡임 지도학습으로 D=16 까지 좁힌 표현이
**학습을 전혀 안 한 랜덤 인코더(0.6261)와 같은 수준**으로 샌다는 것 — 즉 목적함수도
차원 축소도 누출을 줄이지 못한다는 실측이다.

---

## C. 학습된 표현의 신원 누출 — 가장 가까운 선행연구 ★

### From Measurement to Mitigation (arXiv 2604.05296, 2026-04) — **필독**

Persona Identities. **비-얼굴인식(non-FR) 인코더의 신원 누출을 감사하고 완화한다.** [원문 확인]

- **문제 설정이 우리와 같다**: CLIP·DINOv2/v3·SSCD 처럼 **신원 감독 없이 학습된** 인코더를
  얼굴 포함 데이터에 쓸 때 얼마나 새는가
- 측정: open-set TAR@low-FAR, 확산 기반 **템플릿 역변환**, face–context 귀속
- 완화: **ISP (Identity Sanitization Projection)** — 추정한 신원 부분공간을 **선형 사영으로 제거**
- 결과: "ISP drives linear access to near-chance while retaining high non-biometric utility"

**🔴 우리 PROTOCOL §11 후속 게이트와 정확히 겹친다.**
우리는 "사후 투영 제거 전/후 × 선형/MLP 4칸. **선형만 내려가고 MLP 는 그대로면 프라이버시
기전이 아니라 프로브 회피**"라고 적어 뒀다. 이 논문은 그 우려를 스스로 인정한다:

> "Our study is restricted to frozen encoders and evaluates only **linear susceptibility**;
> ISP provides **no guaranteed protection against stronger non-linear or generative attacks**."
> — §5 Limitations

그리고 비선형 검사를 하긴 하지만 **projection-only MLP** (사영된 임베딩 위에서만 학습)이고,
저자들도 "this does not constitute a non-linear certificate" 라고 적었다.

**우리의 차별점**: 우리는 선형·MLP 를 **처음부터 둘 다** 필수로 두고, 그것이 실제로 값을 했다
— 광학 교란이 선형 0.0767 / MLP 0.2018 로 2.6배 달랐다(PROTOCOL §8).
그리고 우리 ours 도 랜덤 인코더 대비 **선형만 내려가고(0.5017→0.4783) MLP 는 그대로**다.
이 논문의 프레임으로 보면 그것을 프라이버시로 읽을 뻔했다.

**활용**: 우리 `src/v2/common/probes.py` 의 `fit_identity_subspace` / `project_out` 이 사실상 ISP 다
(G-1h 로 동작 검증됨). 후속 억제 라운드에서 **ISP 를 대조군으로 두고 MLP 프로브로 깨보는 것**이
바로 쓸 수 있는 실험이다.

### 그 밖

| 연구 | 내용 | 확인 |
|---|---|---|
| Reducing Overlearning through Disentangled Representations by Suppressing Unknown Tasks (arXiv 2005.10220) | 알려지지 않은 과제를 억제하는 disentanglement | [검색요약] |
| Image Privacy Protection: A Survey (arXiv 2412.15228) | 이미지 프라이버시 보호 개관 | [검색요약] |
| Differential Privacy for Transformer Embeddings with Nonparametric Variational IB (arXiv 2601.02307) | 임베딩에 정보 병목 + DP | [검색요약] |
| Cancelable biometrics / random projection 계열 | 랜덤 사영으로 템플릿 보호 — **우리 대조군 3(동일 차원 random projection)의 문헌 근거** | [검색요약] |

---

## D. 눈 데이터 프라이버시 종합

| 연구 | 내용 | 확인 |
|---|---|---|
| **A Data-Driven Review of a Decade of Privacy Research in Eye Tracking** (PACM HCI, 3806024) | **2015–2025, 78편** 체계 분석. 시선·홍채·눈 이미지 전반. "내부 표현과 출력이 생체 신원·외형·행동/인지 특성을 노출할 수 있다" | [검색요약] |
| Privacy Enhancement for Gaze Data Using a Noise-Infused Autoencoder (arXiv 2508.10918) | 잡음 주입 오토인코더로 시선 데이터 보호 | [검색요약] |
| Assessing the Privacy Risk of Cross-Platform Identity Linkage using Eye Movement Biometrics (arXiv 2402.08655) | 플랫폼 간 신원 연결 위험 | [검색요약] |
| An investigation of privacy preservation in deep learning-based eye-tracking (BioMed Eng OnLine, 2022) | 딥러닝 시선 추적의 프라이버시 | [검색요약] |

---

## E. 🔴 깜빡임 **동역학만으로도** 사람이 식별된다 — 로드맵에 직접 영향

| 연구 | 내용 | 확인 |
|---|---|---|
| **Seha, Hatzinakos, Zandi, Comeau**, *Improving eye movement biometrics in low frame rate eye-tracking devices using periocular and eye blinking features*, Image Vis. Comput. **108**:104124 (2021) | 저프레임률 운전자 인증, **55명**. 멀티모달 Rank-1 **96.67%**, **깜빡임 단독 93.94%** | [원문 확인 — Sensors 25:4304 관련연구절에 인용된 형태로 확인. **1차 출처 원문은 미확인**] |
| Towards Improved Eye Movement Biometrics (Sensors 25(14):4304, 2025-07) | GazeBase, LSTM 으로 **96%**. 3년에 걸친 안정성 검토 | [원문 확인] |
| NeuroBiometric (IEEE/CAA JAS) | 이벤트 카메라로 깜빡임 지속·속도·에너지·비율·주파수 특징 → 인증 | [검색요약] |

### 우리 로드맵에 주는 함의

PROTOCOL §11 의 억제 설계안 **"A 동역학 — 세션 서명이 주범이면 입력에서 지우는 게 먼저"**
는 **깜빡임 동역학만 남기면 안전해진다는 가정에 서 있다.** 문헌은 그 가정을 지지하지 않는다.

> 깜빡임 동역학 단독으로 55명 중 **93.94%** 식별 (Seha et al. 2021)

즉 **이미지를 지우고 동역학만 남겨도 신원이 남을 수 있다.** 안 A 를 고르더라도
**동역학 표현에 대해 재식별을 다시 재야 한다** — 이미지 제거가 프라이버시를 준다고
가정하면 안 된다. 우리 §12("측정하지 않는 것")에는 지속시간 라벨이 없다고 적혀 있어
mEBAL2 로는 동역학 특징을 온전히 재현하기 어렵다는 제약도 함께 본다.

---

## 정리 — 이 문헌들이 우리 서술에 강제하는 것

1. **"눈 크롭이 신원을 담는다"를 발견처럼 쓰지 않는다.** periocular 인식이 이미 EER 1–15% 다.
   우리 새로움은 **"깜빡임 학습으로도, D=16 압축으로도 줄지 않는다"** 쪽이다.
2. **mEBAL2 공식 99% 와 우리 0.9886 을 나란히 놓지 않는다.** 스펙트럼·인원·분할·지표가 다르다.
3. **선형 부분공간 제거(ISP 계열)를 프라이버시 해법으로 인용하지 않는다.** 최신 논문
   자신이 "선형만 인증, 비선형 보장 없음"이라고 적었다. 우리 선형·MLP 병기 규칙이
   그대로 차별점이 된다.
4. **"동역학만 쓰면 안전하다"고 쓰지 않는다.** 깜빡임 단독 93.94% 식별 사례가 있다.
5. 억제 라운드를 시작하면 **ISP 를 대조군으로 두고 MLP 프로브로 깨보는 실험**이 가장 값싼
   차별화다 — 우리 `probes.fit_identity_subspace` / `project_out` 이 이미 있다.

## 아직 못 찾은 것 (다음 조사 항목)

- mEBAL2 를 **RGB 단독·피험자 분리**로 재평가한 후속 연구 — 있으면 직접 비교 대상이 된다
- 깜빡임/졸음 검출 표현의 **재식별을 함께 보고한** 연구 — 현재까지 발견 못 함.
  **없다면 그게 우리 기여의 근거**이지만, "없다"는 주장은 체계적 검색 없이 하면 안 된다
- Raspberry Pi 급 엣지에서 깜빡임 검출 **지연 실측**을 보고한 2025~2026 연구
