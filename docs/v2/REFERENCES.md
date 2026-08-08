# REFERENCES — 논문에 들어갈 인용 후보

> **작성 2026-08-06.** IEEE Sensors Letters(약 4페이지) 기준. Related Work 는 약 0.4페이지이므로
> **실제 인용은 15~20편**이 상한이다. 아래는 후보이며 ★ 표시가 우선순위다.
>
> **확인 상태**
> - ✅ **[원문]** — PDF/본문을 직접 열어 확인
> - ⚠️ **[검색요약]** — 검색 결과 요약만 봄. **인용 전 원문 확인 필수**
>
> 상세 분석은 [`RELATED_WORK.md`](RELATED_WORK.md).

---

## A. 동기 (Introduction) — 눈 건강 · 화면 시간

| # | 문헌 | 쓸 내용 | 확인 |
|---|---|---|---|
| A1 ★ | **Rosenfield, M.** *Computer Vision Syndrome: a Review of Ocular Causes and Potential Treatments.* Ophthalmic Physiol Opt 31(5), 502–515, 2011 | 화면 작업과 안구 증상의 표준 리뷰. mEBAL 도 이걸 인용한다 | ⚠️ |
| A2 ★ | **Digital Applications for Videoterminal-Associated Dry Eye Disease** (PMC11728679, 2025) | **디지털 기기 사용이 깜빡임 빈도·동역학을 나쁘게 만들고, 불완전 깜빡임이 마이봄샘 기능을 손상시킨다.** 우리 응용의 직접 근거 | ⚠️ |
| A3 | 화면 응시 시 깜빡임 감소 (분당 3~7회, 평상시의 약 1/3) | Intro 첫 문단의 수치 | ⚠️ **학술 출처 필요** |
| A4 | Schiffman, *Sensation and Perception* / Harvard DB of Useful Biological Numbers | **깜빡임 지속 100~400 ms** (mEBAL 원문이 인용) | ✅ (mEBAL 경유) |

> 🔴 **A3 이 비어 있다.** "화면 보면 덜 깜빡인다"는 Intro 의 출발점인데 현재 임상 블로그 수준
> 출처밖에 없다. **동료심사 저널 출처를 찾아야 한다.**

---

## B. 깜빡임 검출 — 랜드마크 / EAR 계열

| # | 문헌 | 쓸 내용 | 확인 |
|---|---|---|---|
| B1 ★★ | **Soukupová, T. & Čech, J.** *Real-Time Eye Blink Detection using Facial Landmarks.* CVWW 2016 | **EAR 의 원전.** 우리 베이스라인의 근거 | ⚠️ |
| B2 | **Drutarovsky, T. & Fogelton, A.** *Eye Blink Detection using Variance of Motion Vectors.* ECCVW 2014 | 모션 기반 계열 | ⚠️ |
| B3 | Fogelton & Benesova — 모션 벡터 + 유한상태기계 | 위와 같은 계열. **머리 회전이 크면 안 된다는 제약**을 스스로 밝힘 | ⚠️ (Nousias 경유) |

> 🔴 **B1 에 대한 중요한 사실.** Soukupová & Čech 은 EAR 을 **단순 임계값**으로 쓰지 않는다 —
> *"uses an **SVM classifier** to detect eye blinks as a pattern of EAR values in a short
> temporal window."*
> → **우리 `ear_head`(EAR 스칼라 → 학습된 시간 헤드)가 원전에 더 가까운 비교 상대다.**
> `ear_rule`(drop_ratio 임계값)만 이기고 끝내면 원전을 약화시킨 것이 된다.
> 우리는 이미 둘 다 내고 있으므로 문제없다. **논문에 이 점을 명시하면 방어가 강해진다.**

---

## C. 깜빡임 검출 — 딥러닝 / 시퀀스 ★ 핵심 섹션

| # | 문헌 | 쓸 내용 | 확인 |
|---|---|---|---|
| C1 ★★★ | **Daza, R., Morales, A., Fierrez, J., Tolosana, R.** *mEBAL: A Multimodal Database for Eye Blink Detection and Attention Level Estimation.* ICMI '20 Companion | 🔴 **우리 `image_cnn` 베이스라인의 출처.** 50×50, 3 conv(32/32/64,3×3)+3 maxpool, dense 64, sigmoid, dropout 0.5, Adam 1e-3. 시퀀스 판정은 **프레임 점수의 max** | ✅ **[원문]** |
| C2 ★★★ | **Daza et al.** *mEBAL2 Database and Benchmark: Image-based Multispectral Eyeblink Detection.* Pattern Recognition Letters 182, 2024 (arXiv 2309.07880) | **우리 데이터셋의 출처.** 180명·21,100 시퀀스. ConvLSTM 최대 99% | ⚠️ |
| C3 ★★★ | **Nousias, G., Delibasis, K., Labiris, G.** *Blink Detection Using 3D Convolutional Neural Architectures and Analysis of Accumulated Frame Predictions.* J. Imaging 11(27), 2025 | 🔴 **가장 가까운 선행연구.** 3D AE + latent(2048) 분류. **60초 영상 양눈 570 s @ RTX 3060**, 35.4M params, step=1 슬라이딩 | ✅ **[원문]** |
| C4 ★★ | **Hu, G., Xiao, Y., Cao, Z., et al.** *Towards Real-Time Eyeblink Detection in the Wild: Dataset, Theory and Practices.* IEEE TIFS, 2019 | **HUST-LEBW** 데이터셋(673 샘플, 영화 20편, 1280×720/1456×600) + 수정 LSTM | ⚠️ |
| C5 ★★ | **Zeng, W., et al.** *Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video.* **CVPR 2023** (arXiv 2303.16053) | **MPEblink** — 686 미편집 영상, 8,748 이벤트. 얼굴 검출·추적·깜빡임 검출을 **한 단계로 end-to-end** | ⚠️ |
| C6 ★ | **DeFB: Decomposed Feature Learning for Real-Time Multi-Person Eyeblink Detection.** AAAI | C5 의 후속. 실시간 다중 인물 | ⚠️ |
| C7 ★ | **BlinkLinMulT: Transformer-Based Eye Blink Detection.** J. Imaging 9(10):196, 2023 | 선형 복잡도 cross-modal attention. **조명·고개 자세**를 명시적으로 다룸 | ⚠️ |
| C8 | **OptFlowBlinkFormer.** PRCV 2025 | 광류 + RGB 융합 Transformer, in-the-wild | ⚠️ |

### ✅✅ C5 MPEblink 원문 확인 완료 (2026-08-06) — **논문 PDF 직접 확인**

> Zeng, W., Xiao, Y., Wei, S., Gan, J., Zhang, X., Cao, Z., Fang, Z., Zhou, J.T.
> *Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video.*
> **CVPR 2023, pp. 13854–13863.**

#### 🔴 Table 3 — 속도 비교 (원문 표 그대로) **★ 우리 논문에 직접 인용 가능**

> *"The inference speed comparison on a **single NVIDIA 3090 GPU**, assuming the compared
> methods use **InsightFace** for face detection (time consumption **T = 9.3 ms** including
> pre-processing) and landmark detection. #faces denotes the face amount in the scene."*

| 방법 | Time / image (ms) |
|---|---|
| **InstBlink (그들 제안)** | **8.9 + 2.6** (데이터 처리) = **11.5** |
| Soukupová & Čech [40] (**EAR**) | T(9.3) + **5.4 × #faces** |
| Blink detection+ [35] | T + 5.4 × #faces |
| Hu et al. [20] (**HUST-LEBW**) | T + 5.7 × #faces |
| **Daza et al. [9] (mEBAL — 우리 image_cnn 베이스라인)** | T + **9.1 × #faces** |

본문: *"InstBlink is also of high inference speed (i.e., **112 FPS for network forwarding**)."*

#### 🔵 이 표가 우리에게 주는 것 — **매우 크다**

**1. 하드웨어가 확정됐다: NVIDIA 3090 GPU.** 우리 에지/CPU 기여와 충돌하지 않는다.

**2. 얼굴 검출이 병목이라는 우리 발견이 제3자 측정으로 재확인된다.**
그들의 프론트엔드 InsightFace 는 **3090 GPU 에서 9.3 ms** 다.
우리 MediaPipe 는 **데스크탑 CPU 에서 4.12 ms** 다.
→ *"우리 프론트엔드가 GPU 급 검출기보다 CPU 에서 더 싸다"* 를 말할 수 있다.

**3. 우리 베이스라인(mEBAL/Daza)의 제3자 측정값이 생겼다.**
얼굴 1개 기준 **9.3 + 9.1 = 18.4 ms/image @ 3090 GPU**.
우리 파이프라인(디코딩 제외) = detect 4.12 + crop 0.36 + encode 0.26 + head 0.14
= **4.88 ms/image @ 데스크탑 CPU**.

> ⚠️ **"3.8배 빠르다"고 쓰지 마라.** 검출기(InsightFace vs MediaPipe)·프레임워크·해상도·
> 구현이 전부 다르다. 이 값은 **Zeng et al. 이 자기 실험 조건에서 잰 것**이다.
> ✅ **쓸 수 있는 형태**: *"Under a common protocol on an NVIDIA 3090 GPU, [Zeng et al.]
> report 18.4 ms per image for the CNN-based detector of [Daza et al.] with a single face,
> where face and landmark detection alone accounts for 9.3 ms."* — **보고값 인용**.

**4. 깜빡임 지속시간이 또 확인됐다**: *"most eyeblink events (i.e., **0.2–0.4 s**)"*
→ mEBAL 의 100~400 ms 와 일치. Intro·검출 지연 논의에 쓴다.

#### 실험 설정 (원문 §5)

| 항목 | 값 |
|---|---|
| 백본 | **TeViT** + ResNet-50 (`tevit_r50`), MMDetection |
| 입력 해상도 | **640 × 360** 으로 리사이즈 |
| 학습 | 절반 프레임레이트(**~12 FPS**) 표집, clip length **11**, AdamW, 10,000 iter |
| **추론** | 원 프레임레이트, clip length **36**, **stride 18** (겹침 클립을 face box IoU 로 연결) |
| 크로스 데이터셋 | MPEblink 로 학습 → HUST-LEBW 에서 **F1 83.45%** |

> 🔵 **창 stride 비교 — 우리 선택의 위치가 잡힌다**
>
> | 연구 | 창 길이 | stride |
> |---|---|---|
> | Nousias 2025 | 12 | **1** (dense overlap) |
> | **우리** | 19 | **1** (권고 반영) |
> | MPEblink 2023 | 36 | **18** (절반 겹침) |
>
> stride 1 은 극단적 선택이 아니다. 선행 연구가 양쪽에 다 있다.

#### 결론 — 경쟁이 아니라 상보 관계

과제가 다르다: **다중 인물 · 미편집 장편 영상 · 얼굴 검출+추적+깜빡임 동시 수행 · GPU**.
우리는 **단일 사용자 · 이벤트 분류 · 에지 디바이스**다.

> ✅ **논문 문장 (확정)**
> *"Recent work achieves real-time multi-person eyeblink detection in unconstrained
> untrimmed video with a one-stage spatio-temporal transformer, reporting 112 FPS for
> network forwarding on an NVIDIA 3090 GPU [Zeng et al., CVPR 2023]. Our setting is
> complementary: a single user, and inference on edge-class hardware with an
> 84 K-parameter model."*

> ❌ **"우리가 더 빠르다"고 쓰지 마라.** 과제·하드웨어·측정 조건이 전부 다르다.

#### C6 DeFB — 여전히 미확인

같은 그룹의 후속(AAAI). 2026.4 에 **MPEblink2.0 + InstBlink++** 도 공개됐다.
**이 분야는 지금도 활발하다**는 서술 근거는 되지만, **수치는 미확인**이다.

---

## D. 데이터셋

| # | 데이터셋 | 규모 | 해상도 | 확인 |
|---|---|---|---|---|
| D1 | Talking Face (INRIA) | 61 blinks / 1명 | 720×576 | ✅ (mEBAL Table 1) |
| D2 | Pan et al. 2007 (ZJU) | 255 / 20명 | 320×240 | ✅ (동일) |
| D3 | Drutarovsky & Fogelton 2014 (Eyeblink8) | 353 / 4명 | **640×480** | ✅ (동일) |
| D4 | Silesian Deception 2015 | 300 / 5명 | **640×480** | ✅ (동일) |
| D5 ★ | **HUST-LEBW** 2019 | 381 / 172명 | **1280×720** | ✅ (동일) |
| D6 ★★ | **mEBAL** 2020 | 3,000 / 38명 (안경 11명) | 1280×720 | ✅ **[원문]** |
| D7 ★★★ | **mEBAL2** 2024 | 21,100 시퀀스 / 180명 | 1280×720 | ⚠️ |
| D8 ★ | **RT-BENE** (Cortacero, Fischer et al.) ICCVW 2019 | **20만+ 눈 이미지**, 그중 1만+ 감은 눈 | — | ⚠️ |
| D9 | **MPEblink** CVPR 2023 | 686 미편집 영상 / 8,748 이벤트 | — | ⚠️ |

> 🔵 **D3·D4 가 640×480 이고 D5·D6·D7 이 1280×720 이다.**
> `EXPERIMENT_PLAN.md` §6-3 의 **"640×480 주 / 1280×720 보조"** 결정이 문헌 표로 뒷받침된다.

---

## E. 리뷰 / 실패 조건

| # | 문헌 | 쓸 내용 | 확인 |
|---|---|---|---|
| E1 ★★ | **A review of deep learning in blink detection.** PeerJ CS 2594 (PMC11784707), 2025 | 🔴 **전사문 10:29 이 요구한 "일반적 문제 정리".** 조도·가림(안경/머리카락)·고개 자세·slippage·데이터 불균형·실시간/디바이스 제약 | ⚠️ |

---

## F. 에지 / 온디바이스

| # | 문헌 | 쓸 내용 | 확인 |
|---|---|---|---|
| F1 ★★ | **Optimized deep learning architectures for high precision eye blink detection on consumer grade hardware.** Discover AI, 2026 | ResNet / VGG-19 / 경량 맞춤 CNN, 5MP 웹캠, 4개 안구 상태. 50명 3,206 프레임 | ⚠️ |
| F2 ★ | Raspberry Pi 5 기반 졸음 검출 데이터셋, 2025 (PMC12630088) | **640×480 @ 30 fps, H.264** — 해상도 결정 근거 | ⚠️ |
| F3 | Efficient Eye-Blinking Detection on Smartphones: A Hybrid Approach Based on Deep Learning | 모바일 | ⚠️ |
| F4 | Embedded System for Eye Blink Detection Using Machine Learning Technique (IEEE) | 임베디드 | ⚠️ |

---

## G. 파이프라인 구성요소

| # | 문헌 | 쓸 내용 | 확인 |
|---|---|---|---|
| G1 ★★ | **Kartynnik, Y., Ablavatski, A., Grishchenko, I., Grundmann, M.** *Real-time facial surface geometry from monocular video on mobile GPUs.* arXiv 1907.06724, 2019 | **MediaPipe FaceMesh** — 우리 프론트엔드 | ⚠️ |
| G2 | Simonyan & Zisserman, VGG16, ICLR 2015 | mEBAL CNN 이 "inspired by VGG16" 이라 밝힘 | ✅ (mEBAL 경유) |
| G3 | ONNX Runtime | 배포 런타임 | — |

---

## H. ▶ 논문 범위 밖 — 특허·후속 연구용으로 보존

| # | 문헌 | 용도 |
|---|---|---|
| H1 | *From Measurement to Mitigation* (arXiv 2604.05296, 2026) | 비-얼굴인식 인코더의 신원 누출 감사·완화 |
| H2 | *Leveraging Large-Scale Face Datasets for Deep Periocular Recognition* (arXiv 2510.26294) | 눈 주변 크롭의 식별성 (EER 1~15%) |
| H3 | 깜빡임 동역학만으로 **93.94% 식별** | 특허 §1 의 핵심 반례 |
| H4 | VRBiom (Electronics 14(9):1835, 2025) | HMD 눈 주변 생체 |

> 🔴 **이 논문에 넣지 않는다.** 넣으면 "그래서 너희는 신원을 어떻게 했나"를 자초한다.

---

## 다음 할 일 — 원문 확인 우선순위

| 순위 | 대상 | 왜 급한가 |
|---|---|---|
| **1** | **C5 MPEblink / C6 DeFB** | **"real-time" 주장의 직접 경쟁자.** 하드웨어·fps 미확인 |
| **2** | **C2 mEBAL2** | 우리 데이터셋의 출처. 99% 주장과 벤치마크 프로토콜 |
| **3** | **A3 화면 응시 시 깜빡임 감소** | Intro 의 출발점인데 **학술 출처가 없다** |
| 4 | E1 리뷰 | 실패 조건 목록의 근거 |
| 5 | B1 Soukupová & Čech | EAR 원전. SVM 사용 사실 확인 |
| 6 | D8 RT-BENE, F1 | 에지·데이터셋 보조 |

> ⚠️ **[검색요약] 을 그대로 인용하지 않는다.** 이 프로젝트 부록 #10:
> *"문서가 아니라 파일을 열어 확인해라."*
