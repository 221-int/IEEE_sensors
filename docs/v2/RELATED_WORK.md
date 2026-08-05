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
