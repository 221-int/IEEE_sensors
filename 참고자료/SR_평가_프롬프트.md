# SR 모델 선정용 프롬프트 (코딩 에이전트에 던지는 용도)

작성일: 2026-07-20
대상: `src/eyeblink/robust.py` 의 `SuperResolution` 스텁에 얹을 경량 SR 모델 선정.
사용법: 아래 **공용 컨텍스트**를 먼저 붙이고, 목적에 따라 **STAGE 1**(생각 정리·의사결정) 또는 **STAGE 2**(실측 테스트)를 이어 붙여 코딩 에이전트에 준다. 처음이면 STAGE 1 → 결과를 보고 STAGE 2 순서를 권장.

---

## 공용 컨텍스트 (STAGE 1·2 앞에 항상 붙일 것)

```
너는 이 저장소(IEEE_sensors)에서 일하는 시니어 임베디드 CV 엔지니어다.

[프로젝트]
- 라즈베리파이 5(CPU only, GPU/NPU 없음) 위에서 도는 실시간 눈 깜빡임 검출 파이프라인.
- 파이프라인: 카메라(1280x720) → MediaPipe FaceLandmarker → 눈 6점 EAR 계산
  → STD 기반 4단계 상태머신 → 깜빡임 카운트/분류.
- 목표 상황: 눈 영역 품질이 낮을 때(멀거나·저해상도·저조도) 랜드마크·EAR이 흔들린다.
  이때 얼굴/눈 ROI에만 경량 SR을 조건부로 걸어 랜드마크·EAR을 안정화하려 한다.

[관련 코드 — 반드시 먼저 읽어라]
- src/eyeblink/robust.py  : 전처리 훅. Passthrough(none)만 실동작, SuperResolution(sr)은 스텁.
  get_preprocessor(name) 로 파이프라인에 주입. BGR ndarray in → BGR ndarray out 규약.
- src/eyeblink/landmarks.py : build_landmarker / eye_aspect_ratio / frame_ear (EAR 계산부).
- src/eyeblink/config.py   : 눈 랜드마크 인덱스, EAR 임계·EMA·캘리브레이션 파라미터.
- benchmark.py            : 단계별 latency / fps / CPU% / RSS / SoC 온도 측정 하니스.
                            --mock, --video clip.mp4, --stride N 지원.
- data/                   : eyeblink8, talkingFace 데이터셋 + data/blinks.csv(정답 라벨).
- 참고자료/SR모델_후보.md : 후보 정리 문서(FSRCNN/ESPCN/LapSRN/EDSR/edge-SR/Real-ESRGAN).

[하드 제약]
- Pi5 CPU only. 전체 프레임 SR은 금지에 가깝다 — 눈 bbox ROI(예: 40x24px)만 2x~4x 업스케일.
- 파이프라인은 이미 OpenCV 기반. 추가 프레임워크 도입은 비용으로 계산할 것.
- 실시간 예산: 조건부 SR을 켜도 end-to-end 목표 fps를 유지해야 함.
- 핵심 리스크: "순차 SR이 오히려 검출을 해치는 사례"가 있다. SR이 EAR/랜드마크를
  실제로 개선한다는 증거 없이는 채택하지 않는다.

[후보 요약]
FSRCNN / FSRCNN-small (OpenCV dnn_superres) — 1순위 후보, 경량·드롭인.
ESPCN (OpenCV dnn_superres) — sub-pixel, 실시간 영상용, 대안 1순위.
LapSRN — 품질↑ 속도↓. EDSR — 무거움, Pi 부적합(제외 후보).
edge-SR(eSR) — 초경량 PyTorch, FSRCNN도 느릴 때 대안.
Real-ESRGAN(ncnn) — 품질 최고 GAN, 무거움·통합 어려움, 후순위.
```

---

## STAGE 1 — 심층분석·의사결정 (코드 실행 없이 "꼼꼼히 생각")

```
[STAGE 1: 분석 및 후보 압축 — 코드를 실행하지 말고 판단 근거를 만들어라]

목표: 위 후보 중 Pi5에서 "조건부 눈 ROI SR"에 실제로 채택할 1~2개를 근거와 함께 좁혀라.

먼저 robust.py, landmarks.py, config.py, benchmark.py, 참고자료/SR모델_후보.md 를
직접 읽고 우리 파이프라인의 실제 인터페이스(입출력 규약, 눈 ROI 크기, 예산)를 파악하라.
추측하지 말고 코드에서 확인한 사실만 근거로 써라. 모르면 모른다고 하라.

각 후보에 대해 아래 축으로 평가표를 채워라(정성 + 가능한 근거 수치):
  1. Pi5 CPU 실시간성  — 눈 ROI(약 40x24 → 2x/3x/4x) 기준 예상 latency 크기감.
  2. 통합 비용         — 추가 프레임워크/모델 포맷/빌드 부담. OpenCV dnn_superres 드롭인 여부.
  3. SR 품질 vs 목적적합 — 우리 목적은 "예쁜 그림"이 아니라 "랜드마크/EAR 안정화".
                          자연영상 학습 SR이 눈/눈꺼풀 경계에 도움이 될지, 오히려
                          아티팩트로 EAR을 왜곡할 위험은 없는지.
  4. 배율 선택         — 2x/3x/4x 중 어디가 랜드마크 검출 안정화에 유효할지 가설.
  5. 실패·리스크       — "순차 SR이 검출을 해치는" 조건, GAN 환각, 시간적 흔들림(temporal jitter) 등.

그다음:
  - 반드시 반례를 스스로 제기하라: "SR을 아예 안 쓰고 단순 bicubic 업스케일 또는
    입력 해상도만 올리는 편이 낫지 않은가?"를 baseline으로 명시하고, SR이 이겨야 할 대상으로 삼아라.
  - 채택 여부를 가르는 정량 게이트를 제안하라. 예: (a) 저해상도/암전 셋에서 SR-on 이
    SR-off 대비 깜빡임 검출 F1을 최소 X%p 올릴 것, (b) SR 켠 상태 end-to-end ≥ Y fps 유지,
    (c) SR을 켤 트리거(눈 가로 픽셀 < N px)의 N 후보값. X/Y/N은 네가 근거와 함께 초기값 제안.

출력 형식(마크다운):
  1. 후보 평가표 (위 5축)
  2. 1~2개로 좁힌 추천 + 각각 왜 / 언제 켜는지
  3. baseline(무SR·bicubic) 대비 SR이 넘어야 할 정량 게이트(구체 숫자 초안)
  4. STAGE 2에서 검증해야 할 가설 목록(측정 가능한 형태로)
  5. 열린 질문 / 확인 필요한 불확실성

코드는 수정하지 마라. 이 단계 산출물은 "판단 근거 문서" 하나다.
```

---

## STAGE 2 — 실측 테스트·벤치마크 (코드 짜서 검증)

```
[STAGE 2: 구현 및 실측 — STAGE 1에서 좁힌 후보를 데이터로 검증하라]

STAGE 1의 추천(1~2개 SR 모델)과 정량 게이트를 입력으로 받는다. 목표는 "채택/기각"을
근거 수치로 판정하는 것. 예쁜 코드보다 재현 가능한 실험이 우선이다.

1) 구현
   - robust.py 의 SuperResolution 스텁을 실제 구현으로 채운다.
     * 우선 OpenCV cv2.dnn_superres 로 FSRCNN-small / ESPCN 를 로드(.pb).
     * 모델 파일은 models/sr/ 아래에 두고, 없으면 다운로드 안내 + graceful fallback(bicubic).
     * 규약 유지: BGR ndarray in → BGR ndarray out, get_preprocessor("sr", scale=...) 로 주입.
     * ★ 전체 프레임이 아니라 "눈/얼굴 ROI만" 잘라 업스케일하는 경로를 명시적으로 만든다.
   - 비교군(preprocessor)을 최소 3개 준비: none(원본), bicubic(단순 업스케일), sr(모델별·배율별).

2) 저품질 조건 재현
   - data/ 의 eyeblink8/talkingFace 원본에서 저해상도·저조도 버전을 합성하라
     (다운스케일 후 재확대, 감마/밝기 저하, 가우시안 블러/노이즈 등). 원본은 참조(정답 근처)로.
   - data/blinks.csv 를 정답 라벨로 사용해 검출 성능을 계산할 수 있게 매핑하라.

3) 측정 (두 축을 분리)
   A. 속도·자원 — benchmark.py 하니스를 재사용/확장.
      preprocessor별로 end-to-end fps, SR 단계 latency(mean·p95), CPU%, RSS, SoC 온도.
      가능하면 Pi5 실측. Pi가 없으면 PC 상대비교 + "Pi5에서 재실행" 지점을 코드/문서에 표시.
   B. 정확도 — 저품질 셋에서 preprocessor별:
      - 랜드마크/얼굴 검출 성공률, frame_ear 안정성(분산·jitter),
      - blinks.csv 대비 깜빡임 검출 F1/precision/recall (또는 count 오차).
      원본(고품질) 결과를 상한 기준으로 함께 표시.

4) 판정
   - STAGE 1의 게이트(F1 +Xp, ≥Y fps, 트리거 N px)에 실제 수치를 대입해 통과/탈락 표로 정리.
   - 반드시 확인: SR-on 이 none/bicubic 을 실제로 이겼는가? 어느 조건에서 오히려 해쳤는가?
     (순차 SR 손해 사례를 적극적으로 찾아 보고하라 — 없다고 가정하지 말 것.)
   - 배율(2x/3x/4x)·모델별 트레이드오프 곡선(정확도 vs fps)을 제시.

5) 산출물
   - 채워진 robust.py(및 필요한 유틸/실험 스크립트, scripts/ 아래).
   - 결과표 + 짧은 결론: "무엇을, 몇 배로, 어떤 트리거 조건에서 켠다 / 혹은 채택 보류".
   - 재현 커맨드(예: python benchmark.py --video ... --preproc sr:fsrcnn:2 형태)를 문서화.
   - 논문/실험로그(참고자료/실험로그.md)에 붙일 수 있는 표·수치 형태로 정리.

주의: 결과가 SR 채택에 불리하게 나와도 그대로 보고하라. 목표는 SR을 넣는 게 아니라
"넣는 게 맞는지"를 데이터로 답하는 것이다. 측정 방법의 한계도 함께 적어라.
```

---

## 사용 팁
- STAGE 1만 먼저 돌려 후보를 2개 이하로 줄인 뒤 STAGE 2로 가면 실험 시간이 크게 준다.
- STAGE 2를 PC에서 먼저 돌려 파이프라인/지표를 검증하고, 최종 fps·전력만 Pi5에서 재측정하는 2패스 권장.
- 게이트 숫자(X/Y/N)는 STAGE 1이 제안한 값을 STAGE 2 프롬프트에 그대로 채워 넣어라.
