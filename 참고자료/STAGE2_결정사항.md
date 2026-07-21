# STAGE 2 결정사항 (승인 완료 + 이번 라운드 스코프)

작성일: 2026-07-17 · 상태: A·B·C 승인 완료

---

## 0. 이번 라운드 스코프 (★ 중요)

- **이번엔 공개 데이터셋(모델 학습·F1 벤치) 사용 안 함.** → **4명(염한울·정다현·강지민·박덕환) 자체 테스트만.**
- 정답(ground truth) = **4인 촬영 + 큐 기반**(calibrate 프로토콜 + 테스트 세션에서 "지금 깜빡" 시점 기록).
- **SR 평가도 자체 4인 영상으로 완결**: 깨끗한 4인 영상을 **합성 열화(다운스케일/감마/노이즈)** → SR on/off 검출 정확도 비교. (공개셋 `.tag` 불필요)
- 공개셋(Eyeblink8 `.tag`) F1 벤치, 모델 학습·비교 = **이후 확장**으로 미룸.
- 내일까지 최우선 = **SR 모델 확정 + 얼굴 ROI two-pass 최소 구현.**

---

## 1. 승인된 결정

| # | 쟁점 | 확정값 | 상태 |
|---|---|---|---|
| A | 아키텍처 | **얼굴 ROI two-pass SR** (눈 픽셀 w_eye=게이트, SR 대상=얼굴 crop, EAR 스케일 불변→역매핑 불필요) | ✅ |
| B | 목표 fps | **하한 15fps(필수) / 목표 24fps / no-SR 운영점 ~30fps** | ✅ |
| C | 평가 열화 표준 | 다운스케일 {2,3,4}×, 감마 {1.0,0.6,0.4}, 노이즈 σ{0,5,10} | ✅ |

---

## 2. 결정별 상세 (스코프 반영)

- **A. 아키텍처:** pass1 FaceLandmarker(원본)→얼굴 bbox·w_eye → w_eye < 임계값이면 얼굴 crop을 SR(작은 입력, 저렴) → pass2 FaceLandmarker(SR crop) → EAR. 프론트엔드 교체(눈 전용 랜드마크)는 이후.
- **(go/no-go) SR 효과 파일럿:** **자체 4인 영상 1개를 합성 열화**해서 SR 유무 → EAR/blink 정확도 델타 소규모 확인. 개선 없으면 대안(저조도 보정 Zero-DCE / 전용 소형 랜드마크). *이 파일럿 통과 전엔 SR 예산 확정 X.*
- **B. fps:** `TARGET_FPS_MIN=15`, `TARGET_FPS=24`. FPS 스윕으로 근거.
- **C. 평가셋(이번 라운드):** **자체 4인 테스트 영상**에 위 열화 표준 적용. (공개셋은 이후) 깨끗한 원본=참조 → 열화→SR off(정확도↓)→SR on(회복) 곡선.
- **D. 정답정합/eval_public(.tag):** 공개셋 안 쓰므로 **이후로 미룸**. 이번 정답 = 큐 기반.
- **E. w_eye 로깅:** 유지, **STAGE2 첫 작업**(게이트 임계값 데이터 근거).
- **F. blinks.csv:** SR 평가와 무관(legacy 분류기 피처). SR 평가 = 영상.
- **G. 하니스:** `run_live --log`(fps/지연/w_eye) + `fps_sweep.py`. 이번엔 **자체 영상 SR 평가 스크립트**(합성 열화→SR on/off) 추가. `eval_public.py`(.tag F1)는 이후.

---

## 3. 내일까지(당장) 할 일 — SR 중심

1. **SR 모델 확정 + 받기**: `cv2.dnn_superres` **FSRCNN-small(x2)** 1순위, ESPCN(x2) 대안.
   - .pb: FSRCNN https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models · ESPCN https://github.com/fannymonori/TF-ESPCN/tree/master/export
2. **robust.py 얼굴 ROI two-pass SR 최소 구현**(스펙 §5). 우선 "동작만" 목표.
3. (병행) **w_eye 로깅 추가**(run_live --log 컬럼).
4. (가능하면) **go/no-go 파일럿**: 깨끗한 클립 1개 → 합성 열화 → SR 유무 검출 비교(눈으로/수치로).

---

## 4. config 상수 (승인 반영, 코드 미적용)

```python
TARGET_FPS_MIN = 15
TARGET_FPS     = 24

SR_ENABLE      = True
SR_W_EYE_MIN   = 24      # 초안 — w_eye 분포 확인 후 확정
SR_MODEL       = "fsrcnn"   # cv2.dnn_superres: fsrcnn / espcn
SR_SCALE       = 2
SR_FACE_MARGIN = 0.3

# 합성 열화 표준 (이번엔 자체 4인 영상에 적용)
DEGRADE_DOWNSCALE   = (2, 3, 4)
DEGRADE_GAMMA       = (1.0, 0.6, 0.4)   # 낮을수록 어둡게
DEGRADE_NOISE_SIGMA = (0, 5, 10)
```

---

## 5. robust.py 얼굴 ROI two-pass SR — 구현 스펙 (내일 참고)

```
입력: 원본 프레임
1) FaceLandmarker(원본) -> 얼굴 bbox, 눈 좌표 -> w_eye(눈 가로 px)
2) if SR_ENABLE and w_eye < SR_W_EYE_MIN:
     face_crop = 원본에서 얼굴 bbox(여유 SR_FACE_MARGIN) 잘라내기
     face_hr   = dnn_superres.upsample(face_crop)     # FSRCNN x2
     res2 = FaceLandmarker(face_hr) -> EAR (스케일 불변, 역매핑 불필요)
     used_sr = True
   else:
     EAR = pass1 결과 사용;  used_sr = False
3) 로깅: w_eye, used_sr, (SR면) sr_ms
```
- 모델 로드는 1회(초기화). SR은 게이트가 켤 때만.
- 트리거 안 되면 기존 경로 그대로 → 실시간 유지.

---

## 6. 이번 라운드 실험 계획 (4인, 공개셋 없이)

- 피험자별: `calibrate`(개인 임계값·10회 검증) → **테스트 세션**(큐 기반 깜빡임 정답) → 인식 정확도.
- SR: 테스트 영상 **합성 열화** → SR on/off 정확도 + Pi fps/전력.
- 산출물: 개인별 임계값·정확도 표(캘리브 근거) + SR 효과 표 + fps/전력 표 + 파이프라인/STD 그림.

---

## 7. 이후 확장 (이번엔 안 함)

- 공개셋(Eyeblink8/mEBAL2 등) `.tag` F1 벤치(`eval_public.py`) + 모델 학습·비교.
- MediaPipe~판단 전체 자체 학습 모델(end-to-end).
- 완성도/불완전 깜빡임 지표.
