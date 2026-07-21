# STAGE 2 설계노트 — 조건부 얼굴 ROI two-pass SR (파이프라인 계층 배치)

작성일: 2026-07-20 · 범위: **설계 + 스캐폴딩만**(실제 구현·Pi 실측은 다음 라운드 리뷰 후)
승인 근거: `참고자료/STAGE2_결정사항.md`(A·B·C), 전단계: `참고자료/SR_STAGE1_분석.md`

---

## 1. 결정 요약 (무엇을 왜)

- **아키텍처(A):** 얼굴 ROI two-pass SR. 눈 40×24 패치 SR은 MediaPipe FaceLandmarker와 물리지 않음(STAGE1 F2) → 얼굴 crop을 SR 대상으로.
- **배치(이번 세션 확정):** two-pass 오케스트레이션은 **파이프라인 계층**(landmarker 핸들을 쥔 새 모듈 `frontend.py`)에 둔다. `robust.py`는 **순수 SR 유틸**(frame→frame)로 남겨 관심사를 분리.
  - 이유: `robust.Preprocessor`의 `frame→frame` 규약을 깨지 않는다. two-pass는 pass1 랜드마크가 있어야 게이트·crop이 가능한데, 현재 훅 주입점(run_live L84 `pre(frame)`)은 랜드마크보다 앞이라 훅 자리에선 불가능. 그래서 랜드마커를 가진 계층으로 올린다.
- **예산(B):** no-SR ~30fps / 목표 24fps / 하한 15fps(필수).
- **평가(C, 이번 라운드):** 공개셋 `.tag` 미사용. 자체 4인 영상 + 합성 열화(다운스케일·감마·노이즈)로 SR on/off 비교.

---

## 2. 모듈 책임 맵

| 모듈 | 책임 | 이번 세션 변경 |
|---|---|---|
| `robust.py` | **순수 SR 유틸.** `SuperResolution`(cv2.dnn_superres, frame→frame, 지연로드) + `resolve_sr_model_path`. 게이트·crop·pass2 **없음**. | SR 스텁 → dnn_superres 래퍼 스캐폴딩(가드 유지) |
| `frontend.py` (신규) | **파이프라인 계층 오케스트레이션.** `EarFrontend.process(frame,ts)→FrameResult`. pass1→게이트(`SrGate`)→얼굴 crop SR→pass2. 텔레메트리(w_eye/used_sr/lm_ms/sr_ms). | 신규. 게이트·dataclass는 실제, 프레임 SR 경로는 ★1 전까지 폴백 |
| `landmarks.py` | 랜드마크·EAR. **신규 순수 헬퍼** `eye_width_px`(게이트 신호 w_eye), `face_bbox`(crop 영역). | 헬퍼 2개 실제 구현 |
| `config.py` | 상수. STAGE2 §4 상수 추가. | `TARGET_FPS*`, `SR_*`, `DEGRADE_*` |
| `scripts/sr_eval.py` (신규) | 합성 열화→SR on/off 평가 하니스. 열화 함수 실제, 스윕 골격. | 신규 |
| `scripts/run_live.py`, `fps_sweep.py` | 실시간/스윕. **이번 세션 미변경** — 통합은 다음 라운드(§5). | — |

---

## 3. 데이터 흐름 (frontend.EarFrontend.process)

```
frame ─▶ pass1: FaceLandmarker(원본) ─┬─ 얼굴 없음 ─▶ FrameResult(ear=None, has_face=False)
                                      │ 얼굴 있음
                                      ▼
                          w_eye = eye_width_px(lms)         # 양안 눈꼬리 대각거리 평균(px)
                                      ▼
                     SrGate.decide(w_eye)  (히스테리시스: on<24 / off>=30)
                          │ False                     │ True (and sr!=None)
                          ▼                           ▼
                 EAR = frame_ear(pass1)      crop = frame[face_bbox(margin=0.3)]
                 used_sr=False               crop_hr = SR.upsample(crop)     # robust
                                             pass2: FaceLandmarker(crop_hr)
                                             EAR = frame_ear(pass2) | 폴백(pass1)
                                             used_sr=True
                          └───────────────┬────────────┘
                                          ▼
                        FrameResult(ear, has_face, w_eye, used_sr, lm_ms, sr_ms)
```

EAR은 정규화좌표 비율 → **스케일 불변**(STAGE1 F3)이라 pass2 좌표계에서 계산해도 역매핑 불필요. crop 좌표 offset/scale 역매핑은 EAR만 쓰는 한 필요 없음(확인).

---

## 4. 반드시 해결할 미결 (★ — 다음 라운드 진입 전)

- **★1 VIDEO 모드 타임스탬프 단조성.** `detect_for_video`는 단조 증가 ts 필요. pass1/pass2를 같은 ts로 부르면 규약 위반. 후보: (a) pass2용 **IMAGE 모드** 별도 landmarker, (b) pass2용 인스턴스 분리, (c) ts+1ms(임시). → 결정 후 `EarFrontend._detect`/SR 경로 활성화. **현재 SR 경로는 `NotImplementedError`로 막아둠**(조용한 오동작 방지).
- **★2 MediaPipe 내부 리사이즈.** FaceLandmarker가 얼굴 crop을 자체 입력크기로 리사이즈하면 업스케일 이득이 상쇄될 수 있음(STAGE1 열린질문2). **go/no-go 파일럿에서 최우선 확인** — 상쇄되면 SR 자체를 재검토.
- **★3 게이트 발동 조건과 열화 방식 정합.** 합성 열화에서 "블러만(같은 크기)"이면 w_eye가 안 줄어 게이트가 안 켜짐. 그래서 `sr_eval.degrade_downscale`은 프레임을 **실제 축소**해 눈 px를 줄인다(게이트 발동). 이 전제를 실험 설계와 합의할 것.

---

## 5. 통합 체크리스트 (다음 라운드)

1. **★1 결정** → `frontend.EarFrontend`의 SR 경로 주석 해제·활성화.
2. **run_live.py 통합**: L84 `pre(frame)` 방식 제거 → `EarFrontend`로 프레임→EAR 일원화. `--log`에 `w_eye, used_sr, sr_ms` 컬럼 추가.
3. **fps_sweep.py 통합**: `extract_ears`를 `EarFrontend` 경유로 교체(중복 제거).
4. **모델 확보**: FSRCNN-small_x2.pb / ESPCN_x2.pb → `models/`. 파일명 규약은 `robust.resolve_sr_model_path` 참조.
5. **go/no-go 파일럿**(★2): 자체 클립 1개 열화 → SR 유무 EAR/blink 델타 소규모 확인. 통과 후에야 SR 예산 확정.
6. **Pi 실측**: `run_live --log`+`fps_sweep`로 SR-on 지속 fps ≥ 24 / 하한 15 확인, CPU%/RSS/온도.
7. **sr_eval 완성**: 큐 기반 truth 정합 → 프레임단위 F1/정밀도/재현율, off vs on 회복곡선.

---

## 6. 이번 세션 산출물 (파일)

- `참고자료/SR_STAGE2_설계노트.md` (본 문서)
- `src/eyeblink/frontend.py` (신규 스캐폴딩: `EarFrontend`, `SrGate`, `FrameResult`)
- `src/eyeblink/robust.py` (SR 래퍼 스캐폴딩 + `resolve_sr_model_path`)
- `src/eyeblink/landmarks.py` (+`eye_width_px`, +`face_bbox` — 실제 구현)
- `src/eyeblink/config.py` (+STAGE2 상수)
- `src/scripts/sr_eval.py` (신규 스캐폴딩: 열화 함수 실제 + 스윕 골격)

**미변경(의도적):** `run_live.py`, `fps_sweep.py`, `pipeline.py`, `std.py` — 통합은 ★1 결정 후.

---
*스캐폴딩 원칙: 순수·테스트가능 부분(게이트/헬퍼/열화)은 실제 구현, MediaPipe·Pi 실측·모델에 의존하는 부분은 명확한 `NotImplementedError`/`TODO`로 표시해 조용한 오동작을 배제했다.*
