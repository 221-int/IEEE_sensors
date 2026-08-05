# Embedding-based Eye Blink Detection on an Edge Device

> 🔵 **2026-08-05 미팅으로 연구 범위가 축소되었습니다.**
> 이 논문(IEEE Sensors Letters)의 범위는 **임베딩 벡터 기반 눈 깜빡임 검출 + 에지 디바이스
> 실시간 성능 검증** 입니다. **신원 정보 제거 · 원본 복원 방지 · 디코더 공격 방어는
> 이번 논문 범위에서 제외**되고 특허 및 후속 연구로 분리됩니다.
> 근거와 변경 이력: [`docs/PROJECT_DIRECTION.md`](docs/PROJECT_DIRECTION.md) ·
> [`docs/CHANGELOG.md`](docs/CHANGELOG.md)

---

## 1. 프로젝트 개요

눈 원본 이미지를 판정 단계에서 직접 쓰지 않고, **encoder** 가 만든 **embedding vector** 만으로
**eye blink detection** 을 수행하는 온디바이스 시스템. 최종 형태는 **Raspberry Pi 5** 같은
**edge device** 위에서 실시간으로 돌아가는 경량 모델이다.

- 논문 타깃: **IEEE Sensors Letters** (약 4페이지). 초안 `paper/IEEE_Sensors_letters_overleaf.tex`
- 데이터셋: **mEBAL2 배포본 58명 중 57명** (U18 좌석 오염으로 제외), 유효 이벤트 **27,758**

## 2. 현재 연구 목표

| # | 목표 |
|---|---|
| 1 | 원본 눈 이미지 대신 encoder 가 만든 embedding vector 만으로 눈 깜빡임을 정확히 검출한다 |
| 2 | EAR 기반 방법과 동일 조건에서 비교해 embedding vector 방식이 더 안정적인지 검증한다 |
| 3 | 원본 이미지 기반 분류 모델과 정확도 · 추론 시간 · 지연 시간을 비교한다 |
| 4 | 해당 모델을 Raspberry Pi 5 에서 실시간(30 fps, e2e p99 ≤ 33.3 ms)으로 돌린다 |
| 5 | 판정 단계에서 원본 이미지를 쓰지 않는 **구조적** 장점을 기술한다 |

> ⚠️ **목표 5는 구조 서술까지다.** 신원 정보가 제거되었다거나 원본 복원이 불가능하다는
> 주장은 이번 논문에서 하지 않는다. 검증하지 않았기 때문이다.
> 상세: [`docs/PATENT_AND_FUTURE_WORK.md`](docs/PATENT_AND_FUTURE_WORK.md)

## 3. 전체 파이프라인

```
카메라 프레임
  → MediaPipe 얼굴/눈 랜드마크로 눈 위치 검출·추적
  → 양눈 크롭 (그레이스케일 64×160, MARGIN 2.2)
  → encoder (비대칭 CNN, vpres)
  → embedding vector (D차원, D 미확정)
  → 시간 헤드(19프레임 링버퍼) + 판정 헤드
  → blink / unblink
```

- 랜드마크 좌표는 크롭 좌표를 얻는 휘발성 용도로만 쓰고 버린다.
- 디코더는 배포하지 않는다. 개발 중 검증 도구로만 존재한다.
- 배포 아티팩트는 ONNX 로 컴파일한 encoder + 판정 헤드.

## 4. 핵심 차별점

**현재 확인된 사실** (`results/v2/train_encoder.json`, 5 fold × 3 seed = 15런)

| 방법 | PR-AUC |
|---|---|
| **ours** (encoder embedding vector, vpres D=16) | **0.9886 ± 0.0038** |
| ear_head (EAR 스칼라 → 동일 판정 헤드) | 0.9724 ± 0.0096 |
| ear_rule (규칙 기반 EAR) | 0.8931 ± 0.0136 |

차이 (ours − ear_head) = **+0.0151, 95% CI [+0.0106, +0.0203]** (짝지은 피험자 부트스트랩).

**연구 가설 (미검증)** — EAR 은 랜드마크 좌표에서 계산한 기하 비율 스칼라 하나지만,
embedding vector 는 눈 모양 · 눈꺼풀 상태 · 주변 밝기 · 눈 주변 질감 · 부분적 자세 변화 ·
눈이 감기는 과정의 시각적 특징을 더 폭넓게 담을 수 있다.
**이는 아직 검증되지 않았다** — 조건별 강건성 실험(Phase 4)이 이것을 검증하는 실험이다.
자세한 구분: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) §4

## 5. 현재 구현 상태

| 항목 | 상태 |
|---|---|
| mEBAL2 57명 크롭 추출·병합 (532,109장) | ✅ 완료, 검증 8/8 PASS |
| EAR 베이스라인 5-fold | ✅ 완료 — ROC-AUC 0.9091 ± 0.0166 |
| encoder + 판정 헤드 학습 (5 fold × 3 seed) | ✅ 완료 — 위 표 |
| 원본 이미지 기반 분류 모델 (대조군 2) | ❌ 미측정 |
| random projection · 평균 벡터 대조군 (3·4) | ❌ 미측정 |
| 조건별 강건성 실험 (조도·각도·거리·안경) | ⚠️ 안경·배치만 부분 완료. 나머지는 **추가 데이터 수집 필요** |
| 모델 파라미터 수 · 파일 크기 · 메모리 측정 | ❌ v2 인코더 기준 미측정 |
| Raspberry Pi 5 실측 | ⚠️ **v1 인코더로만 측정됨** (e2e p99 12.06 ms, 86.07 fps). v2 인코더로 **재측정 필요** |
| ONNX export 코드 | ⚠️ v1 경로(`src/deploy/export_onnx.py`)만 존재. v2 경로 미작성 |

세부 상태와 검증 여부: [`docs/PROGRESS.md`](docs/PROGRESS.md) · [`docs/TASKS.md`](docs/TASKS.md)

## 6. 실행 방법

리포지토리 루트에서 모듈로 실행한다 (파일 직접 실행 금지 — 상호 임포트가 깨진다).

```bash
# v2 (현재 라운드) — mEBAL2 57명
python -m src.v2.ear_baseline_folds        # EAR 베이스라인 5-fold
python -m src.v2.train_encoder             # encoder + 판정 헤드 학습 (5fold×3seed, 약 4.5h)
python -m src.v2.posthoc_subgroups         # 배치·안경 서브그룹 분석
```

- 데스크탑: `requirements.txt` (Python 3.12, CUDA)
- Raspberry Pi 5: `requirements-pi.txt` (Python **3.11** 필수 — mediapipe 에 aarch64/3.13 휠 없음)
- Pi 실행 절차: [`docs/Pi_실행_가이드.md`](docs/Pi_실행_가이드.md)

## 7. 주요 파일 구조

```
IEEE_sensors/
├─ README.md                     이 문서
├─ src/v2/                       ★ 현재 라운드 코드 (python -m src.v2.<모듈>)
│  ├─ common/                    격자·시드·분할·프로브 — 규칙의 단일 구현
│  ├─ dataset/                   mEBAL2 로더 · 크롭
│  ├─ model/encoder.py           encoder 구조 (vpres) + MMAC 분석
│  ├─ ear_baseline_folds.py      EAR 베이스라인
│  ├─ train_encoder.py           encoder + 판정 헤드 학습
│  └─ phase5_reid.py             재식별 프로브 (▶ 논문 범위 밖, 특허·후속 연구용)
├─ src/                          v1 코드 (Eyeblink8 라운드). v2 에서 재사용 안 함
│  └─ deploy/                    ONNX export · Pi 벤치 하네스 — **v2 로 이식 대상**
├─ data/processed/v2/            크롭 샤드 + 인덱스
├─ results/v2/                   ★ v2 측정 원자료 (논문 표의 1차 증거)
├─ models/                       v1 모델 아티팩트
├─ paper/                        IEEE Sensors Letters 초안
└─ docs/
   ├─ PROJECT_DIRECTION.md       ★ 방향 변경의 근거와 현재/제외 범위
   ├─ T3_WORKORDER.md            ★ Phase 3 실행 순서·명령 (Claude Code 용 프롬프트 포함)
   ├─ CLEANUP.md                 파일 정리 감사 + 실행 스크립트
   ├─ RESEARCH_PLAN.md           연구 질문 · 가설 · 기여점
   ├─ EXPERIMENT_PLAN.md         비교 대상 · 조건 · 지표 · Pi 실험 계획
   ├─ TASKS.md                   Phase 1~7 세부 작업 체크리스트
   ├─ PROGRESS.md                진행 현황과 기존 작업 재분류
   ├─ CHANGELOG.md               방향 변경 이력
   ├─ PAPER_OUTLINE.md           논문 6장 구성 초안
   ├─ PATENT_AND_FUTURE_WORK.md  ▶ 분리된 프라이버시·신원 항목
   ├─ PROJECT.md                 v1 통합 기록 (인용 주의)
   ├─ PI5_BENCHMARK.md           Pi5 측정 조건의 근거 (v1 수치)
   ├─ Pi_실행_가이드.md          Pi 환경 구축·실행 절차
   └─ v2/PROTOCOL.md             측정 규칙(격자·시드·분할·chance)의 단일 기준
```

## 8. 앞으로 해야 할 작업

> 🎯 **목표 일정: 특허 출원 · 논문 투고 둘 다 8월 말.** 최소 경로는
> [`docs/TASKS.md`](docs/TASKS.md) §10 에 있다.

바로 진행할 순서 (상세는 [`docs/TASKS.md`](docs/TASKS.md)):

1. **대조군 — 원본 이미지 기반 분류 모델(image_cnn).** 이게 없으면 "임베딩이 이미지보다
   낫다/비슷하다"를 말할 수 없다. `free` / `matched` 두 변형. **가장 먼저 코드를 짜서 런을 건다** (약 9시간).
2. **경량화 지표 측정** — v2 encoder 의 파라미터 수 · 모델 크기 · MMAC · 추론 시간.
3. **Raspberry Pi 5 재측정** — v2 encoder + ONNX, **640×480 주 / 1280×720 보조**.
   v1 수치는 인용 불가.
4. **관련 연구 재조사** — 에지/경량 깜빡임 검출, encoder 기반 접근, 일반적 실패 조건.
5. **조건별 강건성 실험** — 현 데이터로 가능한 축(광학·안경·배치·얼굴 위치)부터.
6. **논문 그림·표 구성** — [`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md)

> 합성 열화 실험은 **P2 로 하향**했다 (논문 작성 시점에 재판단).
> `eye_dataset/` 은 **삭제 예정**이다 — 단 `src/dataset/capture_eye_dataset.py` 코드는
> `compute_ear` 때문에 남긴다.

## 용어 규약

문서 전체에서 다음 용어만 쓴다.

| 용어 | 표기 |
|---|---|
| embedding vector | 임베딩 벡터 / embedding vector (혼용 가능, "잠재 벡터"·"latent"·"코드" 금지) |
| encoder | 인코더 / encoder |
| eye blink detection | 눈 깜빡임 검출 / eye blink detection ("깜빡임 감지" 금지) |
| EAR | EAR (Eye Aspect Ratio). 첫 등장 시 풀네임 병기 |
| edge device | 에지 디바이스 / edge device ("엣지" 표기 금지) |
| Raspberry Pi | 라즈베리파이 5 / Raspberry Pi 5 / Pi 5 |
