# 눈 깜빡임 감지 — 벡터 인코더 + 학습 판정 모델 (라즈베리파이5)

라즈베리파이5 위에서 동작하는 온디바이스 깜빡임 감지 파이프라인. 영상은 기기 밖으로 나가지 않는다(프라이버시 우선).

## 핵심 전환

기존의 원본 픽셀 · EAR(Eye Aspect Ratio) · 규칙 기반 판정 방식을 폐기하고, **벡터 인코더 + 학습된 판정 모델** 구조로 전환한다.

- 운용 경로: 카메라 → 눈 크롭 → 인코더 → 벡터 → 판정 모델
- 눈 원본 영상은 저장하지 않는다. 인코더가 만든 벡터만 다음 단계로 전달된다.
- 디코더는 배포하지 않는다. 초해상도(SR)·디코더는 로컬 검증용 도구로만 존재한다.
- 배포는 ONNX로 컴파일한 경량 모델. Pi5 실측 벤치마크가 필요하다.

## 데이터셋

눈만 크롭한 영상 + 깜빡임 여부(O/X) 레이블로 구성된 자체 데이터셋을 먼저 구축한다. 기존 공개 데이터셋(eyeblink8 등)은 `data/_legacy_public/`에 별도 보관하며, 크롭·재레이블링 시드로 재활용을 검토 중이다.

## 폴더 구조

```
data/            # 눈 크롭 데이터셋 + 레이블 (data/_legacy_public/ = 공개 데이터셋)
src/
  dataset/       # 크롭 전처리 규격(canonical) · 레이블링
  encoder/       # 벡터 인코더 (오토인코더)
  classifier/    # 벡터 → 깜빡임 여부 판정 모델
  eval/          # 이벤트 단위 평가 · EAR 베이스라인 · 프라이버시 지표
  experiments/   # 잠재 차원 스윕 등 실험 드라이버
  deploy/        # ONNX 변환 · MediaPipe 프론트엔드 · Pi5 벤치마크 하네스
  tools/         # 검증 전용 유틸 (로컬 한정, 배포 안 함)
docs/            # 배경·결과·의사결정 로그
results/         # 측정 원자료 (논문 표의 1차 증거)
paper/           # IEEE 논문 초안
```

## 상태 (2026-07-29)

- [x] 눈 크롭 + 레이블 데이터셋 구축 — Eyeblink8 8명, 양눈 64×160, 35,496 프레임
- [x] 벡터 인코더 구현 (`src/encoder/`) — 오토인코더, 128차원
- [x] 판정 모델 학습 (`src/classifier/`) — MLP
- [x] **ONNX 컴파일 + Pi5 실측 벤치마크** (`src/deploy/run_video.py`)
- [ ] 정식 train/val/**test** 분할 + 시드 반복 ← 다음
- [ ] 다신원 데이터 (**mEBAL2 180명, 약 1주 뒤 도착 예정**)
- [ ] 신원 disentanglement 정식 평가
- [ ] 눈 건강 지표 (빈도 · 불완전 깜빡임 · PERCLOS)

### 지금까지 나온 답

| | |
|---|---|
| **엣지** | ✅ Pi 5에서 e2e p99 **12.06 ms** (30fps 예산의 36%), 86 fps, 스로틀 없음. EAR 대비 +2.67 ms |
| **유용성** | ⚠️ EAR과 대등하나 **8명으로는 우열 판정 불가** (시드에 따라 70–73/74, EAR 70/74) |
| **프라이버시** | ❌ 차원을 16배 좁혀도 선형 재식별 99%+ → **차원 축소는 프라이버시가 아니다.** disentanglement 필수 |

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | 한 장 요약 · 현재 위치 |
| [`docs/PROJECT_DETAILED.md`](docs/PROJECT_DETAILED.md) | 배경·방법·데이터·모델·결과 전체 |
| [`docs/RESULTS.md`](docs/RESULTS.md) | 측정 결과 스냅샷 |
| [`docs/PI5_BENCHMARK.md`](docs/PI5_BENCHMARK.md) | Pi5 실측 결과와 **측정 조건의 근거** |
| [`docs/Pi_실행_가이드.md`](docs/Pi_실행_가이드.md) | Pi 환경 구축·실행 절차 (Python 3.13 함정 포함) |
| [`docs/STATUS_AND_DIRECTION_1.md`](docs/STATUS_AND_DIRECTION_1.md) | 로드맵·지도 의견·남은 고민 |

## 실행

리포지토리 루트에서 `python -m src.<하위>.<모듈>` 로 실행한다 (파일 직접 실행 금지).
데스크탑은 `requirements.txt`, 라즈베리파이는 **`requirements-pi.txt`** 를 쓴다.
