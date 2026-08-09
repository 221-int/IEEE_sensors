> ▶ **2026-08-08 아카이브로 이동.** v1 시절 내용과 **폐기된 프라이버시 목표**가
> 섞여 있다. 현재 상태는 [`docs/STATUS_2026-08-08.md`](../STATUS_2026-08-08.md) 가 기준이다.
> 이 문서는 **연구 기록 보존용**이며 논문·결정의 근거로 인용하지 않는다.

# PROGRESS — 진행 현황과 기존 작업 재분류

> **작성 2026-08-05.** 방향 변경(`PROJECT_DIRECTION.md`) 이후, 지금까지 진행된 작업을
> **삭제하지 않고** 재분류한다. 상태는 실제 파일과 결과 JSON 에서 확인된 것만 적었다.
> 작업 목록은 `TASKS.md`, 실험 설계는 `EXPERIMENT_PLAN.md`.

## 🔵 최신 상태 정정 — 2026-08-08

아래 내용이 이 문서의 기존 진행표보다 우선한다. 논문 본문·표의 문구는 이 갱신에서
수정하지 않고, 연구 산출물과 작업 상태만 정리한다.

| 항목 | 최신 상태 | 근거 |
|---|---|---|
| 확정 인코더 | **vpres / D=16 유지**. `vdrop`은 Pi 실측에서 G-E1 실패 시 대체 후보 | `results/v2/cmp_vpres_vs_vdrop.json`, `results/v2/cmp_D16_vs_D8.json`, `cmp_D16_vs_D32.json`, `cmp_D16_vs_D64.json` |
| T3-1 image_cnn 비교 | 탐색 비교 완료 + `image_cnn_head`/`image_cnn_max` 5-fold×3-seed 확정 런 완료 | `results/v2/train_image_cnn_head_final.json`, `train_image_cnn_max_final.json` |
| T3-6 확정 재실행 | 완료. ours 결과 15/15 비트 동일 재현, Recall/F1·혼동행렬·ear_head 가중치 저장 | `results/v2/train_encoder_final.json` |
| 서브그룹 | 이벤트 가중 풀링 + ear_head 상대 δ 판정 완료 | `results/v2/posthoc_subgroups_final.json` |
| ONNX/배포 경로 | ours·image_cnn 두 변형 export 및 동치 게이트 완료 | `results/v2/export_onnx.json`, `results/v2/check_equivalence.json`, `src/v2/deploy/` |
| Pi 5 실측 | **완료**. 4모드·2해상도 측정 완료. G-E1 결과와 세부 수치는 논문 문서의 Table II-b에 기록됨 | `docs/PAPER_OUTLINE.md` Table II-b, `docs/PI_RUNBOOK.md` |

### 남은 작업의 정확한 범위

- 실제 Raspberry Pi 5에서 `ours`, `ear`, `image_cnn_max`, `image_cnn_head`를 480p/720p로 측정하는 작업은 완료됐다.
- 640×480에서 e2e p99 ≤ 33.3 ms를 통과하면 `vpres`를 유지한다. 실패할 때만 `vdrop`으로 전환한다.
- 측정 결과에 따라 현재 배포 구조는 `vpres`로 유지한다. `vdrop`은 대체 후보로만 남긴다.

---

## 0. 🔵 전사문 지시사항 대비 진행률 (2026-08-05 밤 기준)

미팅에서 박사님이 하라고 하신 것을 그대로 나열하고 현재 상태를 붙인다.
결정 번호는 `PROJECT_DIRECTION.md` §3-1 과 같다.

| # | 전사문 지시 | 시각 | 상태 | 근거 |
|---|---|---|---|---|
| D1 | 논문은 **눈 깜빡임 검출 하나로 끝낸다** | 12:22 | ✅ **완료** | 문서 9종 재작성 |
| D2 | 원본 이미지 대신 **embedding vector 로 깜빡임 측정** | 02:04 | ✅ **완료** | `train_encoder.json` PR-AUC 0.9886 ± 0.0038 |
| D3 | 신원 억제·복원 방지는 **논문에 합치지 않는다** | 02:04, 08:47 | ✅ **완료** | `PATENT_AND_FUTURE_WORK.md` 분리 |
| D6 | **EAR 과 비교**한다 | 04:04~05:04 | ✅ **완료** | ear_rule 0.8931 / ear_head 0.9724 / ours 0.9886 |
| D8 | 깜빡임 검출의 **일반적 문제를 정리**한다 | 10:29 | ✅ **조사 완료** | `RELATED_WORK.md` §A2-5 (F1~F7) |
| D9 | **관련 연구 조사** — 에지 수준 / encoder 기반 | 10:52, 06:11 | ✅ **1차 완료** | `RELATED_WORK.md` §A2. ⚠️ 전부 [검색요약], 원문 확인 필요 |
| D7 | **다양한 조건**(어두운 상태 등)에서 비교 실험 | 06:27 | 🔄 **부분** | 안경·배치만 측정. 조도는 값만 있고 **층화 분석 미실시** |
| D11 | 신원은 **특허로**, 특허를 논문보다 먼저 | 15:42, 16:56 | 🔄 **문서만** | 분리 완료. **특허 기술 내용 정리 미착수** |
| D12 | 순서 = 특허 출원 → 논문 투고 | 16:20, 17:53 | 🔄 **일정만** | 둘 다 8월 말 목표 확정. 실무 미착수 |
| **D4** | 🔴 **이미지 기반 방법과 성능 비교를 전면에** | **03:02** | ✅ **완료** | image_cnn 탐색·확정 런 완료 |
| **D5** | 🔴 비교 축 = 정확도 · **지연 시간 · FPS · 에지** | **03:11** | ✅ **완료** | 정적 비용·ONNX·Pi 단계별 측정 완료 |
| **D10** | 🔴 결론 = **벡터 뽑아 경량화해 Pi 에서 잘 돌아간다** | **08:56** | ✅ **완료** | v2 배포 경로·Pi 5 실측·G-E1 판정 완료 |

### 🔴 정직한 요약

**✅ 로 표시된 것은 대부분 (a) 미팅 이전에 이미 끝나 있던 것 이거나 (b) 문서 작업이다.**

**박사님이 "내세우는 거"라고 명시적으로 말한 세 가지(D4·D5·D10)가 정확히 가장 덜 됐다.**

> "내세우는 거는 이미지 기반으로 하는 애들하고 성능 비교를 해요." (03:02)
> "지연 시간이 조금 더 많을 수도 있고 초당 몇 프레임을 처리할 수 있는지 이런 거에 차이가
> 있을 수도 있잖아요. 그리고 Edge 수준에서 돌릴 거니까 모델 돌리는 거랑 정확도랑
> 이런 부분에서 비교를 해봤을 때…" (03:11)
> "벡터 뽑는 거 해서 경량화해서 라즈베리파이에서 잘 돌아간다, 이거를 하나를 센서스로." (08:56)

이 세 개가 논문의 **본체**다. 나머지는 이미 있던 것이거나 준비 작업이다.

### 판정 기준도 기록해 둔다 (03:11)

> "이렇게 해도 성능이 이미지를 그대로 쓰는 것보다 **오히려 더 좋다** — 그럼 제일 베스트고,
> 아니면 **성능 차이가 별로 없다** — 이러면 그것도 오히려 사용자 입장에서는 이게 더 낫다."

→ **image_cnn 을 이길 필요는 없다.** "차이가 없다"도 합격이다.
단 `PROTOCOL.md` §9-1 대로 **"차이가 유의하지 않다"는 동등의 근거가 아니다** —
비열등 마진 δ=0.02 로 판정해야 한다.

### 지금 무엇이 돌고 있나

- **GPU**: T3-8(구조 절제) + T3-5(D 스윕) 36런 실행 중, 11.7~15h 예상 → **D2 의 보강**
- **사람**: 문헌 조사 완료 → **D8·D9 를 오늘 닫았고, D4 의 설계 근거를 확보**

---

## 1. 완료된 작업 (검증까지 끝남)

| # | 작업 | 산출물 | 검증 |
|---|---|---|---|
| 1 | **코드 규율 정비** — 임계값 격자·시드·분할·프로브를 `src/v2/common/` 하나로 통합 | `src/v2/common/` | Phase −1 게이트 **9/9 PASS** (G-1i 학습 결정성 포함) |
| 2 | **mEBAL2 배포본 실측 조사** | `docs/mEBAL2_실측_2026-07-31.md` | 배포본 = mEBAL1 38명 + mEBAL2 신규 20명 = 58명 |
| 3 | **58명 이벤트 측정** | `results/v2/phase1_csv58.json` | 28,728 이벤트 |
| 4 | **크롭 규격 확정** — 64×160, MARGIN 2.2, 프레임 표준화, 결측 ≤5 | PROTOCOL §7·§8·§8-bis | `results/v2/crop_margin_check.json` |
| 5 | **57명 크롭 추출·병합** — 532,109장 | `data/processed/v2/` | `results/v2/merge_report.json` **8/8 PASS** |
| 6 | **데이터 오염 조사·정리** — U18 좌석 오염 제외, 좌석 오염 이벤트 392개 제외 | `results/v2/apply_flags_report.json` | 유효 이벤트 **27,758** |
| 7 | **안경 라벨링** — 착용 17 / 미착용 40 (57명) | `results/v2/glasses_label_revision.json` | 육안 확인 58장 |
| 8 | **광학·얼굴 위치 사전 측정** | `results/v2/photometrics_58.json`, `face_position.json` | — |
| 9 | **Phase 0 사전 프로브** — 랜덤 인코더 / 픽셀 천장 / 광학 바닥 | `results/v2/phase0_probes.json` | 랜덤 0.6329 / 픽셀 0.9099 / 광학 0.2018 |
| 10 | **EAR 베이스라인 5-fold** | `results/v2/ear_baseline_folds.json` | drop_ratio ROC-AUC **0.9091 ± 0.0166**, PR-AUC 0.8825 ± 0.0203 |
| 11 | **δ = 0.02 · 주 지표 PR-AUC 확정** | PROTOCOL §9-1 | 우리 결과 0건인 시점에 고정 🔒 |
| 12 | **encoder + 판정 헤드 학습** — 5 fold × 3 seed = 15런, 266분 | `results/v2/train_encoder.json` | deterministic ✅ sealed ✅. ours **0.9886 ± 0.0038** |
| 13 | **EAR 대비 판정** | 같은 파일 | 차이 **+0.0151 [+0.0106, +0.0203]** → 비열등 충족 |
| 14 | **대조군 8 (세션 시각 t_rel 교락)** | `results/v2/control8_trel.json` | 교락으로 우위를 설명할 수 없음 |
| 15 | **U1 원인 규명** | `results/v2/u1_audit.json`, `diagnose_subject.json` | **고유한 전이 실패** (train−test 격차 0.1798, 57명 최대) |
| 16 | **서브그룹 풀링 (ours vs ear_rule)** | `results/v2/restored_ours_scores.npz` | 15런 중 12런 정확 일치, 3런 ≤5e-6 |

---

## 2. 부분적으로 완료된 작업

| # | 작업 | 무엇이 됐나 | 무엇이 안 됐나 |
|---|---|---|---|
| 1 | **서브그룹 분석** | 배치(2020/2022) · 안경(17/40) 축 측정 | **완료** — ear_head 상대 이벤트 가중 풀링 δ 판정까지 재실행함 |
| 2 | **관련 연구 조사** | `docs/v2/RELATED_WORK.md` — 프라이버시 축 §B·§C·§E 충실 | **새 방향에 필요한 축이 비어 있다**: 에지/경량 깜빡임 검출(1건뿐), encoder 기반 깜빡임 검출(미확인), 깜빡임 검출의 일반적 실패 조건(미조사) |
| 3 | **Raspberry Pi 5 실측** | v2 네 모드·두 해상도 실측 완료 | G-E1 판정·단계별 지연·FPS·RSS·CPU·온도·스로틀 기록 완료 |
| 4 | **ONNX 배포 경로** | v2 export·동치 게이트·실행 하네스 완료 | `src/v2/deploy/`, `results/v2/export_onnx.json`, `check_equivalence.json` |
| 5 | **embedding vector 차원 D** | **vpres / D=16 확정** | D8·D32·D64 비교 및 확정 런 재현 완료 |
| 6 | **랩미팅 슬라이드** | `docs/랩미팅_2026-08-05_슬라이드.md` v2 결과로 전면 재작성 완료 | **2026-08-05 방향 변경 반영 안 됨** (미팅 당일 자료) |

---

## 3. 검증이 필요한 작업 (코드는 있으나 v2 조건에서 확인 안 됨)

| # | 항목 | 파일 | 왜 검증이 필요한가 |
|---|---|---|---|
| 1 | 시간 헤드 + 판정 헤드 실제 구현 | `src/v2/train_encoder.py` | v2 확정 런·파라미터 기록 완료 |
| 2 | ONNX export 경로 | `src/v2/deploy/export_onnx.py` | v2 encoder·image_cnn export 및 동치 게이트 완료 |
| 3 | MediaPipe 프론트엔드 | `src/v2/deploy/` | v2 크롭 규격과 실행 하네스 검증 완료 |
| 4 | Pi 실행 하네스 | `src/v2/deploy/run_video.py` | 네 모드 실행 및 실제 Pi 5 측정 완료 |
| 5 | ~~자체 웹캠 데이터셋~~ | `eye_dataset/` (4,524프레임, 1명) | 🔵 **2026-08-05: 삭제 결정.** 개인 얼굴 데이터 + n=1. 사용자가 직접 삭제. ⚠️ **코드 `src/dataset/capture_eye_dataset.py` 는 남긴다** — `compute_ear` 를 `ear_baseline.py`·`run_video.py` 가 import 한다 |
| 6 | git 추적 상태 | `src/v2/` | **대부분 미추적**이라 결과 JSON 의 `git_commit` 이 실행 코드를 지목하지 못한다. **새 실행 전에 커밋 필요** |

---

## 4. 새로운 방향에서도 유지되는 작업

전사문 §7 기준으로 "활용 가능성이 높은 것" 부터 확인한 결과.

| # | 작업 | 새 방향에서의 위치 | 수정 필요 여부 |
|---|---|---|---|
| 1 | **눈 영역 검출 및 추적** | 파이프라인 1단. 그대로 사용 | 없음 |
| 2 | **프레임별 눈 크롭** | 파이프라인 2단. 그대로 사용 | 없음 |
| 3 | **크롭 이미지 크기 통일** (64×160, MARGIN 2.2) | 확정값 🔒 유지 | 없음 |
| 4 | **encoder 학습** | 파이프라인 3단. **논문의 핵심** | 목적함수에서 억제 항 계획을 뺀다 (원래 안 들어 있었음) |
| 5 | **embedding vector 추출** | 파이프라인 4단. **논문의 핵심** | 없음 |
| 6 | **MLP/시간 헤드 기반 눈 깜빡임 판정** | 파이프라인 5단. **논문의 핵심** | 없음 |
| 7 | **EAR 베이스라인 + 비열등 판정 규칙** | 주 비교 대상으로 **위상 상승** | 없음. δ=0.02 유지 |
| 8 | **5-fold 분할 · 시드 · 격자 · 부트스트랩** | 모든 비교의 공통 조건 | 없음 |
| 9 | **서브그룹 분석 코드** | 강건성 실험(Phase 4)의 기반 | 광학·위치 층화 축 **추가 필요** |
| 10 | **실시간 영상 처리 하네스** | 에지 실험의 기반 | v2 로 **이식 필요** |
| 11 | **Raspberry Pi 실행 코드 + 측정 조건 근거** | 논문의 두 번째 기여축 | v2 로 **이식 + 재측정 필요** |
| 12 | **기존 정확도 측정 결과** (v2) | 논문 Table I 의 3개 행 | 없음 |
| 13 | **기존 속도 측정 결과** (v1 Pi) | ⚠️ **논문 인용 불가.** 측정 조건 근거(`PI5_BENCHMARK.md`)만 재사용 | 재측정 필요 |
| 14 | **U1 실패 사례 분석** | 논문 한계 서술 / Fig. 3 | 없음 |
| 15 | **프로젝트 사고 기록 11가지** | 실험 위생 규칙 | 없음 |

---

## 5. 방향 변경으로 보류된 작업

> **삭제하지 않는다.** `PATENT_AND_FUTURE_WORK.md` 의 1차 증거다.

| # | 작업 | 상태 | 왜 보류 |
|---|---|---|---|
| 1 | **Phase 5 재식별 프로브** (`results/v2/phase5_reid.json`) | ✅ **측정 완료** — 57-way 0.6345 ± 0.0288, 랜덤 인코더 0.6261 과 동급, 픽셀 천장 0.9080, 광학 바닥 0.0825, EAR 0.0548, 집계 N=100 → 1.0000 | 논문 범위 밖. **특허·후속 연구의 기준선으로 그대로 유효** |
| 2 | **Phase 5-bis 복원 장벽** (PROTOCOL §14) | ❌ 미착수. `src/v2/phase5b_recon.py` 미작성 | 논문 범위 밖 |
| 3 | **Phase 7 신원 억제 + 파레토** (PROTOCOL §15) | ❌ 미착수. `src/v2/phase7_pareto.py` 미작성 | 논문 범위 밖 |
| 4 | **게이트 G-R1 · G-R2 · G-P1 · G-P2 · G-P3** | 미적용 | 논문 범위 밖 |
| 5 | **encoder 목적함수에 억제 항 추가** | 미착수 | 논문 범위 밖. D 는 **깜빡임 성능 + edge 비용**으로 확정 가능해짐 |
| 6 | **눈 건강 지표** (PERCLOS · 불완전 깜빡임 비율) | 미착수 | mEBAL2 에 지속시간 라벨 없음. 방향 변경 이전부터 제외 |
| 7 | **조건부 초해상도(SR)** | 🚫 **폐기** | baseline 이 6×까지 무손실이라 SR 발동 구간이 없고, 8× 에서는 검출을 오히려 해쳤다 (29→25) |

---

## 6. 재사용 가능한 기존 코드 및 결과

### 6-1. 그대로 재사용 (수정 불필요)

| 파일 | 역할 |
|---|---|
| `src/v2/common/thresholds.py` | 임계값 격자 — 유일 구현 |
| `src/v2/common/splits.py` | fold 생성/검증 — `force=True` 없이 덮어쓰지 않음 |
| `src/v2/common/stats.py` | 짝지은 피험자 클러스터 부트스트랩 |
| `src/v2/common/repro.py` | 시드·결정성 |
| `src/v2/common/probes.py` | 프로브 (선형·MLP, `return_scores`) |
| `src/v2/common/folds_5fold.json` | 얼린 fold 배정 (57명, 이중 층화) 🔒 |
| `src/v2/dataset/mebal2.py`, `crop.py` | 로더·크롭 |
| `src/v2/model/encoder.py` | encoder 구조 + MMAC 분석 (`analyse()`) |
| `src/v2/ear_baseline_folds.py` | EAR 베이스라인 |
| `data/processed/v2/` | 크롭 샤드 532,109장 + 인덱스 |

### 6-2. 일부 수정 후 재사용

| 파일 | 필요한 수정 |
|---|---|
| `src/v2/train_encoder.py` | `--encoder {ours,image_cnn,random_proj,mean,random_init}` 분기 추가 (T3-1~T3-4), D 스윕 옵션 (T3-5) |
| `src/v2/posthoc_subgroups.py` | 광학 3분위 · 얼굴 위치 층화 축 추가 (T4-1, T4-2) |
| `src/deploy/export_onnx.py` | → `src/v2/deploy/export_onnx.py` 로 v2 규칙에 맞춰 **새로 작성** |
| `src/deploy/run_video.py` | → `src/v2/deploy/run_video.py`, `--mode` 에 `image_cnn` 추가 |
| `src/deploy/frontend.py` | v2 크롭 규격(MARGIN 2.2, 64×160)과 일치 확인 후 이식 |
| `docs/v2/RELATED_WORK.md` | §A 확장, §B·§C·§E 를 ▶ 특허/후속 연구용으로 표시 |

### 6-3. 비교 실험에 활용

| 자산 | 용도 |
|---|---|
| `results/v2/ear_baseline_folds.json` | Table I 의 ear_rule 행 |
| `results/v2/train_encoder.json` | Table I 의 ear_head · ours 행 |
| `results/v2/phase0_probes.json` | 픽셀 천장 / 광학 바닥 — 강건성 해석의 참조점 |
| `results/v2/control8_trel.json` | 교락 반론 방어 |
| `results/v2/u1_audit.json` | Fig. 3 실패 사례 |
| `results/v2/train_encoder_contaminated.json` | ⚠️ **인용 금지.** 오염 전후 비교용 (fold0: ours 0.9902 / ear_head 0.9608) |

### 6-4. 향후 연구 · 특허에 활용

| 자산 | 용도 |
|---|---|
| `results/v2/phase5_reid.json` | 억제 기전의 **기준선**. 이 값에서 얼마나 내려가는지가 후속 논문의 주장 |
| `src/v2/phase5_reid.py` | 후속 연구에서 그대로 사용 |
| `src/encoder/train_disentangled.py`, `models/disentangled/` (v1) | GRL 설계 참고. ⚠️ 한쪽눈 64×64 예비라 **수치 인용 금지** |
| `src/experiments/recon_attack.py`, `results/recon_attack.json` (v1) | 복원 공격 설계 참고. ⚠️ 8명 v1 |
| `src/experiments/identity_probe.py`, `results/identity_probe.json` (v1) | 신원 프로브 설계 참고. ⚠️ 8명 v1 |
| `docs/v2/RELATED_WORK.md` §B·§C·§E | 특허 명세서 배경기술 / 후속 논문 Related Work |

### 6-5. 현재는 불필요하여 보류

| 자산 | 사유 |
|---|---|
| `models/sweep/` (차원 스윕 인코더 15개, 139 MB) | v1 8명·한쪽눈. T3-5 에서 v2 로 재실행 |
| `results/dim_sweep.json` | 동일 |
| `models/privacy_metrics.json` | ⚠️ **날짜 없는 구버전.** 재식별 0.9978 / PSNR 32.40. 출처 불명이라 인용 금지 |
| ~~`eye_dataset/`~~ | 🔵 **삭제 결정 (2026-08-05).** 개인 얼굴 데이터 + n=1. 데이터 폴더만 삭제하고 `src/dataset/capture_eye_dataset.py` 는 남긴다 |
| `data/processed/eyeblink8_eyes.npz` 및 v1 parts | v1 데이터셋. v2 는 mEBAL2 단독 |
| `temp_ppt_build/`, `tmp/` | 임시 산출물 |

### 6-6. 잘못된 방향이므로 폐기 검토 필요

| 자산 | 사유 |
|---|---|
| `src/tools/robust.py` (조건부 SR 유틸) | 🚫 SR 미채택 확정. baseline 이 6×까지 무손실, 8× 에서 검출 저하(29→25) |
| `results/split_eval.json` | 🚫 임계값 격자 오류. `results/split_eval.RETIRED.md` 에 폐기 기록 |
| `docs/archive/**` | 흡수 완료. **인용하지 않는다** (삭제는 하지 않음) |
| `models/classifier/metrics.json` | ⚠️ 문서에 "이상치, 교체 대상" 으로 기록됨 (v1) |

---

## 7. 다음 세션이 가장 먼저 확인할 것

1. `src/v2/` **git 커밋** — 미추적 상태라 결과 JSON 의 `git_commit` 이 실행 코드를 지목하지 못한다.
   **새 실행 전에 커밋한다.**
2. `TASKS.md` **T3-1 (image_cnn)** — 논문의 핵심 비교가 여기 걸려 있다.
3. `docs/v2/PROTOCOL.md` §0 상단의 **2026-08-05 범위 정정 헤더**를 먼저 읽는다.
   §14·§15 는 범위 밖이다.
4. v1 Pi 5 수치(12.06 ms / 86 fps)를 **논문에 인용하지 않는다.**
