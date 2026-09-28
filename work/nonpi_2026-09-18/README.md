# Raspberry Pi 없이 진행한 원고 강화

작업일: 2026-09-18. 원고 원본과 기존 코드/학습 산출물은 수정하지 않고, 이 폴더에 수정 사본·새 분석·실험 코드를 분리했다.

## 완료한 항목

- 기존 자료 보존: `../../snapshots/nonpi_2026-09-18/prior_evidence_complete.zip`, `manifest.json`, git 상태/변경사항. 읽기 권한이 없는 과거 `docs/archive/HANDOFF.md` 한 파일은 manifest에 미포함 이유를 표시했다. 데이터 대용량 원본은 중복하지 않았다.
- `manuscript/MCE_blink_detection.tex`: 수정 원고. `manuscript_changes.diff`: Downloads 원본과의 차이.
- `results/source_robustness.json`: 원래 paired subject bootstrap 재현, 허용차 민감도, subject-macro 분석, 전체 57명 leave-one-out 영향 분석.
- `results/external_pilot.json`, `external_summary.json`, `external_scores.npz`: EyeBlink8 영상 9 한 개의 실제 외부 추론 결과. 모델 4종 각각 기존 15 fits, source validation threshold 고정. 전체 EyeBlink8 검증이 아니다.
- `CITATION_AUDIT.md`: 인용별 원문/서지 확인 근거 및 남은 제약.
- `FUTURE_RESEARCH.md`: Pi 재측정·프론트엔드 개선 등 이후 연구로 보존할 과제.
- `PROTOCOL.md`: 외부 예측을 보기 전 정한 범위와 평가 규칙. 공개 등록 프로토콜은 아니다.

## 확인된 결과와 해석

| 분석 | 실제 결과 | 해석 |
|---|---|---|
| 원래 pooled AP 차이, ours − Image-CNN+TCN | −0.004021 [−0.006304, −0.001993] | 기존 결과 재현. 정확도 손실은 존재. |
| 사후 NI 허용차 | 0.02 및 0.01 통과, 0.005 보류 | 0.01을 사전 기준처럼 바꾸지 않는다. |
| 피험자 동일가중 AP 차이 | −0.006190 [−0.011438, −0.002274] | 0.02 통과, 0.01 보류. 추정량/가중치에 따른 차이를 보고한다. |
| U1 제외 시 pooled 차이 | −0.003883 | 전체 성능 차이가 U1 하나로 설명되지는 않는다. U1은 본 분석에 유지. |
| 전체 leave-one-subject-out 차이 범위 | −0.004358 ~ −0.003442 | 어느 한 사람을 제외해도 점추정 방향은 유지. 새 CI가 아니므로 우월/열등 검정으로 해석하지 않는다. |
| 외부 단일 영상, ours | window AP 0.9990; continuous recall 0.9984, F1 0.8887; 오검출 3.81회/분 | 15 source fit의 기술적 평균. 외부 피험자 15명이 아니며 일반화 CI/우월성 근거가 아니다. |
| 외부 단일 영상, Image-CNN+TCN | window AP 0.9987; continuous recall 0.9919, F1 0.8665; 오검출 4.93회/분 | 단일 영상·다른 negative 구성. 논문 표의 source AP나 공식 benchmark F1과 직접 우열 비교하지 않는다. |

원고의 전력 조건을 실제 98.65–130.39초 free-run으로 정정했고, active-minus-idle W와 total-active mJ/frame을 구분했다. 30fps 고정률 전력/배터리 수명 주장은 하지 않는다. Image-CNN도 16D+같은 TCN임을 명시하고 TCN 4,625 / BiLSTM 8,577 / LSTM 4,353의 파라미터 설명을 바로잡았다. 비대칭 stride와 encode-once 자체의 독자 기여 주장은 축소했다.

## 진행 중 / 미완료

**새 GAP 대조군 5-fold × 3 seed 학습은 장시간 실행 중이다.** `train_gap.log`가 현재 진행 근거다. 완료 전 파일은 `results/train_gap.partial.json`이며 인용하지 않는다. 최종 파일은 `results/train_gap.json`; 완료 후 비교는 `results/gap_completion_status.json`에 기록된다. 상태 파일이 없다면 완료로 간주하지 않는다. 새 대조군 결과는 아직 수정 원고에 반영하지 않았다.

GAP는 기존 Image-CNN의 convolution/pooling을 유지하고 6,912→64 dense를 GAP+64→64로 교체한다. 37,889 total params, 31.414208 MMAC로 예상 비용을 코드로 확인한다. 원래 비교 모델보다 작으므로 기존 5.7배 파라미터 주장이 비교 모델 선택에 의존하는지 시험한다. 학습은 기존 trainer·split·3 seed·validation AP early stopping을 그대로 사용한다. 신규 가상환경 NumPy 1.26.4와 기존 2.4.2 차이를 기록하며 비트 단위 동일 환경 재현이라고 주장하지 않는다.

EyeBlink8 전체 8개 영상은 아직 확보하지 못했다. 저자 공식 ZIP에 WEDOS 보안 확인이 반환됐고 우회하지 않았다. 로컬에는 9번 영상만 있었다. 전체 원본 경로가 확보되면 동일 정책을 전 영상에 확장해야 한다. RNN 전달 ZIP에는 학습 코드와 요약 문서만 있고 raw JSON/NPZ/checkpoint는 없어 독립 재분석은 미완료다.

## 외부 평가의 데이터 처리 기록

- 컨테이너와 timestamp 파일 5,183 frames, 실제 decode 5,134. 주석은 frame 74–5,133. 49개 초과 timestamp tail을 버림.
- 실제 timestamp를 30Hz 최근접 frame map으로 맞춰 5,133 positions; 복제 174회, 사용하지 않은 native frame 175개. 원본을 고정 30fps의 완전한 시퀀스로 가정하지 않음.
- blink ID 41개 모두 유지. 영상 경계로 제외된 event 0. 양눈 통합 ID 기준이며 per-eye completeness 평가 아님.
- timestamp 길이 불일치는 예측 전에 검사에서 발견해 처리했다. 원래 모든 timestamp와 decoded frame 수가 같다는 가정이 잘못됐으며 이를 metadata로 남겼다.
- `annotation_alignment.png`에서 시작/폐안 표본의 눈 위치와 주석을 시각 확인. 전체 라벨을 수작업 재주석한 것은 아니다.
- 원래 held-out 32 events의 checkpoint logit 재현 최대 오차: ours 2.29e−5, Image-CNN 2.57e−5 (CUDA). 최초 CPU 비교는 0.00160의 수치 차이로 엄격한 검사를 통과하지 못했으며, 원래 GPU 경로로 확인했다.
- interval matching 경계·중복 검출·augmenting path·IoU 검사 4건 통과. 0.1 IoU 민감도도 보고.
- 지연 0.307초는 각 run의 median을 평균한 **영상 시각 기준** 지연이다. Pi wall-clock/queue/display 지연을 재측정한 것이 아니다.
- 사후 오검출 분해(`external_false_event_diagnostic.json`): ours는 영상당 평균 10.67개의 unmatched detections 중 3.53개가 이미 다른 검출과 매칭된 GT blink와 겹친다. 나머지 7.13개는 어떤 GT와도 겹치지 않는다. 중복 억제가 후속 개발 후보지만 이 테스트 영상으로 임계값을 다시 고르지는 않았다.

## 재실행

프로젝트 루트에서 `.venv-nonpi/Scripts/python.exe`를 사용한다. 기존 확정 출력과 경로가 다르다.

```powershell
& .venv-nonpi/Scripts/python.exe work/nonpi_2026-09-18/reanalyse.py
& .venv-nonpi/Scripts/python.exe work/nonpi_2026-09-18/test_external_eval.py
& .venv-nonpi/Scripts/python.exe work/nonpi_2026-09-18/external_eval.py preprocess
& .venv-nonpi/Scripts/python.exe work/nonpi_2026-09-18/external_eval.py evaluate
& .venv-nonpi/Scripts/python.exe work/nonpi_2026-09-18/validate_and_summarize.py
```

현재 실행 중인 학습을 다시 시작하지 말 것. 종료/실패했으면 기존 partial을 확인한 뒤 이미 완료한 runs를 보존하는 재개 방식을 먼저 구성해야 한다. 원래 trainer는 자동 resume 기능이 없으므로 같은 출력으로 재실행하면 기존 새 실험의 진행분을 덮어쓸 수 있다.
