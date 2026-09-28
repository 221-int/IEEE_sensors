# 보존해 둘 후속 연구 묶음

2026-09-18 사용자 요청에 따라 기존 설계·결과·원고는 `snapshots/nonpi_2026-09-18/prior_evidence_complete.zip` 및 SHA-256 manifest에 보존했다. 한 과거 문서 `docs/archive/HANDOFF.md`는 OS 읽기 권한 때문에 압축에 포함되지 못했고 manifest에 오류를 명시했다. 원본은 삭제·이동하지 않았다. 대용량 원데이터는 압축에 중복하지 않았다. `data/mEBAL2`에 원영상 압축들이 존재하며, 추출된 `data/raw/mEBAL2`가 일부라는 사실과 구분해야 한다.

## 현재 원고 다음에 진행할 연구

1. **검출 프론트엔드 최적화**: 얼굴 재검출 주기/신뢰도 기반 재검출, 눈 추적, landmark-free ROI를 같은 영상·검출 지표로 비교. 정확도·실패 복구·p99·드롭률을 공동 보고. 기존 MediaPipe는 이미 tracking 모드를 사용하므로 ‘tracking을 처음 도입한다’는 구상은 부정확하다.
2. **고정률 전력 재측정**: 학습된 가중치, 동일 30fps workload/입력 시각, 10–30분 안정구간, run순서·온도·throttle 기록. USB 입력 전력계로 PMIC rail 합의 의미 교차 검증. 전체 active J/frame과 idle-subtracted J/frame을 구분. 무부하/idle 기준의 의미도 고정.
3. **일반화 확장**: EyeBlink8 전체와 가능하면 ZJU/Researcher's Night/HUST/MPE 등. 원래 입력이 한쪽 눈만인 자료는 양눈 크롭으로 억지 변환하지 말고 별도 모델/비교 설계. 접근 신청·타인에게 메일은 별도 사용자 지시가 필요하다.
4. **연속 검출 학습**: window classification과 event localization을 분리; onset/offset 또는 frame-state supervision, validation-only hysteresis/중복 검출 억제, 매칭 규칙 사전 고정. 이번 외부 영상 결과를 보고 조정하면 그 영상은 이후 개발 데이터로 취급하고 새 외부 테스트셋을 둔다.
5. **실패 원인과 수집**: U1, U16 등 손실이 큰 피험자에서 라벨 재검수·눈 움직임·크롭·랜드마크 출처·부분 깜빡임을 확인. 원인 검증용 개입과 일반 성능 향상을 위한 모델 선택을 분리한다.
6. **압축·다른 하드웨어**: PTQ/QAT, calibration subject 분리, CPU/NPU에서 trained FP32와 정확도·지연·실제 전력 동시 비교. Pi 실측 없이 MMAC로 속도/전력 이득을 확정하지 않는다.

프라이버시·신원 제거·복원 방지는 이 원고의 주장이 아니다. 관련 특허/별도 후속 작업에만 남긴다.
