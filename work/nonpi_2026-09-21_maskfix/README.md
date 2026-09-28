# 수정된 입력으로 전체 비교 재실행

현재 상태: `status.json`의 state와 updated를 함께 확인한다. `running`이면 개별 로그 경로도 확인한다.

- Ours(vpres), Image-CNN+TCN, vdrop, GAP+TCN 각각 5-fold × 3-seed = 60회 신규 학습.
- 발견된 결측 입력의 다른 피험자 픽셀 참조만 수정. 원래 src/·데이터·확정 결과·원고는 보존한다.
- `maskfix.py`는 유효 프레임만 읽고 결측 입력을 0으로 만든다. 학습 시 encoder BatchNorm보다 먼저 적용된다.
- `preflight.json`에서 결측 입력 누출 재현, 수정 후 불변성, 정상 입력·라벨·마스크 유지 검증 결과를 확인한다.
- `PROTOCOL.md`는 새 실험의 고정 규칙이다. 이전 GAP 6회 및 vdrop 1회, 기존 vpres/Image-CNN은 새 결과에 혼합하지 않는다.
- `records/`와 `attempts/`에 fit별 결과·모델·hash·원래 로그를 보존한다. 최종 15회가 되기 전 결과는 partial이다.
- 전체 완료 시 `RESULTS.md`와 `results/comparison.json`을 자동 생성한다. 원고·표·PDF는 생성하거나 수정하지 않는다.
- `post_train.py watch`도 별도 실행했다. 60회 학습과 비교가 완료된 후, 네 후보 모델의 사용자별 진단·저장된 실제 랜드마크 크롭 비교·EyeBlink8 9번 영상 예비 평가를 재추론한다. 상태는 `post_train/status.json`이다. 이 과정은 채팅 알림을 보내지 않는다.
- 기존 외부·실제 랜드마크 감사의 크롭은 재사용할 수 있지만 **수정된 체크포인트로 예측을 다시 해야 한다.**
- 자동 재평가도 전체 외부 데이터 확보·RNN 원자료 검증·최종 배포 구현·Pi 실측을 대신하지 않는다.
- 기존 `../nonpi_2026-09-21/experiment_queue.py`는 재시작하지 않는다.

재개 명령(실행 중인 controller나 학습 child가 없는지 먼저 확인):

```powershell
& .venv-nonpi/Scripts/python.exe work/nonpi_2026-09-21_maskfix/experiment_queue.py prepare
& .venv-nonpi/Scripts/python.exe -u work/nonpi_2026-09-21_maskfix/experiment_queue.py run
```

재개는 완료된 fit 단위다. 중단된 epoch에서 이어지는 것이 아니라 새 attempt에서 해당 fit을 다시 학습한다.

외부 전체 데이터·RNN 원자료 확보, 최종 배포 규칙 통일, Pi 재측정은 여전히 별도 완료 조건이다. 이 큐 완료만으로 투고 준비 전체가 완료되었다고 보지 않는다.
