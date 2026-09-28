# GAP + vdrop 전체 비교 (원고 수정 보류)

> **2026-09-21 중단:** 결측 입력에 frame0(U1)을 대입하는 경로가 encoder BatchNorm을 통해 학습에 영향을 주는 문제를 확인했다. 이 큐를 재시작하지 말 것. 기존 결과는 보존하며, 새 실행은 `../nonpi_2026-09-21_maskfix/`를 사용한다. 아래 내용은 중단 전 계획 기록이다.

2026-09-21 요청에 따라 GAP 완료와 vdrop 전체 검증만 수행한다. 원고/표/PDF를 수정하거나 재생성하는 명령은 실행 큐에 없다.

- **GAP:** 기존 완료 6회는 검증·복사해 보존하고 나머지 9회 학습. 중단된 fold2/seed0은 epoch checkpoint가 없어 처음부터 다시 실행.
- **vdrop:** 5-fold × 3 seed = 15회 새로 학습하고 checkpoint 보존.
- **최종 비교:** 기존 vpres, Image-CNN+TCN, GAP, vdrop의 정확도·paired CI·파라미터·연산량. 성능이 좋은 runs만 선택하지 않는다.
- 기존 두 fold vdrop 결과는 본 비교의 사전 탐색 자료다. 새 결과도 독립적인 사전등록 검증으로 포장하지 않는다.

현재 상태는 `status.json`의 **state와 updated 둘 다** 확인한다. 오래된 running 표시는 살아 있는 프로세스의 증거가 아니다. GPU 프로세스는 창을 띄우지 않고 순차 실행한다. controller PID는 `controller_pid.txt`, 실제 fit 로그는 `status.json`의 `log` 경로다.

실행 상태:

- `running`: 개별 학습 중, 15초 간격 heartbeat.
- `analysing`: 모든 학습 후 통계 비교 중.
- `complete`: GAP 15회·vdrop 15회 모두 검증 완료. `RESULTS.md`, `results/comparison.json` 확인.
- `failed`: 중단/오류. `error`와 해당 `attempts/.../train.log` 확인. partial 값을 최종 결과로 인용하지 않는다.

완료된 fit마다 `records/`에 검증 메타데이터, SHA-256, 실제 모델 위치를 남긴다. 재시작은 완료된 fit을 건너뛴다. 중간 epoch에서의 재개 기능은 없으며, 미완료 fit 재실행 시 새 attempt 폴더를 만들어 기존 로그를 남긴다. 이전 child가 살아 있으면 중복 학습을 거부한다.

프로젝트 루트에서 수동 재개(기존 controller가 실행 중일 때 중복 실행 금지):

```powershell
& .venv-nonpi/Scripts/python.exe work/nonpi_2026-09-21/experiment_queue.py prepare
& .venv-nonpi/Scripts/python.exe -u work/nonpi_2026-09-21/experiment_queue.py run
```

최종 비교 결과가 나와도 추천 모델을 논문에 자동 반영하지 않는다. 원고는 전체 결과와 외부 검증 범위를 검토한 뒤 별도로 수정한다. 새로운 Pi 지연/전력 결과도 생성하지 않는다.
