# Pi 실측 전 PC 준비 — 2026-09-28

범위: 장치 없이 할 수 있는 준비만 했다. **Pi에는 접속하지 않았고 Pi 수치는 하나도 없다.** 원고·표·그림·PDF, 기존 결과·모델·데이터, 기존 소스 파일은 수정하지 않았다(새 파일만 추가). 완료된 학습은 다시 실행하지 않았다.

시작 전 검사: 인수인계 목록 39/39, 9/22 수정 소스 7/7, ONNX ZIP 2/2 해시 일치 → `preflight_inventory.json`. 작업 후 다시 검사해도 같다.

## 1. 30 fps 고정 재생 측정 코드 — 완료(PC 검증)

새 모듈(`src/v2/deploy/`), 기존 `run_video.py`·`run_video_power.py`·`power_bench_runner.py`는 그대로 둠:

| 파일 | 역할 |
|---|---|
| `pacing.py` | monotonic clock 30 Hz 시간축, `drop`/`queue`/`freerun` 정책, 클립 반복·이음매 표시 |
| `run_paced.py` | 1 run: warmup→settle→idle_pre→active→idle_post, 모델 계약·게이트, 결과 JSON/NPZ/PMIC 원문 |
| `bench_telemetry.py` | 온도·스로틀(NOW/sticky 구분)·ARM 클럭·팬 1 Hz, PMIC 원문 기록 |
| `paced_session.py` | preflight, 블록 무작위 순서, 온도 기준 쿨다운, 재개, 최종 QC |
| `check_paced_results.py` | run QC(보고 불가 사유/경고), 모드별 기술 통계, 블록 내 짝 차이 |
| `pi_env_check.py` | Pi 환경·버전·PMIC 원문 3회 기록(읽기 전용) |
| `test_paced_bench.py` | 24개 단위테스트 |

구분하는 양: 도착 aᵢ=t0+i/30, 대기(sleep, 처리 시간 아님), backlog(앞 프레임 초과), dispatch(깨어남·루프 지연), schedule delay=backlog+dispatch, processing(읽기~head), response=도착→종료, deadline miss(종료>도착+33.33 ms 또는 drop). active는 9,000 시간축 프레임이며 모든 모드가 마지막 슬롯까지 같은 300초를 채운다. free-run은 별도 정책으로만 존재하고 deadline을 정의하지 않는다.

모델 연결: 모드명 `ours_vpres`/`image_cnn_head`/`vdrop`/`gap`/`ear_rule`. CNN 모드는 번들 계약 없이는 실행되지 않고, 계약의 variant가 모드와 다르면 거부한다(vdrop/GAP가 Ours로 기록될 수 없음). 번들 manifest 해시, 이 장치에서 만든 `pi_model_verification.json`(machine·ORT 버전 일치), 동치 게이트가 모두 통과해야 측정한다.

## 2. 측정 절차 — 프로토콜 파일로 고정(값은 설계 제안, Pi 미실행)

`protocol/pi5_paced30_v1.json`(SHA-256은 `verification.json`). 요약:

- 입력: EyeBlink8 video9 원본 AVI, OpenCV 디코드 5,134프레임. 늦은 프레임은 `drop`(카메라 방식), 버린 위치는 결측으로 유지.
- run: warmup 10초(30 Hz) → settle 5초 → idle_pre 60초 → active 300초 → idle_post 30초, run마다 새 프로세스.
- 세션: primary(Ours·Image-CNN+TCN·EAR) 5블록, 블록마다 시드 20260928 순열. 최소 120초 쉼 + 기준 온도+3 °C까지 최대 600초 추가 대기. secondary(Ours 기준·vdrop·GAP) 3블록, freerun 진단 3블록은 별도 세션.
- ORT intra 2, spin off, PMIC 7.5 Hz, 온도 1 Hz. governor performance, throttled=0x0에서만 시작.
- QC는 코드가 강제한다. deadline miss·drop은 결과로 남기고 제외 사유로 쓰지 않는다. 값 때문에 run을 빼지 않는다.
- 예상 시간(추정): primary 약 2.3시간, secondary 약 1.4시간, freerun 약 50분.

## 3. Pi 전송 묶음 — 완료

`pi_kit_2026-09-28.zip`(11.7 MB, SHA-256 `5c3f3384…5496`, `.sha256` 동봉) = `pi_kit_2026-09-28/`. 수정 코드(② 포함) 22개 파일, 수정 ONNX 기본 번들(ZIP 해시 확인 후 해제), 클립, 동치 게이트, 프로토콜, `scripts/00~08`, `README_PI.md`, `verify_kit.py`와 60개 파일 해시 manifest. Pi에서 **새 폴더**에 풀어 그 안에서만 실행하므로 기존 Pi 저장소·모델을 덮어쓰지 않는다. 재생성: `build_pi_kit.py`.

## 4. PC 테스트·동작 검증 — 완료

- 단위테스트 47개 OK(새 24 + 기존 PMIC·데모 UI). 9/22 `test_policy.py`, `test_entrypoints.py`, ONNX `test_packaging.py` 종료코드 0.
- 실제 ONNX + MediaPipe 데스크톱 dry run(`pc_dryrun/`): 5개 모드 모두 20초 600/600프레임 처리, drop 0. 모든 run에서 얼굴 587/600, 후보 수는 모델별로 반복 간 동일(Ours 9, Image-CNN 9, vdrop·GAP 12). **데스크톱 지연 수치는 Pi 성능이 아니며 쓰지 않는다.**
- 강제 과부하(fps 400): drop 1,620건, 처리된 모든 프레임에서 클립 위치 = 시간축 위치 mod 5,134, 이음매 5,134 확인.
- 새 ZIP을 새 폴더에 풀고 스크립트 실행 리허설(`pc_rehearsal/`): 00·02·03 통과, 04 smoke는 실행된 뒤 QC가 비-Pi로 표시, 01·05는 비-Pi라서 거부(05는 아무것도 측정하지 않음) — 모두 설계대로. 리허설 뒤 04의 출력 폴더 이름만 바꿔 다시 빌드했고, 최종 ZIP은 00·03으로 다시 확인했다.

**발견·수정한 버그:** 첫 초안의 `argparse.BooleanOptionalAction` 때문에 `--no-spin`이 오히려 spin을 켰다. dry run에서 CNN 모드 프로세스 CPU가 20초 창에 약 41초였고 QC가 잡았다. 명시적 `--no-spin/--allow-spin`으로 고친 뒤 약 3~4초. 회귀 테스트 추가. 문제 run은 `pc_dryrun/PC_DRYRUN_primary_spinbug_superseded/`에 보존. 기존 코드(store_true)에는 해당하지 않는다.

## 확인한 사실

- 클립: 컨테이너·주석은 5,183프레임, OpenCV 4.11 디코드는 5,134프레임.
- 기존 Pi 전력 run `results/v2/power_ours_480p_run1.json`: 10,268프레임(=2×5,134)을 free-run 88.2 fps로 116초 만에 처리하고 EOF로 끝남(요청 300초). 기존 결과가 30 fps 고정 부하가 아니었다는 인수인계의 지적과 일치.
- 이 기존 run의 실제 Pi PMIC 원문은 12개 rail 파서로 누락 없이 파싱된다(테스트 fixture). rail 합이 보드 입력 전력과 같다는 검증은 아니다.
- `.gitignore`가 11:16:30에 이 작업과 무관하게 변경됐다. 건드리지 않았다.

## 제안(미확정, Pi 첫 보고용 run 전에만 변경)

- 반복 5회·idle 60초·쿨다운 규칙은 출발점이다. smoke 결과에서 idle 표준편차나 온도 회복 시간이 크면 **모드 비교를 보기 전에** 새 protocol_id로 조정한다.
- 파일 재생은 H.264 디코드를 read 단계에 포함하므로 실제 카메라 캡처 비용과 다르다. 카메라 경로 비용은 별도 항목으로 보고한다.
- vcgencmd 폴링(7.5 Hz + 1 Hz) 부하는 모든 모드와 idle에 같게 들어가지만, 외부 전력계로 대조하기 전에는 “PMIC rail 합”으로만 쓴다.

## 남은 일 — Pi 필요(미완료)

1. `00`→`01`(PMIC 원문·rail 대조)→`02`(Pi ORT 1.19.2 번들 검사)→`03`→`04` smoke→`05` primary→(선택)`06`·`07`→`08` 회수 후 PC에서 QC 검토.
2. Pi 주소·계정·외부 전력계 정보는 아직 받지 않았다. 필요한 시점에 받는다.
3. 별도로 남은 연구 작업: 외부 데이터 전체 검증, RNN 원자료 검증. 원고 수정은 모든 검증 뒤로 보류.
