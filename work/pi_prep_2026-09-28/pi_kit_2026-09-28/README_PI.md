# Pi 5 30 fps 고정 재생 측정 키트 — 2026-09-28

수정된 fold0/seed0 모델(Ours vpres, Image-CNN+TCN, 후보 vdrop·GAP)과 규칙 EAR을 **같은 클립·같은 30 Hz 시간축·같은 측정 구간**으로 재는 자립형 키트다. 연구 PC에서 만들고 PC에서만 검사했다. **Pi에서는 아직 한 번도 실행하지 않았다.**

## 0. 배치 — 기존 Pi 저장소를 덮어쓰지 않는다

```bash
# PC에서 (주소·계정은 실제 값으로)
scp pi_kit_2026-09-28.zip pi_kit_2026-09-28.zip.sha256 <user>@<pi-address>:~/
# Pi에서
cd ~ && sha256sum -c pi_kit_2026-09-28.zip.sha256
unzip pi_kit_2026-09-28.zip          # -> ~/pi_kit_2026-09-28/ (새 폴더)
cd ~/pi_kit_2026-09-28
conda activate eyeblink              # Python 3.11, requirements-pi.txt
```

모든 명령은 키트 루트(`~/pi_kit_2026-09-28`)에서 실행한다. 기존 `~/IEEE_sensors`와 `models/v2/onnx`는 읽지도 쓰지도 않는다.

## 1. 순서

| 단계 | 명령 | 통과 조건 | 비고 |
|---|---|---|---|
| 0 | `bash scripts/00_verify_kit.sh` | problems 없음 | 전송 무결성 |
| 1 | `bash scripts/set_governor_performance.sh` (sudo, 부팅마다) | 모든 CPU `performance`, `throttled=0x0` | 사용자가 직접 실행 |
| 2 | `bash scripts/01_env_check.sh` | 종료코드 0 | `results/pi_paced/env_check.json`: 버전·팬·PMIC 원문 3회. **PMIC rail 목록을 여기서 원문과 대조** |
| 3 | `bash scripts/02_verify_bundle.sh` | `"passed": true` | Pi ORT로 모델 해시·수치·결측 정책 검사 → `bundle/pi_model_verification.json` |
| 4 | `bash scripts/03_tests.sh` | OK | 30 Hz 스케줄러·drop·이음매·QC 단위테스트 |
| 5 | `bash scripts/04_smoke.sh` | 오류 없이 종료, smoke_check의 제외 사유가 없음 | **보고 불가 점검.** ② 수정 경로(run_video) + 짧은 paced 세션 |
| 6 | `bash scripts/05_primary.sh` | 세션 요약에서 excluded 0 | **보고용 본 측정.** 약 2.3시간 |
| 7 | `bash scripts/06_secondary.sh` | 〃 | 선택. vdrop/GAP 후보 비용, 약 1.4시간 |
| 8 | `bash scripts/07_freerun.sh` | 〃 | 선택. 처리량 진단, 약 50분. 30 fps 결과와 합치지 않음 |
| 9 | `bash scripts/08_collect.sh` | tgz + sha256 | PC로 회수 |

어느 단계든 실패하면 다음 단계로 넘어가지 않고 해당 JSON/로그를 회수한다. 세션이 중단되면 **같은 명령을 다시 실행**하면 이어서 진행된다(끝난 run은 유지, 순서는 재추첨하지 않음).

## 2. 고정 프로토콜 (`protocol.json`, id `pi5_paced30_v1`)

- 입력: EyeBlink8 video9 원본 AVI(640×480, H.264), SHA-256 `d895e84a…e38e0a`. PC OpenCV 4.11 디코드 5,134프레임(컨테이너 메타는 5,183). Pi preflight가 디코드 수를 다시 세고 다르면 시작하지 않는다.
- 시간축: 프레임 i는 t0 + i/30초에 도착. active = 9,000 시간축 프레임(300초) = 클립 1회(5,134) + 2회차 앞 3,866. 이음매에서 19프레임 창과 후보 상태를 초기화한다.
- 늦은 프레임: `drop`(카메라 방식). 새 프레임이 이미 도착했으면 대기 중인 옛 프레임을 버리고, 버린 위치는 창에서 결측으로 남긴다. 버린 프레임도 deadline miss로 센다.
- 한 run(새 프로세스): 모델·FaceMesh 로드 → warmup 10초(30 Hz, 분석 제외) → settle 5초 → idle_pre 60초 → active 300초 → idle_post 30초.
- 세션: primary 3모드 × 5블록. 블록마다 모든 모드를 한 번씩, 시드 20260928로 섞은 순서. run 사이 최소 120초 쉰 뒤 SoC 온도가 세션 기준(시작 전 120초 idle 평균)+3 °C 이하가 될 때까지 최대 600초 추가 대기(충족 여부 기록).
- ORT intra 2, spin off, head stride 1, refine_landmarks off, PMIC 7.5 Hz, 온도·스로틀·ARM 클럭·팬 1 Hz.

프로토콜 값은 PC에서 정한 설계값이다. 바꾸려면 첫 보고용 Pi run 이전에 새 `protocol_id`로 만들고, 모드 비교 결과를 본 뒤에는 바꾸지 않는다.

## 3. 결과 구조와 정의

`results/pi_paced/<SESSION>/`:

- `session_plan.json`: preflight(플랫폼·클립 디코드·게이트), 블록 순서, 기준 온도
- `session.log`, `session_runs.jsonl`: 각 run의 시작/끝 시각, 쿨다운 기록. 외부 전력계를 쓰면 이 벽시계 시각과 run.json의 `phase_log`로 맞춘다.
- `<run_id>/run.json`: 설정·모델 해시·게이트·단계별 지연·pacing·전력·온도·QC
- `<run_id>/frames.npz`: 프레임별 도착/시작/종료/대기, drop, 단계 시간, 판정
- `<run_id>/pmic_raw.jsonl.gz`: 모든 `pmic_read_adc` 원문(시각 포함)
- `session_summary.json/.md`: QC 통과 run만 집계, 제외 run과 사유, 블록 내 짝 차이

| 이름 | 정의 |
|---|---|
| processing_ms | 시작→종료: 읽기(디코드)+FaceMesh+crop+encoder+head (+drop 프레임 grab) |
| schedule_delay_ms | 도착→시작 = backlog(앞 프레임 초과) + dispatch(깨어남·루프 지연) |
| response_ms | 도착→종료 |
| wait | 시작 전 sleep. 처리 시간에 넣지 않음 |
| deadline miss | 종료 > 도착 + 33.33 ms, 또는 drop |
| delta_w | active 평균 − idle_pre 평균 (PMIC rail 합) |
| incremental_mj_per_timeline_frame | delta_w × active 시간 / 9,000 |

창 중심과 출력 시각 사이의 9프레임은 모델 정의상의 지연이며 response_ms와 별개다.

## 4. QC — 코드가 강제

보고 불가(hard): Pi 아님, governor ≠ performance, Pi 모델 검사 미통과/ORT 버전 불일치, 동치 게이트, intra≠2 또는 spin, 1 Hz 표본의 스로틀 NOW 비트 또는 run 중 새 sticky 비트, PMIC 읽기 실패·rail 누락·표본률 < 절반·간격 초과, 클립 해시/패스 길이 불일치, 프로토콜 값/해시 불일치.

경고: idle 전후 차 > max(0.05 W, 3 %), 시작 시 sticky 비트(재부팅 권장), 클립 메타 fps ≠ 30.

**deadline miss·drop은 제외 사유가 아니라 결과다**(`realtime_pass=false`). 값이 크거나 불리하다는 이유로 run을 빼지 않는다. 에너지 비교는 두 모드 모두 drop 0(`workload_identical`)일 때만 한다.

## 5. 해석 한계

- 전력은 PMIC 12개 rail V×I 합이다. 보드 전체 입력 전력과 같다고 검증하지 않았다. 외부 전력계 대조 전에는 “PMIC rail 합”으로만 쓴다.
- 한 장치·한 클립·몇 회 반복의 기술 통계다. 모집단 CI를 주장하지 않는다.
- `ours_vpres`만 Ours다. vdrop/GAP는 후보 이름 그대로 보고한다. `ear_rule`은 가중치 없는 규칙이며 검증된 임계값이 없어 점수만 낸다.
- free-run 결과와 30 fps 결과를 합치거나 서로의 절감으로 바꿔 쓰지 않는다. 기존 18개 Pi run(`results/v2/power_*_480p_run*.json`, 이전 모델·free-run)은 과거 자료로만 둔다. 키트의 `results/v2/power_ours_480p_run1.json`은 PMIC 파서 테스트 fixture다.
- `bundle/local_verification.json`은 연구 PC 결과다. Pi 측정은 `bundle/pi_model_verification.json`만 인정한다.
