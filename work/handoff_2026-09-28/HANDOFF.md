# 새 세션 인수인계 — 2026-09-28

## 0. 가장 먼저 알아야 할 상태

- 프로젝트: `C:\Users\sch\PycharmProjects\IEEE_sensors`
- 연구: 작은 눈 영상 임베딩을 이용한 깜빡임 분류와 Raspberry Pi 5 실시간 비용 분석.
- 현재 원고: `C:\Users\sch\Downloads\MCE_blink_detection.tex` (IEEE Consumer Electronics Magazine). 예전 문서의 Sensors Letters/4페이지 설명을 현재 투고 조건으로 사용하지 말 것.
- **① 수정 결과 검토, ② 전처리·배포 규칙 수정, ③ 수정 체크포인트 ONNX·모델 계약 생성은 완료.**
- **Pi 측정 준비는 아직 전부 완료되지 않았다.** 다음으로 PC에서 30 fps 고정 재생 측정 코드, 고정 측정 절차, 수정 코드 전송 묶음을 준비해야 한다. 그 뒤 Pi 실측이다.
- 외부 데이터 전체 검증·RNN 원자료 검증도 남았다. Pi 실측만 남은 상태가 아니다. 다만 이 둘은 Pi 측정 준비를 막는 필수 선행 작업은 아니다.
- 이번 세션의 마지막 요청은 인수인계 자료 정리다. 이 요청으로 새 실험이나 Pi 측정을 시작하지 않았다.

### 이 인수인계 묶음 사용법

- 같은 PC의 새 세션: 이 프로젝트를 열고 `NEW_SESSION_PROMPT.txt` 내용을 붙여 넣는다. 프로젝트 파일을 다시 업로드할 필요는 없다.
- `current_inventory.json`: 9/28에 확인한 핵심 파일 경로·해시·git 상태. 핵심 파일 누락 없음, 수정 학습 기록 15개씩 총 60개 확인. 9/22 수정 소스 7개와 보존 대상 원고/기존 모델의 해시가 그대로이며, 새 ONNX ZIP 2개의 해시도 일치한다. 이는 파일 보존 검사이며 전체 실험을 다시 실행한 결과가 아니다.
- `handoff_context.zip`: 이 문서·시작 메시지·현재 목록과 주요 보고서/측정 소스의 작은 참고 묶음. **전체 프로젝트 백업이나 Pi 실행 키트가 아니다.** 데이터·체크포인트·ONNX 모델 ZIP은 넣지 않았으며 기존 프로젝트 경로를 계속 사용한다. 다른 PC에서 계속하려면 이 ZIP만으로는 부족하다.

## 1. 사용자 제약과 작업 방식

1. **원고·표·그림·PDF 수정은 모든 실험과 검토를 마친 뒤 한 번에 한다. 지금 수정하지 않는다.** 9/18 작업용 수정본은 이미 존재하지만, 이후 보류 요청에 따라 보존 중이다. Downloads 원본을 작업용 사본과 혼동하지 않는다.
2. 신원 제거·프라이버시·복원 방지는 논문 범위 밖. `privacy-preserving` 주장 금지. 기존 데모에는 신원 기능이 있지만 연구 기여로 다시 끌어오지 않는다. 별도 `eye-wellness-patent` 프로젝트도 이번 작업에 섞지 않는다.
3. 기존 결과·모델·원본 데이터는 보존한다. 새 자료는 날짜별 `work/`에 저장한다. ②는 사용자의 진행 요청에 따라 실제 `src/`에 수정 반영했고, 수정 전 파일은 별도 보존했다.
4. 사용자는 반복 확인 질문보다 실행과 명확한 상태 보고를 선호한다. 이미 허용된 PC 준비를 다시 허락받을 필요는 없다. 다만 장치 주소나 원자료 위치는 확인된 정보만 사용한다.
5. 검증한 사실과 추정, 데스크톱 검사와 Pi 실측을 구분한다. 파일 경로를 근거로 기록한다.
6. 새 모델 선택·임계값 재튜닝·불리한 seed 제외를 임의로 하지 않는다. 별도 에이전트/새 작업 생성도 요청 없이 하지 않는다.

## 2. 읽을 순서 — 상대 경로는 모두 프로젝트 루트 기준

| 순서 | 파일 | 용도 |
|---|---|---|
| 1 | `work/handoff_2026-09-28/HANDOFF.md` | 현재 인수인계 |
| 2 | `work/nonpi_2026-09-22_review/REVIEW.md` | ①·② 결과·통계·코드 수정의 상세 근거 |
| 3 | `work/nonpi_2026-09-22_onnx/DELIVERY.md` | ③ 완료 보고 및 한계 |
| 4 | `work/nonpi_2026-09-22_onnx/README.md` | 새 모델 계약·Pi 연결·검사 명령 |
| 5 | `src/v2/deploy/run_video_power.py`, `power_bench_runner.py`, `pmic_power.py` | 다음에 보완할 측정 경로 |
| 6 | `src/v2/deploy/run_video.py`, `inference_policy.py`, `frontend.py` 및 `src/v2/dataset/crop.py` | 현재 추론·전처리 정책 |
| 7 | `docs/EXPERIMENT_PLAN.md`, `docs/v2/PROTOCOL.md` | 기존 연구 설계. 최신 완료 보고와 비교해서 읽기 |

과거 `docs/STATUS_2026-08-08.md`, `docs/PAPER_OUTLINE.md`, `docs/CHANGELOG.md`, `docs/PATENT_AND_FUTURE_WORK.md`는 배경 자료다. 과거 상태를 최신 결과로 읽지 않는다. 9/22 review 문서의 “③ 남음”은 작성 당시 상태이며 이후 ONNX DELIVERY에서 완료됐다.

## 3. 완료된 실험과 핵심 수치

### 수정 학습의 배경

이전 trainer는 결측 프레임 위치에 `frames[0]`(U1)을 넣었다. encoder BatchNorm이 마스크 적용보다 먼저 실행되어 다른 피험자 픽셀이 학습에 영향을 줄 수 있었다. 정상 입력·라벨·분할은 유지하고 결측은 encoder 이전에 0, 실제 유효 행만 읽도록 수정했다.

그래서 Ours(vpres), Image-CNN+TCN, vdrop, GAP+TCN을 **각 5 fold × 3 seed, 총 60회 새로 학습**했다. 기존 GAP 6회/중단된 vdrop 등과 섞지 않았다. 완료 시각은 2026-09-22 11:50경이다.

| 모델 | 새 평균 AP | 총 파라미터 | MMAC/frame, TCN 포함 |
|---|---:|---:|---:|
| Ours(vpres) | 0.988227 | 84,049 | 12.489536 |
| Image-CNN+TCN | 0.990494 | 476,161 | 31.852480 |
| vdrop | 0.986351 | 55,377 | 3.490624 |
| GAP+TCN | 0.982621 | 37,889 | 31.414208 |

근거: `work/nonpi_2026-09-21_maskfix/results/comparison.json`, `RESULTS.md`. 데이터는 mEBAL2 RGB 57명·27,758 이벤트, 19프레임 창, gray 64×160, D=16. 60회 결과 해시·점수·라벨·혼동행렬 검증 통과, 학습 예산 초과 없음. 당시 모델은 현재도 보존되어 있다.

Ours−Image-CNN의 추정량을 섞지 말 것:

- 15회 AP 차이 평균: −0.002267, 95% CI [−0.004492, −0.000224]. `work/nonpi_2026-09-22_review/run_mean_ci.json`
- pooled AP 차이: −0.003547 [−0.006487, −0.001382]. `work/nonpi_2026-09-21_maskfix/results/comparison.json`
- subject-macro 차이: −0.004473 [−0.008379, −0.001726]. 같은 파일.
- 57명 subject bootstrap 2,000회, 학습된 모델들에 조건부인 CI. 15 fit을 독립 피험자 15명처럼 취급하지 않는다.
- 원래 δ=0.02 비열등성 유지, 동시에 정확도는 유의하게 낮음. δ=0.01도 통과. δ=0.005는 run-mean만 통과하고 pooled/macro는 유보이므로 유리한 추정량으로 바꿔 주장하지 않는다.
- Ours가 GAP보다 평균은 높지만 CI가 0을 포함하므로 유의한 우월성 주장 불가. 비대칭 stride의 필수성도 입증되지 않음.

### 배포 입력과 외부 영상

- 새 U1 AP: Ours 0.863010 vs Image-CNN 0.952204. 개선됐지만 실패가 해결된 것은 아님. U54 Ours 0.993794로 전반적 저성능자가 아님.
- 성능과 무관하게 고정 선정한 8명·160이벤트: Ours recall 0.9875→실제 FaceMesh 입력 0.891667, Image-CNN 0.9750→0.904167. 검출 실패 양성 4개도 FN 포함. 작은 층화 표본이며 전체 일반화 성능이 아님.
- EyeBlink8 로컬 video9 한 개, GT 41 이벤트: 새 Ours any-overlap event recall 0.993496, F1 0.855425, FP/min 5.640619. Image-CNN F1 0.876904, FP/min 4.998017. 15개 source fit의 기술적 평균이며 독립 외부 피험자 CI 아님.
- Ours IoU≥0.5 F1은 0.675465. 이벤트 존재와 정밀한 폐안 경계는 구분한다. 이전 모델의 외부 pilot F1 0.888674를 새 결과로 사용하지 말 것.

근거: `work/nonpi_2026-09-22_review/result_review.json`, `provider_policy_review.json`; 원자료 `work/nonpi_2026-09-21_maskfix/post_train/{subjects,provider_stratified_sample,external_video9}.json`.

## 4. ②에서 실제 반영한 코드와 검증

- `src/v2/train_encoder.py`: 유효 프레임만 읽고 결측 정규화 텐서를 encoder 전에 0으로 초기화. 새 60회 학습의 로컬 수정과 동등하다.
- `src/v2/dataset/crop.py`: NumPy 1.x/2.x 스칼라 승격 차이로 발생한 반 픽셀 경계 문제를 explicit float32 좌표 연산으로 고정했다. U19 frame33969, U55 frame49892 불일치가 재현·해결됨.
- 8명 표본을 원본에서 재추출한 **3,037/3,037 프레임이 저장된 학습 크롭과 비트 일치**. 새 FaceMesh 크롭도 이 표본에서 기존 fresh 크롭과 같았고 재추론 판정 변화 0. 검출 실패 자체를 해결한 것은 아님.
- `src/v2/deploy/inference_policy.py`: 창 19, 최소 유효 14, 현재 크롭 유효, 결측 시간 위치 유지, 계약 해시·임계값 검증.
- `run_video.py`, `run_video_power.py`, `pi_demo.py`: 해당 정책 사용. CNN 크롭 실패를 EAR로 대체하지 않음. 데모의 암묵적 임계값 0.5, hysteresis·wall-clock 기간 필터를 제거하고 고정 임계값 연속 양성 구간으로 후보를 센다. source frame sequence가 건너뛰면 결측 위치를 채운다.
- 계약 없는 CNN 데모는 명시적인 탐색용 `--blink-th` 없이 실행되지 않음. 새 모델 계약을 연결하는 것이 정상 사용법이다.
- 학습 이벤트 분류의 e_valid(min14)와 온라인 현재 프레임 유효 조건은 구분한다. 시간창 중심과 실제 출력에는 9프레임의 차이가 있다.
- **14개 테스트 통과** 및 두 NumPy 환경의 크롭 비교. `work/nonpi_2026-09-22_review/verification.json`, `verification.log`, `changes.patch`.

수정 전 소스는 `work/nonpi_2026-09-22_review/before/`에 있다. 완료된 maskfix 큐의 source hash와 현재 crop/trainer가 달라졌으므로 그 큐의 `prepare/run`을 현재 소스로 재시작하면 보호 검사에 걸리는 것이 정상이다. **완료된 60회 학습을 다시 시작하거나 freeze를 덮어쓰지 않는다.**

## 5. ③ ONNX·계약·전송 파일

기본 경로: `C:\Users\sch\PycharmProjects\IEEE_sensors\work\nonpi_2026-09-22_onnx`

| 파일/폴더 | 내용 |
|---|---|
| `corrected_onnx_pi_default.zip` | fold0/seed0 고정, 네 모델/8그래프, 약 2.63 MB |
| `corrected_onnx_all_60.zip` | 60개 모델/120그래프, 약 39.34 MB |
| `*.zip.sha256` | 전송 ZIP SHA-256 |
| `onnx/fold0_seed0/` | Ours encoder/head/contract, 비교군은 하위 `image_cnn_head/`, `vdrop/`, `gap/` |
| `packages/pi_default/verify_bundle.py` | 패키지 루트에서 독립 실행하는 장치 검사기 |
| `verification.json`, `packages.json`, `entrypoint_verification.json` | 변환·패키지·실제 실행 경로 검사 |
| `gpu_cpu_boundary_diagnosis.json` | TF32 경계 진단 |

모든 모델은 수정된 학습 checkpoint에서 만들었다. 모델별 source validation 임계값을 그대로 쓴다. 기본 fold0/seed0는 기존 공학 검증용 고정 선택이며 test 최고점 선택이 아니다. 전체 데이터 재학습 모델이 아니다.

FP32, ONNX opset 17/IR 8. 실제 이벤트·모델 3,346조합에서 CPU PyTorch→ORT 판정 차이 0, 최대 확률 차이 5.692244e-6. 저장 GPU→CPU 경계 판정 변화 4건(vdrop 3, GAP 1). 동일 batch 재현에서 TF32 on이면 저장 GPU 값과 정확히 일치, off면 CPU에 근접했다. 학습/임계값을 변경하지 않았다. 모든 하드웨어에서 bit-identical 판정을 보장하지 않는다.

Pi에 기본 ZIP을 새 폴더로 풀고 해당 폴더에서:

```bash
python verify_bundle.py --root . --out pi_model_verification.json
```

torch·카메라·학습 데이터 없이 수행 가능하다. 연구 PC ORT는 1.30.0, 저장소 Pi 요구 버전은 1.19.2이므로 **Pi에서 직접 검사해야 한다**. ZIP의 `local_verification.json`은 데스크톱 결과다. 기본 ZIP은 모델/계약/검사기이며 수정된 저장소 전체나 30 fps 측정 코드를 포함한 완성 측정 키트가 아니다.

## 6. 다음 PC 작업: Pi 실측 전 준비

이 목록은 **미완료 작업과 설계 제안**이다. 구현·검증된 기능이나 확정 실측 프로토콜로 오해하지 말 것.

1. **30 fps 고정 workload 재생**: 기존 `--fps 30`은 파일 free-run을 30 fps로 제한하지 않는다. monotonic clock으로 입력 시점을 관리하고, 대기 시간·실제 처리 지연·스케줄 지연·deadline miss/누락을 구분한다. 늦었을 때 프레임을 버리는지 모두 처리하는지도 사전에 고정한다. sleep을 e2e 처리 시간에 섞거나 free-run FPS를 실시간 성능으로 바꾸어 쓰지 않는다.
2. **동일 입력·동일 시간 비교**: 영상 해시, 프레임 수, 해상도, 30 Hz 시간축, 모델 해시, 유효 판정 비율을 기록한다. 짧은 영상 반복 시 이음매와 warmup/ring reset을 명시한다. 길이·내용이 다른 입력으로 모드별 에너지를 비교하지 않는다.
3. **측정 단계·반복 고정**: 모델 및 검출기 워밍업, idle 기준선, active 안정 구간, 반복 횟수, 모드 순서/냉각, 팬·governor·온도·throttling 기록을 문서와 runner에서 맞춘다. 예를 들어 active 300초 이상·3회 이상을 출발점으로 기존 설계와 대조하되 이것을 이미 확정한 값이라고 하지 않는다.
4. **전력 경계 정리**: PMIC rail 12개 합의 정의가 미검증이며 보드 전체 입력전력과 동일하다고 주장하지 않는다. Pi raw `vcgencmd pmic_read_adc` 로그와 rail 검증, 가능하면 외부 전력계 대조가 필요하다. watts와 J/frame, idle 차감 여부를 명시한다. free-run 절감과 30 fps 절감을 분리한다.
5. **전송 코드/명령 묶음**: ② 수정 사항과 30 fps runner, 모델 계약 자동 연결, 결과 검사기·실행 순서를 새 폴더/ZIP으로 묶는다. 기존 Pi 모델을 덮어쓰지 않는다. vdrop/GAP를 `ours`라는 모드명으로 실행해 Ours 측정치로 보고하지 않는다.
6. PC에서 fake clock/가짜 입력으로 pacing·deadline·EOF·결측 동작을 검증한다. 실제 ONNX의 짧은 데스크톱 동작 확인은 Pi 성능 수치와 구분한다. 장치 주소·계정·전력계는 아직 이 대화에서 확인되지 않았다.

준비 완료 뒤 Pi: 독립 모델 검사 → 수정 코드 연결 smoke check → raw PMIC/thermal 확인 → 고정 프로토콜 반복 실측 → 결과 QC. 기존 18 run의 Pi 결과를 새 모델/새 프로토콜 실측으로 재명명하지 않는다.

## 7. 병행할 미완료 연구 작업

### 외부 데이터

- 전체 EyeBlink8는 확보되지 않았으며 이전 공식 ZIP 요청은 HTTP 401이었다. 새 세션에서 접근 가능 여부를 재확인할 수 있지만 인증을 우회하지 않는다.
- 공식 안내: https://www.blinkingmatters.com/research
- 공식 ZIP: https://www.blinkingmatters.com/files/upload/research/eyeblink8.zip
- 로컬 영상: `data/_legacy_public/eyeblink8/eyeblink8/9/27122013_152435_cam.avi`
- 현재 외부 파일은 한 영상의 exploratory pilot. 전체 데이터·독립 피험자 검증으로 부르지 않는다. 이미 본 영상에서 임계값/후처리를 튜닝하고 untouched test라고 쓰지 않는다.
- EyeBlink8 기존 저장 크롭의 정책 재평가는 했지만 explicit float32 수정 후 외부 원본 전체 재추출은 이번 범위에 없었다.

### RNN 원자료

- 다른 PC에서 학습한 BiLSTM/LSTM의 최종 JSON·각 fold/seed 예측 NPZ·checkpoint·분할/학습 설정 필요.
- Downloads의 `표1_추가작업_결과.zip`, `전력측정_전달본.zip`은 확인 당시 코드/Markdown만 포함했고 raw JSON/NPZ/PT는 없었다. 다시 “거기에 원자료가 있다”고 추정하지 않는다.
- 프로젝트의 RNN ONNX/전력 결과 존재와 독립 정확도 검증 완료는 다르다. 자료가 없으면 재학습 또는 논문 비교 범위 조정이 필요하다.

### 인용·원고

`work/nonpi_2026-09-18/CITATION_AUDIT.md`에 주요 인용의 확인 기록이 있다. 일부 배경 인용은 원문 전체가 아닌 서지/요약 확인이라는 한계를 유지한다. 원고 수정은 사용자의 보류 방침을 따른다.

## 8. 실행 환경·보존 주의

- PowerShell, Python: `.venv-nonpi/Scripts/python.exe`.
- 수정 학습/검증: Python 3.12.14, NumPy 1.26.4, Torch 2.11.0+cu128, RTX 5080, OpenCV 4.11.0. ONNX 1.22.0, ORT 1.30.0.
- Windows 콘솔에서 한글/emoji 출력이 필요하면 해당 명령에서 `PYTHONIOENCODING=utf-8`을 사용한다.
- git에는 기존 사용자 변경과 다수의 untracked 파일이 있다. reset/clean/일괄 삭제 금지. 문서나 원고가 git modified라고 해서 이번 단계에서 수정한 것으로 단정하지 않는다.
- `work/nonpi_2026-09-21/experiment_queue.py`는 결측 입력 문제로 중단된 이전 큐다. 재시작하지 않는다. `nonpi_2026-09-21_maskfix`도 이미 60회 완료됐다.
- 9/21 완료 알림 자동화를 만들었지만, 9/22 조회 시 앱에서 사라져 있었다. 현재 알림이 유지된다고 약속하지 않는다. 새로운 장시간 작업을 시작하면 실제 프로세스/로그와 알림 설정을 따로 확인한다.
- 새 세션에서는 이 문서와 `current_inventory.json`의 확인 시점을 보고, 실행 전에 해당 파일/소스 해시가 그대로인지 점검한다. 이 자료를 만들면서 GPU 학습이나 Pi 작업을 새로 시작하지 않았다.
