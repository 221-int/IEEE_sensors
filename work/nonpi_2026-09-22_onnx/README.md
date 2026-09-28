# 수정 체크포인트 ONNX 배포 묶음 — 2026-09-22

Ours(vpres), Image-CNN+TCN, vdrop, GAP+TCN 각각 5 fold × 3 seed의 수정된 학습 체크포인트를 모두 FP32로 변환했다. 임의 가중치·이전 수정 전 체크포인트는 사용하지 않았다. D=16, 창 길이 19, gray 64×160, frame_standardize, source validation 임계값을 그대로 유지한다. 원고·기존 ONNX·기존 결과는 보존했다.

## 묶음 선택

- `corrected_onnx_pi_default.zip`: 사전에 공학 검증용으로 사용했던 **fold0/seed0 고정**, 네 모델. 최고 정확도 모델을 골라 만든 묶음이 아니다. Pi 연결 점검에는 이 묶음부터 사용한다.
- `corrected_onnx_all_60.zip`: 전체 60개 모델, 120개 ONNX 그래프와 모델별 계약. 다른 fold/seed 비교가 필요할 때 사용한다.

두 ZIP에는 카메라 데이터나 원본 눈 영상이 필요 없는 합성 입력 검사 fixture와 독립 검사 스크립트를 포함한다. 실제 mEBAL2 이벤트를 사용한 변환 검증의 상세 원자료는 연구 PC의 `verification.json`, 모델별 `verification_scores.npz`에 보존하며 전송용 ZIP에는 포함하지 않는다.

## 폴더 구조와 계약

```text
onnx/fold0_seed0/
  encoder.onnx              # Ours(vpres)
  head.onnx
  contract.json
  runtime_fixture.npz
  image_cnn_head/
    backbone.onnx
    head.onnx
    contract.json
    runtime_fixture.npz
  vdrop/                    # encoder.onnx + head.onnx + 계약/fixture
  gap/                      # encoder.onnx + head.onnx + 계약/fixture
verify_bundle.py
runtime/inference_policy.py
manifest.json
```

모든 계약은 모델 종류, fold/seed, 수정 학습 버전, 체크포인트/ONNX SHA-256, 텐서 규격, 정규화, 명시적인 float32 크롭 경계 규칙, logit/확률 임계값, 결측 판정 정책, 실제 검사 결과를 포함한다. 계약의 Windows 체크포인트 경로는 출처 기록이다. 실행 시 그 경로에 접근하지 않는다. 그래프는 계약 옆의 상대 파일명으로 읽는다.

ONNX는 opset 17, IR 8이며 head 그래프가 sigmoid 확률을 출력한다. **logit 임계값을 확률 출력에 직접 적용하면 안 된다.** 계약의 `threshold_probability`를 사용한다. 모델/seed마다 다르며 공통 0.5로 바꾸지 않는다.

현재 프레임 유효 + 19개 시간 위치 중 최소 14개 유효가 온라인 판정 조건이다. 누락 위치를 제거하거나 이전 프레임으로 채우지 않는다. 판정 불가 창도 그래프 자체는 유한한 값을 낼 수 있으므로 호출자가 반드시 보류해야 한다. 오프라인 창 중심 라벨과 출력 시각의 9프레임 차이는 별도 지연이며 그래프 실행 시간과 다르다.

## Pi에 옮긴 뒤 먼저 할 검사

기존 Pi Python 환경에서 ZIP을 새 폴더에 풀고 실행한다. 기존 `models/v2/onnx`를 덮어쓰지 않는다.

```bash
python verify_bundle.py --root . --out pi_model_verification.json
```

이 검사는 manifest·모델 계약의 해시, 실제 ONNX 로딩, encoder/head 출력 수치, 결측 판정 조건을 검사한다. torch·학습 데이터·카메라가 필요하지 않다. 실패하면 측정을 시작하지 말고 JSON/오류 로그로 호환성을 확인한다.

데스크톱 검증 환경은 ONNX Runtime 1.30.0 CPUExecutionProvider다. 저장소의 Pi 요구 파일은 1.19.2를 지정하므로 **이번 데스크톱 통과를 Pi 통과로 대체할 수 없다.** Pi에서 위 검사를 직접 통과한 뒤 측정한다. `local_verification.json`은 전송 전에 연구 PC에서 생성된 결과이며 Pi 결과가 아니다.

## 기존 실행 코드 연결

Pi 저장소에는 ②에서 수정된 `crop.py`, `inference_policy.py`, `run_video.py`, `run_video_power.py`, `pi_demo.py`를 먼저 반영해야 한다. 이번 ZIP은 저장소 전체를 교체하지 않는다. 아래는 Pi 저장소 루트에서 실행하는 예이며 `/path/to/bundle`을 압축을 푼 경로로 바꾼다.

```bash
# Ours 시연: 해당 폴더의 contract.json을 자동으로 읽음
python -m src.v2.deploy.pi_demo --source 0 --no-mirror \
  --onnx-dir /path/to/bundle/onnx/fold0_seed0

# 모델/계약 연결 확인용 짧은 실행. 논문용 지연·전력 측정이 아님.
python -m src.v2.deploy.run_video --mode ours --source 0 \
  --onnx-dir /path/to/bundle/onnx/fold0_seed0 \
  --intra-threads 2 --no-spin --duration 10 --out results/v2/corrected_ours_smoke.json
python -m src.v2.deploy.run_video --mode image_cnn_head --source 0 \
  --onnx-dir /path/to/bundle/onnx/fold0_seed0 \
  --intra-threads 2 --no-spin --duration 10 --out results/v2/corrected_image_head_smoke.json
```

vdrop/GAP는 추가 후보 변환물이다. 이들을 기존 `--mode ours` 이름으로 측정하고 Ours 측정치라고 보고하지 않는다. 기존 모드별 비용 주석도 후보마다 다르므로 향후 후보별 Pi 측정 연결에서 모델명·비용 기록을 명시해야 한다. 이번 범위에서는 기본 Ours/Image-CNN 연결과 모든 후보의 독립 ONNX 검사를 준비했다.

## 검증 범위와 남은 일

- 실제 held-out 이벤트의 클래스 균형 고정 표본, 임계값 근처 진단 표본, 결측 창을 합쳐 3,346개 이벤트·모델 조합 검사. 임계값 근처 표본 선택은 변환 오류 진단용이며 성능 추정용 표본이 아니다.
- PyTorch CPU→ONNX CPU의 최대 encoder 차이 0.000003815, 최대 확률 차이 0.000005693. 판정 차이 0건. mask 경계(0/1/5/6/18/19개 결측, 현재 프레임 결측), batch 1과 다중 batch, 결측 임베딩 무관성도 검사했다.
- 저장된 GPU 점수→CPU PyTorch 사이에서는 vdrop 3건, GAP 1건의 임계값 근처 판정 차이가 있었다. Ours/Image-CNN 검사 표본에서는 0건이다. ONNX와 CPU PyTorch의 판정은 일치하므로 이를 ONNX 변환 실패로 분류하지 않는다. GPU/CPU 수치 진단은 연구 PC의 `gpu_cpu_boundary_cases.json`과 `gpu_cpu_boundary_diagnosis.json`에 기록한다. 이 표본에서 계산한 4/3,346을 일반적인 불일치율로 사용하지 않는다.
- 새 학습·임계값 재조정·외부 데이터 최적화·양자화는 하지 않았다. 전체 외부 데이터, RNN 원자료, Pi의 실제 30 fps 지연/전력 재측정은 여전히 별도 작업이다. 기존 Pi 수치를 새 모델 측정치로 재명명하지 않는다.
