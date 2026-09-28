# ③ 수정 체크포인트 ONNX·모델 계약 생성 완료

2026-09-22. 새 학습과 임계값 재선택 없이 수정된 4모델 × 5 fold × 3 seed 전부를 변환했다. 원고·기존 ONNX·기존 결과는 변경하지 않았다.

| 산출물 | 내용 |
|---|---|
| `corrected_onnx_pi_default.zip` | fold0/seed0 고정, 네 모델, 8개 그래프. 약 2.63 MB |
| `corrected_onnx_all_60.zip` | 60개 모델, 120개 그래프. 약 39.34 MB |
| `*.zip.sha256` | 전송 파일 무결성 검사값 |
| `README.md` | 폴더 구조, 모델 계약, Pi 검사 및 기존 코드 연결 방법 |
| `verification.json` | 실제 이벤트를 사용한 전체 변환 검증 및 기존 파일 보존 확인 |
| `packages.json` | 패키지별 파일 해시·크기와 독립 검사 결과 |
| `entrypoint_verification.json` | 실제 새 ONNX로 기존 Ours/Image-CNN 실행 경로 연결 확인 |

## 수치 검증

3,346개 held-out 이벤트·모델 조합에서 CPU PyTorch와 CPU ONNX Runtime을 비교했다. 클래스 균형 표본에 임계값 근처·결측 진단 표본을 추가했으므로 일반적인 오류율 추정 표본은 아니다.

- 최대 encoder 차이: 0.0000038146973.
- 최대 확률 차이: 0.0000056922436.
- **CPU PyTorch→ONNX 임계값 판정 변화: 0건.**
- ONNX checker, batch 1/다중 batch, 결측 수 0/1/5/6/18/19, 현재 프레임 결측, 결측 임베딩 무관성 검사 통과.
- 기본 묶음 4개, 전체 묶음 60개 모두 독립 `verify_bundle.py` 검사 통과. 빈 패키지·Windows 전용 경로·해시 없는 계약을 거부하는 회귀 검사도 통과.
- Ours와 Image-CNN은 기존 `run_video` 실행 루프에서 실제 새 ORT 세션을 사용해 계약·임계값·최종 판정 연결을 확인했다. 카메라와 랜드마크 공급자는 고정 held-out 크롭으로 모사했으며 실시간/전력 측정은 아니다.

## 저장 GPU 결과와 CPU의 경계 4건

vdrop의 3건, GAP의 1건이 저장된 GPU 점수와 CPU 재추론 사이에서 임계값을 넘어갔다. 검사 표본의 Ours/Image-CNN은 0건이다. 모든 ONNX 판정은 대응 CPU PyTorch와 일치했다.

`gpu_cpu_boundary_diagnosis.json`에서 원래 평가 batch=32를 복원해 같은 checkpoint를 CUDA로 추론했다. 네 사건 모두 **cuDNN TF32를 켜면 저장 GPU logit과 정확히 일치**했다. TF32를 끄면 CPU logit과의 차이는 최대 0.0000059605였다. 당시 환경 로그가 TF32 플래그를 직접 기록한 것은 아니므로 이는 네 경계 사건의 재현 증거이며 역사적 환경 전체를 완전히 증명하는 것은 아니다.

따라서 모든 하드웨어에서 bit-identical 판정을 보장한다고 주장하지 않는다. 임계값을 이 사건들에 맞춰 조정하지 않았고, 원래 점수 파일을 교체하지 않았다. 향후 Pi 평가에서 실제 FP/FN을 함께 측정해야 한다.

## Pi에서 이어갈 일

기본 묶음을 **새 폴더**에 풀고 그 폴더에서 실행한다.

```bash
python verify_bundle.py --root . --out pi_model_verification.json
```

이 스크립트는 torch·학습 데이터·카메라 없이 해시, ONNX 로딩, 합성 입력의 수치 일치, 결측 판정 정책을 검사한다. 현재 데스크톱은 ORT 1.30.0이고 저장소 Pi 환경은 ORT 1.19.2를 지정하므로 실제 Pi 검사가 남아 있다. 그래프 형식은 IR 8/opset 17이며 FP32다.

②의 수정된 배포 코드를 Pi 저장소에도 반영한 뒤 README의 Ours/Image-CNN 연결 예제를 사용한다. 후보 vdrop/GAP는 계약에 이름을 명시했고, 기존 Ours 모드 이름으로 수치를 잘못 보고하지 않도록 후보별 측정 연결은 구분해야 한다.

전체 외부 데이터 검증, RNN 원자료 검증, 30 fps Pi 지연·전력 재측정은 이번 작업에 포함하지 않았다. 양자화나 전체 데이터 재학습도 하지 않았다. 이번 완료 범위는 **변환·계약·전송 패키지·데스크톱 검증**이다.
