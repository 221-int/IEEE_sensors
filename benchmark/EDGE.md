# [폐기됨] 엣지 배포 & PC↔라즈베리파이 5 성능 비교 가이드

> **이 문서는 2026-07-29 부로 폐기되었다. 내용을 따르지 말 것.**
>
> 대체 문서:
> - 실행 절차 → [`docs/Pi_실행_가이드.md`](../docs/Pi_실행_가이드.md)
> - 측정 결과와 조건 → [`docs/PI5_BENCHMARK.md`](../docs/PI5_BENCHMARK.md)

## 왜 폐기했나

이 문서가 전제하던 코드와 구조가 더 이상 존재하지 않는다.

| 이 문서의 전제 | 현재 |
|---|---|
| `benchmark.py --video` / `--mock` 로 측정 | 해당 스크립트는 삭제된 `config` · `blink_segmenter` · `model` · `detector` 모듈을 import 하므로 **실행 불가**. 대체: `src/deploy/run_video.py` |
| 분류기 = NumPy 결정론 연산(TinyDNN) | 현재 판정 경로는 ONNX 인코더(128차원 벡터) + MLP. 랜드마크·EAR은 판정 경로에 없다 |
| `face_landmarker.task` (MediaPipe Tasks API) | 현재는 legacy `solutions.face_mesh` 사용. 외부 모델 파일 불필요 |
| Raspberry Pi OS Bookworm 기준 | 실제 장비는 Debian 13 (trixie), 시스템 파이썬 3.13 → **mediapipe 설치 불가**. conda로 3.11 환경 필요 |
| `apt python3-opencv` + `libatlas-base-dev` | conda 환경에서는 apt 패키지가 보이지 않는다. pip `opencv-contrib-python==4.11.0.86` 로 통일. libatlas는 불필요 |
| "MediaPipe 앞단이 병목, 파이에서 fps 급락 예상" | 실측 결과 MediaPipe는 7.76 ms(p50)로 예상보다 훨씬 빠르다. 해상도 축소·스트라이드 대응이 필요 없었다 |

## 살아남은 판단

원래 문서에서 옳았고 그대로 유지된 것:

- **웹캠이 아니라 동일 영상 파일로 비교할 것.** 라이브 카메라는 프레임이 매번 달라 비교가 흐려진다.
- **비교 대상은 정확도가 아니라 실시간 거동** — 단계별 지연시간, 처리량, CPU·메모리·발열.
- **소비전력은 소프트웨어로 정확히 못 잰다.** USB-C 인라인 전력계로 아이들 대비 구동 시 차이를 재는 것이 여전히 권장 방법이다. (미완)

실제로 PC↔Pi 수치 일치는 확인되었다 — 같은 입력에 대해 `blink_prob` 가 소수점 6자리까지 동일하다. 근거는 `PI5_BENCHMARK.md` §4.
