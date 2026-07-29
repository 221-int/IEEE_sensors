# 엣지 배포 & PC↔라즈베리파이 5 성능 비교 가이드

목표: 현재 블링크 파이프라인을 라즈베리파이 5에서 돌리고, **PC와 동일 입력**에
대해 실시간 거동(지연시간·처리량·자원·발열)이 어떻게 달라지는지 측정한다.

## 핵심 개념 — "동일 성능"의 정의

분류기(TinyDNN / BlinkClassifier)는 순수 NumPy 결정론 연산이라, **같은 입력
프레임이면 PC와 파이에서 출력이 완전히 동일**하다. 따라서 비교할 것은 정확도가
아니라 **실시간 거동**이다:

- 단계별 지연시간: landmark 검출 / EAR / 세그먼터 / 분류기
- end-to-end 처리량(fps)과 지속 가능한 최대 fps
- CPU%·메모리(RSS)·SoC 온도(발열 throttling)

공정 비교를 위해 **웹캠 대신 같은 영상 파일**을 두 기기에 입력한다
(`benchmark.py --video clip.mp4`). 라이브 카메라는 프레임이 매번 달라 비교가
흐려진다.

## 1. 라즈베리파이 5 준비

- OS: **Raspberry Pi OS 64-bit (Bookworm)** 권장 (aarch64).
- 시스템 패키지:
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-opencv libatlas-base-dev
  ```
- 가상환경 + 파이썬 패키지:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install numpy psutil opencv-python mediapipe
  ```
  ※ 라즈베리파이 5(aarch64)는 pip에 mediapipe aarch64 휠이 있으나, 버전에 따라
  설치가 까다로울 수 있다. 실패 시 `mediapipe==0.10.x` 특정 버전을 시도하거나,
  MediaPipe Tasks용 대체 빌드를 확인한다.
- MediaPipe 모델 파일 `face_landmarker.task` 를 프로젝트 루트에 둔다.
- 헤드리스(모니터 없이) 실행: `benchmark.py --video`(창을 띄우지 않음)를 사용.

## 2. 비교 실험 절차

1. **테스트 영상 준비** — 깜빡임이 여러 번 든 짧은 클립(예: 60fps로 30초).
   PC와 파이가 *동일 파일*을 쓰도록 같은 clip을 양쪽에 복사.
2. **PC에서 기준 측정**:
   ```bash
   python benchmark.py --video clip.mp4
   ```
3. **라즈베리파이 5에서 동일 실행**:
   ```bash
   python benchmark.py --video clip.mp4
   ```
4. 두 출력의 표를 비교 → 단계별 latency·throughput·CPU·온도 차이가 결과.
5. **fps 다운샘플 ablation**(리뷰어의 저프레임률 지적 대응):
   ```bash
   python benchmark.py --video clip.mp4 --stride 1   # 원본
   python benchmark.py --video clip.mp4 --stride 2   # fps 절반
   python benchmark.py --video clip.mp4 --stride 4   # fps 1/4
   ```

## 3. 전력 측정 (선택)

- SoC 온도는 자동 로깅(`/sys/class/thermal/thermal_zone0/temp`).
- 소비전력은 소프트웨어로 정확히 못 재므로, **USB 전력계(인라인 파워미터)**로
  아이들 vs 파이프라인 구동 시 전력차를 측정하는 것을 권장.

## 4. 주의점

- **MediaPipe 앞단이 병목**이다. 파이에서 fps가 급락하면, (a) 입력 해상도 축소,
  (b) 프레임 스트라이드, (c) landmark 트래킹 간격 조정으로 대응하고 그 영향을
  ablation으로 보고한다.
- 파이는 발열로 클럭이 낮아질 수 있으니(throttling), 방열판/팬 유무와 온도를
  함께 기록한다.
- 파이가 도착하기 전에는 PC에서 `--mock`(harness 점검)과 `--video`(전체 계측)로
  전부 개발·검증해 두면, 파이에선 같은 명령만 실행하면 된다.

## 요약

| 단계 | 명령 |
|---|---|
| harness 점검(카메라 불필요) | `python benchmark.py --mock` |
| PC 기준 측정 | `python benchmark.py --video clip.mp4` |
| 파이 5 측정 | `python benchmark.py --video clip.mp4` (동일 파일) |
| fps ablation | `--stride 2`, `--stride 4` |
