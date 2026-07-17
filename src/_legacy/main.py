"""main.py - 진입점 안내.

이 프로젝트는 작고 단일 목적인 모듈들로 구성된다.
일반적인 작업 흐름:

  1) 라벨된 깜빡임 수집 (웹캠 + mediapipe 필요):
        python collect.py --subject S01
        python collect.py --subject S02      # 여러 명 반복

  2) 학습 + leave-one-subject-out 평가:
        python train.py                       # data/blinks.csv 사용
        python train.py --synthetic           # 카메라 없이 전체 점검

  3) Liveness anti-spoofing 데모 (웹캠 필요):
        python liveness.py

먼저 train.py --synthetic 으로 오프라인 파이프라인을 점검하세요.
"""
import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
