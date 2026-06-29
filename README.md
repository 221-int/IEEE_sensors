# 비접촉 의도-깜빡임 센싱 (Contactless Voluntary-Blink Sensing)

RGB 카메라 한 대로 **각 깜빡임**을 운동학(kinematics)만 보고
**의도(voluntary)** vs **무의식(spontaneous)** 으로 분류하고, 이를
**liveness / anti-spoofing** 프리미티브로 사용한다(챌린지-리스폰스:
"일부러 N번 깜빡이세요").

기존 피로/졸음 프로젝트와는 별개의 새 프로젝트다. 기여는 또 하나의 졸음
감지기가 아니라 *새로운 센싱 능력(깜빡임 의도)* 이다.

## 왜 센서 기여인가

- 비전 센서가 얼굴 미세 운동학으로부터 **새로운 측정값(깜빡임 의도)** 을
  만든다. 단순 "깜빡임/아님" 이 아님.
- **엣지 배포 가능**: 프레임워크 없는 NumPy 분류기 (TF/PyTorch 불필요).
- **강건성/적응**: 개인 캘리브레이션 + 머리자세 보정.
- **응용**: 사진/재생/딥페이크 스푸핑에 대한 카메라 liveness.

## 과학적 근거

의도 깜빡임은 보통 **더 길고, 더 완전히 감기고, 더 느리게 닫힌다**. 무의식
깜빡임은 **더 짧고, 불완전한 경우가 많고, 더 빠르다**. 각 깜빡임을 11개
운동학 특징(닫힘/valley/열림 지속시간, 진폭/완성도, 피크 닫힘/열림 속도,
닫힘-열림 비대칭, 면적결손)으로 기술하고 분류한다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 상수, 랜드마크 인덱스, 특징 스키마, 경로 |
| `detector.py` | MediaPipe 프론트엔드, EAR, 머리자세 보정, 캘리브레이션 |
| `blink_segmenter.py` | 스트리밍 깜빡임 검출 → 깜빡임별 샘플 |
| `features.py` | 깜빡임별 운동학 특징 벡터 (핵심 기술자) |
| `model.py` | 프레임워크 없는 NumPy 분류기 (엣지 배포용) |
| `train.py` | 학습 + **leave-one-subject-out** 평가, 지표 |
| `collect.py` | 라벨 데이터 수집 프로토콜 (voluntary/spontaneous) |
| `liveness.py` | 깜빡임 챌린지-리스폰스 anti-spoofing 데모 |
| `main.py` | 진입점 안내 |

## 방법론 (이전 프로젝트의 약점 해결)

1. **순환 라벨이 아니라 진짜 정답.** 라벨은 *수집 프로토콜* 에서 나온다
   (신호 깜빡임 = voluntary, 자유 시청 = spontaneous). 모델이 자기 규칙을
   다시 학습하는 구조가 아님.
2. **Leave-One-Subject-Out(LOSO)** 평가. 같은 피험자가 train/test 에 동시에
   안 들어가므로 정확도가 새 사람에 대한 일반화를 반영.

## 빠른 시작

```bash
pip install -r requirements.txt

# 1) 합성 데이터로 오프라인 파이프라인 점검 (카메라 불필요)
python train.py --synthetic

# 2) 실제 라벨 깜빡임 수집 (웹캠 + mediapipe + face_landmarker.task 필요)
python collect.py --subject S01
#    여러 명 반복 -> data/blinks.csv

# 3) 실제 데이터로 학습 + LOSO
python train.py

# 4) liveness anti-spoofing 데모
python liveness.py
```

라이브 프론트엔드를 쓰려면 MediaPipe 의 `face_landmarker.task` 를 프로젝트
루트에 받아두세요.

## 상태 / 할 일

- [x] 핵심 특징 추출, 세그먼터, 분류기, LOSO 학습
- [x] 합성 점검 (`train.py --synthetic`)
- [ ] `collect.py` / `liveness.py` 의 카메라/landmarker 루프 연결
      (`TODO` 표시; `detector.py` 의 호출을 미러링)
- [ ] 조명/안경 조건을 다양화한 다중 피험자 데이터 수집
- [ ] `evaluate.py` 추가: 특징분포 그림, 혼동행렬, 스푸핑 공격 평가
      (FAR/FRR/EER), 엣지 latency
- [ ] 선택: 반사(reflex) 깜빡임 3번째 클래스

## 계획된 실험 (논문용)

- 깜빡임 종류 분류: precision/recall/F1, **LOSO** 혼동행렬.
- 운동학 분리성: voluntary vs spontaneous 특징 분포.
- Liveness: 사진/재생/딥페이크 공격 대비 챌린지-리스폰스 성공률
  (FAR/FRR/EER).
- 엣지: latency, 모델 크기, 온디바이스 실현성.
