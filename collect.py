"""
collect.py — 라벨링된 데이터 수집 프로토콜 (뼈대).

정답 voluntary/spontaneous 라벨을 *프로토콜*에서 생성한다. 이것이 이
데이터셋을 유효하게 만든다(순환 자기-라벨링 없음):

  VOLUNTARY 블록 : CUE_INTERVAL_SEC 마다 신호 표시; 피험자는 신호마다 한 번
                   깜빡인다. 신호 직후 짧은 창 안의 깜빡임을 VOLUNTARY 로 라벨.
  SPONTANEOUS 블록: 피험자가 SPONTANEOUS_SEC 동안 자유롭게 읽기/시청. 여기서의
                   깜빡임을 SPONTANEOUS 로 라벨.

두 블록이 N_BLOCKS 번 번갈아 진행된다. 검출된 각 깜빡임의 운동학 특징벡터를
(subject_id, label) 과 함께 data/blinks.csv 에 추가한다.

이 파일은 뼈대다: 카메라 루프는 스케치 + TODO 표시. 라벨링 로직과 CSV
기록은 실제로 동작한다.

실행:
    python collect.py --subject S01
"""
import argparse
import csv
import os
import time

import config

# 라이브 프론트엔드 import 는 main() 안에서 한다. 그래야 오프라인 파이프라인
# (features/model/train)이 opencv/mediapipe 없이도 동작한다.


def append_blink(subject_id, label, features, path=config.BLINK_CSV):
    """라벨된 깜빡임 1건을 데이터셋에 추가(새 파일이면 헤더 기록)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["subject_id", "label"] + config.FEATURE_NAMES)
        w.writerow([subject_id, label] + [f"{v:.6f}" for v in features])


class Protocol:
    """현재 블록을 추적하고, 확정된 깜빡임의 라벨을 결정."""

    def __init__(self):
        # 스케줄 구성: voluntary / spontaneous 블록을 번갈아
        self.blocks = []
        for _ in range(config.N_BLOCKS):
            self.blocks.append(("voluntary", config.N_VOLUNTARY_CUES
                                * config.CUE_INTERVAL_SEC))
            self.blocks.append(("spontaneous", config.SPONTANEOUS_SEC))
        self.t0 = time.time()
        self.last_cue_t = 0.0

    def phase(self, now):
        """('voluntary'|'spontaneous'|'done', 블록 내 경과초) 반환."""
        elapsed = now - self.t0
        acc = 0.0
        for kind, dur in self.blocks:
            if elapsed < acc + dur:
                return kind, elapsed - acc
            acc += dur
        return "done", 0.0

    def due_for_cue(self, now):
        """지금 새 voluntary 신호를 띄워야 하면 True."""
        kind, _ = self.phase(now)
        if kind != "voluntary":
            return False
        if now - self.last_cue_t >= config.CUE_INTERVAL_SEC:
            self.last_cue_t = now
            return True
        return False

    def label_for(self, blink_end_t):
        """확정된 깜빡임을 끝난 블록 기준으로 라벨링."""
        kind, _ = self.phase(blink_end_t)
        if kind == "voluntary":
            # 신호 직후의 깜빡임만 voluntary 로 인정
            if blink_end_t - self.last_cue_t <= config.CUE_INTERVAL_SEC:
                return config.LABEL_MAP["voluntary"]
            return None                      # 애매하면 버림
        if kind == "spontaneous":
            return config.LABEL_MAP["spontaneous"]
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="피험자 id, 예: S01")
    args = ap.parse_args()

    import cv2                               # TODO: pip install opencv-python
    import detector as D
    from blink_segmenter import BlinkSegmenter

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    landmarker = D.build_landmarker()
    calib = D.Calibrator()
    seg = None
    proto = Protocol()
    saved = 0

    print(f"[collect] subject={args.subject}. 캘리브레이션 중; 눈 뜨고 기다리세요...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        ts_ms = int(now * 1000)
        # TODO: frame 을 mediapipe 이미지로 변환 후 landmarker.detect_for_video 호출
        # res = landmarker.detect_for_video(mp_image, ts_ms)
        # if not res.face_landmarks: continue
        # lms = res.face_landmarks[0]
        # ear = D.frame_ear(lms, w, h)
        ear = None  # 위 프론트엔드를 연결하기 전까지의 placeholder

        if ear is None:
            # ── 위 landmarker 블록을 구현하면 이 가드는 삭제 ──
            cv2.imshow("collect", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if not calib.done:
            calib.update(ear)
            continue
        if seg is None:
            seg = BlinkSegmenter(calib.baseline, calib.threshold)

        # voluntary 신호 (TODO: 신호를 크게 표시 / 소리 재생)
        if proto.due_for_cue(now):
            print(">>> 지금 깜빡이세요 (BLINK NOW)")

        ev = seg.update(now, ear)
        if ev is not None:
            label = proto.label_for(ev.t_end)
            if label is not None:
                append_blink(args.subject, label, ev.features)
                saved += 1

        if proto.phase(now)[0] == "done":
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[collect] 완료. 라벨된 깜빡임 {saved}건 저장 -> {config.BLINK_CSV}")


if __name__ == "__main__":
    main()
