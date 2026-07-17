"""
liveness.py — 깜빡임 챌린지-리스폰스 anti-spoofing 데모 (뼈대).

의도-깜빡임 센서를 보여주는 응용: 살아있는 사람임을 증명하기 위해(사진/
재생영상/딥페이크와 구별) 시스템이 CHALLENGE_N_BLINKS 번 의도적으로 깜빡이라고
요청하고, 관측된 깜빡임이 기대 운동학을 가진 VOLUNTARY 로 분류되는지 검증한다.

  - 사진          -> 깜빡임 자체가 없음            -> 거부
  - 재생영상      -> 깜빡임은 있으나 요청 시점/타이밍이 안 맞음
  - 딥페이크      -> 깜빡임은 있으나 신호에 맞춘 올바른 voluntary 운동학을
                     재현하기 어려움
  - 진짜 사용자   -> 올바른 타이밍의 voluntary 깜빡임 N회 -> 통과

이 파일은 뼈대다: 카메라/landmarker 루프는 collect.py 와 동일하게 TODO 로
남겨둠. 챌린지 상태 머신과 통과/거부 로직은 실제 동작하며,
`verify_blinks(...)` 로 단위 테스트가 가능하다.
"""
import time

import numpy as np

import config
from model import BlinkClassifier


class Challenge:
    """하나의 liveness 챌린지 추적: 제한 시간 내 voluntary 깜빡임 N회."""

    def __init__(self, clf, n=config.CHALLENGE_N_BLINKS,
                 timeout=config.CHALLENGE_TIMEOUT_SEC,
                 min_voluntary=config.CHALLENGE_MIN_VOLUNTARY):
        self.clf = clf
        self.n = n
        self.timeout = timeout
        self.min_voluntary = min_voluntary
        self.t_start = time.time()
        self.results = []          # (is_voluntary, proba) 리스트

    def submit(self, blink_event):
        """확정된 깜빡임을 분류하고 voluntary 여부를 기록."""
        proba = self.clf.predict_proba(blink_event.features[None, :])[0]
        v_idx = config.LABEL_MAP["voluntary"]
        is_vol = int(np.argmax(proba)) == v_idx
        self.results.append((is_vol, float(proba[v_idx])))
        return is_vol

    @property
    def expired(self):
        return time.time() - self.t_start > self.timeout

    def verdict(self):
        """'pass' | 'fail' | 'pending' 반환."""
        n_vol = sum(1 for ok, _ in self.results if ok)
        if n_vol >= self.min_voluntary:
            return "pass"
        if self.expired or len(self.results) > 2 * self.n:
            return "fail"
        return "pending"


def verify_blinks(clf, blink_events):
    """오프라인 헬퍼: BlinkEvent 리스트를 받아 판정 반환.

    녹화 클립에서 분리한 깜빡임을 넣어 라이브 루프 없이 스푸핑 공격을
    평가할 때 유용.
    """
    ch = Challenge(clf)
    for ev in blink_events:
        ch.submit(ev)
    n_vol = sum(1 for ok, _ in ch.results if ok)
    return ("pass" if n_vol >= ch.min_voluntary else "fail"), ch.results


def main():
    clf = BlinkClassifier.load()

    import cv2
    import mediapipe as mp
    import detector as D
    from blink_segmenter import BlinkSegmenter

    cap = cv2.VideoCapture(0)
    landmarker = D.build_landmarker()
    calib = D.Calibrator()
    seg = None
    challenge = None
    last_verdict = ""

    print(f"[liveness] 캘리브레이션 중; 눈 뜨고 기다리세요...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        ts_ms = int(now * 1000)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_image, ts_ms)

        if not res.face_landmarks:
            cv2.imshow("liveness", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        lms = res.face_landmarks[0]
        ear = D.frame_ear(lms, w, h)

        if not calib.done:
            prog = calib.update(ear)
            cv2.putText(frame, f"Calibrating {int(prog*100)}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.imshow("liveness", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        if seg is None:
            seg = BlinkSegmenter(calib.baseline, calib.threshold)
        if challenge is None:
            print(f">>> 지금 {config.CHALLENGE_N_BLINKS}번 깜빡이세요")
            challenge = Challenge(clf)

        ev = seg.update(now, ear)
        if ev is not None:
            challenge.submit(ev)

        v = challenge.verdict()
        if v != "pending":
            last_verdict = v.upper()
            print(f"[liveness] 판정: {last_verdict}")
            challenge = None                 # 다음 시도를 위해 재시작

        # ── 화면 피드백 ──
        n_vol = sum(1 for ok_, _ in challenge.results if ok_) if challenge else 0
        cv2.putText(frame, f"Blink {config.CHALLENGE_N_BLINKS}x  "
                    f"voluntary={n_vol}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        if last_verdict:
            col = (0, 200, 0) if last_verdict == "PASS" else (0, 0, 255)
            cv2.putText(frame, last_verdict, (30, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
        cv2.imshow("liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
