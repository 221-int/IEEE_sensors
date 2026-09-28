"""라즈베리파이 5 실시간 시연 화면 — 발표자료(도 9 '실증 결과') 캡처용.

    맥북(카메라 서버) ──MJPEG──▶ 라즈베리파이 5(처리·표시)

  # Pi 에서 (저장소 루트에서 실행)
  cd ~/IEEE_sensors && conda activate eyeblink
  python -m src.v2.deploy.pi_demo --source http://<맥IP>:8000/stream.mjpg

화면 구성 — 흰색 카드·파란색 포인트의 네이티브 대시보드
--------------------------------------------------
  사람 실루엣 + 사각형 안에 실시간 눈 크롭 영상만 표시
  ① 깜빡임 검출 (head.onnx, 창 19, 매 프레임)
  ② 사용자 식별 (같은 임베딩의 코사인 유사도)
  ③ 사용자별 누적 이력 · 개인 기준선 · 권고
  하단 한 줄: 처리 FPS만 표시

  원본 얼굴 영상 대신 눈 크롭만 표시한다. 나머지 얼굴·몸은 단순한 도형이다.
  추론 경로(크롭 -> encoder -> head)는 그대로 사용한다.

🔴 이 화면의 숫자는 **시연용**이다. 논문 Table II 의 측정치는
   `src/v2/deploy/run_video.py` 로 따로 재며, 이 스크립트는 렌더링 비용이 섞여 있어
   그 자리에 쓰지 않는다. --preview의 FPS는 예시이며 실행 중에는 측정값을 표시한다.

조작키
------
  q / ESC   종료(프로필 저장)      f  전체화면 토글        s  화면 캡처 저장(PNG)
  1..6      현재 사람을 그 번호 사용자로 **등록**(재등록도 같은 키)
  r         식별 버퍼 초기화(사용자 교체 시연)
  c         이번 세션 카운터 초기화(누적 이력은 유지)
  x         등록 프로필 **전체 삭제**(비휘발성 파일까지)
  [ / ]     식별 임계값 -0.01 / +0.01
  깜빡임 임계값은 모델 계약에서 읽고 실행 중 고정한다.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

# 저장소 루트를 sys.path 에 넣어 `python src/v2/deploy/pi_demo.py` 로도 돌게 한다.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2  # noqa: E402

from src.v2.dataset import crop as C  # noqa: E402
from src.v2.deploy import inference_policy as IP  # noqa: E402

EVENT_LEN = 19          # head 의 시간창. run_video.py 와 같은 값이어야 한다
EMB_DIM = 16            # encoder.onnx 출력 차원
COMMON_BASELINE_BPM = 15.0   # 관측 시간 부족 구간에 잠정 적용하는 공통 초기값

# ------------------------------------------------- 색 (BGR) — 라이트 테마
BG      = (249, 247, 244)          # 바탕: 흰색
PANEL   = (249, 250, 250)          # 아주 옅은 면(빈 영역에만)
TRACK   = (243, 244, 246)          # 게이지 트랙
LINE    = (235, 231, 229)          # 기본 선
LINE2   = (219, 213, 209)          # 조금 진한 선
FG      = (39, 24, 17)             # 본문
FG2     = (81, 65, 55)
MUTED   = (128, 114, 107)
BLUE    = (221, 105, 48)
GREEN   = (74, 163, 22)
AMBER   = (6, 119, 217)
RED     = (38, 38, 220)
VIOLET  = (237, 58, 124)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansKR-Regular.otf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
]


# ================================================================== 그리기
class Painter:
    """한글 텍스트를 프레임당 **한 번의 PIL 변환**으로 몰아 그린다.

    매 문자열마다 numpy<->PIL 을 왕복하면 Pi 에서 렌더링만 수십 ms 가 된다.
    큐에 모았다가 `flush()` 에서 한 번에 그린다.
    """

    def __init__(self, font_path: str | None = None, bold_path: str | None = None):
        self.ok = False
        self.ko = False
        self._q: list = []
        self._cache: dict = {}
        try:
            from PIL import ImageFont  # noqa: F401
            self._ImageFont = ImageFont
            self.ok = True
        except Exception:
            self._ImageFont = None
        if not self.ok:
            return
        self.regular = font_path or self._find(FONT_CANDIDATES)
        self.bold = bold_path or self._find(BOLD_CANDIDATES) or self.regular
        self.ko = self.regular is not None
        if not self.ko:
            self.ok = False

    @staticmethod
    def _find(cands):
        for p in cands:
            if os.path.exists(p):
                return p
        # fontconfig 로 한 번 더 찾아본다
        try:
            out = subprocess.check_output(
                "fc-match -f '%{file}' 'Nanum Gothic'", shell=True, text=True,
                stderr=subprocess.DEVNULL, timeout=3).strip()
            if out and os.path.exists(out):
                return out
        except Exception:
            pass
        return None

    def _font(self, size: int, bold: bool):
        key = (size, bold)
        f = self._cache.get(key)
        if f is None:
            path = self.bold if bold else self.regular
            try:
                f = self._ImageFont.truetype(path, size)
            except Exception:
                f = self._ImageFont.load_default()
            self._cache[key] = f
        return f

    def text(self, x, y, s, size=16, color=FG, bold=False, anchor="lt"):
        if s is None:
            return
        self._q.append((int(x), int(y), str(s), int(size), color, bool(bold), anchor))

    def flush(self, img):
        if not self._q:
            return img
        if not self.ok:                      # PIL/한글폰트 없음 -> ASCII 근사
            for x, y, s, size, color, bold, anchor in self._q:
                sc = size / 30.0
                s2 = s.encode("ascii", "replace").decode()
                (tw, th), _ = cv2.getTextSize(s2, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
                if anchor[0] == "r":
                    x -= tw
                elif anchor[0] == "m":
                    x -= tw // 2
                cv2.putText(img, s2, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, sc,
                            color, 2 if bold else 1, cv2.LINE_AA)
            self._q.clear()
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for x, y, s, size, color, bold, anchor in self._q:
            d.text((x, y), s, font=self._font(size, bold),
                   fill=(color[2], color[1], color[0]), anchor=anchor)
        self._q.clear()
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def rrect(img, x, y, w, h, r, color, thickness=-1):
    x, y, w, h, r = int(x), int(y), int(w), int(h), int(r)
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy, a in ((x + r, y + r, 180), (x + w - r, y + r, 270),
                          (x + r, y + h - r, 90), (x + w - r, y + h - r, 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, -1)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
        for cx, cy, a in ((x + r, y + r, 180), (x + w - r, y + r, 270),
                          (x + r, y + h - r, 90), (x + w - r, y + h - r, 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness, cv2.LINE_AA)


def fit_into(frame, w, h):
    """레터박스 없이 중앙 크롭해서 (w,h) 에 채운다."""
    fh, fw = frame.shape[:2]
    s = max(w / fw, h / fh)
    nw, nh = int(np.ceil(fw * s)), int(np.ceil(fh * s))
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w], s, x0, y0


# ================================================================== 입력
class StreamReader(threading.Thread):
    """항상 **최신 프레임만** 들고 있는 리더.

    MJPEG 를 동기로 읽으면 디코딩이 느린 쪽에 큐가 쌓여 화면이 몇 초씩 밀린다.
    별도 스레드로 계속 비우고 마지막 프레임만 남긴다. 끊기면 스스로 재접속한다.
    """

    def __init__(self, src, width=640, height=480, fps=30):
        super().__init__(daemon=True)
        self.src, self.width, self.height, self.fps = src, width, height, fps
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.connected = False
        self.err = ""
        self.stop_evt = threading.Event()

    def _open(self):
        src = int(self.src) if str(self.src).isdigit() else self.src
        cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def run(self):
        cap = None
        while not self.stop_evt.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self._open()
                if not cap.isOpened():
                    self.connected, self.err = False, f"열 수 없음: {self.src}"
                    time.sleep(1.5)
                    continue
                self.connected, self.err = True, ""
            ok, f = cap.read()
            if not ok:
                self.connected, self.err = False, "스트림 끊김 — 재접속 중"
                cap.release()
                cap = None
                time.sleep(0.8)
                continue
            with self.lock:
                self.frame = f
                self.seq += 1
        if cap is not None:
            cap.release()

    def latest(self, last_seq, timeout=1.0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            with self.lock:
                if self.seq != last_seq and self.frame is not None:
                    return self.frame, self.seq
            time.sleep(0.002)
        return None, last_seq

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=2)


# ================================================================== 상태
def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL, timeout=3).strip()
    except Exception:
        return None


def device_line():
    model = None
    try:
        with open("/proc/device-tree/model", "rb") as f:
            model = f.read().decode(errors="ignore").strip("\x00").strip()
    except Exception:
        pass
    if not model:
        model = f"{platform.system()} {platform.machine()}"
    py = platform.python_version()
    try:
        import onnxruntime as ort
        ortv = ort.__version__
    except Exception:
        ortv = "?"
    return f"{model} · Python {py} · onnxruntime {ortv} · CPU only"


def read_temp():
    out = _sh("vcgencmd measure_temp")
    if out and "=" in out:
        try:
            return float(out.split("=")[1].split("'")[0])
        except Exception:
            return None
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_throttled():
    out = _sh("vcgencmd get_throttled")
    return out.split("=")[1] if out and "=" in out else None


def pct(v, q):
    return float(np.percentile(np.asarray(v), q)) if len(v) else None


class Profiles:
    """사용자별 임베딩 중심 · 누적 이력 · 개인 기준선. **비휘발성 파일로 보존**한다.

    발표자료 §핵심구성② 의 '세션 종료 후에도 이력·기준선 유지, 재식별 시 연속 누적'
    을 그대로 시연하기 위한 최소 구현이다.
    """

    def __init__(self, path):
        self.path = path
        self.users: dict = {}
        self.load()

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.users = d.get("users", {})
            for u in self.users.values():
                u["centroid"] = np.asarray(u["centroid"], np.float32)
        except Exception:
            self.users = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            d = {"saved_at": datetime.now().isoformat(timespec="seconds"),
                 "users": {k: {**v, "centroid": [float(x) for x in v["centroid"]]}
                           for k, v in self.users.items()}}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"프로필 저장 실패: {e}")

    def enroll(self, slot, name, vec):
        old = self.users.get(slot, {})
        self.users[slot] = {
            "name": name,
            "centroid": np.asarray(vec, np.float32),
            "blinks_total": int(old.get("blinks_total", 0)),
            "observed_sec": float(old.get("observed_sec", 0.0)),
            "enrolled_at": old.get("enrolled_at",
                                   datetime.now().isoformat(timespec="seconds")),
            "last_seen": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()

    def identify(self, vec, th):
        """-> (slot, sim, 2등 sim). 최고 유사도가 임계 미만이면 slot=None(미등록)."""
        if vec is None or not self.users:
            return None, 0.0, 0.0
        sims = []
        for k, u in self.users.items():
            c = u["centroid"]
            n = float(np.linalg.norm(c) * np.linalg.norm(vec)) + 1e-9
            sims.append((float(np.dot(c, vec) / n), k))
        sims.sort(reverse=True)
        best, second = sims[0], (sims[1] if len(sims) > 1 else (0.0, None))
        return (best[1] if best[0] >= th else None), best[0], second[0]

    def baseline(self, slot, min_sec):
        u = self.users.get(slot)
        if not u or u["observed_sec"] < min_sec:
            return COMMON_BASELINE_BPM, False
        bpm = u["blinks_total"] / max(u["observed_sec"] / 60.0, 1e-6)
        return float(np.clip(bpm, 3.0, 40.0)), True


# ================================================================== 렌더
def hline(img, x0, x1, y, color=LINE):
    cv2.line(img, (int(x0), int(y)), (int(x1), int(y)), color, 1, cv2.LINE_AA)


def vline(img, x, y0, y1, color=LINE):
    cv2.line(img, (int(x), int(y0)), (int(x), int(y1)), color, 1, cv2.LINE_AA)


def track(img, x, y, w, h, frac, color):
    """얇은 트랙 + 채움. 라운드는 h//2 로만 — 면을 최소로 쓴다."""
    rrect(img, x, y, w, h, h // 2, TRACK, -1)
    fw = int(max(0.0, min(1.0, frac)) * w)
    if fw > 1:
        rrect(img, x, y, max(fw, h), h, h // 2, color, -1)


def sect(img, ui, x, y, w, num, title, accent):
    """번호 + 제목 + 아래 얇은 선 하나. 카드(면) 대신 선으로만 구획한다."""
    ui.text(x, y + 9, num, 13, accent, bold=True, anchor="lm")
    ui.text(x + 26, y + 9, title, 16, FG, bold=True, anchor="lm")
    hline(img, x, x + w, y + 26)
    return y + 44


def color_eye_crop(frame, mesh, meta):
    """Render a color-only display crop and project landmarks into that crop.

    The model's grayscale crop and tensor are never changed or annotated.
    Warp directly to the display rectangle to avoid rotating the full frame.
    """
    if frame is None or mesh is None or meta is None:
        return None
    # Display only the middle 75% vertically; preserve width and pixel aspect.
    # The encoder still receives the original full-height grayscale crop.
    width, height = 320, 96
    display_height = meta.crop_h_px * 0.75
    matrix = cv2.getRotationMatrix2D(
        (meta.center_x, meta.center_y), meta.tilt_deg, 1.0)
    matrix[0, 2] -= meta.center_x - meta.crop_w_px / 2
    matrix[1, 2] -= meta.center_y - display_height / 2
    matrix[0] *= width / meta.crop_w_px
    matrix[1] *= height / display_height
    crop = cv2.warpAffine(frame, matrix, (width, height),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    points = np.asarray(mesh)[[33, 133, 362, 263, 159, 145, 386, 374], :2]
    projected = points @ matrix[:, :2].T + matrix[:, 2]
    for px, py in projected:
        if np.isfinite(px) and np.isfinite(py) and 0 <= px < width and 0 <= py < height:
            cv2.circle(crop, (round(float(px)), round(float(py))), 3, GREEN, -1, cv2.LINE_AA)
    return crop


def compose_eye_view(eye_crop=None):
    """Only the normalized eye crop enters the display; the rest is vector art."""
    view = np.full((540, 720, 3), (244, 236, 225), np.uint8)
    cv2.ellipse(view, (360, 650), (325, 320), 0, 0, 360, (205, 181, 152), -1)
    cv2.ellipse(view, (360, 245), (168, 215), 0, 0, 360, (228, 210, 191), -1)
    x, y, w, h = 200, 220, 320, 96
    if eye_crop is not None and eye_crop.size:
        crop = cv2.resize(eye_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        view[y:y + h, x:x + w] = crop
    else:
        view[y:y + h, x:x + w] = (237, 226, 213)
    cv2.rectangle(view, (x - 1, y - 1), (x + w, y + h), BLUE, 2, cv2.LINE_AA)
    return view


def render(canvas, ui, S):
    """Native 1280x720 dashboard; other canvases preserve its aspect ratio."""
    H, W = canvas.shape[:2]
    if (W, H) != (1280, 720):
        base = render(np.empty((720, 1280, 3), np.uint8), ui, S)
        scale = min(W / 1280, H / 720)
        sw, sh = max(1, round(1280 * scale)), max(1, round(720 * scale))
        canvas[:] = BG
        x, y = (W - sw) // 2, (H - sh) // 2
        canvas[y:y + sh, x:x + sw] = cv2.resize(base, (sw, sh), interpolation=cv2.INTER_AREA)
        return canvas

    canvas[:] = BG
    white = (255, 255, 255)
    pale_blue = (255, 242, 232)
    con = S['stream']['connected']
    valid = con and S['face'] and S['view'] is not None
    slot = S['ident_slot'] if valid else None
    preview = S.get('preview', False)

    def txt(x, y, text, size=14, color=FG, bold=False, anchor='lt'):
        ui.text(x, y, text, size, color, bold, anchor)

    def short(text, n=24):
        text = str(text)
        return text if len(text) <= n else text[:n - 1] + '…'

    def card(x, y, w, h):
        rrect(canvas, x, y, w, h, 14, white)
        rrect(canvas, x, y, w, h, 14, LINE, 1)

    def title(x, y, num, label):
        txt(x, y, num, 12, BLUE, True)
        txt(x + 28, y - 2, label, 17, FG, True)

    # Header: one title and an honest source/mode indicator.
    cv2.rectangle(canvas, (0, 0), (1280, 75), white, -1)
    rrect(canvas, 24, 18, 42, 40, 12, BLUE)
    cv2.ellipse(canvas, (45, 38), (14, 8), 0, 0, 360, white, 2, cv2.LINE_AA)
    cv2.circle(canvas, (45, 38), 4, white, -1, cv2.LINE_AA)
    txt(80, 15, '안구 건강 모니터링', 24, FG, True)
    txt(81, 47, 'EYE WELLNESS  /  사용자별 실시간 관리', 11, MUTED)
    state = '디자인 미리보기 · 예시 데이터' if preview else ('카메라 연결됨' if con else '카메라 연결 대기')
    cv2.circle(canvas, (990, 28), 4, BLUE if preview else (GREEN if con else AMBER), -1)
    txt(1004, 19, state, 13, BLUE if preview else FG2, True)
    mode = 'EAR 규칙 기반' if S.get('mode') == 'ear' else '단일 임베딩 · 검출 + 식별'
    txt(1254, 48, mode, 12, MUTED, anchor='rt')

    # Large live camera, with a dedicated status line outside the image.
    card(24, 92, 532, 470)
    title(42, 113, '01', '실시간 눈 크롭')
    txt(538, 114, '예시 화면' if preview else ('눈 영역 추적 중' if valid else '입력 대기'),
        12, BLUE if valid else MUTED, anchor='rt')
    if S['view'] is not None:
        canvas[148:523, 40:540] = cv2.resize(S['view'], (500, 375), interpolation=cv2.INTER_AREA)
    else:
        canvas[148:523, 40:540] = cv2.resize(compose_eye_view(), (500, 375), interpolation=cv2.INTER_AREA)
    if not valid:
        txt(290, 330, '입력 대기', 12, MUTED, anchor='mm')
        txt(290, 472, '눈 영역이 검출되면 표시됩니다', 14, FG2, anchor='mt')
    if S['blink_flash'] > 0 and valid:
        rrect(canvas, 424, 162, 100, 28, 10, BLUE)
        txt(474, 176, '깜빡임 감지', 12, white, True, 'mm')
    cap = '컬러 눈 영역' + (' · 좌우 반전' if S.get('mirror', True) else ' · 원본 방향')
    txt(42, 538, cap, 12, MUTED)
    txt(538, 538, '원본 컬러 · 눈 특징점', 12, MUTED, anchor='rt')

    # Three easily readable session metrics.
    metrics = [('이번 세션 깜빡임', str(S['session_blinks']), '회', BLUE),
               ('최근 60초 깜빡임', f"{S['bpm']:.1f}" if valid else '—', '회/분', FG),
               ('마지막 깜빡임 후', f"{S['since_blink']:.1f}" if valid else '—', '초',
                AMBER if valid and S['since_blink'] >= S['nb_alert'] else FG)]
    for x, (label, value, unit, color) in zip((576, 808, 1040), metrics):
        card(x, 92, 216, 108)
        txt(x + 18, 110, label, 13, MUTED)
        txt(x + 18, 136, value, 34, color, True)
        txt(x + 198, 158, unit, 13, MUTED, anchor='rt')

    card(576, 216, 680, 170)
    title(594, 236, '02', '깜빡임 검출')
    txt(1238, 237, f"판정 임계 {S['blink_th']:.2f}", 12, MUTED, anchor='rt')
    txt(594, 270, f"{S['prob']:.2f}" if valid and S.get('decision_valid', True) else '—', 38, BLUE, True)
    txt(596, 326, '깜빡임 확률', 12, MUTED)
    gx, gy, gw, gh = 752, 270, 486, 78
    for yy in (gy, gy + gh // 2, gy + gh):
        hline(canvas, gx, gx + gw, yy)
    thy = gy + gh - round(S['blink_th'] * gh)
    for xx in range(gx, gx + gw, 8):
        cv2.line(canvas, (xx, thy), (xx + 3, thy), LINE2, 1)
    hist = S['prob_hist']
    if len(hist) > 1:
        pts = np.array([(gx + round(i * gw / (len(hist) - 1)),
                         gy + gh - round(float(np.clip(v, 0, 1)) * gh))
                        for i, v in enumerate(hist)], np.int32)
        cv2.polylines(canvas, [pts], False, BLUE if valid else LINE2, 2, cv2.LINE_AA)
    txt(gx, 359, '최근 150프레임', 11, MUTED)
    txt(1238, 359, '현재' if valid else '입력 대기', 11, MUTED, anchor='rt')

    card(576, 402, 324, 230)
    title(594, 423, '03', '사용자 식별')
    name = S['ident_name'] if slot else ('미등록 사용자' if valid else '식별 대기')
    txt(594, 455, short(name, 13), 24, BLUE if slot else FG2, True)
    txt(594, 491, f"유사도 {S['ident_sim']:.3f}  /  임계 {S['id_th']:.2f}" if valid
        else '카메라에서 얼굴을 확인해 주세요', 12, MUTED)
    hline(canvas, 594, 882, 517)
    for i, (key, uname, sim) in enumerate(S['users'][:6]):
        x, y = 594 + (i % 2) * 146, 532 + (i // 2) * 25
        on = key == slot
        cv2.circle(canvas, (x + 3, y + 7), 3, BLUE if on else LINE2, -1)
        txt(x + 13, y, short(uname, 6), 11, FG if on else MUTED, on)
        txt(x + 133, y, f'{sim:.2f}' if valid else '—', 11, MUTED, anchor='rt')
    if not S['users']:
        txt(594, 538, '숫자키 1~6으로 사용자를 등록하세요', 12, MUTED)
    txt(594, 607, '식별된 사용자에게만 이력 누적', 11, BLUE)

    card(916, 402, 340, 230)
    title(934, 423, '04', '개인별 누적 이력')
    txt(934, 461, '누적 깜빡임', 12, MUTED)
    txt(1098, 461, '누적 관측', 12, MUTED)
    txt(934, 483, f"{S['u_blinks']:,}회" if slot else '—', 25, FG, True)
    txt(1098, 483, f"{S['u_obs'] / 60:.1f}분" if slot else '—', 25, FG, True)
    hline(canvas, 934, 1238, 523)
    txt(934, 539, '개인 기준선' if S['baseline_personal'] and slot else '공통 초기 기준선', 12, MUTED)
    txt(1238, 535, f"{S['baseline']:.1f} 회/분" if slot else '—', 18, BLUE, True, 'rt')
    ratio = S['bpm'] / max(S['baseline'], 1e-6) if slot else 0
    track(canvas, 934, 573, 304, 7, min(ratio, 1.5) / 1.5, BLUE)
    cv2.line(canvas, (1137, 568), (1137, 584), FG2, 1, cv2.LINE_AA)
    msg = ('현재 빈도 / 기준선  ' + f'{ratio:.0%}') if slot else '사용자 식별 후 이력을 불러옵니다'
    txt(934, 598, msg, 11, MUTED)
    if slot and not S['baseline_personal']:
        txt(1238, 614, '관측 누적 중 · 초기값 적용', 10, AMBER, anchor='rt')

    # Recommendation is always visible, without claiming a diagnosis.
    rrect(canvas, 24, 578, 532, 108, 14, pale_blue)
    txt(44, 592, '관리 안내', 14, BLUE, True)
    txt(536, 593, '개인 기준선 기반' if slot else '모니터링 준비', 12, BLUE, anchor='rt')
    advice = S['advice'] if valid and slot else ('사용자 등록 후 맞춤 안내를 시작합니다' if valid else '카메라 연결과 얼굴 위치를 확인해 주세요')
    # Wrap by rendered width so long alerts remain readable inside the card.
    lines, line = [], ''
    for char in advice:
        candidate = line + char
        width = ui._font(18, True).getlength(candidate) if ui.ok else len(candidate) * 18
        if width > 488 and line:
            lines.append(line.rstrip())
            line = char.lstrip()
        else:
            line = candidate
    if line:
        lines.append(line)
    for i, line in enumerate(lines[:2]):
        txt(44, 618 + i * 23, line, 18, FG2, True)
    txt(44, 666, '현재 깜빡임 상태를 개인 기준선과 비교해 안내합니다', 12, MUTED)

    # Keep the performance footer limited to the current measured frame rate.
    hline(canvas, 24, 1256, 696, LINE2)
    txt(24, 703, f"처리 {S['fps']:.1f} fps", 12, FG2, True)
    txt(160, 703, short(S['device'].split(' · Python')[0], 48), 11, MUTED)
    txt(1256, 703, '1~6 등록   r 식별초기화   c 세션초기화   s 캡처   f 전체화면   q 종료', 11, MUTED, anchor='rt')
    return ui.flush(canvas)


def preview_state():
    """Explicit synthetic design fixture; never opens cameras or writes profiles."""
    # Synthetic eye crop only for preview; runtime uses crop_both_eyes output.
    eye_crop = np.full((64, 160, 3), (174, 197, 219), np.uint8)
    for x in (43, 117):
        cv2.ellipse(eye_crop, (x, 33), (23, 10), 0, 0, 360, (233, 237, 241), -1, cv2.LINE_AA)
        cv2.ellipse(eye_crop, (x, 33), (23, 10), 0, 0, 360, (75, 85, 100), 1, cv2.LINE_AA)
        cv2.circle(eye_crop, (x, 33), 8, (70, 90, 115), -1, cv2.LINE_AA)
        cv2.circle(eye_crop, (x, 33), 4, (35, 38, 43), -1, cv2.LINE_AA)
        cv2.circle(eye_crop, (x - 2, 31), 2, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.ellipse(eye_crop, (x, 23), (24, 8), 0, 195, 345, (75, 85, 100), 2, cv2.LINE_AA)
        for point in ((x - 23, 33), (x + 23, 33), (x, 23), (x, 43)):
            cv2.circle(eye_crop, point, 2, GREEN, -1, cv2.LINE_AA)
    view = compose_eye_view(eye_crop[8:56])
    return dict(view=view, res='640×480', prob=.08,
                prob_hist=[.035 + .85 * np.exp(-((i - 37) / 4)**2) + .72 * np.exp(-((i - 103) / 5)**2) for i in range(150)],
                blink_th=.5, id_th=.85, session_blinks=42, bpm=16.8,
                since_blink=2.4, nb_alert=10, blink_flash=0, face=True, fps=14.7,
                ident_slot='user1', ident_sim=.932, ident_name='사용자 A',
                users=[('user1', '사용자 A', .932), ('user2', '사용자 B', .714), ('user3', '사용자 C', .658)],
                u_blinks=328, u_obs=1260, baseline=15.6, baseline_personal=True,
                advice='평소의 깜빡임 빈도를 유지하고 있어요',
                stages={}, e2e=(0, 0), temp=None, throttled=None,
                device='디자인 미리보기 · 장치 성능 미측정', src_label='예시 카메라',
                stream={'connected': True, 'err': ''}, preview=True, mode='ours', mirror=True)


# ================================================================== 메인
def main() -> int:
    ap = argparse.ArgumentParser(description="Pi 5 실시간 시연 화면")
    ap.add_argument("--source", default="0",
                    help="맥북 MJPEG URL (예: http://192.168.0.31:8000/stream.mjpg) "
                         "또는 로컬 카메라 인덱스")
    ap.add_argument("--onnx-dir", default="models/v2/onnx")
    ap.add_argument("--mode", default="ours", choices=["ours", "ear"],
                    help="ours = encoder+head, ear = 규칙 기반(모델 없이 화면 확인용)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--intra-threads", type=int, default=2)
    ap.add_argument("--no-spin", action="store_true", default=True)
    ap.add_argument("--spin", dest="no_spin", action="store_false")
    ap.add_argument("--canvas", default="1280x720", help="창 크기 WxH")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--mirror", action="store_true", default=True,
                    help="좌우 반전(셀카 뷰). --no-mirror 로 끔")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--blink-th", type=float, default=None,
                    help="Explicit exploratory threshold; otherwise use the graph-verified model contract")
    ap.add_argument("--model-contract", default=None)
    ap.add_argument("--id-th", type=float, default=0.85, help="식별 코사인 임계값")
    ap.add_argument("--id-window", type=int, default=90,
                    help="식별 벡터를 만들 때 평균낼 개안 프레임 수")
    ap.add_argument("--no-blink-alert", type=float, default=10.0,
                    help="무깜빡임 경고 초")
    ap.add_argument("--baseline-min-sec", type=float, default=60.0,
                    help="이 시간 이상 관측해야 개인 기준선으로 전환")
    ap.add_argument("--profiles", default="results/v2/demo_profiles.json")
    ap.add_argument("--shot-dir", default="results/v2/demo_shots")
    ap.add_argument("--names", default="사용자 A,사용자 B,사용자 C,사용자 D,사용자 E,사용자 F")
    ap.add_argument("--font", default=None, help="한글 TTF 경로(자동 탐색 실패 시)")
    ap.add_argument("--headless", action="store_true",
                    help="창 없이 실행(SSH 전용). --shot-every 로 PNG 만 남긴다")
    ap.add_argument("--shot-every", type=float, default=0.0,
                    help="N초마다 화면을 자동 저장(0=끔)")
    ap.add_argument("--cmd-file", default="",
                    help="이 파일에 키를 한 글자 써 넣으면 그 키를 누른 것으로 친다. "
                         "창 없이(--headless) 돌릴 때 다른 SSH 창에서 "
                         "`echo 1 > /tmp/demo_cmd` 처럼 조작한다")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="N초 후 자동 종료(0=무한)")
    ap.add_argument("--preview", metavar="PNG", help="카메라·모델 없이 예시 데이터 디자인 PNG 저장")
    args = ap.parse_args()

    try:
        W, H = (int(v) for v in args.canvas.lower().split("x"))
        if not (320 <= W <= 3840 and 180 <= H <= 2160):
            raise ValueError
    except ValueError:
        ap.error("--canvas 는 320x180 ~ 3840x2160 범위의 WxH 형식이어야 합니다")
    ui = Painter(args.font)
    if not ui.ko:
        print("⚠️  한글 폰트를 찾지 못했습니다. `sudo apt install fonts-nanum` 후 다시 "
              "실행하면 한글이 제대로 나옵니다 (지금은 물음표로 나옵니다).")

    if args.preview:
        out = render(np.empty((H, W, 3), np.uint8), ui, preview_state())
        os.makedirs(os.path.dirname(os.path.abspath(args.preview)), exist_ok=True)
        ok, data = cv2.imencode(".png", out)
        if not ok:
            raise RuntimeError("미리보기 PNG 인코딩 실패")
        data.tofile(args.preview)
        print(f"디자인 미리보기 저장 (예시 데이터): {args.preview}")
        return 0

    contract = None
    manual_blink_threshold = args.blink_th is not None
    if args.mode == "ours":
        contract = IP.load_contract(args.onnx_dir,
            [os.path.join(args.onnx_dir, n) for n in ("encoder.onnx", "head.onnx")], args.model_contract)
    if args.blink_th is None:
        if contract is not None:
            args.blink_th = contract["threshold_probability"]
        elif args.mode == "ear":
            args.blink_th = 0.5  # Explicitly a rule-demo operating point, not a trained threshold.
        else:
            ap.error("Model contract missing. Supply --model-contract or an explicit exploratory --blink-th; no implicit 0.5.")
    if not 0 < args.blink_th < 1:
        ap.error("--blink-th must be between 0 and 1")
    active_policy = IP.metadata(contract)
    active_policy["threshold_probability"] = args.blink_th
    if manual_blink_threshold:
        active_policy["threshold_source"] = "explicit exploratory CLI override"
    elif args.mode == "ear":
        active_policy["threshold_source"] = "EAR rule demo default; not a trained validation threshold"
    print("Inference policy:", json.dumps(active_policy, ensure_ascii=False))
    print("Active probability threshold:", args.blink_th)

    from src.v2.deploy.frontend import EyeFrontend

    # --- ONNX 세션
    enc = head = None
    if args.mode == "ours":
        import onnxruntime as ort

        def sess(path):
            so = ort.SessionOptions()
            if args.intra_threads:
                so.intra_op_num_threads = args.intra_threads
            if args.no_spin:
                so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                so.add_session_config_entry("session.inter_op.allow_spinning", "0")
            return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

        enc = sess(os.path.join(args.onnx_dir, "encoder.onnx"))
        head = sess(os.path.join(args.onnx_dir, "head.onnx"))

    fe = EyeFrontend(refine_landmarks=False)
    prof = Profiles(args.profiles)
    names = [s.strip() for s in args.names.split(",")]

    reader = StreamReader(args.source, args.width, args.height, args.fps)
    reader.start()

    win = "eye-wellness demo"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, W, H)
        if args.fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

    canvas = np.zeros((H, W, 3), np.uint8)
    ring: deque = deque(maxlen=EVENT_LEN)
    ear_ring: deque = deque(maxlen=EVENT_LEN)
    id_buf: deque = deque(maxlen=args.id_window)
    prob_hist: deque = deque(maxlen=150)
    stage_hist = {k: deque(maxlen=300) for k in
                  ("read", "detect", "crop", "encode", "head", "e2e")}
    fps_hist: deque = deque(maxlen=60)

    blink_times: deque = deque(maxlen=600)
    in_blink = False
    blink_start = 0.0
    session_blinks = 0
    last_blink_t = time.time()
    blink_flash = 0
    z_scale = 1.0
    prob = 0.0
    z = None
    g = None
    face = False
    ident_slot = None
    ident_sim = 0.0
    last_seq = -1
    meta = None
    t_prev = time.perf_counter()
    t_prev_wall = time.time()
    t_start = time.time()
    last_shot = 0.0
    t_temp = 0.0
    temp = None
    throttled = read_throttled()
    dev = device_line()
    last_save = time.time()

    src_label = (args.source if not str(args.source).isdigit()
                 else f"로컬 카메라 {args.source}")
    if src_label.startswith("http"):
        src_label = src_label.replace("http://", "").replace("/stream.mjpg", "")

    print(f"▶ {dev}")
    print(f"▶ 입력: {args.source}")
    print(f"▶ 프로필: {args.profiles} ({len(prof.users)}명 로드)")

    try:
        while True:
            t0 = time.perf_counter()
            decision_valid = False
            previous_seq = last_seq
            frame, last_seq = reader.latest(last_seq, timeout=0.5)
            t1 = time.perf_counter()
            now = time.time()

            if frame is not None:
                # The latest-frame reader may skip frames. Keep their positions
                # instead of compressing the 19-frame window in source time.
                skipped = max(0, last_seq - previous_seq - 1) if previous_seq >= 0 else 0
                IP.advance_missing(ring, skipped)
                for _ in range(min(skipped, EVENT_LEN)):
                    ear_ring.append(np.nan)
                if skipped:
                    in_blink = False
                decision_valid = False
                prob = 0.0
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                stage_hist["read"].append((t1 - t0) * 1e3)

                mesh = fe.mesh(frame)
                t2 = time.perf_counter()
                stage_hist["detect"].append((t2 - t1) * 1e3)
                face = mesh is not None

                if mesh is None:
                    # Missing faces must not retain a previous identity or blink.
                    id_buf.clear()
                    ident_slot, ident_sim = None, 0.0
                    prob, in_blink = 0.0, False
                    meta = None
                    ring.append(None)
                    ear_ring.append(np.nan)
                    g = None
                    stage_hist["e2e"].append((t2 - t1) * 1e3)
                else:
                    g, meta = fe.crop_from_mesh(frame, mesh)
                    x = C.to_input_tensor(g) if g is not None else None
                    t3 = time.perf_counter()
                    stage_hist["crop"].append((t3 - t2) * 1e3)

                    if args.mode == "ours" and x is not None:
                        z = enc.run(None, {"crop": x.astype(np.float32)})[0][0]
                        t4 = time.perf_counter()
                        stage_hist["encode"].append((t4 - t3) * 1e3)
                        ring.append(z)
                        z_scale = max(z_scale * 0.995,
                                      float(np.abs(z).max()) + 1e-6)
                        if IP.ring_ready(ring):
                            zz = np.stack([v if v is not None
                                           else np.zeros(EMB_DIM, np.float32)
                                           for v in ring])[None].astype(np.float32)
                            mk = np.array([[0.0 if v is None else 1.0 for v in ring]],
                                          np.float32)
                            prob = float(head.run(
                                None, {"vectors": zz, "mask": mk})[0].reshape(-1)[0])
                            decision_valid = True
                        t5 = time.perf_counter()
                        stage_hist["head"].append((t5 - t4) * 1e3)
                        stage_hist["e2e"].append((t5 - t1) * 1e3)
                        # 개안 프레임의 임베딩만 모아 식별 벡터를 만든다
                        if prob < 0.2:
                            v = z / (np.linalg.norm(z) + 1e-9)
                            id_buf.append(v.astype(np.float32))
                    elif args.mode == "ear":
                        e = C.ear_both(mesh)
                        ear_ring.append(e["mean"])
                        v = np.asarray(ear_ring, np.float64)
                        ok = np.isfinite(v)
                        if len(v) == EVENT_LEN and IP.eligible_mask(ok):
                            first = v[np.argmax(ok)]
                            last = v[len(v) - 1 - np.argmax(ok[::-1])]
                            edge = (first + last) / 2.0
                            lo = np.nanmin(np.where(ok, v, np.inf))
                            prob = float(np.clip((edge - lo) / (edge + 1e-9), 0, 1))
                            decision_valid = True
                        t4 = time.perf_counter()
                        stage_hist["head"].append((t4 - t3) * 1e3)
                        stage_hist["e2e"].append((t4 - t1) * 1e3)
                    else:
                        # Face exists but eye crop failed: abstain, never switch
                        # a CNN's validation threshold onto the EAR rule.
                        ring.append(None)
                        id_buf.clear()
                        ident_slot, ident_sim = None, 0.0
                        in_blink = False
                        stage_hist["e2e"].append((t3 - t1) * 1e3)

                prob_hist.append(prob)

                # Same contiguous-positive candidates as offline replay. Count
                # at onset; a window score is not a physical closure duration.
                in_blink, onset = IP.candidate_onset(
                    prob if decision_valid else None, args.blink_th, in_blink)
                if onset:
                    session_blinks += 1
                    blink_times.append(now)
                    last_blink_t = now
                    blink_flash = 8
                    if ident_slot:
                        prof.users[ident_slot]["blinks_total"] += 1

                # --- 식별
                if len(id_buf) >= max(10, args.id_window // 6):
                    m = np.mean(np.stack(id_buf), axis=0)
                    m = m / (np.linalg.norm(m) + 1e-9)
                    ident_slot, ident_sim, _ = prof.identify(m, args.id_th)
                    cur_vec = m
                else:
                    cur_vec = None
                    ident_slot, ident_sim = None, 0.0

                if ident_slot:
                    u = prof.users[ident_slot]
                    # 식별돼 있는 동안만 관측 시간을 그 사용자에게 귀속한다
                    u["observed_sec"] += max(0.0, min(0.2, now - t_prev_wall))
                    u["last_seen"] = datetime.now().isoformat(timespec="seconds")

            else:
                face = False
                id_buf.clear()
                ident_slot, ident_sim = None, 0.0
                prob, in_blink = 0.0, False
                ring.clear()
                ear_ring.clear()

            dt = time.perf_counter() - t_prev
            t_prev = time.perf_counter()
            t_prev_wall = now
            if dt > 0:
                fps_hist.append(1.0 / dt)
            if now - t_temp > 2.0:
                temp = read_temp()
                throttled = read_throttled()
                t_temp = now
            if now - last_save > 30 and prof.users:
                prof.save()
                last_save = now

            # --- 상태 묶기
            recent = [t for t in blink_times if now - t <= 60]
            elapsed_for_bpm = max(min(now - (blink_times[0] if blink_times else now - 60),
                                      60.0), 15.0)
            bpm = len(recent) * 60.0 / elapsed_for_bpm if recent else 0.0
            since = now - last_blink_t

            # Separate color display crop; g remains the untouched model input.
            eye_crop = color_eye_crop(frame, mesh, meta) if frame is not None and face and g is not None else None
            view = compose_eye_view(eye_crop) if frame is not None else None

            baseline, personal = prof.baseline(ident_slot, args.baseline_min_sec)
            if since >= args.no_blink_alert:
                advice = f"무깜빡임 {since:.0f}초 — 20-20-20 휴식 권고"
            elif ident_slot and bpm < baseline * 0.6:
                advice = "기준선 대비 깜빡임 감소 — 의식적 깜빡임 권고"
            elif ident_slot and bpm < baseline * 0.8:
                advice = "기준선 근처 — 주의 관찰"
            else:
                advice = "정상 범위"

            users_view = []
            if prof.users:
                mvec = (np.mean(np.stack(id_buf), axis=0) if len(id_buf) >= 10
                        else None)
                if mvec is not None:
                    mvec = mvec / (np.linalg.norm(mvec) + 1e-9)
                for k in sorted(prof.users):
                    u = prof.users[k]
                    s_ = 0.0
                    if mvec is not None:
                        c = u["centroid"]
                        s_ = float(np.dot(c, mvec) /
                                   (np.linalg.norm(c) * np.linalg.norm(mvec) + 1e-9))
                    users_view.append((k, u["name"], s_))

            S = {
                "view": view, "mode": args.mode, "mirror": args.mirror,
                "res": (f"{frame.shape[1]}×{frame.shape[0]}" if frame is not None
                        else f"{args.width}×{args.height}"),
                "prob": prob, "prob_hist": list(prob_hist),
                "decision_valid": decision_valid,
                "blink_th": args.blink_th, "id_th": args.id_th,
                "session_blinks": session_blinks, "bpm": bpm, "since_blink": since,
                "nb_alert": args.no_blink_alert,
                "blink_flash": blink_flash, "face": eye_crop is not None,
                "fps": float(np.mean(fps_hist)) if fps_hist else 0.0,
                "ident_slot": ident_slot, "ident_sim": ident_sim,
                "ident_name": (prof.users[ident_slot]["name"] if ident_slot else ""),
                "users": users_view,
                "u_blinks": (prof.users[ident_slot]["blinks_total"]
                             if ident_slot else 0),
                "u_obs": (prof.users[ident_slot]["observed_sec"] if ident_slot else 0.0),
                "baseline": baseline, "baseline_personal": personal,
                "advice": advice,
                "stages": {k: (pct(stage_hist[k], 50) or 0.0,
                               pct(stage_hist[k], 99) or 0.0)
                           for k in ("read", "detect", "crop", "encode", "head")},
                "e2e": (pct(stage_hist["e2e"], 50) or 0.0,
                        pct(stage_hist["e2e"], 99) or 0.0),
                "temp": temp, "throttled": throttled, "device": dev,
                "src_label": src_label,
                "stream": {"connected": reader.connected, "err": reader.err},
            }
            out = render(canvas, ui, S)
            blink_flash = max(0, blink_flash - 1)

            if args.shot_every > 0 and now - last_shot >= args.shot_every:
                os.makedirs(args.shot_dir, exist_ok=True)
                sp = os.path.join(args.shot_dir,
                                  f"demo_{datetime.now():%Y%m%d_%H%M%S}.png")
                cv2.imwrite(sp, out)
                last_shot = now
                print(f"📸 저장: {sp}")
            if args.duration > 0 and now - t_start >= args.duration:
                break

            k = 255
            if args.cmd_file and os.path.exists(args.cmd_file):
                try:
                    with open(args.cmd_file, "r", encoding="utf-8") as f:
                        c = f.read().strip()
                    open(args.cmd_file, "w").close()      # 한 번만 먹는다
                    if c:
                        k = ord(c[0])
                        print(f"⌨  명령 파일: '{c[0]}'")
                except Exception as e:
                    print(f"명령 파일 읽기 실패: {e}")

            if args.headless:
                if k == 255:
                    time.sleep(0.005)
            else:
                cv2.imshow(win, out)
                kk = cv2.waitKey(1) & 0xFF
                if kk != 255:
                    k = kk
            if k in (ord("q"), 27):
                break
            elif k == ord("f"):
                full = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(
                    win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if full == cv2.WINDOW_FULLSCREEN
                    else cv2.WINDOW_FULLSCREEN)
            elif k == ord("s"):
                os.makedirs(args.shot_dir, exist_ok=True)
                p = os.path.join(args.shot_dir,
                                 f"demo_{datetime.now():%Y%m%d_%H%M%S}.png")
                cv2.imwrite(p, out)
                print(f"📸 저장: {p}")
            elif ord("1") <= k <= ord("6"):
                if len(id_buf) >= 10:
                    m = np.mean(np.stack(id_buf), axis=0)
                    m = m / (np.linalg.norm(m) + 1e-9)
                    slot = f"user{chr(k)}"
                    prof.enroll(slot, names[k - ord("1")], m)
                    print(f"✅ 등록: {slot} = {names[k - ord('1')]}")
                else:
                    print("⚠️ 식별 벡터가 아직 안 모였습니다. 카메라를 정면으로 몇 초 보세요.")
            elif k == ord("r"):
                id_buf.clear()
                ident_slot, ident_sim = None, 0.0
                print("↺ 식별 버퍼 초기화")
            elif k == ord("c"):
                session_blinks = 0
                blink_times.clear()
                last_blink_t = time.time()
                print("↺ 세션 카운터 초기화")
            elif k == ord("x"):
                prof.users.clear()
                prof.save()
                print("🗑 프로필 전체 삭제")
            elif k == ord("["):
                args.id_th = round(max(0.0, args.id_th - 0.01), 3)
            elif k == ord("]"):
                args.id_th = round(min(1.0, args.id_th + 0.01), 3)
            elif k == ord("-"):
                print("Blink threshold is fixed for this run; restart with --blink-th for an exploratory override.")
            elif k == ord("="):
                print("Blink threshold is fixed for this run; restart with --blink-th for an exploratory override.")
    except KeyboardInterrupt:
        pass
    finally:
        if prof.users:
            prof.save()
        reader.stop()
        fe.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
