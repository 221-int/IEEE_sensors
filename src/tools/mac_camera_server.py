#!/usr/bin/env python3
"""맥북 웹캠 -> MJPEG 서버. 라즈베리파이 시연(`src/v2/deploy/pi_demo.py`)의 영상원.

    맥북(이 스크립트) ──HTTP MJPEG──▶ 라즈베리파이 5(처리·표시)

의존성은 **opencv 하나뿐**이다(나머지는 표준 라이브러리). Flask 를 쓰지 않는 것은
의도다 — 시연 당일 맥에 뭘 더 깔 일을 만들지 않는다.

  python3 -m pip install opencv-python
  python3 mac_camera_server.py                    # 기본 640x480 / 30fps / :8000
  python3 mac_camera_server.py --list             # 붙어 있는 카메라 인덱스 훑기
  python3 mac_camera_server.py --camera 1 --width 1280 --height 720

확인
  브라우저에서  http://<맥IP>:8000/   가 열리고 영상이 보이면 성공.
  파이에서      http://<맥IP>:8000/stream.mjpg   를 --source 로 준다.

macOS 주의
  - 첫 실행 때 카메라 권한 팝업을 **허용**해야 한다
    (시스템 설정 → 개인정보 보호 및 보안 → 카메라 → 터미널/IDE).
  - 방화벽이 "들어오는 연결을 허용하시겠습니까?" 를 물으면 **허용**.
  - 맥과 파이가 **같은 WiFi 대역**에 있어야 한다.
  - 절전으로 화면이 꺼지면 스트림도 끊긴다. 시연 전 `caffeinate -d` 를 띄워 둘 것.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frameboundary"


class Camera(threading.Thread):
    """카메라를 한 번만 열고 **최신 JPEG 한 장**만 들고 있는다.

    클라이언트마다 카메라를 열거나 인코딩을 반복하면 맥이 먼저 느려진다.
    캡처·인코딩은 이 스레드 하나가 하고, 모든 접속자는 같은 바이트를 가져간다.
    """

    def __init__(self, index: int, width: int, height: int, fps: int,
                 quality: int, flip: bool):
        super().__init__(daemon=True)
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.quality, self.flip = quality, flip
        self.jpeg: bytes | None = None
        self.seq = 0
        self.cond = threading.Condition()
        self.stop_evt = threading.Event()
        self.n_read = 0
        self.t0 = time.time()
        self.err = ""

    def run(self):
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            self.err = (f"카메라 {self.index} 를 열 수 없습니다. "
                        "--list 로 인덱스를 확인하고, macOS 카메라 권한을 허용하세요. "
                        "(아이폰 연속성 카메라가 0번을 차지했다면 --camera 1)")
            print(f"🔴 {self.err}")
            return
        real = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"✅ 카메라 {self.index} 열림 — 실제 해상도 {real[0]}x{real[1]}")
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        period = 1.0 / max(self.fps, 1)
        nxt = time.perf_counter()
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                self.err = "프레임 읽기 실패 — 카메라를 다른 앱이 쓰고 있는지 확인"
                time.sleep(0.3)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if not ok:
                continue
            with self.cond:
                self.jpeg = buf.tobytes()
                self.seq += 1
                self.n_read += 1
                self.cond.notify_all()
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()
        cap.release()

    def wait_frame(self, last_seq, timeout=5.0):
        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout)
            return self.jpeg, self.seq

    def stats(self):
        el = max(time.time() - self.t0, 1e-6)
        return {"frames": self.n_read, "fps": round(self.n_read / el, 1),
                "clients": Handler.clients, "error": self.err}


INDEX_HTML = """<!doctype html><meta charset=utf-8>
<title>맥북 카메라 서버</title>
<style>
 body{{background:#0e1116;color:#e6edf3;font:15px -apple-system,system-ui,sans-serif;
      margin:0;padding:28px}}
 h1{{font-size:19px;margin:0 0 4px}} p{{color:#8b949e;margin:0 0 18px}}
 img{{max-width:100%;border-radius:10px;border:1px solid #262c36}}
 code{{background:#161b22;padding:3px 7px;border-radius:5px;color:#4c8dff}}
</style>
<h1>맥북 카메라 서버 — 동작 중</h1>
<p>라즈베리파이에서:<br>
<code>python -m src.v2.deploy.pi_demo --source http://{ip}:{port}/stream.mjpg</code></p>
<img src="/stream.mjpg">
"""


class Handler(BaseHTTPRequestHandler):
    cam: Camera = None      # 클래스 변수로 주입
    clients = 0
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):       # 접속 로그로 터미널을 채우지 않는다
        pass

    def handle_one_request(self):
        """브라우저가 탭을 닫거나 새로고침하면 소켓이 그냥 끊긴다.

        기본 구현은 그걸 예외로 올려 **트레이스백을 터미널에 쏟는다.** 시연 중에
        붉은 글씨가 쏟아지면 진짜 문제와 구분이 안 되므로 여기서 삼킨다.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, TimeoutError, OSError):
            self.close_connection = True

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, TimeoutError, OSError):
            pass

    def do_GET(self):
        if self.path.startswith("/stream.mjpg"):
            return self._stream()
        if self.path.startswith("/snapshot.jpg"):
            return self._snapshot()
        if self.path.startswith("/health"):
            import json
            body = json.dumps(self.cam.stats()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = INDEX_HTML.format(ip=lan_ip(), port=self.server.server_address[1]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _snapshot(self):
        jpg, _ = self.cam.wait_frame(-1, timeout=5)
        if jpg is None:
            self.send_error(503, "no frame")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg)))
        self.end_headers()
        self.wfile.write(jpg)

    def _stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        Handler.clients += 1
        print(f"▶ 클라이언트 접속: {self.client_address[0]} (총 {Handler.clients})")
        last = -1
        try:
            while True:
                jpg, last = self.cam.wait_frame(last, timeout=5)
                if jpg is None:
                    continue
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            Handler.clients -= 1
            print(f"■ 클라이언트 종료: {self.client_address[0]} (총 {Handler.clients})")


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # 패킷은 나가지 않는다. 경로만 물어본다
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def probe_cameras(n=5):
    print("카메라 인덱스 훑는 중...")
    found = 0
    for i in range(n):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, f = cap.read()
            if ok:
                print(f"  [{i}] 사용 가능 — {f.shape[1]}x{f.shape[0]}")
                found += 1
            else:
                print(f"  [{i}] 열리지만 프레임 없음")
        cap.release()
    if found == 0:
        print("  없음 — macOS 카메라 권한(시스템 설정 → 개인정보 보호 및 보안 → "
              "카메라 → 터미널)을 확인하세요.")
    elif found > 1:
        print("\n  💡 여러 개가 잡혔다면 0번이 **아이폰(연속성 카메라)** 일 수 있습니다.")
        print("     맥북 내장 캠은 보통 그다음 번호입니다: --camera 1")
        print("     아이폰을 아예 빼려면 아이폰 설정 → 일반 → AirPlay 및 연속성 →")
        print("     연속성 카메라 끄기.")


def main() -> int:
    ap = argparse.ArgumentParser(description="맥북 웹캠 MJPEG 서버")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80, help="JPEG 품질 1~100")
    ap.add_argument("--flip", action="store_true",
                    help="서버에서 좌우 반전(파이 쪽 기본이 이미 반전이라 보통 불필요)")
    ap.add_argument("--list", action="store_true", help="카메라 인덱스만 훑고 종료")
    args = ap.parse_args()

    if args.list:
        probe_cameras()
        return 0

    cam = Camera(args.camera, args.width, args.height, args.fps,
                 args.quality, args.flip)
    cam.start()
    time.sleep(1.2)
    Handler.cam = cam

    ip = lan_ip()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print("─" * 66)
    print(f"  맥북 카메라 서버   http://{ip}:{args.port}/")
    print(f"  스트림             http://{ip}:{args.port}/stream.mjpg")
    print(f"  상태               http://{ip}:{args.port}/health")
    print()
    print("  라즈베리파이에서:")
    print(f"    python -m src.v2.deploy.pi_demo "
          f"--source http://{ip}:{args.port}/stream.mjpg")
    print("─" * 66)
    print("  Ctrl+C 로 종료")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        cam.stop_evt.set()
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
