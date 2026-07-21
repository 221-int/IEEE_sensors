"""client — MJPEG 스트림에서 프레임(ndarray, BGR)을 읽는 클라이언트.

서버(streaming.server) 또는 임의의 MJPEG URL 에 접속해 JPEG 경계를 파싱하고
OpenCV 로 디코딩한 프레임을 제너레이터로 내보낸다. 접속/수신 오류는 원인과
확인사항을 담은 ConnectionError 로 변환해, 트레이스백만 찍고 죽지 않게 한다.

    from eyeblink.streaming.client import mjpeg_frames
    for frame in mjpeg_frames("http://192.168.0.20:8000/stream.mjpg"):
        ...   # frame: BGR np.ndarray

의존성: opencv-python (numpy 필수).
"""
import socket
import urllib.error
import urllib.request

import numpy as np


def mjpeg_frames(url, chunk_size=4096, timeout=10):
    """MJPEG URL -> BGR 프레임 제너레이터."""
    try:
        stream = urllib.request.urlopen(url, timeout=timeout)
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(
            f"Failed to connect to MJPEG server: {url}\n"
            "  Check: (1) is 'python -m scripts.run_server' running on the Mac?\n"
            "         (2) is --url IP:port correct? (Mac IP: ipconfig getifaddr en0)\n"
            "         (3) same WiFi, and firewall allows the python connection?\n"
            f"  (cause: {e})") from None

    import cv2  # 연결 성공 후 디코더 로드 (지연 import)
    buf = b""
    while True:
        try:
            data = stream.read(chunk_size)
        except (OSError, socket.timeout) as e:
            raise ConnectionError(f"MJPEG stream dropped while receiving: {e}") from None
        if not data:
            break
        buf += data
        soi = buf.find(b"\xff\xd8")       # JPEG 시작
        eoi = buf.find(b"\xff\xd9")       # JPEG 끝
        if soi != -1 and eoi != -1 and eoi > soi:
            jpg = buf[soi:eoi + 2]
            buf = buf[eoi + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame
