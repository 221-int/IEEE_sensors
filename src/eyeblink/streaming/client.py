"""client — MJPEG 스트림에서 프레임(ndarray, BGR)을 읽는 클라이언트.

서버(streaming.server) 또는 임의의 MJPEG URL 에 접속해 JPEG 경계를 파싱하고
OpenCV 로 디코딩한 프레임을 제너레이터로 내보낸다.

    from eyeblink.streaming.client import mjpeg_frames
    for frame in mjpeg_frames("http://192.168.0.10:8000/stream.mjpg"):
        ...   # frame: BGR np.ndarray

의존성: opencv-python (numpy 는 필수).
"""
import urllib.request

import numpy as np


def mjpeg_frames(url, chunk_size=4096, timeout=10):
    """MJPEG URL → BGR 프레임 제너레이터."""
    import cv2  # 지연 import (opencv 없이 패키지 import 가능하게)

    stream = urllib.request.urlopen(url, timeout=timeout)
    buf = b""
    while True:
        data = stream.read(chunk_size)
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
