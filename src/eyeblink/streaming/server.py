"""server — Flask 기반 MJPEG 스트리밍 서버.

카메라를 열어 프레임을 JPEG 로 인코딩하고 multipart/x-mixed-replace 로
스트리밍한다. 클라이언트(streaming.client)가 어디서든 프레임 단위로 받아
파이프라인에 넣을 수 있다(서버-클라이언트 구조 → 카메라 위치 무관·이식성↑).

의존성: flask, opencv-python.
실행: python -m scripts.run_server
"""
import cv2
from flask import Flask, Response

from eyeblink import config


def _mjpeg_generator(cap, quality):
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame, enc)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")


def build_app(camera=config.CAM_INDEX, width=config.CAM_WIDTH,
              height=config.CAM_HEIGHT, fps=config.CAM_FPS,
              quality=config.STREAM_QUALITY, path=config.STREAM_PATH):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    app = Flask(__name__)

    @app.route(path)
    def stream():
        return Response(_mjpeg_generator(cap, quality),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        return f'<h3>eyeblink MJPEG server</h3><img src="{path}">'

    return app, cap


def serve(host=config.STREAM_HOST, port=config.STREAM_PORT, **kw):
    app, cap = build_app(**kw)
    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        cap.release()
