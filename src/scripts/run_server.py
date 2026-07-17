"""run_server — MJPEG 스트리밍 서버 실행 (카메라 → 스트림).

노트북/PC 에서 실행해 카메라 영상을 MJPEG 로 내보낸다. 라즈베리파이의
run_live 가 이 스트림을 받아 처리한다(또는 Pi 에 직접 카메라를 물려도 동일 구조).

실행 (src/ 에서):
    python -m scripts.run_server
    python -m scripts.run_server --camera 0 --width 1280 --height 720 --fps 30
"""
import argparse

from eyeblink import config
from eyeblink.streaming import server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=config.STREAM_HOST)
    ap.add_argument("--port", type=int, default=config.STREAM_PORT)
    ap.add_argument("--camera", type=int, default=config.CAM_INDEX)
    ap.add_argument("--width", type=int, default=config.CAM_WIDTH)
    ap.add_argument("--height", type=int, default=config.CAM_HEIGHT)
    ap.add_argument("--fps", type=int, default=config.CAM_FPS)
    ap.add_argument("--quality", type=int, default=config.STREAM_QUALITY)
    a = ap.parse_args()

    print(f"[server] streaming on http://{a.host}:{a.port}{config.STREAM_PATH}")
    server.serve(host=a.host, port=a.port, camera=a.camera,
                 width=a.width, height=a.height, fps=a.fps, quality=a.quality)


if __name__ == "__main__":
    main()
