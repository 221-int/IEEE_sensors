"""streaming — MJPEG 서버-클라이언트 (카메라 위치와 무관한 프레임 소스).

  server  : 카메라를 열어 MJPEG 로 스트리밍 (노트북/PC 또는 라즈베리파이).
  client  : MJPEG URL 에서 프레임(ndarray, BGR)을 읽는 제너레이터.

server 는 flask+opencv, client 는 opencv 가 필요하므로, 무거운 의존성이 패키지
import 시점에 걸리지 않도록 여기서는 하위 모듈을 자동 import 하지 않는다.
필요할 때 `from eyeblink.streaming import server` / `client` 로 가져온다.
"""
