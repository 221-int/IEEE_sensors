# 라즈베리파이5 실행 가이드 — run_live 로 fps 측정

목표: Pi에서 `run_live` 를 돌려 **FaceLandmarker fps(엣지 실현성)** 를 확인.
맥은 이미 카메라 MJPEG 서버(`run_server`) 로 켜져 있다고 가정. 맥 IP 예: `192.168.0.18`.

---

## 0. Pi 접속
- `--show`(영상 창)를 보려면 **VNC 데스크탑 안의 터미널**에서 실행해야 함.
- SSH만 쓰면 창이 안 뜨므로, `--show` 대신 `--seconds 30 --log run.csv` 로 숫자만 뽑는다.

## 1. 코드 가져오기 (데스크탑 -> Pi)
- git 원격이 있으면:  `git clone <저장소주소> ~/IEEE_sensors`
- 없으면 데스크탑에서 복사(scp):
  `scp -r <프로젝트경로>\src  pi@<PiIP>:~/IEEE_sensors/`
  (Pi에 `~/IEEE_sensors/src/eyeblink`, `~/IEEE_sensors/src/scripts` 가 있으면 됨)

## 2. face_landmarker.task 넣기  ★프로젝트 루트에 (src/ 아님)
`.gitignore` 의 `*.task` 때문에 clone 에 안 딸려온다. 직접 받는다.
```
cd ~/IEEE_sensors
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
ls -lh face_landmarker.task        # 수 MB 나오면 OK
```
(config 가 `~/IEEE_sensors/face_landmarker.task` 를 기대함 — 루트에 두어야 함.)

## 3. 의존성 설치 (Pi5 / Raspberry Pi OS Bookworm)
```
sudo apt update && sudo apt install -y python3-opencv libatlas-base-dev
python3 -m venv ~/eyeblink-env --system-site-packages
source ~/eyeblink-env/bin/activate      # 새 터미널마다 다시 실행
pip install mediapipe flask numpy
#   mediapipe 설치 실패 시 버전 지정:  pip install 'mediapipe==0.10.14'
```
- `--system-site-packages` 로 apt 의 python3-opencv 를 venv 에서 그대로 사용(빌드 회피).

## 4. 네트워크 확인
```
hostname -I                                  # Pi IP 가 192.168.0.x 인지
curl -s http://192.168.0.18:8000/ | head     # 맥 서버 응답 오는지(안 오면 서버/방화벽/IP 확인)
```

## 5. 실행
VNC 데스크탑 터미널 (영상 창 표시):
```
source ~/eyeblink-env/bin/activate
cd ~/IEEE_sensors/src
python -m scripts.run_live --source url --url http://192.168.0.18:8000/stream.mjpg --show
```
SSH만 (창 없이 30초 측정 + CSV):
```
source ~/eyeblink-env/bin/activate
cd ~/IEEE_sensors/src
python -m scripts.run_live --source url --url http://192.168.0.18:8000/stream.mjpg --seconds 30 --log run.csv
```
- 캘리브: 화면/안내에 따라 **눈 뜨고 대기** -> `CLOSE eyes` 뜨면 **꼭 감기**.
- 종료(q) 또는 30초 후 출력되는 다음 줄이 목표 숫자:
  `[profile] landmark: mean=..ms p95=..ms -> max ~X fps`

## 6. 트러블슈팅
- `[live] MJPEG 서버 접속 실패` -> 맥에서 run_server 켜져 있나 / IP·포트 맞나 / 같은 WiFi·방화벽 허용?
- `cv2.imshow` 관련 에러 (SSH) -> `--show` 빼고 `--seconds 30 --log run.csv` 로.
- `face_landmarker.task 모델 파일이 없습니다` -> 2단계 다시(루트에 두었는지).
- `mediapipe 가 필요합니다` / 설치 실패 -> 3단계, 버전 핀(`mediapipe==0.10.14`).
- 얼굴 인식 안 됨(계속 `no face`) -> 조명/거리(약 30~60cm) 조정.

## 결과 공유
`[profile] ... -> max ~X fps` 의 X, 그리고 `--log run.csv` 파일을 알려주면 다음 튜닝/실험으로 진행.
