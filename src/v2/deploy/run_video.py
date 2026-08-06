"""T5-7 — v2 파이프라인 e2e 지연 하네스. Pi 5 실측용.

    # Pi (실측)
    python -m src.v2.deploy.run_video --mode ours --source 0 \
        --width 640 --height 480 --duration 300 \
        --intra-threads 2 --no-spin --out results/v2/pi_ours_480p.json

    # 데스크톱 (동작 확인용. 이 숫자는 논문에 쓰지 않는다)
    python -m src.v2.deploy.run_video --mode ours --source clip.mp4 --duration 10

🔴 `--intra-threads 2 --no-spin` 은 선택이 아니다
------------------------------------------------
ORT 스레드는 추론이 끝나도 계속 spin-wait 하며 CPU 를 물고 있어, 같은 코어에서
도는 디코딩·얼굴검출을 느리게 만든다. v1 에서 이걸 빼면 전체가 27% 느려졌고
그 상태의 EAR 비교는 무효다. 안 주면 이 스크립트가 결과 JSON 에 경고를 박는다.

head 를 **매 프레임**(stride 1) 돌린다
-------------------------------------
창이 찰 때마다 한 번(stride 19)만 판정하면 깜빡임을 최대 18프레임(600 ms @30fps)
늦게 잡는다. 연속 검출에서는 프레임마다 판정해야 한다.
비용은 **0.0459 MMAC/frame** 으로, 인코더 12.44 와 합쳐 12.49 MMAC/frame 이다
(head 비중 0.37%). 즉 stride 1 로 올려도 사실상 공짜다 — **병목 설계의 이득이
여기서 나온다.** 원본 픽셀을 시간축에 넣는 구조였다면 이 비용이 F² 로 커진다.

⚠️ head 는 **증분 캐싱이 안 된다.** TCN 이 `padding=dilation` 인 대칭 Conv1d 이고
pooling·BatchNorm 이 창 전체에 걸려서, 직전 프레임 계산을 재사용할 수 없다.
매 프레임 창 19개를 다시 돈다. 0.0459 MMAC/frame 은 그 값이다.

단계 분해를 반드시 함께 낸다
--------------------------
v1 640×480 에서 detect 가 e2e 의 약 67%(7.76/11.53 ms p50)였다. 해상도가 오르면
이 비중이 더 커져 **인코더 차이가 묻힌다.** 그래서 read/detect/crop/encode/head 를
따로 재서 보고한다 (EXPERIMENT_PLAN §6-3).

모드
----
  ours   크롭 -> encoder.onnx -> 링버퍼 19 -> head.onnx (stride 1)
  ear    같은 메시에서 EAR 스칼라 -> drop_ratio 규칙 (가중치 불필요)
         ⚠️ **ear_head(학습된 대조군)는 아직 못 돈다.** 확정 런이 earhead 가중치를
            저장하지 않았다(T3-6 에서 해결). 지금 재는 ear 는 **규칙 기반**이다.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from collections import deque

import numpy as np

from src.v2.common import repro
from src.v2.dataset import crop as C

OUT = "results/v2/pi_run.json"
ONNX_DIR = "models/v2/onnx"
EVENT_LEN = 19


# ------------------------------------------------------------------ 플랫폼 계측
def _sh(cmd: str) -> str | None:
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return None


def read_temp() -> float | None:
    out = _sh("vcgencmd measure_temp")                     # temp=53.6'C
    if out and "=" in out:
        try:
            return float(out.split("=")[1].split("'")[0])
        except (IndexError, ValueError):
            return None
    return None


def read_throttled() -> str | None:
    out = _sh("vcgencmd get_throttled")                    # throttled=0x0
    return out.split("=")[1] if out and "=" in out else None


def decode_throttled(hexstr: str | None) -> list[str]:
    """v1 규격 그대로. bit 2 = 지금 스로틀, bit 18 = 이번 부팅에 스로틀된 적 있음."""
    if not hexstr:
        return []
    try:
        v = int(hexstr, 16)
    except ValueError:
        return []
    bits = [(0, "under-voltage NOW"), (1, "arm-freq-capped NOW"),
            (2, "throttled NOW"), (3, "soft-temp-limit NOW"),
            (16, "under-voltage occurred"), (17, "arm-freq-capped occurred"),
            (18, "throttling occurred"), (19, "soft-temp-limit occurred")]
    return [name for bit, name in bits if v & (1 << bit)]


class Sampler(threading.Thread):
    """배경 표집 — 온도·스로틀·CPU%·RSS. 측정 자체를 방해하지 않게 1초 간격."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.temp, self.cpu, self.rss, self.flags = [], [], [], set()
        try:
            import psutil
            self.proc = psutil.Process()
            self.proc.cpu_percent(None)
        except ImportError:
            self.proc = None

    def run(self):
        while not self.stop_evt.wait(self.interval):
            t = read_temp()
            if t is not None:
                self.temp.append(t)
            self.flags.update(decode_throttled(read_throttled()))
            if self.proc is not None:
                self.cpu.append(self.proc.cpu_percent(None))
                self.rss.append(self.proc.memory_info().rss / 1e6)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=3)


def pct(v: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(v), q)) if v else None


def summarize(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "mean": float(a.mean())}


# ------------------------------------------------------------------ 세션
def make_session(path: str, intra: int, no_spin: bool):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if intra:
        so.intra_op_num_threads = intra
    if no_spin:
        # v1 규격. 이 두 줄이 27% 를 만든다.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ear_drop_ratio(buf: list[float]) -> float:
    """창 안 상대 하강. `train_encoder.Bundle.ear_feature` 와 **같은 식**이다.

    가용한 첫/마지막을 기준선으로 쓴다(고정 위치를 쓰면 그 프레임이 결측인 창에서
    베이스라인만 불리해진다).
    """
    v = np.asarray(buf, np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return 0.0
    first = v[np.argmax(ok)]
    last = v[len(v) - 1 - np.argmax(ok[::-1])]
    edge = (first + last) / 2.0
    lo = np.nanmin(np.where(ok, v, np.inf))
    if not np.isfinite(edge) or edge == 0:
        return 0.0
    return float(np.nan_to_num((edge - lo) / edge, nan=0.0, posinf=0.0, neginf=0.0))


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ours",
                    choices=["ours", "ear", "image_cnn_max", "image_cnn_head"],
                    help="image_cnn 은 두 변형을 **따로** 잰다. 백본이 같고 시간 처리만 "
                         "다르므로, 둘의 차이가 곧 head 의 지연 기여다")
    ap.add_argument("--source", default="0", help="카메라 인덱스 또는 영상 경로")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=300.0, help="초. 실측은 5분")
    ap.add_argument("--onnx-dir", default=ONNX_DIR)
    ap.add_argument("--intra-threads", type=int, default=0,
                    help="🔴 실측은 2. 안 주면 결과에 경고가 박힌다")
    ap.add_argument("--no-spin", action="store_true",
                    help="🔴 실측은 필수. 빼면 27% 느려져 비교가 무효다")
    ap.add_argument("--head-stride", type=int, default=1,
                    help="head 를 몇 프레임마다 돌리나. **기본 1** (매 프레임)")
    ap.add_argument("--refine-landmarks", action="store_true",
                    help="iris 서브모델(468->478). 크롭에 안 쓰는데 CPU 를 먹는다")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import cv2
    from src.v2.deploy.frontend import EyeFrontend

    # 동치 게이트를 통과했는지 먼저 본다. 안 통과했으면 재도 의미가 없다.
    eq_path = "results/v2/check_equivalence.json"
    eq = json.load(open(eq_path, encoding="utf-8")) if os.path.exists(eq_path) else None
    if not (eq and eq.get("pi_measurement_authorized")):
        print("🔴 train/serve 동치 게이트를 통과하지 않았습니다. "
              "`python -m src.v2.deploy.check_equivalence` 를 먼저 통과시키십시오.\n"
              "   배포 경로가 어긋나면 지연은 정상으로 나오고 판정만 조용히 망가집니다.")
        return 2

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print(f"영상원을 열 수 없습니다: {args.source}")
        return 1

    fe = EyeFrontend(refine_landmarks=args.refine_landmarks)
    enc_sess = head_sess = None
    random_weights = False
    if args.mode == "ours":
        enc_sess = make_session(os.path.join(args.onnx_dir, "encoder.onnx"),
                                args.intra_threads, args.no_spin)
        head_sess = make_session(os.path.join(args.onnx_dir, "head.onnx"),
                                 args.intra_threads, args.no_spin)
    elif args.mode.startswith("image_cnn"):
        sub = os.path.join(args.onnx_dir, args.mode)
        enc_sess = make_session(os.path.join(sub, "backbone.onnx"),
                                args.intra_threads, args.no_spin)
        if args.mode == "image_cnn_head":
            head_sess = make_session(os.path.join(sub, "head.onnx"),
                                     args.intra_threads, args.no_spin)
        # 🔴 image_cnn 체크포인트가 없어 무작위 초기화로 export 했다.
        # 지연은 구조·입력·하드웨어로 결정되므로 유효하지만 **정확도는 무의미하다.**
        random_weights = True

    t_read, t_detect, t_crop, t_encode, t_head, t_e2e = ([] for _ in range(6))
    ring: deque = deque(maxlen=EVENT_LEN)
    ear_ring: deque = deque(maxlen=EVENT_LEN)
    n_frames = n_nodetect = n_decisions = 0
    probs: list[float] = []

    smp = Sampler()
    smp.start()
    thr_start = read_throttled()
    temp_start = read_temp()
    t_wall = time.perf_counter()

    while time.perf_counter() - t_wall < args.duration:
        f0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        t_read.append((t1 - f0) * 1e3)
        n_frames += 1

        mesh = fe.mesh(frame)
        t2 = time.perf_counter()
        t_detect.append((t2 - t1) * 1e3)
        if mesh is None:
            n_nodetect += 1
            ring.append(None)
            ear_ring.append(np.nan)
            t_e2e.append((t2 - f0) * 1e3)
            continue

        if args.mode in ("ours", "image_cnn_max", "image_cnn_head"):
            g, _ = fe.crop_from_mesh(frame, mesh)
            x = C.to_input_tensor(g) if g is not None else None
            t3 = time.perf_counter()
            t_crop.append((t3 - t2) * 1e3)
            if x is None:
                ring.append(None)
                t_e2e.append((t3 - f0) * 1e3)
                continue
            z = enc_sess.run(None, {"crop": x.astype(np.float32)})[0][0]
            t4 = time.perf_counter()
            t_encode.append((t4 - t3) * 1e3)
            ring.append(z)

            t5 = t4
            if len(ring) == EVENT_LEN and n_frames % args.head_stride == 0:
                d = len(z)
                zz = np.stack([v if v is not None else np.zeros(d, np.float32)
                               for v in ring])[None].astype(np.float32)
                mk = np.array([[0.0 if v is None else 1.0 for v in ring]], np.float32)
                if head_sess is not None:
                    p = float(head_sess.run(
                        None, {"vectors": zz, "mask": mk})[0].reshape(-1)[0])
                else:
                    # image_cnn_max — mEBAL 원문 §5.1 의 masked max. 학습 파라미터가 없다.
                    s = np.where(mk[0] > 0, zz[0, :, 0], -np.inf)
                    p = float(s.max()) if np.isfinite(s).any() else 0.0
                t5 = time.perf_counter()
                t_head.append((t5 - t4) * 1e3)
                probs.append(p)
                n_decisions += 1
            t_e2e.append((t5 - f0) * 1e3)
        else:
            e = C.ear_both(mesh)
            ear_ring.append(e["mean"] if "mean" in e else float(np.mean(list(e.values()))))
            t3 = time.perf_counter()
            t_crop.append((t3 - t2) * 1e3)
            if len(ear_ring) == EVENT_LEN and n_frames % args.head_stride == 0:
                probs.append(ear_drop_ratio(list(ear_ring)))
                n_decisions += 1
            t4 = time.perf_counter()
            t_head.append((t4 - t3) * 1e3)
            t_e2e.append((t4 - f0) * 1e3)

    elapsed = time.perf_counter() - t_wall
    cap.release()
    fe.close()
    smp.stop()
    thr_end = read_throttled()

    stages = {"read": t_read, "detect": t_detect, "crop": t_crop,
              "encode": t_encode, "head": t_head, "e2e": t_e2e}
    res = {k: summarize(v) for k, v in stages.items()}

    warn = []
    if not args.intra_threads:
        warn.append("--intra-threads 를 주지 않았다 (실측은 2). 비교가 무효일 수 있다")
    if not args.no_spin:
        warn.append("--no-spin 을 주지 않았다. v1 에서 27% 느려졌다. 비교가 무효일 수 있다")
    if args.duration < 300:
        warn.append(f"측정 길이 {args.duration}s < 300s. 지속 성능이 아니다")
    if read_temp() is None:
        warn.append("vcgencmd 가 없다 — Pi 가 아니다. 이 숫자는 논문에 쓰지 않는다")
    if random_weights:
        warn.append("image_cnn 가중치가 **무작위 초기화**다. 지연은 유효하지만 "
                    "이 런의 판정 출력(probs)은 의미가 없다")

    out = {
        "env": repro.env_fingerprint(),
        "config": vars(args),
        "platform": {"machine": platform.machine(), "platform": platform.platform(),
                     "cpu_governor": _sh("cat /sys/devices/system/cpu/cpu0/"
                                         "cpufreq/scaling_governor"),
                     "ort_intra_threads": args.intra_threads or "default(all cores)",
                     "ort_spinning": "disabled" if args.no_spin else "default(ENABLED)",
                     "has_vcgencmd": bool(shutil.which("vcgencmd"))},
        "head_stride": args.head_stride,
        "weights": "random (latency only)" if random_weights else "trained",
        "head_mmac_per_frame": 0.045888 if args.head_stride == 1 else None,
        "_head_cost_note": "stride 1 이므로 head 는 매 프레임 0.0459 MMAC 이다. "
                           "인코더 12.44 와 합쳐 12.49 MMAC/frame (head 0.37%).",
        "frames": n_frames, "seconds": elapsed,
        "sustained_fps": n_frames / elapsed if elapsed else None,
        "n_no_detect": n_nodetect,
        "detect_miss_rate": n_nodetect / n_frames if n_frames else None,
        "n_decisions": n_decisions,
        "stages_ms": res,
        "thermal": {"temp_start": temp_start, "temp_end": read_temp(),
                    "temp_p95": pct(smp.temp, 95),
                    "throttled_start": thr_start, "throttled_end": thr_end,
                    "throttled_flags": sorted(smp.flags),
                    "throttled_clean": not smp.flags},
        "cpu_percent_mean": float(np.mean(smp.cpu)) if smp.cpu else None,
        "rss_peak_mb": float(np.max(smp.rss)) if smp.rss else None,
        "gate_G_E1": {"budget_ms": 33.3, "e2e_p99": res["e2e"].get("p99"),
                      "pass": (res["e2e"].get("p99") is not None
                               and res["e2e"]["p99"] <= 33.3 and not smp.flags)},
        "warnings": warn,
        "equivalence_gate": {"path": eq_path,
                             "authorized": eq.get("pi_measurement_authorized")},
    }

    print(f"\n모드 {args.mode}  {n_frames} 프레임 / {elapsed:.1f}s "
          f"= {out['sustained_fps']:.1f} fps  (검출 실패 {n_nodetect})")
    print(f"  {'stage':<8}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}   (ms)")
    for k in ("read", "detect", "crop", "encode", "head", "e2e"):
        s = res[k]
        if s["n"]:
            print(f"  {k:<8}{s['p50']:>9.2f}{s['p95']:>9.2f}{s['p99']:>9.2f}{s['max']:>9.2f}")
    if res["e2e"]["n"]:
        share = {k: res[k]["p50"] / res["e2e"]["p50"] for k in
                 ("read", "detect", "crop", "encode", "head") if res[k]["n"]}
        print("  p50 비중: " + "  ".join(f"{k} {v:.0%}" for k, v in share.items()))
    print(f"  G-E1 (e2e p99 <= 33.3ms, 스로틀 없음): "
          f"{'PASS' if out['gate_G_E1']['pass'] else 'FAIL'}")
    for w in warn:
        print(f"  ⚠️ {w}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
