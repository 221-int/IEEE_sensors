"""T5-7 확장 — v2 파이프라인 전력/에너지 소비 측정 하네스. Pi 5 실측용.

`run_video.py`(지연시간 벤치마크)를 복제해 만들었다. **원본은 건드리지
않는다** — latency 벤치마크 결과·코드가 이 작업으로 꼬이면 안 된다.

    # Pi (실측) — 4모드 × 최소 3회, 매 모드 사이 쿨다운 60s+, 순서 무작위화는
    # power_bench_runner.py 가 담당한다. 이 파일 자체는 "1회 실행"만 담당.
    python -m src.v2.deploy.run_video_power --mode ours --source 0 \
        --width 640 --height 480 --duration 300 \
        --intra-threads 2 --no-spin --power-hz 7.5 --idle-seconds 30 \
        --run-index 1

    # 데스크톱 (동작 확인용 — 전력 수치는 참고용일 뿐 공식 수치 아님)
    python -m src.v2.deploy.run_video_power --mode ours --source clip.mp4 \
        --duration 10 --idle-seconds 5 --run-index 1

⚠️ **주 지표는 idle 대비 증분(power_delta_w)과 프레임당 에너지
(energy_per_frame_mJ)다.** Pi5 유휴 소비가 수 W대라, 방법별 차이를 총
전력·총 전력 기준 배터리 시간으로만 보면 4개 모드가 사실상 같은 값으로
나와 "1~2ms라 체감 안 된다"는 원래 문제가 "0.0xW라 체감 안 된다"로
반복된다. 총 전력/배터리 환산은 참고용 보조 지표로만 함께 낸다.

⚠️ PMIC rail 이름은 **아직 실제 Pi5에서 검증되지 않았다.**
    `pmic_power.py` 모듈 docstring과 결과 JSON의 `power_sampling.rail_verified`
    필드를 볼 것. 착수 시 `vcgencmd pmic_read_adc` 1회 원문을
    `power_sampling.pmic_raw_sample_first` 와 대조할 것.

🔴 `--intra-threads 2 --no-spin` 은 선택이 아니다 (원본과 동일한 이유).
🔴 4개 모드 실행에서 `--power-hz` 값을 반드시 동일하게 유지할 것 —
    폴링 주기가 다르면 `vcgencmd` 호출 자체의 부하 차이가 결과로 둔갑한다.
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
from src.v2.deploy.pmic_power import (
    CURRENT_RAILS,
    PowerSampler,
    VOLT_ONLY_RAILS,
    battery_life_hours,
    integrate_energy_j,
    mean_power_in_window,
)

ONNX_DIR = "models/v2/onnx"
EVENT_LEN = 19
RESULTS_DIR = "results/v2"


# ------------------------------------------------------------------ 플랫폼 계측 (run_video.py 와 동일)
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
    """배경 표집 — 온도·스로틀·CPU%·RSS. 1초 간격 (PowerSampler 와는 별도 스레드)."""

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
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ear_drop_ratio(buf: list[float]) -> float:
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
                    choices=["ours", "ear", "image_cnn_max", "image_cnn_head",
                             "image_cnn_bilstm", "image_cnn_lstm"])
    ap.add_argument("--source", default="0", help="카메라 인덱스 또는 영상 경로")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=300.0, help="초. 실측은 5분")
    ap.add_argument("--onnx-dir", default=ONNX_DIR)
    ap.add_argument("--intra-threads", type=int, default=0,
                    help="🔴 실측은 2. 안 주면 결과에 경고가 박힌다")
    ap.add_argument("--no-spin", action="store_true",
                    help="실측은 필수. 빼면 27%% 느려져 비교가 무효다")
    ap.add_argument("--head-stride", type=int, default=1)
    ap.add_argument("--refine-landmarks", action="store_true")
    # ---- 전력 측정 전용 인자 ----
    ap.add_argument("--power-hz", type=float, default=7.5,
                    help="PMIC 폴링 주기(Hz). 5~10Hz 권장. 🔴 4모드 실행에서 동일값 고정")
    ap.add_argument("--idle-seconds", type=float, default=30.0,
                    help="추론 시작 전 유휴 상태 측정 시간(초). 30초 이상 권장")
    ap.add_argument("--run-index", type=int, default=1,
                    help="반복 실행 인덱스 (1..N). 결과 파일명에 들어간다")
    ap.add_argument("--battery-mah", type=float, default=5000.0,
                    help="배터리 환산용 예시 용량(mAh). 실제 타깃 기기 스펙 아님")
    ap.add_argument("--battery-cell-v", type=float, default=3.7)
    ap.add_argument("--battery-eta", type=float, default=0.87,
                    help="셀->5V 승압 컨버터 효율 가정치 (0.85~0.90)")
    ap.add_argument("--out", default=None,
                    help="기본값: results/v2/power_{mode}_480p_run{run-index}.json")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(
            RESULTS_DIR, f"power_{args.mode}_480p_run{args.run_index}.json")

    import cv2
    from src.v2.deploy.frontend import EyeFrontend

    eq_path = "results/v2/check_equivalence.json"
    eq = json.load(open(eq_path, encoding="utf-8")) if os.path.exists(eq_path) else None
    if not (eq and eq.get("pi_measurement_authorized")):
        print("🔴 train/serve 동치 게이트를 통과하지 않았습니다. "
              "`python -m src.v2.deploy.check_equivalence` 를 먼저 통과시키십시오.")
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
        if args.mode in ("image_cnn_head", "image_cnn_bilstm", "image_cnn_lstm"):
            head_sess = make_session(os.path.join(sub, "head.onnx"),
                                     args.intra_threads, args.no_spin)
        random_weights = True

    t_read, t_detect, t_crop, t_encode, t_head, t_e2e = ([] for _ in range(6))
    ring: deque = deque(maxlen=EVENT_LEN)
    ear_ring: deque = deque(maxlen=EVENT_LEN)
    n_frames = n_nodetect = n_decisions = 0
    probs: list[float] = []

    # ---- 배경 표집 시작: 온도/CPU(1Hz, 기존 Sampler) + 전력(power-hz, 별도 스레드) ----
    smp = Sampler()
    smp.start()
    pw = PowerSampler(hz=args.power_hz)
    pw.start()
    thr_start = read_throttled()
    temp_start = read_temp()

    # ---- IDLE 구간: 모델은 이미 로드됐고(워밍업 끝), 추론은 아직 시작 안 함 ----
    # 카메라는 열려 있지만 이 구간에서는 cap.read() 를 호출하지 않는다
    # (idle 이 진짜 idle 이 되도록). 실제 Pi에서 카메라 스트리밍 자체가
    # idle 전력에 영향을 주는지는 확인이 필요한 가정 — MD 참고.
    t_idle_start = time.perf_counter()
    time.sleep(max(0.0, args.idle_seconds))
    t_idle_end = time.perf_counter()

    # ---- ACTIVE 구간: 첫 추론 직전을 측정 시작 마커로 둔다 ----
    t_active_start = time.perf_counter()
    t_wall = t_active_start

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

        if args.mode in ("ours", "image_cnn_max", "image_cnn_head",
                        "image_cnn_bilstm", "image_cnn_lstm"):
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

    t_active_end = time.perf_counter()
    elapsed = t_active_end - t_active_start
    cap.release()
    fe.close()
    smp.stop()
    pw.stop()
    thr_end = read_throttled()

    # ---- 전력 계산 ----
    power_idle_w = mean_power_in_window(pw.samples, t_idle_start, t_idle_end)
    power_active_w = mean_power_in_window(pw.samples, t_active_start, t_active_end)
    power_delta_w = (power_active_w - power_idle_w
                      if power_idle_w is not None and power_active_w is not None
                      else None)
    active_samples = [(t, w) for t, w in pw.samples if t_active_start <= t <= t_active_end]
    energy_j_total = integrate_energy_j(active_samples)
    energy_per_frame_mJ = (energy_j_total / n_frames * 1000.0) if n_frames else None

    battery_wh = args.battery_mah / 1000.0 * args.battery_cell_v * args.battery_eta
    battery_hours_delta = battery_life_hours(
        power_delta_w, args.battery_mah / 1000.0, args.battery_cell_v, args.battery_eta)
    battery_hours_total_power = battery_life_hours(
        power_active_w, args.battery_mah / 1000.0, args.battery_cell_v, args.battery_eta)

    stages = {"read": t_read, "detect": t_detect, "crop": t_crop,
              "encode": t_encode, "head": t_head, "e2e": t_e2e}
    res = {k: summarize(v) for k, v in stages.items()}

    governor = _sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")

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
        warn.append("image_cnn 가중치가 **무작위 초기화**다. 지연/전력은 유효하지만 "
                    "이 런의 판정 출력(probs)은 의미가 없다")
    if args.idle_seconds < 30:
        warn.append(f"--idle-seconds {args.idle_seconds}s < 30s. idle 기준선이 불안정할 수 있다")
    if governor is not None and governor.strip() != "performance":
        warn.append(f"CPU governor 가 'performance' 가 아니다 (현재: {governor!r}). "
                    "고정하지 않으면 4모드 비교가 무효일 수 있다")
    if smp.flags:
        warn.append("이번 run 에서 스로틀이 발생했다 (throttled_flags 참고). "
                    "지침상 스로틀 발생 run 은 결과 집계에서 폐기해야 한다")
    if pw.parse_errors:
        warn.append(f"PMIC 폴링 중 파싱 실패 {pw.parse_errors}건 — rail 포맷이 실제와 "
                    "다를 수 있다. power_sampling.pmic_raw_sample_first 로 확인할 것")
    if pw.missing_rails:
        warn.append(f"일부 샘플에서 누락된 rail: {sorted(pw.missing_rails)} — "
                    "이름 철자가 실제와 다를 가능성. 실제 Pi 원문과 대조 필요")
    if power_idle_w is None or power_active_w is None:
        warn.append("전력 샘플이 부족해 idle/active 평균을 못 냈다 — "
                    "Pi 가 아니거나 vcgencmd pmic_read_adc 가 실패했을 가능성")

    discard_recommended = bool(smp.flags)

    out = {
        "run_index": args.run_index,
        "env": repro.env_fingerprint(),
        "config": vars(args),
        "platform": {"machine": platform.machine(), "platform": platform.platform(),
                     "cpu_governor": governor,
                     "ort_intra_threads": args.intra_threads or "default(all cores)",
                     "ort_spinning": "disabled" if args.no_spin else "default(ENABLED)",
                     "has_vcgencmd": bool(shutil.which("vcgencmd"))},
        "head_stride": args.head_stride,
        "weights": "random (latency only)" if random_weights else "trained",
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
        # ---- 전력/에너지 (§2-3 핵심 필드) ----
        "power_idle_w": power_idle_w,
        "power_active_w": power_active_w,
        "power_delta_w": power_delta_w,
        "energy_j_total": energy_j_total,
        "energy_per_frame_mJ": energy_per_frame_mJ,
        "battery_estimate": {
            "_note": "예시 참고용 — capacity_mah/cell_v 는 요청자가 임의로 넣은 값이지 "
                     "실제 타깃 기기 스펙이 아니다. 논문 수치로 쓰지 말 것.",
            "capacity_mah": args.battery_mah,
            "cell_v": args.battery_cell_v,
            "boost_converter_eta": args.battery_eta,
            "battery_wh": battery_wh,
            "battery_hours_at_delta_power": battery_hours_delta,
            "battery_hours_at_total_active_power": battery_hours_total_power,
            "_primary_metric": "battery_hours_at_delta_power",
        },
        "power_sampling": {
            "hz": args.power_hz,
            "idle_window_s": [t_idle_start, t_idle_end],
            "active_window_s": [t_active_start, t_active_end],
            "n_idle_samples": sum(1 for t, _ in pw.samples if t_idle_start <= t <= t_idle_end),
            "n_active_samples": len(active_samples),
            "parse_errors": pw.parse_errors,
            "missing_rails": sorted(pw.missing_rails),
            "current_rails": CURRENT_RAILS,
            "volt_only_rails_excluded": VOLT_ONLY_RAILS,
            "rail_verified": False,
            "rail_source": "https://github.com/jfikar/RPi5-power (커뮤니티 자료, 미검증)",
            "pmic_raw_sample_first": pw.first_raw_sample,
            "_verification_todo": "착수 시 `vcgencmd pmic_read_adc` 1회 원문과 "
                                  "current_rails 목록·pmic_raw_sample_first 포맷을 대조할 것",
        },
        "discard_recommended": discard_recommended,
        "gate_G_E1": {"budget_ms": 33.3, "e2e_p99": res["e2e"].get("p99"),
                      "pass": (res["e2e"].get("p99") is not None
                               and res["e2e"]["p99"] <= 33.3 and not smp.flags)},
        "warnings": warn,
        "equivalence_gate": {"path": eq_path,
                             "authorized": eq.get("pi_measurement_authorized")},
    }

    print(f"\n모드 {args.mode} run{args.run_index}  {n_frames} 프레임 / {elapsed:.1f}s "
          f"= {out['sustained_fps']:.1f} fps  (검출 실패 {n_nodetect})")
    print(f"  idle {power_idle_w if power_idle_w is None else f'{power_idle_w:.3f}'} W"
          f"  active {power_active_w if power_active_w is None else f'{power_active_w:.3f}'} W"
          f"  delta {power_delta_w if power_delta_w is None else f'{power_delta_w:.3f}'} W"
          f"  {energy_per_frame_mJ if energy_per_frame_mJ is None else f'{energy_per_frame_mJ:.3f}'} mJ/frame")
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