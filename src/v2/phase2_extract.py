"""Phase 2/3 — 원본 영상에서 양눈 크롭을 뽑는다. 사용자당 샤드 1개.

    python -m src.v2.phase2_extract --users 1-5            # 소규모 검증
    python -m src.v2.phase2_extract --users 1-58           # 본 실행 (수 시간)

동작
----
사용자 1명당:
  1. `Processed_Data.zip` 에서 landmarks/box 를 **스트리밍**으로 읽어 얼굴을 해소한다
     (파서는 `dataset/mebal2.scan_stream` 하나만 쓴다 — phase1 과 동일)
  2. color.mp4 를 확보한다. 이미 풀려 있으면 그대로, 아니면 Webcams zip 에서
     **임시 파일로 꺼냈다가 끝나면 지운다** (사용자당 약 5 GB)
  3. 라벨된 이벤트 프레임만 **순차 디코딩**해 크롭한다 (시킹 금지)
  4. `data/processed/v2/shards/u{NN}.npz` 로 저장한다

왜 샤드인가
----------
58명을 한 번에 돌리면 몇 시간이 걸리고, 중간에 끊기면 처음부터입니다. 사용자 단위로
저장하면 이어서 돌릴 수 있고, 실패한 사용자만 다시 할 수 있습니다. 합치는 것은
`phase3_merge` 가 따로 합니다.

메모리
------
가장 큰 사용자(이벤트 2,110개)가 약 40,000 프레임 = 410 MB 입니다. 사용자 단위로
처리하고 바로 디스크에 쓰므로 그 이상 쌓이지 않습니다.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import tempfile
import time
import zipfile

import numpy as np

from src.v2.common import repro
from src.v2.dataset import crop as C
from src.v2.dataset import mebal2 as M

PD_ZIP = "data/mEBAL2/Processed_Data.zip"
WEBCAM_ZIPS = "data/mEBAL2/Webcams-EEG *.zip"
RAW = "data/raw/mEBAL2"
PROBE = f"{RAW}/_probe"
OUT_DIR = "data/processed/v2/shards"


def parse_users(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------- 영상 확보
def _find_in_webcam_zips(user: int) -> tuple[str, str] | None:
    member = f"User {user}/RealSense/Color_Webcam/color.mp4"
    for zp in sorted(glob.glob(WEBCAM_ZIPS)):
        try:
            with zipfile.ZipFile(zp) as z:
                if member in z.namelist():
                    return zp, member
        except zipfile.BadZipFile:
            continue
    return None


class VideoSource:
    """이미 풀려 있으면 그대로, 아니면 zip 에서 꺼냈다가 **반드시 지운다**.

    원본 zip 은 지우지 않는다(크롭 검증이 끝날 때까지 보관). 지우는 것은
    우리가 꺼낸 임시 사본뿐이다.
    """

    def __init__(self, user: int, keep: bool = False):
        self.user, self.keep = user, keep
        self.path: str | None = None
        self._tmp: str | None = None
        self.extracted_from: str | None = None
        self.extract_seconds = 0.0

    def __enter__(self) -> str:
        direct = os.path.join(RAW, f"User {self.user}", "RealSense",
                              "Color_Webcam", "color.mp4")
        if os.path.exists(direct):
            self.path = direct
            return direct
        found = _find_in_webcam_zips(self.user)
        if found is None:
            raise FileNotFoundError(f"User {self.user} 의 color.mp4 를 찾을 수 없습니다.")
        zp, member = found
        self.extracted_from = zp
        t0 = time.perf_counter()
        self._tmp = tempfile.mkdtemp(prefix=f"mebal2_u{self.user}_")
        dst = os.path.join(self._tmp, "color.mp4")
        with zipfile.ZipFile(zp) as z, z.open(member) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out, length=8 << 20)
        self.extract_seconds = time.perf_counter() - t0
        self.path = dst
        return dst

    def __exit__(self, *exc):
        if self._tmp and not self.keep:
            shutil.rmtree(self._tmp, ignore_errors=True)


# --------------------------------------------------------------- 사용자 처리
def _tag(m: float) -> str:
    """2.2 -> 'm22'. npz 키에 쓰므로 점을 없앤다."""
    return "m" + f"{m:.2f}".replace(".", "").rstrip("0").rjust(2, "0")[:3]


def process_user(user: int, out_dir: str, max_events: int | None = None,
                 out_h: int = C.OUT_H, out_w: int = C.OUT_W,
                 margins: tuple[float, ...] = (C.MARGIN,), keep_video: bool = False,
                 max_frame: int | None = None) -> dict:
    t_all = time.perf_counter()
    sess = M.Session.from_probe(PROBE, user)
    ev, isb = sess.events()

    if max_frame is not None:
        # 시킹이 금지돼 있어 마지막 표본까지 순차로 넘어가야 하고, 그 시간이 곧
        # 실행 시간입니다. 실행 시간 상한이 있는 환경에서만 쓰는 디버그 옵션입니다.
        keep = ev[:, 1] <= max_frame
        ev, isb = ev[keep], isb[keep]

    if max_events and len(ev) > max_events:
        # 클래스별 균형 서브샘플. 결정적(시드 없음) — 균등 간격으로 고른다.
        idx = []
        for cls in (1, 0):
            w = np.flatnonzero(isb == cls)
            take = np.linspace(0, len(w) - 1, max_events // 2).astype(int)
            idx.append(w[np.unique(take)])
        sel = np.sort(np.concatenate(idx))
        ev, isb = ev[sel], isb[sel]

    need = {f for s, e, _ in ev for f in range(int(s), int(e) + 1)}

    t0 = time.perf_counter()
    with zipfile.ZipFile(PD_ZIP) as z:
        with z.open(f"Processed_Data/User {user}/box.csv") as fh:
            box_counts, box_parsed = M.scan_stream(fh, need, 4, 1)
        with z.open(f"Processed_Data/User {user}/landmarks.csv") as fh:
            lm_counts, lm_parsed = M.scan_stream(fh, need, 3, M.N_MESH)
    t_scan = time.perf_counter() - t0
    session_frames = len(lm_counts)

    # 얼굴이 해소되는 프레임만 디코딩 대상으로 삼는다
    sel_of: dict[int, M.FaceSel] = {}
    for f in sorted(need):
        s = M.select_face(f, lm_parsed.get(f), box_parsed.get(f))
        if s.status == M.OK:
            sel_of[f] = s
    frames = sorted(sel_of)

    # --- 디코딩 + 크롭 ---
    # 배율을 여러 개 뽑는 이유: 디코딩이 전체 시간의 대부분이고 크롭은 0.1 ms 수준이라,
    # 한 번 디코딩할 때 여러 배율을 함께 뽑으면 **배율 결정을 학습 시점까지 미룰 수 있다.**
    # 각 벌이 모두 원본 프레임에서 crop_both_eyes 로 직접 만들어지므로 배포 경로와
    # 픽셀 단위로 동일하다(저장본에서 파생하면 이중 리샘플링이 되어 달라진다).
    rows: dict[int, int] = {}
    imgs: dict[float, list] = {m: [] for m in margins}
    per: dict[float, list] = {m: [] for m in margins}
    meta, ear, n_partial = [], [], 0
    t0 = time.perf_counter()
    with VideoSource(user, keep=keep_video) as vpath:
        for idx, frame in M.iter_frames(vpath, frames):
            xy = sel_of[idx].mesh[:, :2]
            got = {}
            for mg in margins:
                g, cm = C.crop_both_eyes(frame, xy, out_h, out_w, mg)
                if g is None:
                    break
                got[mg] = (g, cm)
            if len(got) != len(margins):      # 일부 배율만 성공하면 행 정렬이 깨진다
                n_partial += 1
                continue
            rows[idx] = len(meta)
            for mg in margins:
                g, cm = got[mg]
                ph = C.photometrics(g)
                imgs[mg].append(g)
                per[mg].append((cm.crop_w_px, float(cm.interp_cubic), float(cm.padded),
                                ph["brightness"], ph["contrast"], ph["sharpness"]))
            cm0 = got[margins[0]][1]
            e = C.ear_both(xy)
            meta.append((idx, cm0.span_px, cm0.tilt_deg))
            ear.append((e["left"], e["right"], e["mean"], e["min"]))
    t_decode = time.perf_counter() - t0

    if not meta:
        raise RuntimeError(f"User {user}: 크롭이 하나도 생성되지 않았습니다.")

    F = np.asarray(meta, np.float64)
    f_frame_idx = F[:, 0].astype(np.int32)
    lo, hi = int(f_frame_idx.min()), int(f_frame_idx.max())
    f_t_rel = ((f_frame_idx - lo) / max(hi - lo, 1)).astype(np.float32)

    e_rows = np.full((len(ev), M.EVENT_LEN), -1, np.int32)
    for i, (s, e, _) in enumerate(ev):
        for k, f in enumerate(range(int(s), int(e) + 1)):
            if f in rows:
                e_rows[i, k] = rows[f]
    e_n_missing = (e_rows < 0).sum(1).astype(np.int16)
    e_centre = (ev[:, 0] + ev[:, 1]) / 2.0
    e_t_rel = ((e_centre - lo) / max(hi - lo, 1)).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"u{user:02d}.npz")
    tmp = path + ".tmp.npz"
    payload = dict(
        f_frame_idx=f_frame_idx, f_t_rel=f_t_rel,
        f_span_px=F[:, 1].astype(np.float32), f_tilt_deg=F[:, 2].astype(np.float32),
        f_ear=np.asarray(ear, np.float32),           # (n, 4) left/right/mean/min
        e_rows=e_rows, e_is_blink=isb.astype(np.uint8),
        e_start=ev[:, 0].astype(np.int32), e_end=ev[:, 1].astype(np.int32),
        e_blink_flag=ev[:, 2].astype(np.uint8),
        e_t_rel=e_t_rel, e_n_missing=e_n_missing,
        user=np.int32(user), session_frames=np.int32(session_frames),
        margins=np.asarray(margins, np.float32),
        out_hw=np.array([out_h, out_w], np.int32),
    )
    for mg in margins:
        t = _tag(mg)
        P = np.asarray(per[mg], np.float64)
        payload[f"frames_{t}"] = np.stack(imgs[mg])
        payload[f"f_crop_w_px_{t}"] = P[:, 0].astype(np.float32)
        payload[f"f_interp_cubic_{t}"] = P[:, 1].astype(np.uint8)
        payload[f"f_padded_{t}"] = P[:, 2].astype(np.uint8)
        payload[f"f_brightness_{t}"] = P[:, 3].astype(np.float32)
        payload[f"f_contrast_{t}"] = P[:, 4].astype(np.float32)
        payload[f"f_sharpness_{t}"] = P[:, 5].astype(np.float32)
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)

    out = {
        "user": user, "path": path,
        "session_frames": session_frames,
        "n_events": int(len(ev)), "n_blink": int((isb == 1).sum()),
        "n_frames_needed": len(need), "n_frames_face_ok": len(frames),
        "n_crops": len(meta), "n_partial_skipped": n_partial,
        "events_fully_missing": int((e_n_missing == M.EVENT_LEN).sum()),
        "mean_missing_per_event": float(e_n_missing.mean()),
        "seconds": {"scan": round(t_scan, 1), "decode": round(t_decode, 1),
                    "total": round(time.perf_counter() - t_all, 1)},
        "size_mb": round(os.path.getsize(path) / 1e6, 1),
        "span_mean": float(F[:, 1].mean()), "span_std": float(F[:, 1].std()),
        "tilt_mean": float(F[:, 2].mean()),
        "by_margin": {},
    }
    for mg in margins:
        P = np.asarray(per[mg], np.float64)
        out["by_margin"][f"{mg}"] = {
            "brightness_mean": float(P[:, 3].mean()), "brightness_std": float(P[:, 3].std()),
            "sharpness_mean": float(P[:, 5].mean()), "sharpness_std": float(P[:, 5].std()),
            "interp_cubic_rate": float(P[:, 1].mean()), "padded_rate": float(P[:, 2].mean()),
            "crop_w_px_mean": float(P[:, 0].mean()),
        }
    return out


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="1-5")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--margins", default="2.2,1.8",
                    help="쉼표로 구분. 여러 개면 한 번 디코딩해 전부 저장한다 "
                         "(크롭은 0.1ms, 디코딩이 시간의 대부분이므로 사실상 공짜)")
    ap.add_argument("--max-events", type=int, default=None,
                    help="사용자당 이벤트 상한(클래스 균형 서브샘플). 파이프라인 검증용")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="이 프레임 이후 이벤트는 제외(디버그 전용). 본 실행에서는 빼십시오")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--keep-video", action="store_true",
                    help="zip 에서 꺼낸 임시 영상을 지우지 않는다(디버깅용, 사용자당 5 GB)")
    args = ap.parse_args()

    users = parse_users(args.users)
    margins = tuple(float(x) for x in args.margins.split(","))
    print(f"대상 {len(users)}명  배율 {list(margins)}  ->  {args.out_dir}")
    if args.max_events:
        print(f"  ⚠ 검증 모드: 사용자당 이벤트 {args.max_events}개로 제한 (본 실행에서는 빼십시오)")

    tot_t = tot_mb = 0.0
    for u in users:
        path = os.path.join(args.out_dir, f"u{u:02d}.npz")
        if os.path.exists(path) and not args.redo:
            print(f"  User {u:2d}  건너뜀 (이미 있음)")
            continue
        try:
            r = process_user(u, args.out_dir, args.max_events, margins=margins,
                             keep_video=args.keep_video, max_frame=args.max_frame)
        except Exception as e:                       # noqa: BLE001
            print(f"  User {u:2d}  실패: {type(e).__name__}: {e}")
            continue
        s = r["seconds"]
        tot_t += s["total"]; tot_mb += r["size_mb"]
        line = (f"  User {u:2d} {s['total']:6.1f}s (scan {s['scan']:4.1f}/decode {s['decode']:5.1f}) "
                f"crop {r['n_crops']:6d} {r['size_mb']:6.1f}MB "
                f"결측/ev {r['mean_missing_per_event']:.2f} span {r['span_mean']:5.1f} |")
        for mg in margins:
            b = r["by_margin"][f"{mg}"]
            line += (f" m{mg}: 밝기 {b['brightness_mean']:5.1f} "
                     f"cubic {b['interp_cubic_rate']:5.2%} pad {b['padded_rate']:5.2%}")
        print(line)

    n = len([u for u in users])
    if tot_t:
        print(f"\n  합계 {tot_t/60:.1f}분  {tot_mb:.0f}MB")
        if args.max_events or args.max_frame:
            print("  ⚠ --max-events / --max-frame 가 걸려 있어 58명 환산은 의미가 없습니다.")
        else:
            print(f"  58명 환산 추정: {tot_t/max(n,1)*58/3600:.1f}시간, "
                  f"{tot_mb/max(n,1)*58/1000:.1f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
