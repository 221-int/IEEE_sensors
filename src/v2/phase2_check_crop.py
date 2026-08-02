"""크롭 배율 후보 검증 — **영상 없이** 랜드마크만으로.

    python -m src.v2.phase2_check_crop --users 1,9,46,53,57 --margins 2.2,1.8

무엇을 확인하는가
----------------
1. **눈이 크롭 경계에서 잘리지 않는가** — 눈꼬리 4점과 눈꺼풀 8점을 크롭 좌표계로
   투영해, 출력 이미지 밖으로 나가는 프레임 비율과 경계까지의 여유(px)를 잰다.
2. **패딩·업스케일·크롭 실패율이 나빠지지 않는가** — 배율을 줄이면 상자가 작아져
   화면 밖으로 나갈 일은 줄지만(패딩↓), 업스케일은 늘어난다. 그 교환을 수치로 본다.

영상 디코딩이 전혀 필요 없다. 크롭 기하는 랜드마크와 배율만으로 완전히 결정되므로,
**58명을 다시 뽑기 전에 이 검사로 배율을 확정**하는 것이 맞다.

산출물: results/v2/crop_margin_check.json (사용자별 누적, 재개 가능)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile

import cv2
import numpy as np

from src.v2.dataset import crop as C
from src.v2.dataset import mebal2 as M

PD_ZIP = "data/mEBAL2/Processed_Data.zip"
PROBE = "data/raw/mEBAL2/_probe"
OUT = "results/v2/crop_margin_check.json"
FRAME_W, FRAME_H = 1280, 720          # mEBAL2 color.mp4 (User 1 실측, 전 사용자 동일 가정)

# 반드시 크롭 안에 들어와야 하는 점들
EYE_PTS = [C.EYE_PAIR_A[0], C.EYE_PAIR_A[1], C.EYE_PAIR_B[0], C.EYE_PAIR_B[1]]
LID_PTS = [C.EAR_EYE_A[k] for k in ("top1", "bot1", "top2", "bot2")] + \
          [C.EAR_EYE_B[k] for k in ("top1", "bot1", "top2", "bot2")]
# 들어와도 되고 안 들어와도 되는 참고점 (얼마나 잘려나가는지 보기용)
REF_PTS = {"eyebrow_L": 105, "eyebrow_R": 334, "nose_tip": 1}


def project(xy: np.ndarray, margin: float, out_h: int, out_w: int):
    """랜드마크 -> 크롭 좌표계. crop_both_eyes 와 동일한 변환을 쓴다."""
    la, lb, ra, rb = C.eye_corners(xy)
    le, re = (la + lb) / 2, (ra + rb) / 2
    ctr = (le + re) / 2
    span = float(np.linalg.norm(le - re))
    if span < C.MIN_SPAN_PX:
        return None
    cw = span * margin
    ch = cw * out_h / out_w
    x0 = int(round(ctr[0] - cw / 2)); y0 = int(round(ctr[1] - ch / 2))
    x1 = int(round(ctr[0] + cw / 2)); y1 = int(round(ctr[1] + ch / 2))
    pad = max(0, -x0, -y0, x1 - FRAME_W, y1 - FRAME_H)
    d = re - le
    ang = float(np.degrees(np.arctan2(d[1], d[0])))
    ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)
    Mt = cv2.getRotationMatrix2D((float(ctr[0]), float(ctr[1])), ang, 1.0).copy()
    Mt[0, 2] -= x0; Mt[1, 2] -= y0
    p = (Mt[:, :2] @ xy.T).T + Mt[:, 2]
    p = p * np.array([out_w / max(x1 - x0, 1), out_h / max(y1 - y0, 1)])
    return {"pts": p, "span": span, "pad": pad, "box": (x1 - x0, y1 - y0),
            "cubic": (x1 - x0) < out_w}


def check_user(zf, user: int, margins: list[float], out_h: int, out_w: int,
               max_frames: int = 8000) -> dict:
    t0 = time.perf_counter()
    sess = M.Session.from_probe(PROBE, user)
    ev, _ = sess.events()
    need = sorted({f for s, e, _ in ev for f in range(int(s), int(e) + 1)})
    if len(need) > max_frames:                      # 균등 서브샘플 (결정적)
        need = [need[i] for i in np.linspace(0, len(need) - 1, max_frames).astype(int)]
    need = set(need)

    with zf.open(f"Processed_Data/User {user}/box.csv") as fh:
        _, bp = M.scan_stream(fh, need, 4, 1)
    with zf.open(f"Processed_Data/User {user}/landmarks.csv") as fh:
        _, lp = M.scan_stream(fh, need, 3, M.N_MESH)

    meshes = []
    for f in sorted(need):
        s = M.select_face(f, lp.get(f), bp.get(f))
        if s.status == M.OK:
            meshes.append(s.mesh[:, :2])

    res: dict = {"user": user, "n_frames": len(meshes),
                 "n_labelled": len(need), "seconds": None, "margins": {}}
    for mg in margins:
        n_fail = 0
        eye_out = lid_out = pad_n = cubic_n = 0
        slack_x, slack_y, ref = [], [], {k: [] for k in REF_PTS}
        for xy in meshes:
            r = project(xy, mg, out_h, out_w)
            if r is None:
                n_fail += 1
                continue
            p = r["pts"]
            pad_n += int(r["pad"] > 0)
            cubic_n += int(r["cubic"])
            e, l = p[EYE_PTS], p[LID_PTS]
            both = np.vstack([e, l])
            if (e[:, 0] < 0).any() or (e[:, 0] >= out_w).any() or \
               (e[:, 1] < 0).any() or (e[:, 1] >= out_h).any():
                eye_out += 1
            if (l[:, 0] < 0).any() or (l[:, 0] >= out_w).any() or \
               (l[:, 1] < 0).any() or (l[:, 1] >= out_h).any():
                lid_out += 1
            slack_x.append(min(both[:, 0].min(), out_w - 1 - both[:, 0].max()))
            slack_y.append(min(both[:, 1].min(), out_h - 1 - both[:, 1].max()))
            for k, i in REF_PTS.items():
                ref[k].append(float(p[i, 1]))
        n = max(len(meshes) - n_fail, 1)
        sx, sy = np.array(slack_x), np.array(slack_y)
        res["margins"][f"{mg}"] = {
            "crop_fail_rate": n_fail / max(len(meshes), 1),
            "eye_clipped_rate": eye_out / n, "lid_clipped_rate": lid_out / n,
            "padded_rate": pad_n / n, "cubic_rate": cubic_n / n,
            "slack_x_px": {"min": float(sx.min()), "p1": float(np.percentile(sx, 1)),
                           "median": float(np.median(sx))},
            "slack_y_px": {"min": float(sy.min()), "p1": float(np.percentile(sy, 1)),
                           "median": float(np.median(sy))},
            "ref_y_median": {k: float(np.median(v)) for k, v in ref.items()},
        }
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="1,9,46,53,57")
    ap.add_argument("--margins", default="2.2,1.8")
    ap.add_argument("--out-h", type=int, default=C.OUT_H)
    ap.add_argument("--out-w", type=int, default=C.OUT_W)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    users = [int(x) for x in args.users.split(",")]
    margins = [float(x) for x in args.margins.split(",")]

    done = {}
    if os.path.exists(args.out) and not args.redo and os.path.getsize(args.out) > 0:
        try:
            done = json.load(open(args.out, encoding="utf-8")).get("users", {})
        except json.JSONDecodeError:
            pass

    zf = zipfile.ZipFile(PD_ZIP)
    for u in users:
        if str(u) in done and not args.redo:
            print(f"  User {u:2d} 건너뜀")
            continue
        r = check_user(zf, u, margins, args.out_h, args.out_w)
        done[str(u)] = r
        line = f"  User {u:2d} {r['seconds']:5.1f}s  n={r['n_frames']:5d} |"
        for mg in margins:
            m = r["margins"][f"{mg}"]
            line += (f"  m{mg}: 눈밖 {m['eye_clipped_rate']:.2%} 눈꺼풀밖 {m['lid_clipped_rate']:.2%}"
                     f" 여유 x{m['slack_x_px']['min']:.0f}/y{m['slack_y_px']['min']:.0f}"
                     f" pad {m['padded_rate']:.1%} cubic {m['cubic_rate']:.1%}")
        print(line)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"out_hw": [args.out_h, args.out_w], "users": done}, f,
                      ensure_ascii=False, indent=1)
        os.replace(tmp, args.out)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
