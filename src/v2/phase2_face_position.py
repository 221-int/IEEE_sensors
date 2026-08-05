"""선택된 얼굴의 화면 위치를 재서 **옆자리 사람이 뽑힌 프레임**을 찾는다.

    python -m src.v2.phase2_face_position --batch 2022     # 신규 20명
    python -m src.v2.phase2_face_position                   # 58명 전량

왜 필요한가
----------
mEBAL2 2022 배치는 **여러 학생이 한 방에서 동시에 녹화**됐다(다중 얼굴 비율 2020 배치
0.1% vs 2022 배치 11.2%). 우리 얼굴 선택 규칙은 "메시가 있는 얼굴 중 박스 면적 최대"
인데, **주 피험자가 검출되지 않은 프레임에서는 옆자리 사람만 남아 그 사람이 선택된다.**
U18 에서 라벨 프레임의 27% 가 그런 경우였고, 그중 88% 는 얼굴이 하나만 검출된
프레임이라 "프레임 간 급변" 탐지로는 잡히지 않았다.

방법
----
얼굴이 둘 이상인 프레임에서 각 얼굴의 가로 위치를 모으면 **자리 배치**를 알 수 있다.
주 피험자는 카메라 앞(자기 노트북 앞)에 앉으므로 위치가 좁게 분포하고, 옆자리는
멀리 떨어진 다른 봉우리에 나타난다. 그래서:

    1. 라벨 프레임마다 선택된 얼굴의 눈 중심 x 를 구한다
    2. 최빈 위치(주 좌석)를 히스토그램 모드로 잡는다
    3. 모드에서 `--tol` px 이상 떨어진 프레임을 **옆자리 선택**으로 플래그한다

영상 디코딩이 전혀 필요 없다(랜드마크 CSV 만 사용). 결과는 재추출 없이
인덱스에 붙일 수 있는 프레임 단위 플래그다.

산출물: results/v2/face_position.json (사용자별 누적, 재개 가능)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
import zipfile

import numpy as np

from src.v2.dataset import crop as C
from src.v2.dataset import mebal2 as M

PD_ZIP = "data/mEBAL2/Processed_Data.zip"
WEBCAM = "data/mEBAL2/Webcams-EEG *.zip"
PROBE = "data/raw/mEBAL2/_probe"
OUT = "results/v2/face_position.json"
FRAME_W = 1280


def exam_days() -> dict[int, str]:
    """StudentData.csv 에서 시험일을 읽는다. 배치(2020=mEBAL1 / 2022=신규) 구분용."""
    out: dict[int, str] = {}
    for zp in sorted(glob.glob(WEBCAM)):
        try:
            z = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            continue
        for n in z.namelist():
            if n.endswith("StudentData.csv"):
                u = int(re.search(r"User (\d+)", n).group(1))
                L = z.read(n).decode("utf-8", "replace").splitlines()
                if len(L) > 1:
                    out[u] = dict(zip(L[0].split(), L[1].split())).get("Exam_day_is", "")
        z.close()
    return out


def scan_user(zf: zipfile.ZipFile, user: int, tol: float, bins: int = 64) -> dict:
    t0 = time.perf_counter()
    sess = M.Session.from_probe(PROBE, user)
    ev, isb = sess.events()
    need = {f for s, e, _ in ev for f in range(int(s), int(e) + 1)}

    with zf.open(f"Processed_Data/User {user}/box.csv") as fh:
        _, bp = M.scan_stream(fh, need, 4, 1)
    with zf.open(f"Processed_Data/User {user}/landmarks.csv") as fh:
        _, lp = M.scan_stream(fh, need, 3, M.N_MESH)

    xs, nmesh, frames = [], [], []
    seat_all = []                      # 다중 얼굴 프레임의 모든 얼굴 위치 (자리 배치 파악)
    for f in sorted(need):
        faces = lp.get(f)
        if faces is not None and faces.shape[0] > 1:
            for i in range(faces.shape[0]):
                xy = faces[i, :, :2]
                la, lb, ra, rb = C.eye_corners(xy)
                seat_all.append(float((((la + lb) / 2 + (ra + rb) / 2) / 2)[0]))
        sel = M.select_face(f, faces, bp.get(f))
        if sel.status != M.OK:
            continue
        xy = sel.mesh[:, :2]
        la, lb, ra, rb = C.eye_corners(xy)
        xs.append(float((((la + lb) / 2 + (ra + rb) / 2) / 2)[0]))
        nmesh.append(int(sel.n_mesh))
        frames.append(f)

    x = np.asarray(xs)
    if x.size == 0:
        return {"user": user, "error": "no face"}
    # 주 좌석 추정: **최빈 구간이 아니라 질량이 가장 큰 구간**을 쓴다.
    # 주 피험자는 노트북 앞에서 움직이므로 분포가 넓게 퍼지고, 옆자리는 가만히 앉아
    # 봉우리가 뾰족하다. 그래서 최빈 구간을 쓰면 **옆자리를 주 좌석으로 오인**한다
    # (U18 에서 실제로 발생: 모드 430 = 옆자리, 실제 주 좌석은 x>=700 쪽 73%).
    h, edges = np.histogram(x, bins=bins, range=(0, FRAME_W))
    ctr = (edges[:-1] + edges[1:]) / 2
    mass = np.array([h[np.abs(ctr - c) <= tol].sum() for c in ctr])
    win = ctr[np.abs(ctr - ctr[int(mass.argmax())]) <= tol]
    inwin = (x >= win.min() - (edges[1] - edges[0]) / 2) & (x <= win.max() + (edges[1] - edges[0]) / 2)
    mode = float(np.median(x[inwin]))          # 그 구간 안에서의 중앙값
    off = np.abs(x - mode) > tol
    nm = np.asarray(nmesh)

    # 이벤트 단위 오염
    pos = {f: i for i, f in enumerate(frames)}
    ev_off = 0
    for s, e, _ in ev:
        idxs = [pos[f] for f in range(int(s), int(e) + 1) if f in pos]
        if idxs and off[idxs].any():
            ev_off += 1

    seat = np.asarray(seat_all)
    return {
        "user": user, "seconds": round(time.perf_counter() - t0, 1),
        "n_labelled_faces": int(x.size), "mode_x": mode, "tol": tol,
        "off_seat_rate": float(off.mean()),
        "off_seat_single_face": int((off & (nm == 1)).sum()),
        "off_seat_multi_face": int((off & (nm > 1)).sum()),
        "events_touched": ev_off, "events_total": int(len(ev)),
        "events_touched_rate": ev_off / max(len(ev), 1),
        "x_percentiles": {str(q): float(np.percentile(x, q)) for q in (1, 5, 50, 95, 99)},
        "multiface_seat_hist": (np.histogram(seat, bins=16, range=(0, FRAME_W))[0].tolist()
                                if seat.size else []),
        "n_multiface_faces": int(seat.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default=None, help="예: 1-58 또는 1,5,9")
    ap.add_argument("--batch", default=None, choices=["2020", "2022"],
                    help="시험 연도로 대상 선택")
    ap.add_argument("--tol", type=float, default=250.0,
                    help="주 좌석 모드에서 이 px 이상 벗어나면 옆자리 선택으로 본다")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    days = exam_days()
    if args.users:
        us = []
        for part in args.users.split(","):
            if "-" in part:
                a, b = part.split("-"); us += list(range(int(a), int(b) + 1))
            else:
                us.append(int(part))
    elif args.batch:
        us = [u for u, d in days.items() if d[:4] == args.batch]
    else:
        us = sorted(days)
    us = sorted(set(us))

    done = {}
    if os.path.exists(args.out) and not args.redo and os.path.getsize(args.out) > 0:
        try:
            done = json.load(open(args.out, encoding="utf-8")).get("users", {})
        except json.JSONDecodeError:
            pass

    todo = [u for u in us if str(u) not in done]
    print(f"대상 {len(us)}명 / 남은 {len(todo)}명   tol {args.tol:.0f}px")
    if not todo:
        print("모두 완료")
    zf = zipfile.ZipFile(PD_ZIP)
    for u in todo:
        r = scan_user(zf, u, args.tol)
        r["exam_day"] = days.get(u, "")
        done[str(u)] = r
        if "error" in r:
            print(f"  U{u:02d} {r['error']}")
        else:
            print(f"  U{u:02d} {r['seconds']:5.1f}s  {r['exam_day']}  모드x {r['mode_x']:6.0f}  "
                  f"이탈 {r['off_seat_rate']:6.2%} (단일얼굴 {r['off_seat_single_face']:5d} / "
                  f"다중 {r['off_seat_multi_face']:4d})  이벤트오염 {r['events_touched_rate']:6.2%}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"tol": args.tol, "users": done}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, args.out)
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
