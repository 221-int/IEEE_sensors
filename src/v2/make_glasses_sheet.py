"""안경 라벨 재검토용 컨택트시트 — **우리 크롭**으로 만든다.

    python -m src.v2.make_glasses_sheet

배포자가 제공한 `_probe/glasses_check/*.png` 는 눈에 밀착된 작은 크롭이라 안경테가
잘려 나가 판단이 어렵다(실제로 User 1 이 미착용으로 잘못 라벨됐다 — 컨택트시트에서는
검은 뿔테가 명확히 보인다). 우리 크롭은 MARGIN 2.2 로 넓게 잘려 테가 남으므로
재라벨링에 적합하다.

사용자당 세션 전체에 고르게 퍼진 프레임 여러 장을 나란히 놓는다. 한 장만 보면
손이 얼굴을 가린 순간 같은 것에 속을 수 있다.

산출물: docs/v2/figures/glasses_sheet_58.png
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

DATA = "data/processed/v2"
LABELS = "data/raw/mEBAL2/glasses_labels_58.csv"
OUT = "docs/v2/figures/glasses_sheet_58.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tag", default="m22")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--per-user", type=int, default=3)
    ap.add_argument("--users-per-row", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    idx = np.load(os.path.join(args.data, "index.npz"))
    frames = np.load(os.path.join(args.data, f"frames_{args.tag}.npy"), mmap_mode="r")
    subj = idx["f_subject"].astype(int)

    lab: dict[int, str] = {}
    if os.path.exists(args.labels):
        with open(args.labels, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab[int(r["user"])] = r["glasses"]

    users = sorted(np.unique(subj))
    h, w = frames.shape[1:]
    sh, sw = int(h * args.scale), int(w * args.scale)
    cell_w = sw * args.per_user
    rows = int(np.ceil(len(users) / args.users_per_row))
    sheet = np.zeros((rows * sh, cell_w * args.users_per_row), np.uint8)

    for n, u in enumerate(users):
        pos = np.flatnonzero(subj == u)
        pick = pos[np.linspace(0, len(pos) - 1, args.per_user).astype(int)]
        strip = [cv2.resize(np.asarray(frames[p]), (sw, sh),
                            interpolation=cv2.INTER_LINEAR) for p in pick]
        tile = np.hstack(strip)
        tag = lab.get(int(u), "?")
        txt = f"U{int(u):02d} g={tag}"
        cv2.putText(tile, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 4, cv2.LINE_AA)
        cv2.putText(tile, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 1, cv2.LINE_AA)
        r, c = divmod(n, args.users_per_row)
        sheet[r * sh:(r + 1) * sh, c * cell_w:(c + 1) * cell_w] = tile

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print(f"사용자 {len(users)}명 x {args.per_user}장  ->  {args.out}  {sheet.shape}")
    print("  라벨 표기 g=0 미착용 / 1 착용 / 2 불확실 / ? 라벨없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
