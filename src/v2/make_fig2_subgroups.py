"""Fig. 2 — 서브그룹 forest plot (ours − ear_head, 이벤트 가중 풀링).

    python -m src.v2.make_fig2_subgroups

값을 **코드에 하드코딩하지 않는다.** `results/v2/posthoc_subgroups_final.json` 의
`pooled_groups` 에서 읽는다. 그래야 런을 다시 돌리면 그림이 따라온다.

왜 forest plot 인가
------------------
Table 로 5행을 내면 CI 가 0 을 넘는지/δ 를 넘는지 **눈으로 안 들어온다.** 이 그림의
목적은 값 나열이 아니라 *"어느 그룹에서도 CI 가 0 아래로 내려가지 않는다"* 를
한눈에 보이는 것이다. 그래서 0 기준선과 δ=0.02 마진선을 함께 그린다.

IEEE Sensors Letters 단 컬럼 폭(약 3.4 in)에 맞추고, 축소돼도 읽히도록 글꼴을
7~8 pt 로 둔다. 벡터(PDF)로 저장한다 — 래스터로 넣으면 인쇄에서 뭉갠다.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

SRC = "results/v2/posthoc_subgroups_final.json"
OUT = "docs/v2/figures/fig2_subgroups.pdf"
DELTA = 0.02

# JSON 키 -> 그림 라벨. 순서가 곧 위에서 아래 순서다.
ROWS = [("all", "All"),
        ("batch2020", "2020 batch"),
        ("batch2022", "2022 batch"),
        ("glasses", "Glasses"),
        ("no_glasses", "No glasses")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--width", type=float, default=3.4, help="단 컬럼 폭(inch)")
    ap.add_argument("--height", type=float, default=2.1)
    args = ap.parse_args()

    d = json.load(open(args.src, encoding="utf-8"))
    pooled = d.get("pooled_groups")
    if not pooled:
        raise SystemExit(f"{args.src} 에 pooled_groups 가 없습니다. "
                         "사이드카가 있는 런으로 posthoc_subgroups 를 다시 도십시오.")

    rows = []
    for key, label in ROWS:
        g = pooled.get(key)
        if not g or not g.get("n_subjects"):
            print(f"  ⚠️ {key} 없음 — 건너뜀")
            continue
        lo, hi = g["gain_ci"]
        rows.append({"label": label, "point": g["gain_pooled"], "lo": lo, "hi": hi,
                     "n_sub": g["n_subjects"], "n_ev": g["n_events"],
                     "verdict": g["verdict"]})
    if not rows:
        raise SystemExit("그릴 행이 없습니다.")

    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType 埋め込み
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    y = list(range(len(rows)))[::-1]          # 첫 행이 맨 위
    for yi, r in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color="#333", lw=1.0,
                solid_capstyle="butt", zorder=3)
        for xv in (r["lo"], r["hi"]):         # CI 끝 캡
            ax.plot([xv, xv], [yi - 0.13, yi + 0.13], color="#333", lw=0.8, zorder=3)
        ax.plot([r["point"]], [yi], "o", ms=3.4, color="#1a56b0",
                zorder=4, clip_on=False)

    ax.axvline(0.0, color="#b3261e", lw=0.8, zorder=2)
    ax.axvline(-DELTA, color="#888", lw=0.8, ls="--", zorder=2)
    ax.text(-DELTA, len(rows) - 0.42, f"$-\\delta$ = {-DELTA:g}", color="#888",
            fontsize=6, ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['label']}\n" r"$\it{n}$" f"={r['n_sub']}, "
                        f"{r['n_ev']:,} ev." for r in rows], fontsize=6.5)
    ax.set_xlabel(r"$\Delta$ PR-AUC   (ours $-$ EAR-head)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5, length=2.5, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)

    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    pad = (hi - lo) * 0.18
    ax.set_xlim(min(-DELTA - 0.004, lo - pad), hi + pad)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.25)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)  # 확인용
    plt.close(fig)

    print(f"{'group':<12}{'Δ PR-AUC':>10}{'95% CI':>22}{'판정':>14}")
    for r in rows:
        print(f"{r['label']:<12}{r['point']:>+10.4f}"
              f"{f'[{r[chr(108)+chr(111)]:+.4f}, {r[chr(104)+chr(105)]:+.4f}]':>22}"
              f"{r['verdict']:>14}")
    print(f"\n  -> {args.out}  (+ {png})")
    print(f"  출처: {args.src}  (하드코딩 없음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
