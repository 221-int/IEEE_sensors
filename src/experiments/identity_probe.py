"""
identity_probe.py

"EAR 은 스칼라 하나라 신원이 없을 텐데, 그럼 EAR 이 더 안전한 것 아닌가?"
— 지도 미팅에서 나올 가장 아픈 질문. 추측 대신 재서 답한다.

공정한 비교란 무엇인가
----------------------
우리 시스템은 프레임당 **128개 숫자**를 내보내고 EAR 시스템은 **1개**를 내보낸다.
그대로 비교하면 우리가 더 새는 게 당연하다. 그래서 두 층위로 나눠 측정한다.

  [A] 시스템이 실제로 내보내는 것끼리
      ours    프레임당 128차원 벡터
      ear     프레임당 스칼라 1개

  [B] ★ 숫자 개수를 맞춘 조건 (이쪽이 논쟁의 핵심)
      ours    벡터 1개            = 128개 숫자
      ear_win EAR 연속 128프레임  = 128개 숫자 (15fps 기준 약 8.5초)

  [C] 현실적인 EAR 시스템이 실제로 저장할 법한 것
      ear_stats  창의 요약통계(평균·표준편차·분위수·깜빡임률 등)

[B] 가 중요한 이유: 깜빡임 **동역학**(빈도·지속시간·파형)은 그 자체로 생체 신호다.
스칼라라서 안전한 게 아니라, **시간축을 주면 스칼라도 샐 수 있다.** 이걸 재지 않으면
"EAR 이 더 안전하다"에 반박할 근거가 없다.

프로토콜
--------
모든 표현에 **동일한 선형 프로브**와 **동일한 분할**을 쓴다(공격자 과제:
같은 피험자, 다른 프레임). 표현마다 프로브 용량이 다르면 비교가 오염되므로
선형으로 통일하고, 참고용으로 MLP 프로브도 같이 낸다.

주의: 창(window)은 겹치지 않게 자른다. 겹치면 학습/테스트 창이 사실상 같은
구간을 공유해 정확도가 부풀려진다.

Run:
    python -m src.experiments.identity_probe
    python -m src.experiments.identity_probe --window 128 --seeds 0 1 2
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.encoder.train_autoencoder import Encoder


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def encode_all(encoder, images, device, batch=512):
    encoder.eval()
    out = []
    for i in range(0, len(images), batch):
        xb = torch.from_numpy(images[i:i + batch]).float().div_(255.0).unsqueeze(1)
        out.append(encoder(xb.to(device)).cpu())
    return torch.cat(out).numpy()


def probe(X, y, n_cls, device, linear, epochs, seed, hidden=128, t=None):
    """동일 조건 선형/MLP 프로브. -> test 정확도

    t 가 주어지면 **시간 블록 분할**(피험자별 앞 70% 학습 / 뒤 30% 평가)을 쓴다.
    무작위 프레임 분할은 인접 프레임이 학습·평가에 나뉘어 들어가 정확도를
    부풀린다. 창(window)을 겹쳐 만들 때는 특히 그렇다.
    """
    set_seed(seed)
    X = np.asarray(X, np.float32)
    # 표준화: 스케일 차이로 인한 학습 난이도 차이를 제거한다(표현의 정보량만 비교)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd

    if t is None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        tr, te = perm[:cut], perm[cut:]
    else:
        tr = np.where(t <= 0.7)[0]
        te = np.where(t > 0.7)[0]
        if len(te) == 0 or len(tr) == 0:
            return float("nan")

    d = X.shape[1]
    net = (nn.Linear(d, n_cls) if linear else
           nn.Sequential(nn.Linear(d, hidden), nn.ReLU(True), nn.Dropout(0.3),
                         nn.Linear(hidden, n_cls))).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X[tr]),
                                  torch.from_numpy(y[tr])),
                    batch_size=256, shuffle=True)
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(X[te]).to(device)).argmax(1).cpu().numpy()
    return float((pred == y[te]).mean())


def join_frames(subject, frame_id, e_sub, e_fid, has_fid):
    """npz 와 EAR 덤프의 행을 정렬한다.

    덤프에 frame_id 가 있으면 (subject, frame_id) 로 조인한다.
    없으면(구 버전 덤프) **피험자별 위치 기준**으로 맞춘다 — 두 파일 모두 피험자
    안에서는 시간순이고 동일한 stride/tag 필터를 거치므로 개수가 같으면 1:1 대응이
    성립한다. 개수가 다르면 즉시 실패시킨다(조용히 잘못 맞추는 것보다 낫다).
    전역 위치로 맞추면 안 된다 — 두 파일의 피험자 순서가 다르다.
    """
    if has_fid:
        key = {(s, f): i for i, (s, f) in enumerate(zip(subject, frame_id))}
        rows = [(key[(s, f)], j) for j, (s, f) in enumerate(zip(e_sub, e_fid))
                if (s, f) in key]
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), "frame_id"

    ni, ei = [], []
    for s in sorted(set(subject.tolist())):
        a = np.where(subject == s)[0]
        a = a[np.argsort(frame_id[a])]
        b = np.where(e_sub == s)[0]          # 덤프는 이미 시간순
        if len(a) != len(b):
            raise SystemExit(
                f"[{s}] npz {len(a)} vs EAR dump {len(b)} 프레임 — 개수가 달라 "
                f"위치 기준 조인을 신뢰할 수 없습니다.\n"
                f"  해결: ear_baseline 을 --dump 로 다시 실행해 frame_id 를 포함시키세요.")
        ni.append(a); ei.append(b)
    return np.concatenate(ni), np.concatenate(ei), "피험자별 위치(구 덤프)"


def make_windows(values, subject, frame_id, W, stride=None):
    """피험자별 시간순 정렬 후 길이 W 창으로 자른다.
    -> (X [n_win, W], y [n_win], t [n_win]) — t 는 피험자 내 상대 시각 0~1"""
    stride = stride or W
    Xs, ys, ts = [], [], []
    subs = sorted(set(subject.tolist()))
    idx_of = {s: i for i, s in enumerate(subs)}
    for s in subs:
        sel = np.where(subject == s)[0]
        sel = sel[np.argsort(frame_id[sel])]
        v = values[sel]
        v = v[np.isfinite(v)]                      # 검출 실패 프레임 제거
        if len(v) < W:
            continue
        starts = np.arange(0, len(v) - W + 1, stride)
        Xs.append(np.stack([v[i:i + W] for i in starts]))
        ys.append(np.full(len(starts), idx_of[s], np.int64))
        ts.append(starts / max(len(v) - W, 1))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ts))


def window_stats(Xw):
    """창 -> 현실적인 EAR 시스템이 저장할 법한 요약통계."""
    q = np.percentile(Xw, [5, 25, 50, 75, 95], axis=1).T
    feats = [Xw.mean(1, keepdims=True), Xw.std(1, keepdims=True),
             Xw.min(1, keepdims=True), Xw.max(1, keepdims=True), q]
    # 깜빡임률 대용: 창 내에서 자기 중앙값보다 크게 낮은 프레임 비율
    thr = np.median(Xw, axis=1, keepdims=True) - 2 * np.std(Xw, axis=1, keepdims=True)
    feats.append((Xw < thr).mean(1, keepdims=True))
    return np.concatenate(feats, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/eyeblink8_eyes.npz")
    ap.add_argument("--ear-dump", default="results/ear_frames.npz")
    ap.add_argument("--ear-variant", default="mean")
    ap.add_argument("--encoder", default="models/autoencoder/encoder.pt")
    ap.add_argument("--config", default="models/autoencoder/config.json")
    ap.add_argument("--window", type=int, default=128,
                    help="EAR 창 길이. 우리 벡터 차원과 맞추는 것이 핵심 비교")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--json", default="results/identity_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    latent = (json.load(open(args.config))["latent"]
              if os.path.exists(args.config) else 128)
    d = np.load(args.data, allow_pickle=True)
    images = d["images"]
    subject = d["subject"].astype(str)
    frame_id = d["frame_id"].astype(int)

    e = np.load(args.ear_dump, allow_pickle=True)
    e_sub = e["subject"].astype(str)
    has_fid = "frame_id" in e
    e_fid = e["frame_id"].astype(int) if has_fid else np.zeros(len(e_sub), int)
    e_val = e[args.ear_variant].astype(np.float64)

    ni, ei, how = join_frames(subject, frame_id, e_sub, e_fid, has_fid)
    print(f"joined {len(ni)} frames by {how}  "
          f"(npz {len(subject)}, ear dump {len(e_sub)})")
    if len(ni) < 0.9 * len(subject):
        raise SystemExit("조인된 프레임이 너무 적습니다 — 덤프를 재생성하세요.")

    subj_j = subject[ni]
    fid_j = frame_id[ni]
    ear_j = e_val[ei]

    subs = sorted(set(subj_j.tolist()))
    idx_of = {s: i for i, s in enumerate(subs)}
    y_frame = np.array([idx_of[s] for s in subj_j], np.int64)
    chance = 1.0 / len(subs)

    print(f"subjects {len(subs)}  chance {chance:.3f}  "
          f"window {args.window} frames (~{args.window/15:.1f}s @15fps)  "
          f"device {args.device}\n")

    enc = Encoder(latent).to(args.device)
    enc.load_state_dict(torch.load(args.encoder, map_location=args.device))
    Z = encode_all(enc, images, args.device)[ni]
    print(f"vector {Z.shape}")

    ok = np.isfinite(ear_j)
    W, S = args.window, max(args.window // 4, 1)
    # 프레임 단위 표현에도 시간 좌표를 붙여 동일한 블록 분할을 쓴다
    t_frame = np.empty(len(subj_j))
    for s in subs:
        sel = np.where(subj_j == s)[0]
        sel = sel[np.argsort(fid_j[sel])]
        t_frame[sel] = np.linspace(0, 1, len(sel))

    Xw, yw, tw = make_windows(ear_j, subj_j, fid_j, W, S)
    Xs = window_stats(Xw)
    # 벡터도 같은 창으로 평균 요약 (창 단위 조건을 맞춘 참고값)
    Zw, yzw, tzw = [], [], []
    for s in subs:
        sel = np.where(subj_j == s)[0]
        sel = sel[np.argsort(fid_j[sel])]
        z = Z[sel]
        if len(z) < W:
            continue
        st = np.arange(0, len(z) - W + 1, S)
        Zw.append(np.stack([z[i:i + W].mean(0) for i in st]))
        yzw.append(np.full(len(st), idx_of[s], np.int64))
        tzw.append(st / max(len(z) - W, 1))
    Zw = np.concatenate(Zw); yzw = np.concatenate(yzw); tzw = np.concatenate(tzw)

    reps = [
        ("[A] ours  벡터 1개 (프레임)", Z[ok], y_frame[ok], "128", t_frame[ok]),
        ("[A] ear   스칼라 1개 (프레임)", ear_j[ok][:, None], y_frame[ok], "1", t_frame[ok]),
        (f"[B] ours  벡터 1개", Z[ok], y_frame[ok], "128", t_frame[ok]),
        (f"[B] ear   창 {W}프레임", Xw, yw, str(W), tw),
        ("[C] ear   창 요약통계", Xs, yw, str(Xs.shape[1]), tw),
        ("[참고] ours 창 평균벡터", Zw, yzw, "128", tzw),
    ]

    print(f"\n  분할: 피험자별 시간 블록 (앞 70% 학습 / 뒤 30% 평가)")
    print(f"  창 stride {S} 프레임 (겹침 허용 — 시간 분할이라 인접 누출 없음)\n")
    print(f"  {'표현':<28}{'차원':>5}{'표본':>8}{'선형 프로브':>14}{'MLP 프로브':>13}")
    print("  " + "-" * 70)
    out = []
    for name, X, y, dim, t in reps:
        lin = [probe(X, y, len(subs), args.device, True, args.epochs, s, t=t)
               for s in args.seeds]
        mlp = [probe(X, y, len(subs), args.device, False, args.epochs, s, t=t)
               for s in args.seeds]
        print(f"  {name:<28}{dim:>5}{len(y):>8}"
              f"{np.mean(lin):>9.3f}±{np.std(lin):<4.3f}"
              f"{np.mean(mlp):>8.3f}±{np.std(mlp):<4.3f}")
        out.append({"rep": name, "dim": dim, "n": int(len(y)),
                    "linear_mean": float(np.mean(lin)),
                    "linear_std": float(np.std(lin)),
                    "mlp_mean": float(np.mean(mlp)),
                    "mlp_std": float(np.std(mlp))})

    # 기존 RESULTS.md 의 0.998 과 이어보기 위한 무작위 프레임 분할 참고값
    r_lin = [probe(Z[ok], y_frame[ok], len(subs), args.device, True, args.epochs, s)
             for s in args.seeds]
    print("  " + "-" * 70)
    print(f"  {'[참고] ours, 무작위 프레임 분할':<28}{'128':>5}{int(ok.sum()):>8}"
          f"{np.mean(r_lin):>9.3f}±{np.std(r_lin):<4.3f}"
          f"{'  (기존 프로토콜)':>13}")
    out.append({"rep": "ours random-frame split", "dim": "128",
                "n": int(ok.sum()), "linear_mean": float(np.mean(r_lin)),
                "linear_std": float(np.std(r_lin)), "mlp_mean": None, "mlp_std": None})
    print(f"\n  chance = {chance:.3f}  ({len(subs)} subjects)")

    print("\n  읽는 법")
    print("   [A] 시스템이 실제로 내보내는 단위. 벡터가 높은 건 당연하다(128 vs 1).")
    print("   [B] ★ 숫자 개수를 맞춘 비교. EAR 도 시간축을 주면 얼마나 새는가.")
    print("       여기서 EAR 이 chance 근처면 -> 'EAR 이 프라이버시상 유리'가 사실이므로")
    print("       우리 주장을 '프라이버시 우위'가 아니라 '능력 대비 제어 가능성'으로 잡아야 한다.")
    print("       EAR 도 높게 나오면 -> '스칼라라서 안전한 게 아니다'가 성립한다.")
    print("   [C] 현실적인 EAR 시스템이 저장할 요약통계 수준의 누출.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"config": vars(args), "chance": chance,
                   "n_subjects": len(subs), "results": out},
                  open(args.json, "w"), indent=2, default=str)
        print(f"\n  -> {args.json}")


if __name__ == "__main__":
    main()
