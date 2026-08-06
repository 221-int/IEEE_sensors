"""인코더 + 시간 헤드 학습, 그리고 EAR 비열등 판정. v2 본 실험.

    python -m src.v2.train_encoder --folds 0            # 한 fold 먼저
    python -m src.v2.train_encoder                      # 5-fold x 3시드 전체

프로토콜 (docs/v2/PROTOCOL.md)
-----------------------------
    분할      피험자 분리 5-fold 회전. test=fold i, val=fold i+1, train=나머지 3
    선택      에폭은 **val PR-AUC** 로 고르고 얼려서 test 에 적용
    시드      3개. 조건 차이가 시드 std 보다 작으면 결론은 "미측정"
    격자      임계값은 src/v2/common/thresholds.py 하나만 사용
    판정      EAR(drop_ratio) 대비 **비열등**. delta=0.02, 주 지표 PR-AUC
              (ours - EAR) 차이를 **같은 부트스트랩 표본에서 짝지어** 계산한다

EAR 베이스라인은 같은 이벤트 집합에서 인덱스의 `f_ear` 로 즉석 계산한다. 우리가 쓰는
이벤트와 **정확히 같은 집합**이어야 비교가 성립한다(PROTOCOL §10 S6).

이번 라운드는 **억제 기전 없는 기준선**이다. 적대 학습·정보 병목·사후 투영은 범위 밖.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from src.v2.common import repro, splits, stats
from src.v2.common import thresholds as TH
from src.v2.dataset import crop as C

DATA = "data/processed/v2"
OUT = "results/v2/train_encoder.json"
MODELS = "models/v2"
EVENT_LEN = 19
DELTA = 0.02                      # PROTOCOL §9-1, 2026-08-01 확정. 여기서 바꾸지 말 것


# ------------------------------------------------------------------ 산출물 보존
# 2026-08-03 추가. 이전 버전은 아래 세 가지를 잃었다.
#
#   1) 런별 test 점수(_test_scores 등)를 JSON 기록 직전에 버렸다. 그래서
#      **배치별·안경별 풀링 PR-AUC 와 그 짝지은 부트스트랩을 사후에 계산할 수 없었다.**
#      by_subject(피험자별 PR-AUC)만 남는데, 그것을 평균하는 것은 풀링 값과 다른 숫자다
#      (PROTOCOL §9-1: EAR 이 사용자별 0.930 / 섞으면 0.909. 배포 조건의 값은 후자).
#   2) `ear_head` 가중치를 저장하지 않아 (1) 을 체크포인트로 복원할 길도 없었다.
#      `ours` 가중치만 있으면 비교 상대가 없다.
#   3) 15런이 전부 끝나야 JSON 을 썼다. 런당 ~17분이므로 마지막 런에서 죽으면
#      4시간이 통째로 날아간다.
#
# 점수는 JSON 이 아니라 **npz 사이드카**에 넣는다. JSON 에 넣으면 런당 수천 개
# float 가 텍스트로 들어가 파일이 비대해지고 사람이 읽을 수 없게 된다.

def scores_dir(args) -> str:
    """`--out` 마다 다른 디렉터리를 쓴다. 실험 변형끼리 사이드카가 섞이면 안 된다."""
    return os.path.splitext(args.out)[0] + "_scores"


def _atomic_write(path: str, write_fn) -> None:
    """tmp 에 쓰고 os.replace. 쓰는 도중 죽어도 기존 파일이 반쯤 망가지지 않는다."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write_fn(f)
    os.replace(tmp, path)


def save_run_scores(args, fold: int, seed: int, scores: dict,
                    y: np.ndarray, subject: np.ndarray, baseline_used: str) -> str:
    """런 하나의 test 점수·정답·피험자를 npz 로 남긴다.

    세 경로(ours / ear_head / ear_rule)를 **모두** 남긴다. `baseline_used` 는
    test PR-AUC 로 고른 값이라 런마다 달라질 수 있으므로, 하나만 남기면
    사후에 "어느 베이스라인과의 비교였나"가 섞인다.
    """
    path = os.path.join(scores_dir(args), f"fold{fold}_seed{seed}.npz")
    _atomic_write(path, lambda f: np.savez_compressed(
        f,
        ours=np.asarray(scores["ours"], np.float64),
        ear_head=np.asarray(scores["ear_head"], np.float64),
        ear_rule=np.asarray(scores["ear_rule"], np.float64),
        y=np.asarray(y, np.int64),
        subject=np.asarray(subject, np.int64),
        fold=np.int64(fold), seed=np.int64(seed),
        baseline_used=np.array(baseline_used),
    ))
    return path


def lean_runs(runs: list[dict]) -> list[dict]:
    """JSON 에 넣을 형태 — 원시 배열(`_` 접두사)은 사이드카로 갔으므로 뺀다."""
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]


def write_partial(args, runs: list[dict]) -> None:
    """런 하나가 끝날 때마다 부분 결과를 기록한다(판정 없음).

    중간에 죽어도 **끝난 런까지는 남는다.** 이 파일은 최종 산출물이 아니므로
    `.partial.json` 으로 두어 인용 대상과 헷갈리지 않게 한다.
    """
    path = os.path.splitext(args.out)[0] + ".partial.json"
    payload = {"_note": "미완료 런의 중간 기록. 인용 금지. 판정(verdict) 없음.",
               "env": repro.env_fingerprint(), "config": vars(args),
               "input_norm": C.INPUT_NORM, "n_done": len(runs),
               "runs": lean_runs(runs)}
    _atomic_write(path, lambda f: f.write(
        json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")))


# ------------------------------------------------------------------ 데이터
class Bundle:
    """index.npz + memmap 을 묶고, 이벤트 단위로 꺼내 쓴다."""

    def __init__(self, data_dir: str, tag: str = "m22"):
        self.idx = np.load(os.path.join(data_dir, "index.npz"))
        self.frames = np.load(os.path.join(data_dir, f"frames_{tag}.npy"), mmap_mode="r")
        if len(self.idx["f_subject"]) != self.frames.shape[0]:
            raise SystemExit("index 와 memmap 길이가 다릅니다. phase3_merge 재실행.")
        keep = self.idx["e_valid"].astype(bool)          # 결측 <=5 (PROTOCOL §7)
        self.rows = self.idx["e_rows"][keep].astype(np.int64)
        self.y = self.idx["e_is_blink"][keep].astype(np.int64)
        self.subject = self.idx["e_subject"][keep].astype(int)
        self.mask = (self.rows >= 0).astype(np.float32)
        self.n_dropped = int((~keep).sum())

    def batch(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """이벤트 인덱스 -> (x (B,19,1,H,W) float32, mask (B,19), y (B,))"""
        r = self.rows[ids]
        flat = np.where(r >= 0, r, 0).ravel()
        imgs = np.asarray(self.frames[flat])             # (B*19, H, W) uint8
        x = C.batch_input(imgs)                          # 정규화는 crop.INPUT_NORM 하나만
        x = x.reshape(len(ids), EVENT_LEN, 1, *x.shape[-2:])
        return x, self.mask[ids], self.y[ids]

    def _ear_windows(self, ids: np.ndarray, cols=(2,)) -> np.ndarray:
        """(B, 19, len(cols)) EAR 창. 결측은 NaN."""
        e = self.idx["f_ear"][:, list(cols)].astype(np.float64)
        r = self.rows[ids]
        ok = r >= 0
        v = e[np.where(ok, r, 0)]
        return np.where(ok[..., None], v, np.nan)

    def ear_feature(self, ids: np.ndarray) -> np.ndarray:
        """EAR drop_ratio 스칼라 (규칙 기반 베이스라인).

        창의 0·18번이 아니라 **가용한 첫/마지막** 프레임을 기준선으로 쓴다.
        고정 위치를 쓰면 그 프레임이 결측인 이벤트(0.38%)에서 특징이 0 으로 뭉개져
        베이스라인만 불리해진다 — 우리 쪽은 마스크로 처리하므로 비대칭이 생긴다.
        """
        v = self._ear_windows(ids)[..., 0]
        ok = np.isfinite(v)
        any_ok = ok.any(axis=1)
        first_i = np.argmax(ok, axis=1)
        last_i = v.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
        first = np.take_along_axis(v, first_i[:, None], 1)[:, 0]
        last = np.take_along_axis(v, last_i[:, None], 1)[:, 0]
        edge = (first + last) / 2.0
        with np.errstate(invalid="ignore"):
            lo = np.where(any_ok, np.nanmin(np.where(ok, v, np.inf), axis=1), np.nan)
            out = (edge - lo) / edge
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def ear_batch(self, ids: np.ndarray, cols=(0, 1, 2, 3)):
        """대조군 1용: (B, 19, k) EAR 시퀀스 + 마스크.

        **PROTOCOL §9 대조군 1 = "EAR 스칼라 x 19프레임 -> 우리와 동일한 판정 헤드"** 다.
        규칙 기반 drop_ratio 는 그보다 약한 베이스라인이므로, 그것만 이기고
        "표현의 기여"를 주장하면 안 된다.
        결측은 0 으로 채우고 마스크로 알린다(헤드가 pooling 에서 제외한다).
        """
        v = self._ear_windows(ids, cols)
        m = np.isfinite(v[..., 0]).astype(np.float32)
        return np.nan_to_num(v, nan=0.0).astype(np.float32), m


# ------------------------------------------------------------------ 학습
def train_one(bundle: Bundle, mode: str, tr, va, seed: int, args, dev):
    """mode='cnn' (우리) 또는 'earhead' (대조군 1). **학습 절차는 완전히 동일하다.**

    같은 옵티마이저·에폭 수·배치·선택 기준을 쓴다. 다른 것은 프레임마다 무엇을
    뽑는가 하나뿐이다.
    """
    import torch
    from torch import nn
    from src.v2.model import encoder as E

    repro.seal(seed)
    if mode == "cnn":
        # --front 는 **프론트엔드 종류**, --arch 는 **인코더 구조**다. 다른 축이다.
        #   ours            크롭 -> vpres/sym16/vdrop/vfull -> D차원   (--arch 가 여기 붙는다)
        #   image_cnn_head  크롭 -> mEBAL CNN -> D차원 -> 우리와 동일한 시간 헤드
        #   image_cnn_max   크롭 -> mEBAL CNN -> 점수 1개 -> max pooling (원문 §5.1)
        if args.front == "ours":
            front = E.build(args.arch, args.latent).to(dev)
        elif args.front == "image_cnn_head":
            front = E.build_image_cnn(args.latent).to(dev)
        elif args.front == "image_cnn_max":
            front = E.build_image_cnn(1).to(dev)
        else:
            raise ValueError(f"unknown --front {args.front!r}")
        def fetch(ids):
            x, mk, y = bundle.batch(ids)
            b, t = x.shape[:2]
            return torch.from_numpy(x.reshape(b * t, *x.shape[2:])), mk, y, b, t
    else:
        cols = (0, 1, 2, 3) if args.ear_feats == "all4" else (2,)
        front = E.build_ear_frontend(len(cols), args.latent).to(dev)
        def fetch(ids):
            v, mk = bundle.ear_batch(ids, cols)
            b, t, k = v.shape
            return torch.from_numpy(v.reshape(b * t, k)), mk, bundle.y[ids], b, t

    # image_cnn_max 만 학습 파라미터 없는 max pooling 을 쓴다(mEBAL 원문 §5.1).
    # 대조군 1(earhead)은 어떤 --front 에서도 우리와 같은 시간 헤드를 쓴다 — 그래야
    # "무엇을 프레임마다 뽑는가"만 달라진다.
    if mode == "cnn" and args.front == "image_cnn_max":
        head = E.build_max_head().to(dev)
    else:
        head = E.build_head(args.latent, EVENT_LEN).to(dev)
    opt = torch.optim.Adam(list(front.parameters()) + list(head.parameters()), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    g = np.random.default_rng(seed)

    def infer(ids):
        front.eval(); head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), args.batch):
                xb, mk, _, b, t = fetch(ids[i:i + args.batch])
                z = front(xb.to(dev)).reshape(b, t, -1)
                out.append(head(z, torch.from_numpy(mk).to(dev)).cpu().numpy())
        return np.concatenate(out)

    # 고정 에폭 수는 **공정하지 않다.** 한쪽이 일찍 포화하고 다른 쪽이 아직 오르는
    # 중이면, 에폭 예산이 같아도 덜 학습된 쪽이 손해를 본다(1차 실행에서 실제로 발생:
    # ours 는 ep5 에 정점, ear_head 는 ep30 에서도 상승 중이었다).
    # -> **양쪽에 같은 규칙**을 적용한다: val AP 가 patience 에폭 동안 갱신되지 않으면 중단.
    best, hist, since = {"val_ap": -1.0}, [], 0
    for ep in range(1, args.max_epochs + 1):
        front.train(); head.train()
        perm = g.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            ids = perm[i:i + args.batch]
            xb, mk, y, b, t = fetch(ids)
            z = front(xb.to(dev)).reshape(b, t, -1)
            logit = head(z, torch.from_numpy(mk).to(dev))
            loss = lossf(logit, torch.from_numpy(np.asarray(y)).float().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(ids)
        ap = TH.average_precision(TH.canonical(infer(va), True), bundle.y[va])
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "val_ap": ap})
        if ap > best["val_ap"]:
            best = {"val_ap": ap, "epoch": ep,
                    "front": {k: v.detach().cpu().clone() for k, v in front.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            since = 0
        else:
            since += 1
        if args.log and (ep % 5 == 0 or ep == 1):
            print(f"      [{mode}] ep {ep:3d}  loss {tot/len(perm):.4f}  val AP {ap:.4f}"
                  f"{'  *' if since == 0 else ''}", flush=True)
        if since >= args.patience:
            if args.log:
                print(f"      [{mode}] val AP 가 {args.patience} 에폭 정체 -> ep {ep} 에서 중단"
                      f" (최고 ep {best['epoch']})", flush=True)
            break

    front.load_state_dict(best["front"]); head.load_state_dict(best["head"])
    front.to(dev); head.to(dev)
    best["stopped_at"] = ep
    best["converged"] = ep < args.max_epochs      # False 면 예산이 모자랐다는 뜻
    return front, head, infer, best, hist


def run_fold(bundle: Bundle, fold: int, seed: int, args) -> dict:
    t_run = time.perf_counter()
    import torch
    from src.v2.model import encoder as E  # noqa: F401

    repro.seal(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assign = splits.load_folds()
    m = splits.subject_masks(bundle.subject, assign, splits.fold_rotation(fold))
    tr, va, te = (np.flatnonzero(m[k]) for k in ("train", "val", "test"))

    _, _, infer, best, hist = train_one(bundle, "cnn", tr, va, seed, args, dev)
    _, _, infer_e, best_e, hist_e = train_one(bundle, "earhead", tr, va, seed, args, dev)

    TH.AUDIT.reset()
    res, scores = {}, {}
    # 세 경로를 **동일한 격자·동일한 선택 절차**로 평가한다
    for name, sv, st in (
        ("ours", infer(va), infer(te)),
        ("ear_head", infer_e(va), infer_e(te)),
        ("ear_rule", bundle.ear_feature(va), bundle.ear_feature(te)),
    ):
        cv, ct = TH.canonical(sv, True), TH.canonical(st, True)
        pick = TH.select_threshold(cv, bundle.y[va], TH.crit_accuracy(), name=name)
        ev = TH.evaluate_at(ct, bundle.y[te], pick["thr"])
        res[name] = {"accuracy": ev["accuracy"], "precision": ev["precision"],
                     "pr_auc": TH.average_precision(ct, bundle.y[te]),
                     "roc_auc": TH.roc_auc(ct, bundle.y[te]), "thr": pick["thr"]}
        scores[name] = ct
    TH.AUDIT.require_uniform()

    # 비교 상대는 **더 강한 쪽**이다. 약한 쪽을 골라 이기면 방법의 성능이 아니다.
    # ⚠️ 이 선택은 **test PR-AUC** 로 한다. 방향은 우리에게 불리한 쪽(베이스라인을
    # 최대한 강하게)이지만 test 의존 선택인 것은 사실이므로, 보고할 때 런별
    # `baseline_used` 를 반드시 병기한다.
    strong = max(("ear_head", "ear_rule"), key=lambda k: res[k]["pr_auc"])

    if args.save_models:
        d = os.path.join(MODELS, f"fold{fold}_seed{seed}")
        os.makedirs(d, exist_ok=True)
        torch.save(best["front"], os.path.join(d, "encoder.pt"))
        torch.save(best["head"], os.path.join(d, "head.pt"))
        # 대조군 가중치도 저장한다. 이것이 없으면 사후에 ours 만 복원되고
        # 비교 상대가 없어 서브그룹 비교 자체가 성립하지 않는다.
        torch.save(best_e["front"], os.path.join(d, "earhead_front.pt"))
        torch.save(best_e["head"], os.path.join(d, "earhead_head.pt"))

    # 점수 사이드카 — `--save-models` 와 무관하게 **항상** 남긴다(런당 수백 KB).
    scores_file = save_run_scores(args, fold, seed, scores,
                                  bundle.y[te], bundle.subject[te], strong)

    # 피험자별 분해 — "이 이득이 어디서 나오는가"에 답하기 위해 반드시 남긴다.
    # 이득이 EAR 이 약한 사용자(랜드마크가 눈꺼풀을 못 따라가는 사람)에만 몰려 있다면,
    # 그건 "표현이 좋다"가 아니라 "베이스라인이 그 사용자에서 불리하다"는 뜻이다.
    yt, st_ = bundle.y[te], bundle.subject[te]
    by_subject = {}
    for u in np.unique(st_):
        k = st_ == u
        if len(np.unique(yt[k])) < 2:
            continue
        by_subject[int(u)] = {
            "n": int(k.sum()),
            "ours_pr_auc": TH.average_precision(scores["ours"][k], yt[k]),
            "ear_head_pr_auc": TH.average_precision(scores["ear_head"][k], yt[k]),
            "ear_rule_pr_auc": TH.average_precision(scores["ear_rule"][k], yt[k]),
        }
    return {
        "fold": fold, "seed": seed,
        "best_epoch": best["epoch"], "best_epoch_earhead": best_e["epoch"],
        "converged": bool(best["converged"]), "converged_earhead": bool(best_e["converged"]),
        "stopped_at": best["stopped_at"], "stopped_at_earhead": best_e["stopped_at"],
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "val_optimism_ap": float(best["val_ap"] - res["ours"]["pr_auc"]),
        "val_optimism_ap_earhead": float(best_e["val_ap"] - res["ear_head"]["pr_auc"]),
        "baseline_used": strong,
        **{k: res[k] for k in res},
        "_test_scores": scores["ours"].tolist(),
        "_ear_scores": scores[strong].tolist(),
        "_ear_rule_scores": scores["ear_rule"].tolist(),
        "_ear_head_scores": scores["ear_head"].tolist(),
        "_test_y": bundle.y[te].tolist(), "_test_subject": bundle.subject[te].tolist(),
        "by_subject": by_subject, "seconds": round(time.perf_counter() - t_run, 1),
        "scores_file": scores_file.replace("\\", "/"),
        "history": hist, "history_earhead": hist_e,
    }


def verdict(runs: list[dict], n_boot: int, seed: int = 0) -> dict:
    """test 결과를 모아 **짝지은** 피험자 부트스트랩으로 비열등을 판정한다.

    두 방법의 CI 를 따로 구해 비교하면 피험자 효과가 상쇄되지 않아 불필요하게 넓어진다.
    같은 재표집 표본에서 둘 다 재고 **차이**를 부트스트랩해야 한다.
    """
    s_ours = np.concatenate([np.asarray(r["_test_scores"]) for r in runs])
    s_ear = np.concatenate([np.asarray(r["_ear_scores"]) for r in runs])
    y = np.concatenate([np.asarray(r["_test_y"]) for r in runs])
    sub = np.concatenate([np.asarray(r["_test_subject"]) for r in runs])

    def diff(rows):
        return (TH.average_precision(s_ours[rows], y[rows])
                - TH.average_precision(s_ear[rows], y[rows]))

    boot = stats.subject_bootstrap(diff, sub, n_boot=n_boot, seed=seed)
    v = stats.non_inferiority(boot["ci_lo"], boot["ci_hi"], DELTA)
    per_fold_diff = [r["ours"]["pr_auc"] - r[r["baseline_used"]]["pr_auc"] for r in runs]
    return {"delta": DELTA, "metric": "pr_auc", "paired_bootstrap": boot,
            "verdict": v, "per_run_diff": per_fold_diff,
            "fold_bootstrap": stats.fold_bootstrap(per_fold_diff)}


def main() -> int:
    repro.ensure_hashseed()
    repro.seal(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tag", default="m22")
    ap.add_argument("--arch", default="vpres",
                    help="인코더 **구조** (vpres/sym16/vdrop/vfull). --front ours 일 때만 쓰인다")
    ap.add_argument("--front", default="ours",
                    choices=["ours", "image_cnn_head", "image_cnn_max"],
                    help="프론트엔드 **종류**. --arch 와 다른 축이다. "
                         "image_cnn_* 는 mEBAL 원문 구조(RELATED_WORK §A2-1)")
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=80,
                    help="상한. 실제 종료는 patience 로 결정된다(양쪽 동일 규칙)")
    ap.add_argument("--patience", type=int, default=10,
                    help="val AP 가 이 에폭 동안 갱신 안 되면 중단. ours/베이스라인 공통")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ear-feats", default="all4", choices=["mean", "all4"],
                    help="대조군 1 이 쓰는 EAR 특징. all4 = left,right,mean,min (베이스라인에 관대한 쪽)")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--log", action="store_true", default=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    b = Bundle(args.data, args.tag)
    print(f"이벤트 {len(b.y):,} (결측정책으로 제외 {b.n_dropped:,})  "
          f"{splits.chance_report(b.y)}")
    desc = (f"구조 {args.arch}" if args.front == "ours"
            else f"프론트 {args.front} (mEBAL 원문 구조)")
    print(f"{desc}  D={args.latent}  정규화 {C.INPUT_NORM}  "
          f"max_epochs {args.max_epochs} patience {args.patience}  batch {args.batch}")

    runs = []
    t0 = time.perf_counter()
    for fold in args.folds:
        for seed in args.seeds:
            print(f"\n  fold {fold} seed {seed}")
            r = run_fold(b, fold, seed, args)
            runs.append(r)
            write_partial(args, runs)     # 중간에 죽어도 끝난 런은 남는다
            warn = "" if r["converged"] and r["converged_earhead"] else "   ! 예산 부족"
            print(f"    ours PR-AUC {r['ours']['pr_auc']:.4f} | "
                  f"ear_head {r['ear_head']['pr_auc']:.4f} | "
                  f"ear_rule {r['ear_rule']['pr_auc']:.4f} | "
                  f"비교상대 {r['baseline_used']} 차이 "
                  f"{r['ours']['pr_auc']-r[r['baseline_used']]['pr_auc']:+.4f}"
                  f"  (ep {r['best_epoch']}/{r['best_epoch_earhead']}, {r['seconds']:.0f}s){warn}")

    vd = verdict(runs, args.n_boot)
    out = {"env": repro.env_fingerprint(), "config": vars(args),
           "input_norm": C.INPUT_NORM, "runs": lean_runs(runs), "verdict": vd,
           "scores_dir": scores_dir(args).replace("\\", "/"),
           "minutes": round((time.perf_counter() - t0) / 60, 1)}

    def ms(key):
        a = np.array([r[key]["pr_auc"] for r in runs])
        sd = a.std(ddof=1) if len(a) > 1 else float("nan")
        return a.mean(), sd
    print(f"\n{'='*64}")
    for k in ("ours", "ear_head", "ear_rule"):
        m, s = ms(k)
        print(f"  {k:9s} PR-AUC {m:.4f}" + (f" ± {s:.4f}" if len(runs) > 1 else "  (n=1)"))
    pb = vd["paired_bootstrap"]
    print(f"  차이(ours-EAR) {pb['point']:+.4f}  95% CI [{pb['ci_lo']:+.4f}, {pb['ci_hi']:+.4f}]")
    print(f"  delta = {DELTA}  ->  판정: **{vd['verdict']}**")
    print(f"  val 낙관(AP) 평균 {np.mean([r['val_optimism_ap'] for r in runs]):+.4f}")

    _atomic_write(args.out, lambda f: f.write(
        json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8")))
    partial = os.path.splitext(args.out)[0] + ".partial.json"
    if os.path.exists(partial):
        os.remove(partial)            # 최종본이 생겼으니 중간본을 남겨 혼동시키지 않는다
    print(f"  -> {args.out}")
    print(f"  -> {scores_dir(args)}/  (런별 test 점수. 서브그룹 분석은 여기서 한다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
