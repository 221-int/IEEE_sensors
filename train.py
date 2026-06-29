"""
train.py — 깜빡임-종류 분류기를 피험자 단위 평가로 학습.

핵심 방법론 선택 (이전 프로젝트의 누수 문제 해결):
  ► Leave-One-Subject-Out(LOSO) 교차검증. 같은 피험자의 깜빡임이 train 과
    test 에 동시에 들어가지 않으므로, 보고 정확도가 외운 프레임이 아니라
    *새로운 사람*에 대한 일반화를 반영한다.

데이터 형식 (data/blinks.csv), 깜빡임 1건당 한 행:
    subject_id, label, f0..f10   (특징명은 config.FEATURE_NAMES)

실행:
    python train.py                 # data/blinks.csv 로 학습 + LOSO
    python train.py --synthetic     # 합성 데이터 생성 후 전체 파이프라인 점검
"""
import argparse
import csv
import os

import numpy as np

import config
from model import BlinkClassifier


# ── 데이터 입출력 ─────────────────────────────────────────────────────────────
def load_csv(path=config.BLINK_CSV):
    subj, y, X = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            subj.append(row["subject_id"])
            y.append(int(row["label"]))
            X.append([float(row[name]) for name in config.FEATURE_NAMES])
    return np.array(X, np.float32), np.array(y, np.int64), np.array(subj)


# ── 지표 ─────────────────────────────────────────────────────────────────────
def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf(cm):
    """클래스별 precision/recall/F1 과 macro 평균."""
    out = {}
    ps, rs, fs = [], [], []
    for i in range(len(cm)):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[config.LABELS[i]] = (prec, rec, f1, int(cm[i].sum()))
        ps.append(prec); rs.append(rec); fs.append(f1)
    out["macro"] = (np.mean(ps), np.mean(rs), np.mean(fs), int(cm.sum()))
    return out


# ── LOSO 교차검증 ────────────────────────────────────────────────────────────
def loso(X, y, subj):
    n = len(config.LABELS)
    cm = np.zeros((n, n), int)
    for s in np.unique(subj):
        te = subj == s
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        clf = BlinkClassifier().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        cm += confusion(y[te], pred, n)
    return cm


# ── 합성 데이터 (카메라 없이 파이프라인 점검용) ──────────────────────────────
def make_synthetic(n_subjects=8, per_class=40, seed=0):
    """그럴듯한 voluntary/spontaneous 깜빡임 특징을 생성.

    문헌 가정을 인코딩: voluntary 는 더 길고/완전하고/느리며, spontaneous 는
    더 짧고/빠르고/얕다. LOSO 가 의미 있도록 피험자별 오프셋을 더한다.
    ※ 합성 수치이므로 결과 자체는 의미 없음 — 파이프라인 동작 확인용.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sj in range(n_subjects):
        soff = rng.normal(0, 0.05, size=config.FEATURE_LEN)
        for label in (0, 1):                 # 0 spontaneous, 1 voluntary
            for _ in range(per_class):
                if label == 1:               # voluntary: 길고, 완전하고, 느림
                    dur = rng.normal(0.35, 0.05)
                    amp = rng.normal(0.85, 0.08)
                    vel = rng.normal(6.0, 1.2)
                else:                         # spontaneous: 짧고, 빠르고, 얕음
                    dur = rng.normal(0.20, 0.04)
                    amp = rng.normal(0.60, 0.12)
                    vel = rng.normal(9.0, 1.8)
                dur = max(dur, 0.05); amp = float(np.clip(amp, 0.1, 1.0))
                dc, do = dur * 0.45, dur * 0.55
                f = np.array([
                    dur, dc, dur * 0.2, do,
                    amp, 1 - amp,
                    vel, vel * 0.8,
                    dc / do, 1.25,
                    amp * dur,
                ], np.float32) + soff.astype(np.float32)
                rows.append((f"S{sj:02d}", label, f))
    return rows


def save_rows(rows, path=config.BLINK_CSV):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "label"] + config.FEATURE_NAMES)
        for sid, label, feat in rows:
            w.writerow([sid, label] + [f"{v:.6f}" for v in feat])
    print(f"{len(rows)}개 깜빡임 기록 -> {path}")


def report(cm, title):
    m = prf(cm)
    acc = np.trace(cm) / max(cm.sum(), 1)
    print(f"\n=== {title} ===")
    print(f"혼동행렬 (행=정답 {config.LABELS}):\n{cm}")
    print(f"{'class':12s}{'Prec':>7s}{'Rec':>7s}{'F1':>7s}{'N':>7s}")
    for k in config.LABELS + ["macro"]:
        p, r, f1, n = m[k]
        print(f"{k:12s}{p:7.3f}{r:7.3f}{f1:7.3f}{n:7d}")
    print(f"전체 정확도: {acc*100:.1f}%")
    return acc, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="합성 데이터를 먼저 생성 (카메라 불필요)")
    args = ap.parse_args()

    if args.synthetic or not os.path.exists(config.BLINK_CSV):
        if not os.path.exists(config.BLINK_CSV):
            print("[info] 실제 데이터가 없어 합성 데이터를 사용합니다.")
        save_rows(make_synthetic())

    X, y, subj = load_csv()
    print(f"{len(X)}개 깜빡임 로드 ({len(np.unique(subj))}명 피험자, "
          f"클래스={np.bincount(y).tolist()})")

    # LOSO 교차검증 (헤드라인 수치)
    cm = loso(X, y, subj)
    report(cm, "LEAVE-ONE-SUBJECT-OUT")

    # 전체 데이터로 최종 모델 학습 후 저장 (liveness.py 용)
    clf = BlinkClassifier().fit(X, y)
    clf.save()
    print(f"\n모델 저장 ({clf.param_count()} params) -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
