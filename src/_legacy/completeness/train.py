"""train — 깜빡임 분류기(완전/불완전) 학습 + Leave-One-Subject-Out 평가.

방법론: LOSO 교차검증으로 같은 피험자가 train/test 에 동시에 안 들어가게 해,
보고 정확도가 새 사람에 대한 일반화를 반영하도록 한다.

data/blinks.csv 형식(깜빡임 1건당 1행): subject_id, label, <FEATURE_NAMES...>

실행 (src/ 에서):
    python -m scripts.train                # data/blinks.csv 로 학습 + LOSO
    python -m scripts.train --synthetic    # 합성 데이터로 전체 파이프라인 점검
"""
import argparse
import csv
import os

import numpy as np

from eyeblink import config
from eyeblink.classifier import BlinkClassifier
from eyeblink.synth import make_synthetic


def load_csv(path=config.BLINK_CSV):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [n for n in config.FEATURE_NAMES if n not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV 특징 스키마 불일치(누락: {missing}).")
        subj, y, X = [], [], []
        for row in reader:
            subj.append(row["subject_id"])
            y.append(int(row["label"]))
            X.append([float(row[n]) for n in config.FEATURE_NAMES])
    return np.array(X, np.float32), np.array(y, np.int64), np.array(subj)


def save_rows(rows, path=config.BLINK_CSV):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "label"] + config.FEATURE_NAMES)
        for sid, label, feat in rows:
            w.writerow([sid, label] + [f"{v:.6f}" for v in feat])
    print(f"{len(rows)}개 깜빡임 기록 -> {path}")


def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf(cm):
    out, ps, rs, fs = {}, [], [], []
    for i in range(len(cm)):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[config.LABELS[i]] = (prec, rec, f1, int(cm[i].sum()))
        ps.append(prec); rs.append(rec); fs.append(f1)
    out["macro"] = (np.mean(ps), np.mean(rs), np.mean(fs), int(cm.sum()))
    return out


def loso(X, y, subj):
    n = len(config.LABELS)
    cm = np.zeros((n, n), int)
    for s in np.unique(subj):
        te = subj == s
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        clf = BlinkClassifier().fit(X[tr], y[tr])
        cm += confusion(y[te], clf.predict(X[te]), n)
    return cm


def report(cm, title):
    m = prf(cm)
    acc = np.trace(cm) / max(cm.sum(), 1)
    print(f"\n=== {title} ===")
    print(f"혼동행렬 (행=정답 {config.LABELS}):\n{cm}")
    print(f"{'class':12s}{'Prec':>7s}{'Rec':>7s}{'F1':>7s}{'N':>7s}")
    for k in config.LABELS + ["macro"]:
        p, r, f1, nn = m[k]
        print(f"{k:12s}{p:7.3f}{r:7.3f}{f1:7.3f}{nn:7d}")
    print(f"전체 정확도: {acc*100:.1f}%")
    return acc, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="합성 데이터를 먼저 생성 (카메라 불필요)")
    args = ap.parse_args()

    need_synth = args.synthetic or not os.path.exists(config.BLINK_CSV)
    if need_synth:
        if not args.synthetic:
            print("[info] 실제 데이터가 없어 합성 데이터를 사용합니다.")
        save_rows(make_synthetic())
    try:
        X, y, subj = load_csv()
    except ValueError as e:
        print(f"[warn] {e} 합성 데이터로 대체합니다.")
        save_rows(make_synthetic())
        X, y, subj = load_csv()

    print(f"{len(X)}개 깜빡임 로드 ({len(np.unique(subj))}명, "
          f"클래스={np.bincount(y).tolist()})")

    cm = loso(X, y, subj)
    report(cm, "LEAVE-ONE-SUBJECT-OUT")

    clf = BlinkClassifier().fit(X, y)
    clf.save()
    print(f"\n모델 저장 ({clf.param_count()} params) -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
