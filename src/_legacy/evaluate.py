"""
evaluate.py — 논문용 평가/그림 생성 (오프라인, 카메라 불필요).

생성물 (results/):
  Table  : LOSO 분류 성능 (precision/recall/F1) + 추론 latency·모델 크기
  Fig C  : LOSO 혼동행렬 히트맵
  Fig D  : voluntary vs spontaneous 특징 분포 (분리도)
  Fig R  : liveness ROC 곡선 + EER (genuine vs 스푸핑 공격셋)

데이터가 없으면 train.make_synthetic 으로 합성 데이터를 만들어 파이프라인을
끝까지 점검한다(합성 수치는 의미 없음, 동작 확인용).

실행:
    python evaluate.py
"""
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model import BlinkClassifier
import train as T

OUT = config.RESULTS_DIR
os.makedirs(OUT, exist_ok=True)


# ── 분류 성능 + 혼동행렬 그림 ────────────────────────────────────────────────
def eval_classification(X, y, subj):
    cm = T.loso(X, y, subj)
    acc, m = T.report(cm, "LEAVE-ONE-SUBJECT-OUT")

    n = len(config.LABELS)
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(config.LABELS, rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels(config.LABELS, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_confusion.pdf"))
    plt.close(fig)
    return cm, acc, m


# ── 특징 분포 그림 (분리도) ──────────────────────────────────────────────────
def fig_feature_distributions(X, y):
    keys = ["dur_total", "amplitude", "peak_close_vel", "asym_vel"]
    idx = [config.FEATURE_NAMES.index(k) for k in keys]
    fig, axes = plt.subplots(1, len(keys), figsize=(9, 2.4))
    for ax, k, j in zip(axes, keys, idx):
        data = [X[y == c, j] for c in range(len(config.LABELS))]
        ax.boxplot(data, labels=config.LABELS, showfliers=False)
        ax.set_title(k, fontsize=9)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    fig.suptitle("Voluntary vs spontaneous — kinematic feature distributions",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_feature_dist.pdf"))
    plt.close(fig)


# ── 추론 latency / 모델 크기 ─────────────────────────────────────────────────
def latency_benchmark(clf, X, repeats=2000):
    x = X[:1].astype(np.float64)
    # warm-up
    for _ in range(50):
        clf.predict(x)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        clf.predict(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    size_kb = os.path.getsize(config.MODEL_PATH) / 1024 if os.path.exists(
        config.MODEL_PATH) else 0
    print(f"\n=== EDGE BENCHMARK ===")
    print(f"params={clf.param_count()}  model={size_kb:.1f} KB")
    print(f"latency/blink: mean={ts.mean():.4f} ms  p95={np.percentile(ts,95):.4f} ms")
    return ts.mean(), float(np.percentile(ts, 95)), clf.param_count(), size_kb


# ── liveness ROC / EER ───────────────────────────────────────────────────────
def _roc_eer(genuine_scores, attack_scores):
    """genuine(높을수록 진짜) vs attack 점수로 ROC/EER 계산."""
    g = np.asarray(genuine_scores); a = np.asarray(attack_scores)
    thr = np.unique(np.concatenate([g, a]))
    tpr, fpr = [], []
    for t in thr:
        tpr.append(np.mean(g >= t))       # genuine 통과율
        fpr.append(np.mean(a >= t))       # attack 오통과율(FAR)
    tpr, fpr = np.array(tpr), np.array(fpr)
    frr = 1 - tpr
    eer_i = int(np.argmin(np.abs(fpr - frr)))
    eer = (fpr[eer_i] + frr[eer_i]) / 2
    order = np.argsort(fpr)
    auc = np.trapz(tpr[order], fpr[order])
    return fpr, tpr, eer, auc


def liveness_eval(clf, n_challenges=300, seed=1):
    """챌린지 단위 liveness 평가.

    genuine 시도 = voluntary 깜빡임 N회(살아있는 사용자).
    attack  시도 = spontaneous 깜빡임 N회(재생/딥페이크가 의도 깜빡임 운동학을
                  못 만든다는 가정). 각 시도 점수 = 평균 P(voluntary).
    """
    rng = np.random.default_rng(seed)
    N = config.CHALLENGE_N_BLINKS
    v_idx = config.LABEL_MAP["voluntary"]

    def synth_block(label):
        rows = T.make_synthetic(n_subjects=1, per_class=N, seed=int(rng.integers(1e9)))
        return np.array([f for _, lb, f in rows if lb == label][:N])

    g_scores, a_scores = [], []
    for _ in range(n_challenges):
        gv = clf.predict_proba(synth_block(1))[:, v_idx]      # genuine: voluntary
        av = clf.predict_proba(synth_block(0))[:, v_idx]      # attack: spontaneous
        g_scores.append(gv.mean())
        a_scores.append(av.mean())

    fpr, tpr, eer, auc = _roc_eer(g_scores, a_scores)
    print(f"\n=== LIVENESS (challenge-level) ===")
    print(f"AUC={auc:.3f}  EER={eer*100:.1f}%")

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot(fpr, tpr, color="#2563eb", lw=1.8, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.scatter([eer], [1 - eer], color="#dc2626", zorder=5,
               label=f"EER={eer*100:.1f}%")
    ax.set_xlabel("FAR (attack accept)"); ax.set_ylabel("TAR (genuine accept)")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_roc.pdf"))
    plt.close(fig)
    return eer, auc


def main():
    # 데이터 확보 (없으면 합성)
    if not os.path.exists(config.BLINK_CSV):
        print("[info] 실제 데이터가 없어 합성 데이터를 사용합니다.")
        T.save_rows(T.make_synthetic())
    X, y, subj = T.load_csv()
    print(f"{len(X)}개 깜빡임 / {len(np.unique(subj))}명 피험자")

    eval_classification(X, y, subj)
    fig_feature_distributions(X, y)

    clf = BlinkClassifier().fit(X, y)
    clf.save()
    latency_benchmark(clf, X)
    liveness_eval(clf)

    print(f"\n결과 저장 -> {OUT}")


if __name__ == "__main__":
    main()
