"""classifier — 프레임워크 없는 깜빡임 분류기 (완전 vs 불완전).

순수 NumPy 로지스틱/얕은 MLP 라서 TF·PyTorch 없이 엣지에 배포 가능
(엣지 기여의 핵심). 표준화 통계를 가중치와 함께 저장한다.

참고: 완성도(completeness)는 기하학적으로도 계산되지만, 조명·자세·개인차에
강건한 판정을 위해 운동학 특징 전체를 쓰는 경량 분류기를 함께 둔다.
"""
import numpy as np

from . import config


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class BlinkClassifier:
    def __init__(self, in_dim=config.FEATURE_LEN, n_classes=len(config.LABELS),
                 hidden=16, seed=42):
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        if hidden > 0:
            self.W1 = rng.standard_normal((in_dim, hidden)) * np.sqrt(2 / in_dim)
            self.b1 = np.zeros(hidden)
            self.W2 = rng.standard_normal((hidden, n_classes)) * np.sqrt(2 / hidden)
            self.b2 = np.zeros(n_classes)
        else:
            self.W2 = rng.standard_normal((in_dim, n_classes)) * 0.01
            self.b2 = np.zeros(n_classes)
        self.mu = np.zeros(in_dim)
        self.sd = np.ones(in_dim)

    def _fit_scaler(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6

    def _scale(self, X):
        return (X - self.mu) / self.sd

    def _forward(self, Xs):
        if self.hidden > 0:
            self._z1 = Xs @ self.W1 + self.b1
            self._a1 = np.maximum(0, self._z1)
            logits = self._a1 @ self.W2 + self.b2
        else:
            self._a1 = Xs
            logits = Xs @ self.W2 + self.b2
        return _softmax(logits)

    def fit(self, X, y, epochs=400, lr=0.05, batch=32, l2=1e-4, verbose=False):
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.int64)
        self._fit_scaler(X)
        Xs = self._scale(X)
        Y = np.eye(self.n_classes)[y]
        n = len(Xs)
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                xb, yb = Xs[b], Y[b]
                p = self._forward(xb)
                dlogits = (p - yb) / len(b)
                if self.hidden > 0:
                    dW2 = self._a1.T @ dlogits + l2 * self.W2
                    db2 = dlogits.sum(0)
                    da1 = dlogits @ self.W2.T
                    dz1 = da1 * (self._z1 > 0)
                    dW1 = xb.T @ dz1 + l2 * self.W1
                    db1 = dz1.sum(0)
                    self.W1 -= lr * dW1; self.b1 -= lr * db1
                    self.W2 -= lr * dW2; self.b2 -= lr * db2
                else:
                    dW2 = xb.T @ dlogits + l2 * self.W2
                    db2 = dlogits.sum(0)
                    self.W2 -= lr * dW2; self.b2 -= lr * db2
            if verbose and (ep + 1) % 100 == 0:
                acc = (self.predict(X) == y).mean()
                print(f"  ep{ep+1}/{epochs}  train_acc={acc*100:.1f}%")
        return self

    def predict_proba(self, X):
        return self._forward(self._scale(np.asarray(X, np.float64)))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path=config.MODEL_PATH):
        d = dict(mu=self.mu, sd=self.sd, W2=self.W2, b2=self.b2,
                 hidden=self.hidden, n_classes=self.n_classes, in_dim=self.in_dim)
        if self.hidden > 0:
            d.update(W1=self.W1, b1=self.b1)
        np.savez(path, **d)
        return path

    @classmethod
    def load(cls, path=config.MODEL_PATH):
        d = np.load(path)
        clf = cls(in_dim=int(d["in_dim"]), n_classes=int(d["n_classes"]),
                  hidden=int(d["hidden"]))
        clf.mu, clf.sd = d["mu"], d["sd"]
        clf.W2, clf.b2 = d["W2"], d["b2"]
        if clf.hidden > 0:
            clf.W1, clf.b1 = d["W1"], d["b1"]
        return clf

    def param_count(self):
        ws = [self.W2, self.b2] + ([self.W1, self.b1] if self.hidden > 0 else [])
        return int(sum(w.size for w in ws))
