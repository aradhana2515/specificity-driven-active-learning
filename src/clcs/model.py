from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

@dataclass
class MultiTaskRidge:
    alpha: float = 5.0

    def fit(self, X, y: np.ndarray):
        base = Ridge(alpha=self.alpha, random_state=0)
        self.model = MultiOutputRegressor(base)
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X, y: np.ndarray) -> dict:
        pred = self.predict(X)
        return {
            "r2_tnorm": float(r2_score(y[:, 0], pred[:, 0])),
            "r2_onorm": float(r2_score(y[:, 1], pred[:, 1])),
            "r2_logE_expr": float(r2_score(y[:, 2], pred[:, 2])),
            "r2_logE_enrich": float(r2_score(y[:, 3], pred[:, 3])),
        }

def bootstrap_ensemble(
    X, y: np.ndarray, X_pool,
    n_models: int = 10,
    alpha: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit bootstrapped multitask ridge ensemble; return mean prediction and scalar uncertainty per sample.
    Uncertainty is mean std across outputs.
    """
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    preds = []
    for m in range(n_models):
        idx = rng.integers(0, n, size=n)
        model = MultiTaskRidge(alpha=alpha).fit(X[idx], y[idx])
        preds.append(model.predict(X_pool))
    preds = np.stack(preds, axis=0)  # (M, N, D)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0).mean(axis=1)  # scalar
    return mu, sigma