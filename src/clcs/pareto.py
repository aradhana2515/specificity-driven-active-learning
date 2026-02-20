from __future__ import annotations
import numpy as np

def pareto_frontier(points: np.ndarray, maximize: np.ndarray) -> np.ndarray:
    """
    Return indices of non-dominated points.
    maximize[j]=True means higher is better for dim j.
    """
    P = points.copy()
    for j, mx in enumerate(maximize):
        if mx:
            P[:, j] = -P[:, j]  # convert maximize to minimize

    n = P.shape[0]
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        dominates_i = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if np.any(dominates_i):
            is_nd[i] = False
            continue
        dominated_by_i = np.all(P[i] <= P, axis=1) & np.any(P[i] < P, axis=1)
        is_nd[dominated_by_i] = False
        is_nd[i] = True
    return np.where(is_nd)[0]

def frontier_proxy_score(mu: np.ndarray, nd_idx: np.ndarray) -> float:
    """
    Simple proxy: mean(t_norm - o_norm) among ND points.
    mu columns expected: [t_norm, o_norm, logE_expr, logE_enrich]
    """
    if len(nd_idx) == 0:
        return float("nan")
    t = mu[nd_idx, 0]
    o = mu[nd_idx, 1]
    return float(np.mean(t - o))