from __future__ import annotations
import numpy as np
from .pareto import pareto_frontier

def diversity_greedy(X_feat: np.ndarray, candidates: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Greedy diversity selection on dense feature vectors:
    pick a high-scoring seed then iteratively maximize min distance.
    """
    rng = np.random.default_rng(seed)
    if len(candidates) <= k:
        return candidates

    # start with random candidate (or best later by ordering candidates beforehand)
    chosen = [int(candidates[0])]
    cand = candidates[1:].tolist()

    # precompute candidate matrix
    Xc = X_feat[candidates]
    # map global idx -> local pos
    pos = {int(idx): i for i, idx in enumerate(candidates)}

    while len(chosen) < k and len(cand) > 0:
        chosen_local = np.array([pos[c] for c in chosen], dtype=int)
        remaining = np.array(cand, dtype=int)
        rem_local = np.array([pos[c] for c in remaining], dtype=int)

        # distances to nearest chosen
        # (use squared euclid)
        d2 = []
        for rl in rem_local:
            v = Xc[rl]
            mnd = np.inf
            for cl in chosen_local:
                u = Xc[cl]
                mnd = min(mnd, float(np.sum((v - u) ** 2)))
            d2.append(mnd)
        d2 = np.array(d2)

        best = int(remaining[int(np.argmax(d2))])
        chosen.append(best)
        cand.remove(best)

    return np.array(chosen, dtype=int)

def propose_next_round(
    seqs: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    X_dense_for_diversity: np.ndarray,
    batch_size: int = 256,
    expr_min: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Acquisition for next batch:
      - Objective: maximize t_norm and minimize o_norm
      - Constraint: predicted logE_expr >= expr_min
      - Score: (t_norm - o_norm) + 0.5 * sigma
      - Prefer Pareto set, then fill by score, then apply diversity.

    Returns:
      selected_idx, acquisition_score, pareto_flag
    """
    t = mu[:, 0]
    o = mu[:, 1]
    logE = mu[:, 2]

    feasible = logE >= expr_min
    base_score = (t - o) + 0.5 * sigma
    base_score = np.where(feasible, base_score, -np.inf)

    # Pareto on [t, o] (maximize t, minimize o)
    nd = pareto_frontier(mu[:, [0, 1]], maximize=np.array([True, False]))
    pareto_flag = np.zeros(len(seqs), dtype=int)
    pareto_flag[nd] = 1

    # rank candidates: prioritize pareto, then score
    order = np.lexsort((-base_score, -pareto_flag))  # last key primary, so use lexsort carefully
    # lexsort sorts by first key then second; we want pareto then score descending.
    # easiest: compute composite rank
    composite = pareto_flag * 1e6 + base_score
    ranked = np.argsort(composite)[::-1]

    # take a bigger candidate pool to allow diversity filtering
    pool = ranked[: min(len(ranked), batch_size * 6)]
    pool = pool[np.isfinite(base_score[pool])]

    if len(pool) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)

    # sort pool by score desc so diversity starts with best
    pool = pool[np.argsort(base_score[pool])[::-1]]

    # apply greedy diversity on dense features
    sel = diversity_greedy(X_dense_for_diversity, pool, k=min(batch_size, len(pool)), seed=seed)

    return sel, base_score[sel], pareto_flag[sel]