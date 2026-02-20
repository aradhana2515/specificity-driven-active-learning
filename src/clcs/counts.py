from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_sequencing_counts(
    meas: pd.DataFrame,
    selected_idx: np.ndarray,
    n_in_reads: int = 2_000_000,
    n_out_reads: int = 200_000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate NGS counts for input library (uniform sampling) and output (selected gate).
    """
    rng = np.random.default_rng(seed)
    n = len(meas)

    # input reads ~ uniform multinomial
    in_counts = rng.multinomial(n_in_reads, np.full(n, 1.0 / n))

    # output reads: uniform over selected set (if empty, all zeros)
    out_counts = np.zeros(n, dtype=int)
    if len(selected_idx) > 0:
        p = np.zeros(n, dtype=float)
        p[selected_idx] = 1.0 / len(selected_idx)
        out_counts = rng.multinomial(n_out_reads, p)

    out = pd.DataFrame({
        "seq": meas["seq"].to_numpy(),
        "round": meas["round"].to_numpy(),
        "count_in": in_counts,
        "count_out": out_counts,
    })
    return out