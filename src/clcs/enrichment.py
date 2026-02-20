from __future__ import annotations
import numpy as np
import pandas as pd

def log_enrichment(count_in: np.ndarray, count_out: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    cin = count_in.astype(float) + pseudocount
    cout = count_out.astype(float) + pseudocount
    fin = cin / cin.sum()
    fout = cout / cout.sum()
    return np.log(fout / fin)

def compute_round_enrichment(counts: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """
    For each round, compute log enrichment for every sequence.
    """
    out = []
    for r, g in counts.groupby("round", sort=True):
        le = log_enrichment(g["count_in"].to_numpy(), g["count_out"].to_numpy(), pseudocount=pseudocount)
        tmp = pd.DataFrame({
            "seq": g["seq"].to_numpy(),
            "round": int(r),
            "logE": le,
        })
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

def build_training_table(
    meas: pd.DataFrame,
    counts: pd.DataFrame,
    final_round: int,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """
    Build training table for final round:
      - labels: t_norm=log(T/E), o_norm=log(O/E), logE enrichment, logE_expr=log(E)
    """
    # enrichment
    enr = compute_round_enrichment(counts, pseudocount=pseudocount)
    enr_f = enr[enr["round"] == final_round][["seq", "logE"]].rename(columns={"logE": "y_logE"})

    # measurements in final round
    m_f = meas[meas["round"] == final_round].copy()
    eps = 1e-9
    m_f["y_tnorm"] = np.log((m_f["T"] + eps) / (m_f["E"] + eps))
    m_f["y_onorm"] = np.log((m_f["O"] + eps) / (m_f["E"] + eps))
    m_f["y_logE_expr"] = np.log(m_f["E"] + eps)

    keep = m_f[["seq", "y_tnorm", "y_onorm", "y_logE_expr"]].merge(enr_f, on="seq", how="left")
    # if enrichment missing (shouldn't happen), fill with 0
    keep["y_logE"] = keep["y_logE"].fillna(0.0)

    return keep