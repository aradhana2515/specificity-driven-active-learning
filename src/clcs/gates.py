from __future__ import annotations
import numpy as np
import pandas as pd

def compute_gate_thresholds_from_quantiles(
    meas: pd.DataFrame,
    e_quantile: float = 0.50,
    t_quantile: float = 0.75,
    o_quantile: float = 0.40,
) -> dict:
    """
    Compute absolute thresholds from quantiles on a reference round (usually round 1).
    Returns a dict with keys: e_thr, t_thr, o_thr
    """
    E = meas["E"].to_numpy()
    T = meas["T"].to_numpy()
    O = meas["O"].to_numpy()

    e_thr = float(np.quantile(E, e_quantile))
    pass_e = E > e_thr
    if pass_e.sum() == 0:
        return {"e_thr": e_thr, "t_thr": float("inf"), "o_thr": -float("inf")}

    t_thr = float(np.quantile(T[pass_e], t_quantile))
    pass_et = pass_e & (T > t_thr)
    if pass_et.sum() == 0:
        return {"e_thr": e_thr, "t_thr": t_thr, "o_thr": -float("inf")}

    o_thr = float(np.quantile(O[pass_et], o_quantile))
    return {"e_thr": e_thr, "t_thr": t_thr, "o_thr": o_thr}


def apply_facs_gates_fixed(
    meas: pd.DataFrame,
    e_thr: float,
    t_thr: float,
    o_thr: float,
    max_collect: int = 200_000,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Apply fixed absolute gates:
      E > e_thr
      T > t_thr
      O < o_thr
    Then bottleneck to max_collect.
    """
    rng = np.random.default_rng(seed)

    E = meas["E"].to_numpy()
    T = meas["T"].to_numpy()
    O = meas["O"].to_numpy()

    n0 = len(meas)

    pass_e = E > e_thr
    n1 = int(pass_e.sum())

    pass_et = pass_e & (T > t_thr)
    n2 = int(pass_et.sum())

    pass_eto = pass_et & (O < o_thr)
    n3 = int(pass_eto.sum())

    idx = np.where(pass_eto)[0]
    if len(idx) > max_collect:
        idx = rng.choice(idx, size=max_collect, replace=False)
    idx = np.sort(idx)

    stats = {
        "n0": n0,
        "nE": n1,
        "nET": n2,
        "nETO": n3,
        "nCollect": int(len(idx)),
        "e_thr": float(e_thr),
        "t_thr": float(t_thr),
        "o_thr": float(o_thr),
        "max_collect": int(max_collect),
        "gate_mode": "fixed_absolute",
    }
    return idx, stats