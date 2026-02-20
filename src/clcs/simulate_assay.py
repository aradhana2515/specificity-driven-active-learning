from __future__ import annotations
import numpy as np
import pandas as pd

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))

def random_sequences(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    idx = rng.integers(0, len(AA), size=(n, L))
    seqs = ["".join(AA[row]) for row in idx]
    return np.array(seqs, dtype=object)

def seq_to_char_matrix(seqs: np.ndarray) -> np.ndarray:
    return np.array([list(s) for s in seqs])

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def simulate_library(n: int = 20000, L: int = 14, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic 'antibody-like' sequence library with latent affinity + stickiness.
    Tuned so sequence has meaningful predictive signal (less noise than the first v1).
    """
    rng = np.random.default_rng(seed)
    seqs = random_sequences(rng, n=n, L=L)
    Xc = seq_to_char_matrix(seqs)

    # motif-ish affinity: aromatics in middle window
    mid = slice(L//3, 2*L//3)
    aromatic = np.isin(Xc[:, mid], np.array(list("YWF")))
    affinity_motif = aromatic.mean(axis=1)

    # hydrophobic load drives stickiness
    hydrophobic = np.isin(Xc, np.array(list("AILMFWVYV")))
    hydro_load = hydrophobic.mean(axis=1)

    # net charge proxy
    pos = np.isin(Xc, np.array(list("KRH"))).sum(axis=1)
    neg = np.isin(Xc, np.array(list("DE"))).sum(axis=1)
    net_charge = (pos - neg) / L

    # >>> STRONGER signal, LOWER noise (key for learnable mapping)
    true_affinity = 2.6 * affinity_motif + 1.0 * net_charge + rng.normal(0, 0.18, size=n)
    true_stickiness = 2.2 * hydro_load - 0.7 * net_charge + rng.normal(0, 0.18, size=n)

    # stability proxy influences expression
    stability = 0.9 * (1 - hydro_load) + 0.5 * (np.abs(net_charge) < 0.2).astype(float) + rng.normal(0, 0.12, size=n)

    lib = pd.DataFrame({
        "seq": seqs,
        "true_affinity": true_affinity,
        "true_stickiness": true_stickiness,
        "stability": stability,
        "L": L,
    })
    return lib

def simulate_assay_measurements(
    lib: pd.DataFrame,
    round_id: int,
    target_conc: float,
    seed: int,
    batch_effect_sd: float = 0.03,   # smaller drift so learning isn't destroyed
) -> pd.DataFrame:
    """
    Generate per-variant assay measurements: expression E, target binding T, off-target binding O.

    Biology-ish assumptions:
    - E varies with stability + noise + mild batch effect
    - T ~ E * saturating(true_affinity, [target]) + noise
    - O ~ E * f(true_stickiness) + noise

    Tuned to create:
    - realistic tradeoff
    - better learnability from sequence features
    """
    rng = np.random.default_rng(seed)

    n = len(lib)
    aff = lib["true_affinity"].to_numpy()
    stick = lib["true_stickiness"].to_numpy()
    stability = lib["stability"].to_numpy()

    # round-specific multiplicative drift
    be = rng.normal(0, batch_effect_sd)
    be_mult = np.exp(be)

    # Expression: log-normal-ish
    logE = 0.0 + 0.95 * stability + rng.normal(0, 0.28, size=n)
    E = np.exp(logE) * be_mult

    # Target binding: occupancy increases with affinity and target concentration
    occ = logistic((aff + np.log(max(1e-6, target_conc))) / 1.0)

    # >>> More signal, less noise
    T_mean = E * (0.10 + 3.2 * occ)
    T = T_mean + rng.normal(0, 0.08 * np.mean(E), size=n)
    T = np.clip(T, 1e-6, None)

    # Off-target binding: increases with stickiness
    nons = logistic(stick / 0.95)
    O_mean = E * (0.10 + 2.4 * nons)
    O = O_mean + rng.normal(0, 0.08 * np.mean(E), size=n)
    O = np.clip(O, 1e-6, None)

    out = pd.DataFrame({
        "seq": lib["seq"].to_numpy(),
        "round": int(round_id),
        "target_conc": float(target_conc),
        "E": E,
        "T": T,
        "O": O,
    })
    return out