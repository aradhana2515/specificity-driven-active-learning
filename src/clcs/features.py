from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# simple physchem approximations
HYDRO = set(list("AILMFWVYV"))
POS = set(list("KRH"))
NEG = set(list("DE"))

def kmer_vectorizer(k: int = 3) -> CountVectorizer:
    def tokenizer(s: str):
        return [s[i:i+k] for i in range(0, len(s) - k + 1)]
    return CountVectorizer(analyzer=tokenizer)

def physchem_features(seqs: np.ndarray) -> pd.DataFrame:
    feats = []
    for s in seqs:
        L = len(s)
        hydro = sum(aa in HYDRO for aa in s) / L
        pos = sum(aa in POS for aa in s) / L
        neg = sum(aa in NEG for aa in s) / L
        net = pos - neg
        arom = sum(aa in set("YWF") for aa in s) / L
        feats.append((hydro, pos, neg, net, arom, L))
    return pd.DataFrame(feats, columns=["hydro_frac", "pos_frac", "neg_frac", "net_charge", "arom_frac", "length"])

def fit_featurizer(seqs: np.ndarray, k: int = 3):
    vec = kmer_vectorizer(k=k)
    Xk = vec.fit_transform(seqs.tolist())
    Xp = physchem_features(seqs).to_numpy(dtype=float)
    return vec, Xk, Xp

def transform_featurizer(vec: CountVectorizer, seqs: np.ndarray):
    Xk = vec.transform(seqs.tolist())
    Xp = physchem_features(seqs).to_numpy(dtype=float)
    return Xk, Xp

def hstack_sparse_dense(Xk, Xp):
    # manual sparse+dense concat without scipy dependency by converting dense to sparse-ish
    # simplest: convert sparse to dense if small; but our Xk can be big. We'll instead
    # keep Xk sparse and append dense using sklearn's FeatureUnion-like trick:
    # return a scipy sparse hstack IF scipy exists; scikit-learn depends on scipy typically,
    # but to avoid explicit import errors, we import inside.
    try:
        from scipy.sparse import hstack, csr_matrix
        return hstack([Xk, csr_matrix(Xp)], format="csr")
    except Exception:
        # fallback: densify (may be heavy). For n=20k and k=3 still manageable on laptop
        return np.hstack([Xk.toarray(), Xp])