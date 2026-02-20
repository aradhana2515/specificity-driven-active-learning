import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from clcs.features import fit_featurizer, transform_featurizer, hstack_sparse_dense
from clcs.model import MultiTaskRidge

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/train_table.csv")
seqs = df["seq"].to_numpy()

# labels: t_norm, o_norm, logE_expr, logE_enrich
y = df[["y_tnorm", "y_onorm", "y_logE_expr", "y_logE"]].to_numpy(dtype=float)

idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=0)

seq_tr, seq_te = seqs[tr], seqs[te]
y_tr, y_te = y[tr], y[te]

vec, Xk_tr, Xp_tr = fit_featurizer(seq_tr, k=3)
X_tr = hstack_sparse_dense(Xk_tr, Xp_tr)

Xk_te, Xp_te = transform_featurizer(vec, seq_te)
X_te = hstack_sparse_dense(Xk_te, Xp_te)

model = MultiTaskRidge(alpha=5.0).fit(X_tr, y_tr)
metrics = model.score(X_te, y_te)

pd.DataFrame([metrics]).to_csv("results/metrics.csv", index=False)

# also predict for all sequences (pool)
Xk_all, Xp_all = transform_featurizer(vec, seqs)
X_all = hstack_sparse_dense(Xk_all, Xp_all)
pred_all = model.predict(X_all)

pred_df = pd.DataFrame({
    "seq": seqs,
    "pred_tnorm": pred_all[:, 0],
    "pred_onorm": pred_all[:, 1],
    "pred_logE_expr": pred_all[:, 2],
    "pred_logE_enrich": pred_all[:, 3],
})
pred_df.to_csv("results/predictions_all.csv", index=False)

# save featurizer vocabulary (tiny provenance)
vocab = pd.DataFrame({"kmer": list(vec.vocabulary_.keys()), "idx": list(vec.vocabulary_.values())})
vocab.to_csv("results/kmer_vocab.csv", index=False)

print("Wrote:")
print(" - results/metrics.csv")
print(" - results/predictions_all.csv")
print(" - results/kmer_vocab.csv")
print("Metrics:", metrics)