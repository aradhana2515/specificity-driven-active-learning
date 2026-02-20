import os
import numpy as np
import pandas as pd
from clcs.features import fit_featurizer, transform_featurizer, hstack_sparse_dense, physchem_features
from clcs.model import bootstrap_ensemble
from clcs.acquire import propose_next_round
from clcs.viz import ensure_dir, plot_gate_yields, plot_pareto, plot_uncertainty_vs_score

ensure_dir("results")

train = pd.read_csv("data/train_table.csv")
seqs = train["seq"].to_numpy()
y = train[["y_tnorm", "y_onorm", "y_logE_expr", "y_logE"]].to_numpy(dtype=float)

# fit featurizer on all (stable space)
vec, Xk, Xp = fit_featurizer(seqs, k=3)
X = hstack_sparse_dense(Xk, Xp)

# For diversity, we need dense vectors. Use physchem only (small, interpretable).
Xdense = physchem_features(seqs).to_numpy(dtype=float)

# ensemble predictions on same set (acts like pool)
mu, sigma = bootstrap_ensemble(X, y, X, n_models=12, alpha=5.0, seed=123)

# propose with expression constraint (logE_expr)
expr_min = float(np.quantile(mu[:, 2], 0.30))  # require above lower tail
sel, score, pflag = propose_next_round(
    seqs=seqs,
    mu=mu,
    sigma=sigma,
    X_dense_for_diversity=Xdense,
    batch_size=256,
    expr_min=expr_min,
    seed=0,
)

# assemble next_round.csv
out = pd.DataFrame({
    "seq": seqs[sel],
    "pred_tnorm": mu[sel, 0],
    "pred_onorm": mu[sel, 1],
    "pred_logE_expr": mu[sel, 2],
    "pred_logE_enrich": mu[sel, 3],
    "uncertainty": sigma[sel],
    "pareto_flag": pflag,
    "acq_score": score,
})
# human-readable reason
out["selected_reason"] = np.where(
    out["pareto_flag"] == 1,
    "on_pareto_frontier; diversity_filtered",
    "high_acq_score; diversity_filtered",
)
out.to_csv("results/next_round.csv", index=False)

# plots
gate_stats = pd.read_csv("data/gate_stats.csv")
plot_gate_yields(gate_stats, "results/gate_yields.png")
plot_pareto(mu, sigma, "results/pareto_pred.png", title="Predicted Pareto (t_norm vs o_norm)")
plot_uncertainty_vs_score(mu, sigma, "results/uncertainty_vs_score.png")

# save full ensemble preds for inspection
pred = pd.DataFrame({
    "seq": seqs,
    "mu_tnorm": mu[:, 0],
    "mu_onorm": mu[:, 1],
    "mu_logE_expr": mu[:, 2],
    "mu_logE_enrich": mu[:, 3],
    "sigma": sigma,
})
pred.to_csv("results/ensemble_predictions.csv", index=False)

print("Wrote:")
print(" - results/next_round.csv")
print(" - results/gate_yields.png")
print(" - results/pareto_pred.png")
print(" - results/uncertainty_vs_score.png")
print(" - results/ensemble_predictions.csv")
print(f"Expression constraint expr_min (logE_expr) = {expr_min:.3f}")
print(f"Selected {len(out)} sequences.")