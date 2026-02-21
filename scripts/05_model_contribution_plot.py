import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs("results", exist_ok=True)

    # Inputs produced by your AL loop
    hv_path = "results/al_hypervolume.csv"
    gate_path = "results/al_gate_stats.csv"

    # Round-wise pool prediction files (you have these)
    pred_files = {
        1: "results/round1_ensemble_predictions_pool.csv",
        2: "results/round2_ensemble_predictions_pool.csv",
        3: "results/round3_ensemble_predictions_pool.csv",
    }
    next_files = {
        1: "results/round1_next_round.csv",
        2: "results/round2_next_round.csv",
        3: "results/round3_next_round.csv",
    }

    # We will plot: predicted Pareto space (t_norm vs o_norm)
    # and highlight the actually selected "next_round" sequences.
    for r in [1, 2, 3]:
        if not (os.path.exists(pred_files[r]) and os.path.exists(next_files[r])):
            print(f"Missing round {r} files. Skipping.")
            continue

        preds = pd.read_csv(pred_files[r])
        nxt = pd.read_csv(next_files[r])

        # Try common column names (adjust if yours differ)
        # Expect columns like: seq, mu_tnorm, mu_onorm OR pred_tnorm, pred_onorm
        cand_cols = preds.columns

        # Find predicted t_norm / o_norm columns
        t_candidates = [c for c in cand_cols if "tnorm" in c.lower() and ("mu" in c.lower() or "pred" in c.lower())]
        o_candidates = [c for c in cand_cols if "onorm" in c.lower() and ("mu" in c.lower() or "pred" in c.lower())]

        if len(t_candidates) == 0 or len(o_candidates) == 0:
            raise ValueError(f"Could not find predicted t_norm/o_norm columns in {pred_files[r]}.\nColumns: {list(cand_cols)}")

        t_col = t_candidates[0]
        o_col = o_candidates[0]

        # Selected sequences set
        if "seq" not in nxt.columns:
            raise ValueError(f"'seq' column not found in {next_files[r]}. Columns: {list(nxt.columns)}")
        selected = set(nxt["seq"].astype(str).tolist())

        preds["is_selected"] = preds["seq"].astype(str).isin(selected)

        plt.figure(figsize=(7, 5))
        plt.scatter(preds[t_col], preds[o_col], alpha=0.2, s=10, label="Candidate pool")
        plt.scatter(
            preds.loc[preds["is_selected"], t_col],
            preds.loc[preds["is_selected"], o_col],
            s=35,
            alpha=0.9,
            label="Selected for next round"
        )
        plt.xlabel(f"Predicted {t_col} (higher better)")
        plt.ylabel(f"Predicted {o_col} (lower better)")
        plt.title(f"Round {r}: model-selected sequences in predicted objective space")
        plt.legend()
        out = f"results/model_contribution_round{r}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()
