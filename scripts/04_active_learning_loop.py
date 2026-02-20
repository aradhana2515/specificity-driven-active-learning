import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clcs.simulate_assay import simulate_library, simulate_assay_measurements
from clcs.gates import compute_gate_thresholds_from_quantiles, apply_facs_gates_fixed
from clcs.counts import simulate_sequencing_counts
from clcs.enrichment import build_training_table
from clcs.features import fit_featurizer, transform_featurizer, hstack_sparse_dense, physchem_features
from clcs.model import bootstrap_ensemble
from clcs.acquire import propose_next_round
from clcs.hypervolume import pareto_front_2d_tmax_omin, hypervolume_2d_tmax_omin


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def make_next_pool(
    lib: pd.DataFrame,
    meas_round: pd.DataFrame,
    gate_idx: np.ndarray,
    model_sel_seqs: np.ndarray,
    N: int,
    seed: int,
    w_gate: float = 0.70,
    w_model: float = 0.20,
    w_rand: float = 0.10,
) -> pd.DataFrame:
    """
    Create the next round pool (what grows up after sorting / seeding).
    This is a realistic-ish hack:
      - mostly gate-selected survivors (what you physically sorted)
      - some model-proposed sequences (what you'd seed/synthesize)
      - some random background (diversity / carryover)

    Returns a DataFrame with same schema as lib (seq + latent props).
    """
    rng = np.random.default_rng(seed)
    assert abs((w_gate + w_model + w_rand) - 1.0) < 1e-9

    n_gate = int(round(N * w_gate))
    n_model = int(round(N * w_model))
    n_rand = N - n_gate - n_model

    # Gate-selected pool (subset of current pool)
    if len(gate_idx) > 0:
        gate_seqs = meas_round.iloc[gate_idx]["seq"].to_numpy()
        gate_df = lib[lib["seq"].isin(gate_seqs)]
        if len(gate_df) == 0:
            gate_df = lib.sample(n=min(len(lib), n_gate), replace=True, random_state=int(seed))
        gate_samp = gate_df.sample(n=n_gate, replace=True, random_state=int(seed))
    else:
        gate_samp = lib.sample(n=n_gate, replace=True, random_state=int(seed))

    # Model-proposed sequences (subset of current lib in v1)
    if len(model_sel_seqs) > 0:
        model_df = lib[lib["seq"].isin(model_sel_seqs)]
        if len(model_df) == 0:
            model_df = lib.sample(n=min(len(lib), n_model), replace=True, random_state=int(seed + 1))
        model_samp = model_df.sample(n=n_model, replace=True, random_state=int(seed + 1))
    else:
        model_samp = lib.sample(n=n_model, replace=True, random_state=int(seed + 1))

    # Random background
    rand_samp = lib.sample(n=n_rand, replace=True, random_state=int(seed + 2))

    pool = pd.concat([gate_samp, model_samp, rand_samp], ignore_index=True)
    pool = pool.sample(frac=1.0, random_state=int(seed + 3)).reset_index(drop=True)  # shuffle
    return pool


def main():
    ensure_dirs()

    # --- Campaign settings ---
    N = 20000
    L = 14
    ROUNDS = 3
    target_concs = [1.0, 0.3, 0.1]  # decreasing antigen

    # Gate quantiles to define R1 absolute thresholds
    e_q, t_q, o_q = 0.50, 0.78, 0.45

    # Sequencing depth
    N_IN = 2_000_000
    N_OUT = 200_000

    # Acquisition
    BATCH = 256

    # Pool mixing weights for next round (tweakable)
    W_GATE, W_MODEL, W_RAND = 0.70, 0.20, 0.10

    # --- Initialize library + initial pool ---
    lib = simulate_library(n=N, L=L, seed=0)
    lib.to_csv("data/library.csv", index=False)

    pool = lib.copy()

    # Storage
    gate_rows = []
    hv_rows = []
    pareto_pts_by_round = {}

    # Gate thresholds (absolute) learned from round 1
    gate_thresholds = None

    for r in range(1, ROUNDS + 1):
        conc = target_concs[r - 1]

        # Simulate assay readouts on current pool
        meas = simulate_assay_measurements(pool, round_id=r, target_conc=conc, seed=100 + r)

        # Set absolute thresholds from round 1 quantiles
        if r == 1:
            gate_thresholds = compute_gate_thresholds_from_quantiles(
                meas,
                e_quantile=e_q,
                t_quantile=t_q,
                o_quantile=o_q,
            )

        # Scale the T threshold with antigen concentration (prevents collapse)
        scale = conc / target_concs[0]
        scaled_t_thr = gate_thresholds["t_thr"] * scale

        # Apply fixed gates
        gate_idx, stats = apply_facs_gates_fixed(
            meas,
            e_thr=gate_thresholds["e_thr"],
            t_thr=scaled_t_thr,
            o_thr=gate_thresholds["o_thr"],
            max_collect=200_000,
            seed=200 + r,
        )
        stats["round"] = r
        stats["target_conc"] = conc
        stats["t_thr_scaled"] = float(scaled_t_thr)
        gate_rows.append(stats)

        # Simulate sequencing counts for this round
        counts = simulate_sequencing_counts(meas, gate_idx, n_in_reads=N_IN, n_out_reads=N_OUT, seed=300 + r)

        # Compute observed (assay) t_norm and o_norm
        eps = 1e-9
        meas_obs = meas.copy()
        meas_obs["t_norm"] = np.log((meas_obs["T"] + eps) / (meas_obs["E"] + eps))
        meas_obs["o_norm"] = np.log((meas_obs["O"] + eps) / (meas_obs["E"] + eps))

        # Observed Pareto + hypervolume
        pts = meas_obs[["t_norm", "o_norm"]].to_numpy(dtype=float)
        pareto = pareto_front_2d_tmax_omin(pts)

        # Reference point: deliberately "worse than most"
        # (More negative t, higher o)
        ref_t = float(np.quantile(pts[:, 0], 0.01)) - 0.5
        ref_o = float(np.quantile(pts[:, 1], 0.99)) + 0.5
        hv = hypervolume_2d_tmax_omin(pareto, ref_t=ref_t, ref_o=ref_o)

        hv_rows.append({
            "round": r,
            "target_conc": conc,
            "hv_observed": hv,
            "ref_t": ref_t,
            "ref_o": ref_o,
            "pareto_size": int(len(pareto)),
        })
        pareto_pts_by_round[r] = pareto

        # Build training table using this round
        # (Treat this round as "final_round" for labels)
        train = build_training_table(meas, counts, final_round=r, pseudocount=1.0)

        # Prepare training features + labels
        seqs = train["seq"].to_numpy()
        y = train[["y_tnorm", "y_onorm", "y_logE_expr", "y_logE"]].to_numpy(dtype=float)

        # Fit featurizer on this round’s training sequences
        vec, Xk, Xp = fit_featurizer(seqs, k=3)
        X = hstack_sparse_dense(Xk, Xp)

        # Predict on the *current pool* (candidate space)
        pool_seqs = meas["seq"].to_numpy()
        Xk_pool, Xp_pool = transform_featurizer(vec, pool_seqs)
        X_pool = hstack_sparse_dense(Xk_pool, Xp_pool)

        # Diversity space: physchem only (fast dense)
        Xdense_pool = physchem_features(pool_seqs).to_numpy(dtype=float)

        # Train ensemble for mean + uncertainty
        mu, sigma = bootstrap_ensemble(X, y, X_pool, n_models=12, alpha=5.0, seed=123 + r)

        # Expression feasibility constraint (logE_expr)
        expr_min = float(np.quantile(mu[:, 2], 0.30))

        sel_idx, sel_score, sel_pflag = propose_next_round(
            seqs=pool_seqs,
            mu=mu,
            sigma=sigma,
            X_dense_for_diversity=Xdense_pool,
            batch_size=BATCH,
            expr_min=expr_min,
            seed=10 + r,
        )
        sel_seqs = pool_seqs[sel_idx]

        # Save per-round outputs
        meas.to_csv(f"data/round{r}_measurements.csv", index=False)
        counts.to_csv(f"data/round{r}_counts.csv", index=False)
        train.to_csv(f"data/round{r}_train_table.csv", index=False)

        pd.DataFrame({
            "seq": pool_seqs,
            "mu_tnorm": mu[:, 0],
            "mu_onorm": mu[:, 1],
            "mu_logE_expr": mu[:, 2],
            "mu_logE_enrich": mu[:, 3],
            "sigma": sigma,
        }).to_csv(f"results/round{r}_ensemble_predictions_pool.csv", index=False)

        pd.DataFrame({
            "seq": sel_seqs,
            "acq_score": sel_score,
            "pareto_flag": sel_pflag,
            "pred_tnorm": mu[sel_idx, 0],
            "pred_onorm": mu[sel_idx, 1],
            "pred_logE_expr": mu[sel_idx, 2],
            "pred_logE_enrich": mu[sel_idx, 3],
            "uncertainty": sigma[sel_idx],
        }).to_csv(f"results/round{r}_next_round.csv", index=False)

        # Construct next pool
        pool = make_next_pool(
            lib=lib,
            meas_round=meas,
            gate_idx=gate_idx,
            model_sel_seqs=sel_seqs,
            N=N,
            seed=999 + r,
            w_gate=W_GATE,
            w_model=W_MODEL,
            w_rand=W_RAND,
        )

    # Save summary tables
    gate_df = pd.DataFrame(gate_rows)
    hv_df = pd.DataFrame(hv_rows)
    gate_df.to_csv("results/al_gate_stats.csv", index=False)
    hv_df.to_csv("results/al_hypervolume.csv", index=False)

    # --- Plot: Gate yields over rounds ---
    plt.figure()
    plt.plot(gate_df["round"], gate_df["n0"], marker="o", label="start")
    plt.plot(gate_df["round"], gate_df["nE"], marker="o", label="E gate")
    plt.plot(gate_df["round"], gate_df["nET"], marker="o", label="E+T gates")
    plt.plot(gate_df["round"], gate_df["nETO"], marker="o", label="E+T+O gates")
    plt.plot(gate_df["round"], gate_df["nCollect"], marker="o", label="collected")
    plt.xlabel("Round")
    plt.ylabel("Variants passing")
    plt.title("Active learning campaign: FACS-like gate yields")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/al_gate_yields.png", dpi=160)
    plt.close()

    # --- Plot: Observed Pareto fronts per round ---
    plt.figure()
    for r in range(1, ROUNDS + 1):
        pareto = pareto_pts_by_round[r]
        plt.scatter(pareto[:, 0], pareto[:, 1], s=20, label=f"Round {r}")
    plt.xlabel("Observed t_norm = log(T/E) (higher better)")
    plt.ylabel("Observed o_norm = log(O/E) (lower better)")
    plt.title("Observed Pareto fronts across rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/al_pareto_fronts.png", dpi=160)
    plt.close()

    # --- Plot: Hypervolume vs round ---
    plt.figure()
    plt.plot(hv_df["round"], hv_df["hv_observed"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Hypervolume (observed Pareto)")
    plt.title("Closed-loop improvement proxy: Pareto hypervolume vs round")
    plt.tight_layout()
    plt.savefig("results/al_hypervolume.png", dpi=160)
    plt.close()

    print("=== Active learning loop complete ===")
    print("Gate thresholds from Round 1:")
    print(gate_thresholds)
    print("Wrote:")
    print(" - results/al_gate_stats.csv")
    print(" - results/al_hypervolume.csv")
    print(" - results/al_gate_yields.png")
    print(" - results/al_pareto_fronts.png")
    print(" - results/al_hypervolume.png")
    print(" - per-round: data/round*_measurements.csv, data/round*_counts.csv, data/round*_train_table.csv")
    print(" - per-round: results/round*_ensemble_predictions_pool.csv, results/round*_next_round.csv")


if __name__ == "__main__":
    main()