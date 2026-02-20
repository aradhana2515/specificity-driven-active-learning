import os
import pandas as pd
from clcs.simulate_assay import simulate_library, simulate_assay_measurements
from clcs.gates import compute_gate_thresholds_from_quantiles, apply_facs_gates_fixed
from clcs.counts import simulate_sequencing_counts

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

N = 20000
L = 14
ROUNDS = 3

lib = simulate_library(n=N, L=L, seed=0)
lib.to_csv("data/library.csv", index=False)

all_meas = []
all_counts = []
gate_rows = []

# target concentration decreases each round (more stringent)
target_concs = [1.0, 0.3, 0.1]

# We will compute fixed thresholds from round 1 and reuse them
gate_thresholds = None

for r in range(1, ROUNDS + 1):
    meas = simulate_assay_measurements(lib, round_id=r, target_conc=target_concs[r-1], seed=100 + r)

    if r == 1:
        # Set gates based on round 1 quantiles (like a real "gate design" step)
        gate_thresholds = compute_gate_thresholds_from_quantiles(
            meas,
            e_quantile=0.50,
            t_quantile=0.78,
            o_quantile=0.45,
        )
    # Scale target threshold by antigen concentration ratio
    scale = target_concs[r-1] / target_concs[0]
    scaled_t_thr = gate_thresholds["t_thr"] * scale
    idx, stats = apply_facs_gates_fixed(
        meas,
        e_thr=gate_thresholds["e_thr"],
        t_thr=scaled_t_thr,
        o_thr=gate_thresholds["o_thr"],
        max_collect=200_000,
        seed=200 + r,
    )

    counts = simulate_sequencing_counts(meas, idx, n_in_reads=2_000_000, n_out_reads=200_000, seed=300 + r)

    all_meas.append(meas)
    all_counts.append(counts)
    stats["round"] = r
    gate_rows.append(stats)

meas_all = pd.concat(all_meas, ignore_index=True)
counts_all = pd.concat(all_counts, ignore_index=True)
gate_stats = pd.DataFrame(gate_rows)

meas_all.to_csv("data/round_measurements.csv", index=False)
counts_all.to_csv("data/round_counts.csv", index=False)
gate_stats.to_csv("data/gate_stats.csv", index=False)

print("Wrote:")
print(" - data/library.csv")
print(" - data/round_measurements.csv")
print(" - data/round_counts.csv")
print(" - data/gate_stats.csv")
print("Fixed thresholds from round 1:")
print(gate_thresholds)