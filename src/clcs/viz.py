from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .pareto import pareto_frontier

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_gate_yields(gate_stats: pd.DataFrame, outpath: str):
    """
    gate_stats columns: round, n0, nE, nET, nETO, nCollect
    """
    r = gate_stats["round"].to_numpy()
    plt.figure()
    plt.plot(r, gate_stats["n0"], marker="o", label="start")
    plt.plot(r, gate_stats["nE"], marker="o", label="E gate")
    plt.plot(r, gate_stats["nET"], marker="o", label="E+T gates")
    plt.plot(r, gate_stats["nETO"], marker="o", label="E+T+O gates")
    plt.plot(r, gate_stats["nCollect"], marker="o", label="collected")
    plt.xlabel("Round")
    plt.ylabel("Variants passing")
    plt.title("FACS-like gate yields per round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_pareto(mu: np.ndarray, sigma: np.ndarray | None, outpath: str, title: str):
    """
    mu columns: [t_norm, o_norm, logE_expr, logE_enrich]
    """
    t = mu[:, 0]
    o = mu[:, 1]
    nd = pareto_frontier(mu[:, [0, 1]], maximize=np.array([True, False]))

    plt.figure()
    plt.scatter(t, o, s=8, alpha=0.35)
    plt.scatter(t[nd], o[nd], s=18)
    plt.xlabel("Predicted t_norm = log(T/E) (higher better)")
    plt.ylabel("Predicted o_norm = log(O/E) (lower better)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_uncertainty_vs_score(mu: np.ndarray, sigma: np.ndarray, outpath: str):
    score = (mu[:, 0] - mu[:, 1]) + 0.5 * sigma
    plt.figure()
    plt.scatter(score, sigma, s=8, alpha=0.35)
    plt.xlabel("Acquisition base score (t_norm - o_norm + 0.5*sigma)")
    plt.ylabel("Uncertainty (ensemble std)")
    plt.title("Exploration vs exploitation")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()