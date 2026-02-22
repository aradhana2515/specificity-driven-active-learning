# Assay-Aware Closed-Loop Active Learning for Antibody Specificity Optimization

A simulation framework for antibody-like sequence optimization under realistic yeast display counter-selection constraints. This project explicitly models assay physics, FACS gating, sequencing-based enrichment, and multi-objective acquisition to demonstrate closed-loop expansion of the specificity Pareto frontier across iterative rounds.

> **Why this matters:** Most ML-based protein optimization frameworks treat fitness as a clean scalar. Real antibody campaigns don't. This project models the messy reality — expression-confounded signals, hard FACS gates, and the affinity/polyspecificity tradeoff — and shows that accounting for these constraints produces measurably better optimization.

---

## Results

| Pareto Frontier Across Rounds | Hypervolume Expansion |
|:---:|:---:|
| ![Pareto Front](results/al_pareto_fronts.png) | ![Hypervolume](results/al_hypervolume.png) |

Observed hypervolume increases monotonically across rounds (example run):

| Round | Hypervolume |
|:---:|:---:|
| 1 | ~2.0 |
| 2 | ~2.6 |
| 3 | ~3.65 |

---

## Quickstart

```bash
git clone https://github.com/aradhana2515/specificity-driven-active-learning.git
cd specificity-driven-active-learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/04_active_learning_loop_cli.py --preset demo
```

Available presets:
- `demo` — small candidate pool for fast execution
- `full` — larger simulation for extended experiments

---

## Scientific Motivation

In real antibody engineering campaigns:

- Fluorescence signal depends on antigen concentration
- Binding measurements are confounded by expression levels
- FACS gating imposes hard selection bottlenecks
- Counter-selection creates a tradeoff between affinity and polyspecificity

Most ML-based protein optimization frameworks ignore these dynamics. This project explicitly models assay-aware selection and demonstrates that incorporating these constraints enables measurable expansion of the observed specificity frontier.

---

## Problem Formulation

Each variant has three latent properties: true affinity, true stickiness (polyspecificity propensity), and stability (expression proxy).

Observed assay signals are generated as:

```
E ~ exp(stability + noise)
T ~ E * f(affinity, antigen_concentration) + noise
O ~ E * g(stickiness) + noise
```

Antigen concentration decreases across rounds to simulate increasing stringency. Optimization objectives are defined in expression-normalized space:

```
t_norm = log(T / E)   # maximize
o_norm = log(O / E)   # minimize
```

This mirrors real display campaigns where specificity must improve without conflating signal with expression.

---

## FACS Gating Model

Round 1 establishes absolute fluorescence thresholds from quantiles (`E > e_thr`, `T > t_thr`, `O < o_thr`). These remain fixed for subsequent rounds, with the target threshold scaled to antigen concentration — mimicking real FACS practice where gates are set once and applied consistently. Sequencing counts are simulated pre- and post-sort to compute enrichment.

---

## Closed-Loop Active Learning

Each round:

1. Simulates assay readouts under current antigen concentration
2. Applies fixed FACS gates
3. Computes sequencing enrichment
4. Trains a bootstrap ensemble regressor on `y_tnorm`, `y_onorm`, `y_logE_expr`, `y_logE_enrich`
5. Predicts mean and epistemic uncertainty on candidate pool
6. Scores candidates via: `(t_norm - o_norm) + λ * σ`
7. Enforces expression feasibility, Pareto filtering, and physicochemical diversity
8. Selects next batch

The next round pool is a mixture of gate-selected survivors, model-proposed variants, and background diversity — simulating a realistic design–build–test–learn loop.

---

## Modeling & Representation

**Sequence features:** k-mer composition + physicochemical descriptors

**Model:** Bootstrap ensemble (Ridge regression) with mean prediction and ensemble standard deviation as uncertainty proxy

**Acquisition:** Multi-objective exploitation (`t_norm - o_norm`) + uncertainty-driven exploration, with hard expression constraint and diversity enforcement in physicochemical space

---

## Key Technical Contributions

- Explicit modeling of assay-dependent signal generation
- Expression-normalized optimization objectives
- Fixed fluorescence-space gating across rounds
- Multi-objective acquisition with epistemic uncertainty
- Closed-loop simulation with quantitative frontier tracking
- Hypervolume-based evaluation of specificity improvement

---

## Outputs

```
results/
├── al_pareto_fronts.png     # Pareto frontier per round
├── al_hypervolume.png       # Hypervolume expansion across rounds
├── al_gate_yields.png       # Gate yield statistics
└── al_hypervolume.csv       # Raw hypervolume values
```

---

## Context

An independent side project built to develop hands-on skills in 
closed-loop active learning for protein engineering — complementing 
my PhD work in directed evolution and immune specificity at Duke.
