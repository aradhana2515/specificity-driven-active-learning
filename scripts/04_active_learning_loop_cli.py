import argparse
import importlib.util
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="CLI wrapper for active learning loop")
    p.add_argument("--preset", choices=["demo", "full"], default="demo")
    p.add_argument("--N", type=int)
    p.add_argument("--candidate-pool", type=int)
    p.add_argument("--n-models", type=int)
    p.add_argument("--kmer-k", type=int)
    return p.parse_args()


def apply_preset(args):
    cfg = {
        "N": 6000,
        "candidate_pool": 2000,
        "n_models": 5,
        "kmer_k": 2,
    }

    if args.preset == "full":
        cfg.update({
            "N": 20000,
            "candidate_pool": 20000,
            "n_models": 12,
            "kmer_k": 3,
        })

    if args.N is not None:
        cfg["N"] = args.N
    if args.candidate_pool is not None:
        cfg["candidate_pool"] = args.candidate_pool
    if args.n_models is not None:
        cfg["n_models"] = args.n_models
    if args.kmer_k is not None:
        cfg["kmer_k"] = args.kmer_k

    return cfg


def load_original_script():
    script_path = os.path.join("scripts", "04_active_learning_loop.py")

    spec = importlib.util.spec_from_file_location("active_loop", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["active_loop"] = module
    spec.loader.exec_module(module)

    return module


def main():
    args = parse_args()
    cfg = apply_preset(args)

    al = load_original_script()

    # Patch globals BEFORE calling main()
    if hasattr(al, "N"):
        al.N = cfg["N"]
    if hasattr(al, "CANDIDATE_POOL"):
        al.CANDIDATE_POOL = cfg["candidate_pool"]
    if hasattr(al, "N_MODELS"):
        al.N_MODELS = cfg["n_models"]
    if hasattr(al, "KMER_K"):
        al.KMER_K = cfg["kmer_k"]

    print("Running with config:", cfg)

    al.main()


if __name__ == "__main__":
    main()
