"""Preliminary experiment on a vCache benchmark: GMM cache vs. cosine-threshold cache.

Prompts are split at random into a pre-filled cache and a query stream. Each class (ID_Set)
is one answer. A served answer is correct if it has the query's class; for a query whose class
is not in the cache, every hit is an error.

Usage:
    python scripts/run_vcache_prelim.py --data data/processed/lmarena_gte.npz --k 600
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from semcache.cache import CosineNNCache, GMMCache, KNNVoteCache
from semcache.embed import l2_normalize
from semcache.gmm import DiagGMM
from semcache.metrics import hit_precision_curve, hit_rate_at_precision
from semcache.plots import plot_prelim
from semcache.vcache_data import load_prepared, merge_identical_text_classes, split_cache_stream


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/lmarena_gte.npz")
    p.add_argument("--k", type=int, default=600)
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--cache-frac", type=float, default=0.5)
    p.add_argument("--density-quantile", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--knn", type=int, default=10, help="neighbours for the vote baseline")
    p.add_argument("--merge-identical", action="store_true",
                   help="merge classes that contain identical prompt text (repairs label noise)")
    args = p.parse_args()

    name = Path(args.data).stem
    X, labels, prompts = load_prepared(args.data)
    n_merged = 0
    if args.merge_identical:
        labels, n_merged = merge_identical_text_classes(labels, prompts)
        name += "_merged"
        sizes = np.bincount(labels)
        print(f"merged classes sharing identical prompt text: {n_merged} classes removed; "
              f"largest class now {sizes.max()} prompts (check this is not a runaway chain)")
    X = l2_normalize(X)
    exp = split_cache_stream(X, labels, prompts, cache_frac=args.cache_frac, seed=args.seed)
    print(f"{name}: {len(labels):,} prompts, dim {X.shape[1]}")
    print(f"cache {len(exp.y_cache):,} prompts / {exp.n_answers:,} classes   "
          f"stream {len(exp.y_query):,} queries ({exp.answerable.mean():.1%} answerable)")

    pca = PCA(args.pca_dim, whiten=True, random_state=args.seed).fit(exp.X_cache)
    Zc, Zq = pca.transform(exp.X_cache), pca.transform(exp.X_query)
    print(f"PCA {X.shape[1]} -> {args.pca_dim} dims keeps {pca.explained_variance_ratio_.sum():.1%} of variance")

    print(f"fitting GMM with K={args.k} by EM...")
    gmm = DiagGMM(args.k, max_iter=args.max_iter, seed=args.seed).fit(Zc)
    gcache = GMMCache(gmm, args.density_quantile).fit(Zc, exp.y_cache, exp.n_answers)
    g_pred, g_score, logp = gcache.query(Zq)
    c_pred, c_score = CosineNNCache().fit(exp.X_cache, exp.y_cache).query(exp.X_query)
    print(f"running {args.knn}-nearest-neighbour vote baseline...")
    v_pred, v_score = KNNVoteCache(args.knn).fit(exp.X_cache, exp.y_cache).query(exp.X_query)

    y = exp.y_query
    unanswerable = ~exp.answerable
    curves = {
        "cosine NN threshold": hit_precision_curve(c_score, c_pred, y),
        f"{args.knn}-NN vote": hit_precision_curve(v_score, v_pred, y),
        "GMM posterior + density gate": hit_precision_curve(g_score, g_pred, y),
    }

    results = {
        "data": name,
        "k": args.k,
        "pca_dim": args.pca_dim,
        "seed": args.seed,
        "merged_identical_classes": n_merged,
        "n_cache": int(len(exp.y_cache)),
        "n_query": int(len(y)),
        "answerable_fraction": float(exp.answerable.mean()),
        "em_iterations": gmm.n_iter_,
        "final_avg_loglik": gmm.loglik_history_[-1],
        "bic": float(gmm.bic(Zc)),
        "mean_cluster_purity": float(gcache.purity_.mean()),
        "unanswerable_rejected_by_density_gate": float((logp[unanswerable] <= gcache.density_threshold_).mean()),
        "answerable_rejected_by_density_gate": float((logp[~unanswerable] <= gcache.density_threshold_).mean()),
        "hit_rate_at_precision": {
            name_: {str(t): hit_rate_at_precision(c, t) for t in (0.90, 0.95, 0.98)}
            for name_, c in curves.items()
        },
    }
    print(json.dumps(results, indent=2))

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    tag = f"{name}_k{args.k}_seed{args.seed}"
    Path(f"results/vcache_{tag}.json").write_text(json.dumps(results, indent=2))
    fig_path = f"figures/vcache_{tag}.png"
    plot_prelim(gmm.loglik_history_, curves, logp, unanswerable, gcache.density_threshold_, args.k,
                fig_path, labels=("class in cache", "class not in cache"))
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
