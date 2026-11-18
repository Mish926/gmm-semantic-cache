"""Preliminary experiment: GMM semantic cache vs. cosine-threshold cache on CLINC150."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from semcache.cache import CosineNNCache, GMMCache
from semcache.data import load_clinc
from semcache.embed import SIFEmbedder, l2_normalize
from semcache.gmm import DiagGMM
from semcache.metrics import hit_precision_curve, hit_rate_at_precision
from semcache.plots import plot_prelim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clinc", default="data/raw/clinc_full.json")
    p.add_argument("--glove", default="data/raw/glove100.gz")
    p.add_argument("--k", type=int, default=600)
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--density-quantile", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = load_clinc(args.clinc)
    print(f"train {len(ds.train.texts)}  test {len(ds.test.texts)}  intents {ds.n_intents}")

    print("embedding (loading GloVe takes ~1 min)...")
    emb = SIFEmbedder(args.glove).fit(ds.train.texts, vocab_texts=ds.all_texts)
    Xtr = l2_normalize(emb.transform(ds.train.texts))
    Xte = l2_normalize(emb.transform(ds.test.texts))
    pca = PCA(args.pca_dim, whiten=True, random_state=args.seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

    print(f"fitting GMM with K={args.k} by EM...")
    gmm = DiagGMM(args.k, max_iter=args.max_iter, seed=args.seed).fit(Ztr)
    gcache = GMMCache(gmm, args.density_quantile).fit(Ztr, ds.train.y, ds.n_intents)
    g_pred, g_score, logp = gcache.query(Zte)
    c_pred, c_score = CosineNNCache().fit(Xtr, ds.train.y).query(Xte)

    y = ds.test.y
    is_oos = y == -1
    curves = {
        "cosine NN threshold": hit_precision_curve(c_score, c_pred, y),
        "GMM posterior + density gate": hit_precision_curve(g_score, g_pred, y),
    }

    results = {
        "k": args.k,
        "seed": args.seed,
        "em_iterations": gmm.n_iter_,
        "final_avg_loglik": gmm.loglik_history_[-1],
        "bic": float(gmm.bic(Ztr)),
        "mean_cluster_purity": float(gcache.purity_.mean()),
        "oos_rejected_by_density_gate": float((logp[is_oos] <= gcache.density_threshold_).mean()),
        "inscope_rejected_by_density_gate": float((logp[~is_oos] <= gcache.density_threshold_).mean()),
        "hit_rate_at_precision": {
            name: {str(t): hit_rate_at_precision(c, t) for t in (0.90, 0.95, 0.98)}
            for name, c in curves.items()
        },
    }
    print(json.dumps(results, indent=2))

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path(f"results/prelim_k{args.k}_seed{args.seed}.json").write_text(json.dumps(results, indent=2))
    fig_path = f"figures/prelim_k{args.k}_seed{args.seed}.png"
    plot_prelim(gmm.loglik_history_, curves, logp, is_oos, gcache.density_threshold_, args.k, fig_path)
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
