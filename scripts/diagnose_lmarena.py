"""Why can't the cosine cache get above ~95% precision on LM-Arena?

Checks two things on the same cache/stream split the experiment uses:
  1. Label noise: identical prompt texts that carry different class IDs (ID_Set).
  2. Cosine nearest-neighbour precision as a function of similarity, including for
     queries whose nearest cached prompt has exactly the same text.
Prints example errors at very high similarity.

Usage:
    python scripts/diagnose_lmarena.py --data data/processed/lmarena_gte.npz
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

from semcache.embed import l2_normalize
from semcache.vcache_data import load_prepared, normalize_text as normalize


def short(text, width=90):
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[:width] + "..."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/lmarena_gte.npz")
    p.add_argument("--cache-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, labels, prompts = load_prepared(args.data)
    X = l2_normalize(X)
    texts = [normalize(t) for t in prompts]

    # ---------- 1. identical texts with different classes ----------
    classes_of = defaultdict(set)
    for t, lab in zip(texts, labels):
        classes_of[t].add(int(lab))
    count = Counter(texts)
    conflicted = {t for t, s in classes_of.items() if len(s) > 1}
    rows_conflicted = sum(count[t] for t in conflicted)
    print("=" * 80)
    print("1. Identical prompt texts that carry different class IDs")
    print("=" * 80)
    print(f"distinct texts: {len(count):,}   texts in >1 class: {len(conflicted):,}   "
          f"rows affected: {rows_conflicted:,} ({rows_conflicted / len(texts):.1%})")
    worst = sorted(conflicted, key=lambda t: -len(classes_of[t]))[:15]
    for t in worst:
        print(f"  {len(classes_of[t]):4d} classes  {count[t]:5d} rows   '{short(t, 70)}'")

    # ---------- 2. cosine NN precision by similarity (same split as the experiment) ----------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(labels))
    n_cache = int(round(args.cache_frac * len(labels)))
    ci, qi = perm[:n_cache], perm[n_cache:]
    Xc, Xq = X[ci], X[qi]
    nn, sim = [], []
    for i in range(0, len(qi), 2048):
        S = Xq[i:i + 2048] @ Xc.T
        j = S.argmax(1)
        nn.append(j)
        sim.append(S[np.arange(len(j)), j])
    nn, sim = np.concatenate(nn), np.concatenate(sim)
    nn_global = ci[nn]
    correct = labels[nn_global] == labels[qi]
    same_text = np.array([texts[a] == texts[b] for a, b in zip(qi, nn_global)])

    print()
    print("=" * 80)
    print("2. Cosine nearest-neighbour precision by similarity")
    print("=" * 80)
    print(f"overall: {correct.mean():.3f} precision when every query is served")
    edges = [-1.0, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0001]
    print(f"  {'similarity':>16s} {'queries':>8s} {'precision':>10s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sim >= lo) & (sim < hi)
        if m.any():
            print(f"  [{lo:6.3f}, {min(hi, 1.0):5.3f}) {m.sum():8,d} {correct[m].mean():10.3f}")
    print(f"\nqueries whose nearest cached prompt has IDENTICAL text: {same_text.sum():,}")
    if same_text.any():
        print(f"precision on those: {correct[same_text].mean():.3f}   "
              f"(anything below 1.000 is label noise, not a caching error)")
    if (~same_text).any():
        print(f"precision on all other queries: {correct[~same_text].mean():.3f}")

    errors = np.where(~correct & (sim >= 0.99))[0]
    print(f"\nwrong answers at similarity >= 0.99: {len(errors):,}. Examples:")
    for e in errors[:12]:
        print(f"  sim {sim[e]:.4f}")
        print(f"    query  [{labels[qi[e]]}]: {short(prompts[qi[e]])}")
        print(f"    cached [{labels[nn_global[e]]}]: {short(prompts[nn_global[e]])}")


if __name__ == "__main__":
    main()
