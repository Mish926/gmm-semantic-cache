"""Load a prepared vCache benchmark and split it into cache contents and a query stream."""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class CacheExperiment:
    X_cache: np.ndarray      # (n_cache, d)
    y_cache: np.ndarray      # answer index 0..n_answers-1
    X_query: np.ndarray      # (n_query, d)
    y_query: np.ndarray      # answer index, or -1 if the query's class is not in the cache
    prompts_query: List[str]
    n_answers: int

    @property
    def answerable(self) -> np.ndarray:
        return self.y_query >= 0


def normalize_text(text):
    """Lower-case, collapse whitespace, drop leading quote markers like '>' and quotes."""
    text = " ".join(text.lower().split())
    return re.sub(r"^[>\s\"']+|[\s\"']+$", "", text)


def merge_identical_text_classes(labels, prompts):
    """Merge classes that contain the same (normalized) prompt text.

    Identical prompts must share an answer, so a benchmark that puts them in different
    classes would count a perfect cache hit as an error. Union-find over classes linked
    by shared text; returns new contiguous labels and the number of classes removed.
    """
    parent = {}

    def find(a):
        root = a
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(a, a) != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    first_class = {}
    for text, lab in zip((normalize_text(t) for t in prompts), labels):
        lab = int(lab)
        if text in first_class:
            ra, rb = find(lab), find(first_class[text])
            if ra != rb:
                parent[ra] = rb
        else:
            first_class[text] = lab
    roots = np.array([find(int(l)) for l in labels])
    _, merged = np.unique(roots, return_inverse=True)
    return merged.astype(labels.dtype), len(np.unique(labels)) - len(np.unique(merged))


def load_prepared(path):
    path = Path(path)
    d = np.load(path)
    prompts_path = path.with_name(path.stem + "_prompts.json")
    prompts = json.loads(prompts_path.read_text()) if prompts_path.exists() else [""] * len(d["labels"])
    return d["X"].astype(np.float64), d["labels"], prompts


def split_cache_stream(X, labels, prompts, cache_frac=0.5, seed=0) -> CacheExperiment:
    """Random split of prompts into a pre-filled cache and a query stream.

    Each class becomes one 'answer'. A query whose class has no prompt in the cache is
    unanswerable (y = -1): the only correct action for it is a cache miss.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    n_cache = int(round(cache_frac * len(labels)))
    ci, qi = perm[:n_cache], perm[n_cache:]

    cached_classes = np.unique(labels[ci])
    remap = {c: i for i, c in enumerate(cached_classes)}
    y_cache = np.array([remap[c] for c in labels[ci]])
    y_query = np.array([remap.get(c, -1) for c in labels[qi]])

    return CacheExperiment(
        X_cache=X[ci], y_cache=y_cache,
        X_query=X[qi], y_query=y_query,
        prompts_query=[prompts[i] for i in qi],
        n_answers=len(cached_classes),
    )
