"""SIF-weighted average GloVe sentence embeddings (Arora et al., 2017)."""
import gzip
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(s):
    return TOKEN.findall(s.lower())


def l2_normalize(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def load_glove(path, vocab):
    """Load only the vectors for words in `vocab` (keeps memory small)."""
    vecs = {}
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) <= 2:  # skip a "count dim" header line if present
                continue
            if parts[0] in vocab:
                vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float64)
    return vecs


class SIFEmbedder:
    def __init__(self, glove_path, a=1e-3):
        self.glove_path = glove_path
        self.a = a

    def fit(self, texts, vocab_texts=None):
        vocab_texts = texts if vocab_texts is None else vocab_texts
        vocab = {w for s in vocab_texts for w in tokenize(s)}
        self.vecs = load_glove(self.glove_path, vocab)
        self.dim = len(next(iter(self.vecs.values())))
        counts = Counter(w for s in texts for w in tokenize(s))
        total = sum(counts.values())
        self.weight = {w: self.a / (self.a + c / total) for w, c in counts.items()}
        X = self._average(texts)
        # common component removal: first right singular vector of training embeddings
        self.pc = np.linalg.svd(X, full_matrices=False)[2][0]
        return self

    def _average(self, texts):
        out = np.zeros((len(texts), self.dim))
        for i, s in enumerate(texts):
            words = [w for w in tokenize(s) if w in self.vecs]
            if words:
                a = np.array([self.weight.get(w, 1.0) for w in words])
                V = np.stack([self.vecs[w] for w in words])
                out[i] = (a[:, None] * V).sum(0) / len(words)
        return out

    def transform(self, texts):
        X = self._average(texts)
        return X - np.outer(X @ self.pc, self.pc)
