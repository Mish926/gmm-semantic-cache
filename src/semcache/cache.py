"""Cache policies: GPTCache-style cosine nearest neighbour, and our GMM posterior cache."""
import numpy as np

from .embed import l2_normalize


class CosineNNCache:
    """Serve the answer of the most similar cached query; score = cosine similarity."""

    def fit(self, X, answers):
        self.X = l2_normalize(X)
        self.answers = np.asarray(answers)
        return self

    def query(self, Xq, batch=2048):
        Xq = l2_normalize(Xq)
        preds, scores = [], []
        for i in range(0, len(Xq), batch):
            S = Xq[i:i + batch] @ self.X.T
            j = S.argmax(1)
            preds.append(self.answers[j])
            scores.append(S[np.arange(len(j)), j])
        return np.concatenate(preds), np.concatenate(scores)


class GMMCache:
    """p(a | x) = sum_k p(k | x) theta_{k,a}; serve only if confident AND x is in a dense region."""

    def __init__(self, gmm, density_quantile=0.05):
        self.gmm = gmm
        self.density_quantile = density_quantile

    def fit(self, Z, answers, n_answers):
        R = self.gmm.predict_proba(Z)                     # N x K
        onehot = np.zeros((len(answers), n_answers))
        onehot[np.arange(len(answers)), answers] = 1.0
        counts = R.T @ onehot                             # K x A soft counts
        self.theta_ = counts / counts.sum(1, keepdims=True)
        self.purity_ = self.theta_.max(1)
        self.density_threshold_ = float(np.quantile(self.gmm.score_samples(Z), self.density_quantile))
        return self

    def query(self, Zq):
        p_answer = self.gmm.predict_proba(Zq) @ self.theta_   # N x A
        pred = p_answer.argmax(1)
        conf = p_answer.max(1)
        logp = self.gmm.score_samples(Zq)
        score = np.where(logp > self.density_threshold_, conf, 0.0)
        return pred, score, logp
