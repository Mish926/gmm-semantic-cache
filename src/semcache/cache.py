"""Cache policies: GPTCache-style cosine nearest neighbour, and our GMM posterior cache."""
import numpy as np
from scipy import sparse

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


class KNNVoteCache:
    """Serve the majority answer of the k most similar cached queries; score = vote share.

    A non-probabilistic baseline that, like the GMM, uses agreement among nearby cached
    answers. Ties in vote share are broken by the mean similarity of the winning voters.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, answers):
        self.X = l2_normalize(X)
        self.answers = np.asarray(answers)
        return self

    def query(self, Xq, batch=2048):
        Xq = l2_normalize(Xq)
        k = min(self.k, len(self.X))
        preds, scores = [], []
        for i in range(0, len(Xq), batch):
            S = Xq[i:i + batch] @ self.X.T
            top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            top_sim = np.take_along_axis(S, top, axis=1)
            top_lab = self.answers[top]
            for labs, sims in zip(top_lab, top_sim):
                values, counts = np.unique(labs, return_counts=True)
                best = counts.max()
                winners = values[counts == best]
                if len(winners) > 1:  # tie: pick the label whose voters are most similar
                    means = [sims[labs == w].mean() for w in winners]
                    winner = winners[int(np.argmax(means))]
                else:
                    winner = winners[0]
                preds.append(winner)
                scores.append(best / k + 1e-3 * sims[labs == winner].mean())
        return np.array(preds), np.array(scores)


class GMMCache:
    """p(a | x) = sum_k p(k | x) theta_{k,a}; serve only if confident AND x is in a dense region."""

    def __init__(self, gmm, density_quantile=0.05):
        self.gmm = gmm
        self.density_quantile = density_quantile

    def fit(self, Z, answers, n_answers):
        R = self.gmm.predict_proba(Z)                                   # N x K
        answers = np.asarray(answers)
        onehot = sparse.csr_matrix(
            (np.ones(len(answers)), (np.arange(len(answers)), answers)),
            shape=(len(answers), n_answers))                            # N x A, sparse
        counts = np.asarray((onehot.T @ R).T)                           # K x A soft counts
        self.theta_ = counts / counts.sum(1, keepdims=True)
        self.purity_ = self.theta_.max(1)
        self.density_threshold_ = float(np.quantile(self.gmm.score_samples(Z), self.density_quantile))
        return self

    def query(self, Zq, batch=2048):
        preds, confs, logps = [], [], []
        for i in range(0, len(Zq), batch):
            Zb = Zq[i:i + batch]
            p_answer = self.gmm.predict_proba(Zb) @ self.theta_   # batch x A
            preds.append(p_answer.argmax(1))
            confs.append(p_answer.max(1))
            logps.append(self.gmm.score_samples(Zb))
        pred, conf, logp = map(np.concatenate, (preds, confs, logps))
        score = np.where(logp > self.density_threshold_, conf, 0.0)
        return pred, score, logp
