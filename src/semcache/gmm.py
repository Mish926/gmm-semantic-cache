"""Diagonal-covariance Gaussian mixture fit by EM (our own implementation)."""
import numpy as np
from scipy.special import logsumexp


class DiagGMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6, reg_covar=1e-3, seed=0):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

    # ---------- initialisation ----------
    def _init_kmeanspp(self, Z, rng):
        N = len(Z)
        centers = [Z[rng.integers(N)]]
        d2 = ((Z - centers[0]) ** 2).sum(1)
        for _ in range(1, self.K):
            c = Z[rng.choice(N, p=d2 / d2.sum())]
            centers.append(c)
            d2 = np.minimum(d2, ((Z - c) ** 2).sum(1))
        return np.array(centers)

    # ---------- core quantities ----------
    def _log_joint(self, Z):
        """log pi_k + log N(z | mu_k, diag(var_k)), shape (N, K)."""
        iv = 1.0 / self.var_
        maha = (Z ** 2) @ iv.T - 2.0 * Z @ (self.means_ * iv).T + (self.means_ ** 2 * iv).sum(1)
        log_norm = np.log(2.0 * np.pi * self.var_).sum(1)
        return np.log(self.weights_) - 0.5 * (log_norm + maha)

    def score_samples(self, Z):
        """log p(z) for each row."""
        return logsumexp(self._log_joint(Z), axis=1)

    def predict_proba(self, Z):
        """Posterior responsibilities p(k | z), shape (N, K)."""
        lj = self._log_joint(Z)
        return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))

    # ---------- EM ----------
    def fit(self, Z):
        rng = np.random.default_rng(self.seed)
        N, D = Z.shape
        self.means_ = self._init_kmeanspp(Z, rng)
        self.var_ = np.tile(Z.var(0) + self.reg_covar, (self.K, 1))
        self.weights_ = np.full(self.K, 1.0 / self.K)
        self.loglik_history_ = []

        for it in range(self.max_iter):
            # E-step
            lj = self._log_joint(Z)
            lse = logsumexp(lj, axis=1)
            self.loglik_history_.append(float(lse.mean()))
            R = np.exp(lj - lse[:, None])
            # M-step
            Nk = R.sum(0) + 1e-10
            self.weights_ = Nk / N
            self.means_ = (R.T @ Z) / Nk[:, None]
            second = (R.T @ Z ** 2) / Nk[:, None]
            self.var_ = np.maximum(second - self.means_ ** 2, 0.0) + self.reg_covar
            if it > 0 and abs(self.loglik_history_[-1] - self.loglik_history_[-2]) < self.tol:
                break

        self.loglik_history_.append(float(self.score_samples(Z).mean()))
        self.n_iter_ = len(self.loglik_history_) - 1
        return self

    def bic(self, Z):
        N, D = Z.shape
        n_params = (self.K - 1) + 2 * self.K * D
        return -2.0 * self.score_samples(Z).sum() + n_params * np.log(N)
