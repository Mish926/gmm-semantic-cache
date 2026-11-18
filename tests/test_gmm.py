import numpy as np

from semcache.gmm import DiagGMM


def make_blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    return np.vstack([c + rng.normal(size=(300, 2)) for c in centers])


def test_em_loglik_never_decreases():
    Z = make_blobs()
    gmm = DiagGMM(3, max_iter=50, reg_covar=1e-6, seed=0).fit(Z)
    assert np.all(np.diff(gmm.loglik_history_) > -1e-6)


def test_posteriors_sum_to_one():
    Z = make_blobs()
    gmm = DiagGMM(3, seed=0).fit(Z)
    assert np.allclose(gmm.predict_proba(Z).sum(1), 1.0)


def test_recovers_three_clusters():
    Z = make_blobs()
    gmm = DiagGMM(3, seed=0).fit(Z)
    means = gmm.means_[np.lexsort(gmm.means_.T[::-1])]
    expected = np.array([[0.0, 0.0], [0.0, 6.0], [6.0, 0.0]])
    assert np.allclose(means, expected, atol=0.5)
