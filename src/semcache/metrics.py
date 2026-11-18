"""Hit rate vs. precision of served answers, swept over decision thresholds."""
import numpy as np


def hit_precision_curve(score, pred, y_true, n_thresholds=400):
    thresholds = np.unique(np.quantile(score, np.linspace(0.0, 1.0, n_thresholds)))
    rows = []
    for t in thresholds:
        hit = score >= t
        n_hit = hit.sum()
        precision = float((pred[hit] == y_true[hit]).mean()) if n_hit else 1.0
        rows.append((t, float(hit.mean()), precision))
    return np.array(rows)  # columns: threshold, hit_rate, precision


def hit_rate_at_precision(curve, target):
    ok = curve[curve[:, 2] >= target]
    return float(ok[:, 1].max()) if len(ok) else 0.0
