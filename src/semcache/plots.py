import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_prelim(loglik, curves, logp, is_oos, density_threshold, k, out_path):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))

    ax[0].plot(np.arange(len(loglik)), loglik, color="#2f5d8a", lw=1.8)
    ax[0].set(xlabel="EM iteration", ylabel="avg. log-likelihood / query",
              title=f"(a) EM objective (K={k}, diag. cov.)")

    colors = {"cosine NN threshold": "#9a9a9a", "GMM posterior + density gate": "#c0532b"}
    for name, c in curves.items():
        ax[1].plot(c[:, 1], c[:, 2], lw=1.8, label=name, color=colors.get(name))
    ax[1].set(xlabel="cache hit rate", ylabel="precision of served answers",
              title="(b) Hit rate vs. correctness", ylim=(0.5, 1.01))
    ax[1].legend(frameon=False, loc="lower left")

    bins = np.linspace(np.quantile(logp, 0.002), logp.max(), 50)
    ax[2].hist(logp[~is_oos], bins, alpha=0.7, density=True, color="#2f5d8a", label="in-scope test")
    ax[2].hist(logp[is_oos], bins, alpha=0.7, density=True, color="#c0532b", label="out-of-scope test")
    ax[2].axvline(density_threshold, color="k", ls="--", lw=1)
    ax[2].set(xlabel="log p(x) under GMM", ylabel="density", title="(c) Novel-query detection")
    ax[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
