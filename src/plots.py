"""Result figures. Run after run_experiments.py.

Usage:  python -m src.plots [n_subjects]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from . import dataset, evaluate, features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RES = ROOT / "results"
N_PAIRS = 1


def fig_per_subject(results: dict) -> None:
    """The distribution, not the mean. Some people simply do not decode."""
    raw = results["loso_trial-wise pre-cue (injects)"]["per_subject"]
    cor = results["loso_pooled-rest (primary)"]["per_subject"]
    subs = sorted(raw, key=lambda k: cor[k])
    a_raw = [raw[s] for s in subs]
    a_cor = [cor[s] for s in subs]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(subs))
    ax.bar(x - 0.2, a_raw, 0.4, label="trial-wise pre-cue baseline (injects carryover)", color="#bbbbbb")
    ax.bar(x + 0.2, a_cor, 0.4, label="pooled-rest baseline (primary)", color="#1f77b4")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.axhline(np.mean(a_cor), color="#1f77b4", ls=":", lw=1.5,
               label=f"primary mean {np.mean(a_cor):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(s):03d}" for s in subs], rotation=90, fontsize=7)
    ax.set_ylabel("LOSO accuracy")
    ax.set_ylim(0.35, 1.0)
    ax.set_title("Per-subject accuracy: the spread is the result, not the mean")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "per_subject.png", dpi=140)
    plt.close(fig)


def fig_weights(ds, coefs: np.ndarray) -> None:
    """Mean |weight| over time bins, separated into d and s.

    If the model learned physiology the d weights should concentrate after the
    cue, not before it.
    """
    # Only post-cue bins reach the model, so the weight vector is indexed by
    # those bins alone -- not by the full epoch.
    t = features.BIN_CENTERS[features.postcue_mask()]
    n_band, n_pair, n_bin = len(ds.band_names), N_PAIRS, len(t)
    w = coefs.mean(axis=0).reshape(n_band, n_pair, n_bin, 2)

    fig, axes = plt.subplots(1, n_band, figsize=(11, 4.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for b, band in enumerate(ds.band_names):
        ax = axes[b]
        ax.plot(t, np.abs(w[b, :, :, 0]).mean(axis=0), "o-", color="#1f77b4",
                lw=2, label="d (lateralised)")
        ax.plot(t, np.abs(w[b, :, :, 1]).mean(axis=0), "s--", color="#999999",
                lw=1.5, label="s (common-mode)")
        ax.axvline(0, color="k", lw=0.9)
        # movement 0-4.1 s, then the beta rebound window
        ax.axvspan(0, 4.1, color="gray", alpha=0.10, zorder=0)
        ax.axvspan(4.1, 5.5, color="#2ca02c", alpha=0.12, zorder=0)
        ax.set_title(band)
        ax.set_xlabel("time from cue (s)")
        if b == 0:
            ax.set_ylabel("mean |weight| across LOSO folds")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Learned weights over time (grey = movement, green = rebound window)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG / "weights_over_time.png", dpi=140)
    plt.close(fig)


def fig_permutation(results: dict, null: np.ndarray) -> None:
    obs = results["loso_pooled-rest (primary)"]["mean"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(null, bins=20, color="#bbbbbb", edgecolor="w")
    ax.axvline(0.5, color="k", ls="--", lw=1, label="chance 0.500")
    ax.axvline(obs, color="#d62728", lw=2,
               label=f"observed {obs:.3f} (p={results['permutation']['p_value']:.3f})")
    ax.set_xlabel("LOSO accuracy under shuffled labels")
    ax.set_ylabel("count")
    ax.set_title(f"Permutation null ({results['permutation']['n_perm']} shuffles)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "permutation_null.png", dpi=140)
    plt.close(fig)


def main(n_subjects: int = 20) -> int:
    FIG.mkdir(exist_ok=True)
    results = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    ds = dataset.build(list(range(1, n_subjects + 1)), n_pairs=N_PAIRS, verbose=False)

    fig_per_subject(results)
    fig_weights(ds, np.load(RES / "coefs.npy"))
    fig_permutation(results, np.load(RES / "permutation_null.npy"))
    print(f"wrote figures to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
