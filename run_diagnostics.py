"""Diagnostics the plan left to be decided by the data.

These are the experiments the design document flagged as "verify against data
before locking", plus the label-structure control. They do not feed the
headline pipeline -- they check the assumptions it rests on.

The rebound-visibility check lives in make_decay_figure.py, which measures the
whole 8.2 s trial rather than just the analysis window.

Usage:  python run_diagnostics.py [n_subjects]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from src import controls, dataset, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "results"
N_PAIRS = 1


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# 2. Feature layout A vs B
# ---------------------------------------------------------------------------


def layout_a_vs_b(ds: dataset.Dataset, results: dict) -> None:
    """A: 0.5 s bins for mu AND beta.  B: fine beta, mu collapsed to one bin.

    Mu has no rebound, so fine time resolution should buy it little. If
    coarsening mu does not cost accuracy, that is evidence the motor-specific
    signal is carried by BETA -- which is the claim the anti-attention argument
    rests on. B is a test of the story, not just a simplification.
    """
    banner("2. FEATURE LAYOUT A vs B -- does coarsening mu cost anything?")

    # Same pooled-rest referencing the headline model uses.
    trial_ds = dataset.normalised_ds(ds, norm_mode="rest", baseline=False)

    bands = list(ds.band_names)
    mu_i, beta_i = bands.index("mu"), bands.index("beta")

    X_a = trial_ds.reshape(len(ds.y), -1)
    # B: mu averaged across the whole post-cue window, beta left time-resolved
    mu_coarse = trial_ds[:, mu_i : mu_i + 1].mean(axis=-2, keepdims=True)
    beta_fine = trial_ds[:, beta_i : beta_i + 1]
    X_b = np.concatenate(
        [mu_coarse.reshape(len(ds.y), -1), beta_fine.reshape(len(ds.y), -1)], axis=1
    )
    # Control: the mirror of B -- coarsen BETA instead, keep mu fine. If the
    # story is right, THIS is the one that should hurt.
    beta_coarse = trial_ds[:, beta_i : beta_i + 1].mean(axis=-2, keepdims=True)
    mu_fine = trial_ds[:, mu_i : mu_i + 1]
    X_c = np.concatenate(
        [mu_fine.reshape(len(ds.y), -1), beta_coarse.reshape(len(ds.y), -1)], axis=1
    )

    out = {}
    for tag, X in [
        ("A: mu fine + beta fine", X_a),
        ("B: mu COARSE + beta fine", X_b),
        ("C: mu fine + beta COARSE", X_c),
    ]:
        r = evaluate.run_loso(X, ds.y, ds.subjects, tune_C=True)
        print(f"  {r.summary(tag)}   [{X.shape[1]} features]")
        out[tag] = {"mean": r.mean, "std": r.std, "n_features": int(X.shape[1])}
    results["layout"] = out

    d_b = out["B: mu COARSE + beta fine"]["mean"] - out["A: mu fine + beta fine"]["mean"]
    d_c = out["C: mu fine + beta COARSE"]["mean"] - out["A: mu fine + beta fine"]["mean"]
    print(f"\n  coarsening mu   costs {d_b:+.4f}")
    print(f"  coarsening beta costs {d_c:+.4f}")
    print(
        "  If |coarsening mu| << |coarsening beta|, beta carries the "
        "motor-specific signal."
    )


# ---------------------------------------------------------------------------
# 8. Label-structure control
# ---------------------------------------------------------------------------


def label_structure(ds: dataset.Dataset, results: dict) -> None:
    """Does a trial's POSITION in the run predict its label?

    If it did, any model with access to ordering could score above chance
    without touching the EEG at all. Should be ~chance.
    """
    banner("8. LABEL STRUCTURE -- does trial position predict the hand?")

    pos = np.zeros(len(ds.y), dtype=int)
    for subject in np.unique(ds.subjects):
        for run in np.unique(ds.runs):
            idx = np.where((ds.subjects == subject) & (ds.runs == run))[0]
            pos[idx] = np.arange(len(idx))

    X_pos = pos.reshape(-1, 1).astype(float)
    r = evaluate.run_loso(X_pos, ds.y, ds.subjects, tune_C=False)
    print("  " + r.summary("position-only classifier"))

    # Also the raw picture: P(right) at each position, and the majority-class
    # rate, which is the accuracy a position-only rule could actually reach.
    frac = [float(ds.y[pos == p].mean()) for p in range(pos.max() + 1)]
    major = float(np.mean([max(f, 1 - f) for f in frac]))
    print("  P(right) by position: " + " ".join(f"{f:.2f}" for f in frac))
    print(f"  best achievable by position alone (majority per slot): {major:.3f}")

    # switch rate, the number the whole carryover story rests on
    is_switch, valid = controls.switch_flags(ds.y, ds.subjects, ds.runs)
    p_switch = float(is_switch[valid].mean())
    print(f"\n  P(consecutive trials use DIFFERENT hands) = {p_switch:.3f}")
    print("  (0.500 would be independent randomisation; the plan claims ~0.76)")

    results["label_structure"] = {
        "position_loso": {"mean": r.mean, "std": r.std},
        "p_right_by_position": frac,
        "majority_by_position": major,
        "p_switch": p_switch,
    }


def main(n_subjects: int = 57) -> int:
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    subjects = list(range(1, n_subjects + 1))
    ds = dataset.build(subjects, n_pairs=N_PAIRS, verbose=True)

    results: dict = {"n_subjects": n_subjects, "n_trials": len(ds)}
    layout_a_vs_b(ds, results)
    label_structure(ds, results)

    path = OUT / "diagnostics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 57))
