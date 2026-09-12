"""Fit the final model on all usable subjects and save it.

Usage:  python train.py [n_subjects]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

from src import dataset, evaluate, features  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "models"
N_PAIRS = 1  # fixed a priori; see run_experiments.py
NORM_MODE = "rest"  # pooled T0 rest; see run_experiments.build_arms
BASELINE = False    # no trial-wise pre-cue subtraction


def main(n_subjects: int = 20) -> int:
    MODEL_DIR.mkdir(exist_ok=True)
    subjects = list(range(1, n_subjects + 1))

    ds = dataset.build(subjects, n_pairs=N_PAIRS, verbose=True)
    # Pooled rest (T0 blocks inside the same file) is the baseline: it makes
    # subjects comparable AND makes a feature literally mean "below this
    # person's resting level", which is what ERD is defined as. It is one
    # constant per subject, so unlike a trial-wise pre-cue baseline it cannot
    # carry previous-trial information. See run_experiments.build_arms.
    X, y, g = dataset.make_xy(ds, norm_mode=NORM_MODE, baseline=BASELINE)

    # Choose C by leave-one-subject-out over the training subjects.
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

    best_C, best_acc = 1.0, -1.0
    for C in evaluate.C_GRID:
        acc = cross_val_score(
            evaluate.make_model(C),
            X,
            y,
            groups=g,
            cv=LeaveOneGroupOut(),
            scoring="accuracy",
        ).mean()
        print(f"  C={C:<7g} LOSO accuracy {acc:.4f}")
        if acc > best_acc:
            best_C, best_acc = C, acc

    print(f"\nselected C={best_C} (LOSO {best_acc:.4f}); refitting on all subjects")
    model = evaluate.make_model(best_C)
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "n_pairs": N_PAIRS,
            "bands": features.BANDS_MOTOR,
            "bin_centers": features.BIN_CENTERS,
            "tmin": features.TMIN,
            "tmax": features.TMAX,
            "norm_mode": NORM_MODE,
            "baseline": BASELINE,
            "trained_on_subjects": subjects,
            "loso_accuracy": float(best_acc),
        },
        MODEL_DIR / "model.joblib",
    )
    (MODEL_DIR / "model_card.json").write_text(
        json.dumps(
            {
                "task": "left vs right fist, executed movements (runs 3/7/11)",
                "n_train_subjects": len(subjects),
                "n_train_trials": int(len(y)),
                "loso_accuracy": float(best_acc),
                "C": best_C,
                "n_features": int(X.shape[1]),
                "normalisation": "pooled T0 rest within the same recording, label-free",
                "baseline": "pooled rest (no trial-wise pre-cue subtraction)",
                "carryover": (
                    "A trial-wise pre-cue baseline was tested and rejected: the "
                    "beta rebound peaks ~2.3 s after movement offset, so the "
                    "pre-cue window sits on the previous trial's rebound and "
                    "subtracting it injects carryover. Residual carryover is "
                    "reported as the switch/stay split, not corrected."
                ),
                "caveat": (
                    "Requires all trials from a recording at once (per-recording "
                    "normalisation). Not valid for single-trial online use."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved -> {MODEL_DIR / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
