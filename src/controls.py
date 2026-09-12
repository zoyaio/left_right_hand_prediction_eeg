"""Control experiments: rule out boring explanations for the headline number.

The most important one here is ``switch_stay``, which is not a standard control
-- it came out of diagnosing this dataset. Consecutive trials use different
hands 76% of the time, and the previous trial's beta rebound is still present
when the next cue arrives, so a model can score well by decoding the PREVIOUS
trial and betting on alternation. Splitting accuracy by whether the hand
switched or repeated exposes that directly: a carryover-exploiting model is
good on switch trials and BELOW chance on stay trials, while a genuine decoder
performs alike on both.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from . import evaluate


def switch_flags(
    y: np.ndarray, subjects: np.ndarray, runs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(is_switch, has_predecessor) per trial, computed within each run."""
    is_switch = np.zeros(len(y), dtype=bool)
    valid = np.zeros(len(y), dtype=bool)
    for subject in np.unique(subjects):
        for run in np.unique(runs):
            idx = np.where((subjects == subject) & (runs == run))[0]
            if len(idx) < 2:
                continue
            is_switch[idx[1:]] = y[idx[1:]] != y[idx[:-1]]
            valid[idx[1:]] = True
    return is_switch, valid


def loso_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions under leave-one-subject-out."""
    pred = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = evaluate.make_model()
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    return pred


def switch_stay(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
) -> dict[str, float]:
    """Accuracy split by whether the hand alternated or repeated.

    A large positive gap means the model is riding the previous trial rather
    than decoding the current one.
    """
    pred = loso_predictions(X, y, subjects)
    correct = pred == y
    is_switch, valid = switch_flags(y, subjects, runs)
    sw, st = valid & is_switch, valid & ~is_switch
    return {
        "overall": float(correct[valid].mean()),
        "switch": float(correct[sw].mean()),
        "stay": float(correct[st].mean()),
        "gap": float(correct[sw].mean() - correct[st].mean()),
        "n_switch": int(sw.sum()),
        "n_stay": int(st.sum()),
    }


def permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_perm: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Shuffle labels WITHIN each subject and re-run the full LOSO evaluation.

    Gives an empirical null. Catches leakage that the analytic 50% chance line
    cannot -- if scrambled labels score above chance, the pipeline is
    manufacturing accuracy.
    """
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        y_perm = y.copy()
        for g in np.unique(groups):
            sel = groups == g
            y_perm[sel] = rng.permutation(y[sel])
        null.append(evaluate.run_loso(X, y_perm, groups, tune_C=False).mean)
    return np.array(null)


def naive_random_split(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0
) -> float:
    """The WRONG evaluation, run deliberately to show how much it inflates.

    Pools every trial and splits at random, so trials from the same person land
    in both train and test and the model can score by recognising the person.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    return float(
        cross_val_score(
            evaluate.make_model(),
            X,
            y,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            scoring="accuracy",
        ).mean()
    )


def subject_identity_probe(X: np.ndarray, subjects: np.ndarray, seed: int = 0) -> dict:
    """How well do the features identify the PERSON rather than the task?

    Quantifies the "how much of this is about the individual" question directly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]
    )
    acc = cross_val_score(
        clf,
        X,
        subjects,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="accuracy",
    ).mean()
    return {"accuracy": float(acc), "chance": 1.0 / len(np.unique(subjects))}
