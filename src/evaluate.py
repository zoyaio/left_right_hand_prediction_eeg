"""Leave-one-subject-out evaluation.

Train on N-1 people, test on the held-out person, repeat. The alternative --
pooling all trials and splitting at random -- puts the same person in both
train and test, letting the model score well by recognising the individual
rather than the task.

The regularisation strength C is chosen by an inner CV *within the training
folds only*; tuning it against the held-out subject would inflate the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def make_model(C: float | None = None) -> Pipeline:
    """L2 logistic regression.

    L2 rather than L1 because adjacent time bins overlap by 50% and are
    therefore highly correlated; L1 would arbitrarily keep one of a correlated
    pair and zero the other, making the learned weights unstable to read.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0 if C is None else C,
                    solver="liblinear",
                    max_iter=2000,
                ),
            ),
        ]
    )


@dataclass
class LosoResult:
    accuracies: np.ndarray  # one per held-out subject
    subjects: np.ndarray
    chosen_C: list[float] = field(default_factory=list)
    coefs: np.ndarray | None = None  # (n_folds, n_features)

    @property
    def mean(self) -> float:
        return float(self.accuracies.mean())

    @property
    def std(self) -> float:
        return float(self.accuracies.std())

    def summary(self, label: str = "") -> str:
        acc = self.accuracies
        return (
            f"{label:<34s} mean {self.mean:.3f}  sd {self.std:.3f}  "
            f"median {np.median(acc):.3f}  "
            f"range [{acc.min():.3f}, {acc.max():.3f}]  "
            f">chance {(acc > 0.5).sum()}/{len(acc)}"
        )


def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tune_C: bool = True,
    keep_coefs: bool = False,
) -> LosoResult:
    logo = LeaveOneGroupOut()
    accs, subs, chosen, coefs = [], [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        if tune_C:
            # Inner CV is itself leave-one-subject-out, over TRAINING subjects
            # only -- the held-out subject is never seen during tuning.
            # materialise the splits: a generator cannot be pickled to workers
            inner_cv = list(
                LeaveOneGroupOut().split(X[train_idx], y[train_idx], groups[train_idx])
            )
            search = GridSearchCV(
                make_model(),
                {"clf__C": C_GRID},
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            search.fit(X[train_idx], y[train_idx])
            model = search.best_estimator_
            chosen.append(search.best_params_["clf__C"])
        else:
            model = make_model()
            model.fit(X[train_idx], y[train_idx])
            chosen.append(1.0)

        accs.append(model.score(X[test_idx], y[test_idx]))
        subs.append(groups[test_idx][0])
        if keep_coefs:
            coefs.append(model.named_steps["clf"].coef_.ravel())

    return LosoResult(
        accuracies=np.array(accs),
        subjects=np.array(subs),
        chosen_C=chosen,
        coefs=np.array(coefs) if keep_coefs else None,
    )


def run_within_subject(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       runs: np.ndarray) -> LosoResult:
    """Leave-one-run-out inside each subject, for the person-vs-task gap."""
    accs, subs = [], []
    for subject in np.unique(groups):
        sel = groups == subject
        Xs, ys, rs = X[sel], y[sel], runs[sel]
        fold_accs = []
        for run in np.unique(rs):
            tr, te = rs != run, rs == run
            if len(np.unique(ys[tr])) < 2:
                continue
            model = make_model()
            model.fit(Xs[tr], ys[tr])
            fold_accs.append(model.score(Xs[te], ys[te]))
        if fold_accs:
            accs.append(float(np.mean(fold_accs)))
            subs.append(subject)
    return LosoResult(accuracies=np.array(accs), subjects=np.array(subs))
