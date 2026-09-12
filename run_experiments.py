"""Full experiment sweep. Produces every number in the README.

Usage:  python run_experiments.py [n_subjects] [n_perm]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from src import controls, dataset, evaluate, features  # noqa: E402

# Fixed a priori at the single C3/C4 pair -- the textbook anatomical choice for
# hand motor cortex. Deliberately NOT selected by comparing cross-subject scores
# across pair counts: that would be tuning a hyperparameter on the evaluation
# metric and would make the headline figure optimistic.
N_PAIRS = 1
OUT = Path(__file__).resolve().parent / "results"

PRIMARY = "pooled-rest (primary)"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def build_arms(ds: dataset.Dataset):
    """The three baselining schemes, and what each is for.

    A "baseline" is doing up to three separable jobs, and it matters which:

      1. make subjects comparable   -- people differ enormously in absolute
                                       band power, so a raw number is
                                       meaningless until referenced to the
                                       person.
      2. make ERD mean "below rest" -- ERD is DEFINED as a drop below resting
                                       level, so a rest reference is what makes
                                       the feature literally interpretable.
      3. remove the previous trial  -- the carryover confound.

    POOLED REST does 1 and 2. It is one constant per subject, estimated from the
    T0 blocks inside the same file, so it structurally CANNOT carry
    trial-specific information about which hand moved last.

    TRIAL-WISE PRE-CUE attempts 3 and fails, for a reason measured in
    figures/carryover_decay.png: the beta rebound peaks ~2.3 s after movement
    ends, not the 0.5-1 s assumed, so the (-2,-1) s window sits at ~92% of the
    previous trial's rebound peak. Subtracting that constant OVER-subtracts,
    because the residue has decayed further by the time the current trial runs.
    The over-subtraction points opposite the previous hand -- and since hands
    alternate 76% of the time, that looks like the current hand. It injects the
    confound it was meant to remove. Kept as a reported contrast, not used.

    NO BASELINE is the reference point showing what the corrections cost.
    """
    X_rest, y, g = dataset.make_xy(ds, norm_mode="rest", baseline=False)
    X_trial, _, _ = dataset.make_xy(ds, norm_mode="per_subject", baseline=True)
    X_none, _, _ = dataset.make_xy(ds, norm_mode="per_subject", baseline=False)
    arms = [
        (PRIMARY, X_rest),
        ("trial-wise pre-cue (injects)", X_trial),
        ("no baseline", X_none),
    ]
    return arms, y, g


def main(n_subjects: int = 20, n_perm: int = 100) -> int:
    OUT.mkdir(exist_ok=True)
    subjects = list(range(1, n_subjects + 1))
    results: dict = {"n_subjects": n_subjects}

    ds = dataset.build(subjects, n_pairs=N_PAIRS, verbose=True)
    results["n_trials"] = len(ds)

    arms, y, g = build_arms(ds)
    X_primary = dict(arms)[PRIMARY]

    # ---------------------------------------------------------------- headline
    banner("1. HEADLINE: leave-one-subject-out, by baselining scheme")
    for tag, X in arms:
        r = evaluate.run_loso(X, y, g, tune_C=True, keep_coefs=(tag == PRIMARY))
        print("  " + r.summary(tag))
        results[f"loso_{tag}"] = {
            "mean": r.mean,
            "std": r.std,
            "per_subject": dict(zip(map(int, r.subjects), map(float, r.accuracies))),
        }
        if r.coefs is not None:
            np.save(OUT / "coefs.npy", r.coefs)

    # --------------------------------------------------- the carryover confound
    banner("2. THE CONFOUND: accuracy split by switch vs stay trials")
    print("   Consecutive trials use different hands ~76% of the time, and the")
    print("   previous trial's beta rebound is still large when the next cue")
    print("   arrives. A model riding that residue scores well when the hand")
    print("   SWITCHES and poorly when it REPEATS.\n")
    print("   Drift in attention/effort hits switch and repeat trials alike, so a")
    print("   gap HERE is carryover specifically, not slow drift.\n")
    for tag, X in arms:
        s = controls.switch_stay(X, y, g, ds.runs)
        print(
            f"  {tag:30s} overall {s['overall']:.3f} | "
            f"switch {s['switch']:.3f} (n={s['n_switch']}) | "
            f"stay {s['stay']:.3f} (n={s['n_stay']}) | gap {s['gap']:+.3f}"
        )
        results[f"switch_stay_{tag}"] = s
    print(
        "\n   Note the trial-wise arm has the LARGEST gap. Subtracting a pre-cue\n"
        "   baseline makes carryover worse here, not better -- see\n"
        "   figures/carryover_decay.png for why."
    )

    # ---------------------------------------------------------------- controls
    banner(f"3. CONTROLS (on the {PRIMARY} model)")

    ds_emg = dataset.build(
        subjects, bands=features.BANDS_EMG, n_pairs=N_PAIRS, verbose=False
    )
    X_emg, _, _ = dataset.make_xy(ds_emg, norm_mode="rest", baseline=False)
    r = evaluate.run_loso(X_emg, y, g, tune_C=False)
    print("  " + r.summary("30-50 Hz muscle band"))
    print("     (if this separates L/R we may be reading forearm muscle, not cortex)")
    results["emg_band"] = {"mean": r.mean, "std": r.std}

    null = controls.permutation_null(X_primary, y, g, n_perm=n_perm)
    obs = results[f"loso_{PRIMARY}"]["mean"]
    p = float((null >= obs).sum() + 1) / (len(null) + 1)
    print(
        f"  {'label permutation null':<34s} mean {null.mean():.3f}  "
        f"sd {null.std():.3f}  95th pct {np.percentile(null, 95):.3f}"
    )
    print(f"     observed {obs:.3f} -> empirical p = {p:.4f}  ({n_perm} permutations)")
    results["permutation"] = {
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": p,
        "n_perm": n_perm,
    }
    np.save(OUT / "permutation_null.npy", null)

    # ------------------------------------------------------- person vs the task
    banner("4. HOW MUCH IS THE PERSON RATHER THAN THE TASK")
    w = evaluate.run_within_subject(X_primary, y, g, ds.runs)
    print("  " + w.summary("within-subject (leave-one-run-out)"))
    print("  " + f"{'cross-subject (LOSO)':<34s} mean {obs:.3f}")
    print(f"  {'gap':<34s} {w.mean - obs:+.3f}")
    results["within_subject"] = {"mean": w.mean, "std": w.std}
    results["person_task_gap"] = w.mean - obs

    # Probe identity on UN-NORMALISED features only. On per-subject-referenced
    # features the probe is degenerate rather than informative: within-subject
    # centring makes each subject's features sum to zero, so under CV a held-out
    # sample sits opposite its own class centroid and is never predicted. That
    # returns ~0.000, which looks dramatic and is really an artifact of
    # combining within-group centring with cross-validation.
    X_nonorm, _, _ = dataset.make_xy(ds, norm_mode="none", baseline=False)
    probe = controls.subject_identity_probe(X_nonorm, g)
    print(
        f"\n  {'subject-identity probe (un-normalised)':<44s} "
        f"{probe['accuracy']:.3f}  (chance {probe['chance']:.3f})"
    )
    print(
        f"     Features identify WHO at {probe['accuracy'] / probe['chance']:.1f}x chance, "
        f"while identifying WHICH HAND at only {obs:.3f}."
    )
    print("     The per-subject reference is what removes this -- it is doing its job.")
    results["identity_probe"] = probe

    for tag, Xp in [("un-normalised", X_nonorm), ("rest-referenced", X_primary)]:
        naive = controls.naive_random_split(Xp, y, g)
        loso_p = evaluate.run_loso(Xp, y, g, tune_C=False).mean
        print(
            f"\n  {tag}: naive random split {naive:.3f} vs correct LOSO "
            f"{loso_p:.3f}  ->  inflation {naive - loso_p:+.3f}"
        )
        results[f"naive_split_{tag}"] = {
            "naive": naive,
            "loso": loso_p,
            "inflation": naive - loso_p,
        }
    print(
        "\n     Note: the usual random-split inflation does NOT appear here, and the\n"
        "     reason is worth stating. That inflation comes from subject identity\n"
        "     carrying label information. Every subject here is ~50/50 left/right,\n"
        "     so knowing WHO a trial came from says nothing about WHICH HAND, and\n"
        "     leaking subject identity buys the model nothing. The dangerous leak in\n"
        "     this dataset is temporal (the previous trial), not per-subject."
    )

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    perms = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    sys.exit(main(n, perms))
