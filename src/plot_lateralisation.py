"""Phase 2 checkpoint + the headline diagnostic figure.

Originally just a go/no-go gate: does d separate left from right at all?
It also serves as the EMPIRICAL check on the T1/T2 label mapping, since
physiology says the contralateral side desynchronises:

    right-hand trials -> C3 quietens -> d = C3-C4 goes NEGATIVE
    left-hand  trials -> C4 quietens -> d goes POSITIVE

Running it is what exposed the carryover confound: the classes separate well
BEFORE the cue, which no amount of motor decoding can explain.

The second row shows why the obvious fix does not work. Subtracting a pre-cue
baseline cannot remove the residue, because that window sits on the PEAK of the
previous trial's beta rebound -- the rebound peaks ~2.3 s after movement ends,
not the 0.5-1 s usually assumed. Subtracting it over-subtracts and injects the
previous hand. Measured in figures/carryover_decay.png.

Run as:  python -m src.plot_lateralisation [n_subjects]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import dataset, features

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def subject_centre(d: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Remove each subject's own mean.

    Between-subject offsets (skull thickness, electrode placement) otherwise
    swamp the within-subject effect. NOTE: this forces the two class curves to
    be near mirror images of each other, so that symmetry is an artifact of the
    display. Only the SEPARATION between the curves is evidence.
    """
    out = np.empty_like(d)
    for subject in np.unique(subjects):
        sel = subjects == subject
        out[sel] = d[sel] - d[sel].mean(axis=0)
    return out


def main(n_subjects: int = 20) -> int:
    FIG_DIR.mkdir(exist_ok=True)
    ds = dataset.build(list(range(1, n_subjects + 1)), n_pairs=1)

    trial_ds = features.to_ds(ds.log_power)
    variants = [
        ("no baseline", trial_ds),
        ("after trial-wise pre-cue baseline", features.baseline_correct(trial_ds)),
    ]

    t = features.BIN_CENTERS
    n_band = len(ds.band_names)
    fig, axes = plt.subplots(2, n_band, figsize=(11, 7.8), sharex=True)
    axes = np.atleast_2d(axes)

    verdict_rows: list[tuple[str, float, float]] = []
    precue_rows: list[tuple[str, float]] = []

    for row, (variant, arr) in enumerate(variants):
        d = subject_centre(arr[:, :, 0, :, 0], ds.subjects)

        for band_idx, band in enumerate(ds.band_names):
            ax = axes[row, band_idx]
            for label, name, color in [
                (0, "left fist", "#1f77b4"),
                (1, "right fist", "#d62728"),
            ]:
                sel = ds.y == label
                mean = d[sel, band_idx].mean(axis=0)
                sem = d[sel, band_idx].std(axis=0) / np.sqrt(sel.sum())
                ax.plot(t, mean, label=name, color=color, lw=2)
                ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.2)

            ax.axhline(0, color="k", lw=0.7, ls=":")
            ax.axvline(0, color="k", lw=0.9)
            ax.axvspan(-2, -1, color="#d62728", alpha=0.10, zorder=0)  # baseline
            ax.axvspan(0, 4.1, color="gray", alpha=0.10, zorder=0)  # movement
            ax.axvspan(4.1, 5.5, color="#2ca02c", alpha=0.10, zorder=0)  # rebound
            ax.set_title(f"{band} -- {variant}", fontsize=10)
            if row == 1:
                ax.set_xlabel("time from cue (s)")
            if band_idx == 0:
                ax.set_ylabel("d = log P(C3) - log P(C4)\n(subject-centred)")
            ax.legend(frameon=False, fontsize=8)

            erd = (t >= 0.5) & (t <= 2.5)
            pre = t < -0.5
            if row == 0:
                verdict_rows.append(
                    (
                        band,
                        float(d[ds.y == 0, band_idx][:, erd].mean()),
                        float(d[ds.y == 1, band_idx][:, erd].mean()),
                    )
                )
                precue_rows.append(
                    (
                        band,
                        float(
                            d[ds.y == 0, band_idx][:, pre].mean()
                            - d[ds.y == 1, band_idx][:, pre].mean()
                        ),
                    )
                )

    fig.suptitle(
        "TOP: the classes already separate 1.5 s BEFORE the cue -- carryover from\n"
        "the previous trial, not decoding.   BOTTOM: after trial-wise baselining.\n"
        f"{n_subjects} subjects, executed movements, n={len(ds)} trials",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIG_DIR / "phase2_lateralisation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print(f"\nsaved {out}\n")
    print("Uncorrected, mean d over the 0.5-2.5 s ERD window:")
    print("  band    left-fist    right-fist    separation")
    print("  " + "-" * 46)
    ok = True
    for band, left, right in verdict_rows:
        print(f"  {band:6s}  {left:+9.4f}   {right:+9.4f}    {left - right:+.4f}")
        ok &= (left - right) > 0

    print("\nSame separation measured BEFORE the cue (t < -0.5 s):")
    for band, sep in precue_rows:
        print(f"  {band:6s}  {sep:+.4f}   <-- should be ~0 if this were motor decoding")

    print()
    if ok:
        print("Sign check PASSES: left-fist d > right-fist d, so the contralateral")
        print("side desynchronises as physiology predicts, and the label mapping")
        print("T1=left / T2=right is consistent with the data.")
        print()
        print("But note the pre-cue separation above. That is the confound, and it")
        print("is why the uncorrected accuracy figure cannot be taken at face value.")
    else:
        print("Sign check FAILS: separation has the wrong sign in at least one band.")
        print("Either T1/T2 are swapped or something upstream is broken.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
