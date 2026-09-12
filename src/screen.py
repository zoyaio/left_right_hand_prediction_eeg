"""Phase 1 checkpoint: screen subjects and report trial counts.

Run as:  python -m src.screen [n_subjects]
"""

from __future__ import annotations

import sys
from collections import Counter

import mne
import numpy as np

from . import data


def main(max_subject: int = 20) -> int:
    mne.set_log_level("ERROR")

    all_rejections: list[str] = []
    per_subject: dict[int, Counter] = {}
    usable_subjects: list[int] = []

    for subject in range(1, max_subject + 1):
        runs, rejections = data.load_subject(subject, verbose=False)
        all_rejections.extend(rejections)

        if not runs:
            continue

        counts: Counter = Counter()
        for rd in runs:
            epochs = data.make_epochs(rd.raw)
            labels = data.epoch_labels(epochs)
            counts.update(data.CLASS_NAMES[int(v)] for v in labels)
            # drop_log has an entry per *considered* event, and unselected T0
            # rest events show up as 'IGNORED'. Only count real drops.
            n_dropped = sum(
                1 for reasons in epochs.drop_log if reasons and "IGNORED" not in reasons
            )
            if n_dropped:
                counts["dropped_edge"] += n_dropped

        per_subject[subject] = counts
        usable_subjects.append(subject)

    print("=" * 62)
    print(f"SCREENING: subjects 1-{max_subject}, runs {data.RUNS_EXECUTED}")
    print("=" * 62)

    if all_rejections:
        print(f"\nRejected runs ({len(all_rejections)}):")
        for reason in all_rejections:
            print(f"  {reason}")
    else:
        print("\nNo runs rejected.")

    print(f"\nUsable subjects: {len(usable_subjects)}/{max_subject}")
    print("\n  subj   left  right   edge-dropped   balance")
    print("  " + "-" * 46)

    lefts, rights = [], []
    for subject in usable_subjects:
        c = per_subject[subject]
        left, right = c["left"], c["right"]
        lefts.append(left)
        rights.append(right)
        ratio = left / (left + right) if (left + right) else float("nan")
        flag = "  <-- imbalanced" if abs(ratio - 0.5) > 0.15 else ""
        print(
            f"  S{subject:03d}   {left:4d}   {right:4d}   {c['dropped_edge']:8d}"
            f"       {ratio:5.2f}{flag}"
        )

    total_left, total_right = sum(lefts), sum(rights)
    print("  " + "-" * 46)
    print(f"  TOTAL  {total_left:4d}   {total_right:4d}")
    print(
        f"\n  trials/subject: mean {np.mean(np.array(lefts) + np.array(rights)):.1f}, "
        f"min {min(np.array(lefts) + np.array(rights))}, "
        f"max {max(np.array(lefts) + np.array(rights))}"
    )
    print(f"  overall balance: {total_left / (total_left + total_right):.3f} left")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
