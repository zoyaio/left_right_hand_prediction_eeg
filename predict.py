"""Predict left/right fist labels for every annotated trial in a raw EDF file.

Usage:
    python predict.py path/to/S042R03.edf

Notes:
  * Trial times are read from the file's own T1/T2 annotations.
  * Features are referenced to the T0 rest blocks inside this same file, which
    is what makes a number mean "below this person's resting level" (an ERD).
    That uses the recording's EEG and its event timing, never its labels.
  * The whole file is needed at once, since the rest reference is estimated
    from it. This is not a single-trial online decoder. See README
    "Assumptions".
  * The final trial of a run is skipped: the recording stops as the last
    movement ends, so the analysis window runs past the end of the data.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

import mne  # noqa: E402

from src import data, dataset, features  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"

# Runs where T1/T2 mean left/right fist. In runs 5/6/9/10/13/14 they mean
# both-fists / both-feet instead, which this model was not trained for.
LR_RUNS = {3, 4, 7, 8, 11, 12}


def run_number(path: Path) -> int | None:
    """EEGMMIDB files are named S###R##.edf."""
    stem = path.stem.upper()
    if "R" in stem:
        try:
            return int(stem.split("R")[-1])
        except ValueError:
            return None
    return None


def predict_file(edf_path: Path, model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"no trained model at {model_path} -- run `python train.py` first"
        )
    bundle = joblib.load(model_path)

    # A model trained under a different time-binning would still LOAD -- the old
    # -2..4 s / 1.0 s layout and the current 0..5.5 s / 0.5 s one happen to give
    # the same 44 columns -- and would then predict confidently from bins that
    # mean something else. Check the layout explicitly rather than trusting the
    # column count.
    saved_bins = np.asarray(bundle.get("bin_centers", []))
    live_bins = features.BIN_CENTERS
    if saved_bins.shape != live_bins.shape or not np.allclose(saved_bins, live_bins):
        raise RuntimeError(
            f"{model_path.name} was trained on a different time-bin layout "
            f"({len(saved_bins)} bins spanning "
            f"[{saved_bins.min():+.2f}, {saved_bins.max():+.2f}] s) than this code "
            f"produces ({len(live_bins)} bins spanning "
            f"[{live_bins.min():+.2f}, {live_bins.max():+.2f}] s). "
            "Re-run `python train.py` to retrain the model."
        )

    run = run_number(edf_path)
    if run is not None and run not in LR_RUNS:
        print(
            f"WARNING: run {run} is a fists-vs-feet run. T1/T2 do not mean "
            f"left/right fist there, so these predictions are not meaningful.",
            file=sys.stderr,
        )

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="warn", verbose="ERROR")

    problems = data.screen_run(raw, subject=-1, run=run or -1)
    if problems:
        print(f"WARNING: {edf_path.name}: " + "; ".join(problems), file=sys.stderr)

    rd = data.RunData(subject=-1, run=run or -1, raw=raw)
    rf = features.extract_run(rd, bands=bundle["bands"], n_pairs=bundle["n_pairs"])

    ds = dataset.Dataset(
        log_power=rf.log_power,
        y=rf.y,
        subjects=np.zeros(len(rf.y), dtype=int),
        runs=np.full(len(rf.y), run or -1),
        rest_log_power=rf.rest_log_power,
        rest_subjects=np.zeros(len(rf.rest_log_power), dtype=int),
        band_names=rf.band_names,
    )
    # Normalise against the T0 rest blocks inside THIS file -- the same scheme
    # the model was trained under, read back from the bundle so the two cannot
    # drift apart. Uses this recording's EEG and its event TIMING, never its
    # labels.
    X, _, _ = dataset.make_xy(
        ds,
        norm_mode=bundle.get("norm_mode", "rest"),
        baseline=bundle.get("baseline", False),
    )

    pred = bundle["model"].predict(X)
    proba = bundle["model"].predict_proba(X)[:, 1]

    return {
        "file": str(edf_path),
        "n_trials": int(len(pred)),
        # Read onsets off the SURVIVING epochs, not the raw annotation list.
        # The final trial of each run is dropped (the recording stops before the
        # window closes), so the two lists are not the same length.
        "onsets": list(map(float, rf.onsets)),
        "pred": [data.CLASS_NAMES[int(p)] for p in pred],
        "p_right": list(map(float, proba)),
        "true": [data.CLASS_NAMES[int(v)] for v in rf.y] if len(rf.y) else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("edf", type=Path, help="path to a raw EEGMMIDB .edf file")
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--quiet", action="store_true", help="print labels only")
    args = ap.parse_args()

    out = predict_file(args.edf, args.model)

    if args.quiet:
        print("\n".join(out["pred"]))
        return 0

    print(f"{out['file']}  ({out['n_trials']} trials)\n")
    print("  trial   onset(s)    predicted   P(right)   annotated")
    print("  " + "-" * 52)
    for i, (onset, p, pr) in enumerate(zip(out["onsets"], out["pred"], out["p_right"])):
        truth = out["true"][i] if out["true"] else "-"
        print(f"  {i:5d}   {onset:8.2f}    {p:<9s}   {pr:6.3f}     {truth}")

    if out["true"]:
        acc = np.mean([p == t for p, t in zip(out["pred"], out["true"])])
        print(f"\n  accuracy against this file's own annotations: {acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
