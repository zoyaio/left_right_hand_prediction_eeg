"""Loading, screening and epoching of the EEGMMIDB executed-movement runs.

Design notes that matter downstream:

* Runs are kept SEPARATE rather than concatenated. Filtering and the Hilbert
  transform run on continuous data (see features.py), and concatenating runs
  first would create a discontinuity at each seam that the filter smears into
  a large artifact. Epochs are concatenated instead, after filtering.

* Epochs are cut with ``baseline=None``. MNE's ``baseline`` argument subtracts
  a pre-cue mean from the raw *voltage*; we work in log *power*, so that
  correction would not mean what we want. Physiological baselining is handled
  in features.py.

* Label meaning is documented by PhysioNet rather than recoverable from the
  EDF: in runs 3/4/7/8/11/12, T1 is the LEFT fist and T2 is the RIGHT fist.
  This is asserted structurally here, and checked *empirically* by the Phase 2
  lateralisation plot -- if the ERD shows up on the wrong side, these are
  swapped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import mne
import numpy as np

RUNS_EXECUTED = [3, 7, 11]

# PhysioNet EEGMMIDB: for runs 3/4/7/8/11/12, T1 = left fist, T2 = right fist.
LABEL_MAP = {"T1": 0, "T2": 1}
CLASS_NAMES = {0: "left", 1: "right"}

REST_CODE = "T0"

EXPECTED_SFREQ = 160.0
EXPECTED_NCHAN = 64
MIN_TASK_EVENTS_PER_RUN = 10

# Symmetric electrode pairs, each with its four nearest neighbours for the small
# Laplacian. Kept explicit so screening can verify the channels exist before we
# rely on them. The first pair (C3/C4) is the v1 feature set; the rest extend the
# same antisymmetric construction to more of sensorimotor cortex.
LAPLACIAN_PAIRS = [
    (("C3", ["FC3", "C1", "C5", "CP3"]), ("C4", ["FC4", "C2", "C6", "CP4"])),
    (("C1", ["FC1", "Cz", "C3", "CP1"]), ("C2", ["FC2", "Cz", "C4", "CP2"])),
    (("C5", ["FC5", "C3", "T7", "CP5"]), ("C6", ["FC6", "C4", "T8", "CP6"])),
    (("FC3", ["F3", "FC1", "FC5", "C3"]), ("FC4", ["F4", "FC2", "FC6", "C4"])),
    (("CP3", ["C3", "CP1", "CP5", "P3"]), ("CP4", ["C4", "CP2", "CP6", "P4"])),
]

PAIR_NAMES = [f"{left[0]}/{right[0]}" for left, right in LAPLACIAN_PAIRS]

# Only the first pair is required for the pipeline to run at all; the rest are
# checked when the multi-pair feature set is requested.
REQUIRED_CHANNELS = sorted(
    {name for side in LAPLACIAN_PAIRS[0] for name in [side[0], *side[1]]}
)
ALL_LAPLACIAN_CHANNELS = sorted(
    {name for pair in LAPLACIAN_PAIRS for side in pair for name in [side[0], *side[1]]}
)


@dataclass
class RunData:
    """One run of one subject, loaded and screened."""

    subject: int
    run: int
    raw: mne.io.BaseRaw


def load_run(subject: int, run: int) -> mne.io.BaseRaw:
    """Fetch (from cache) one run, with channel names standardised."""
    paths = mne.datasets.eegbci.load_data(subject, [run], update_path=True, verbose="ERROR")
    with warnings.catch_warnings():
        # EEGMMIDB EDF headers trip a benign "not setting meas date" style warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")

    # Raw files name channels 'C3..', 'Fc3.', 'Cp3.' -- nothing matches a montage
    # until this runs.
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="warn", verbose="ERROR")
    return raw


def screen_run(raw: mne.io.BaseRaw, subject: int, run: int) -> list[str]:
    """Return a list of reasons this run is unusable. Empty list means fine."""
    problems: list[str] = []

    if raw.info["sfreq"] != EXPECTED_SFREQ:
        problems.append(f"sfreq={raw.info['sfreq']:g} (expected {EXPECTED_SFREQ:g})")

    if raw.info["nchan"] != EXPECTED_NCHAN:
        problems.append(f"nchan={raw.info['nchan']} (expected {EXPECTED_NCHAN})")

    missing = [ch for ch in REQUIRED_CHANNELS if ch not in raw.ch_names]
    if missing:
        problems.append(f"missing channels {missing}")

    descriptions = set(raw.annotations.description)
    missing_codes = {"T1", "T2"} - descriptions
    if missing_codes:
        problems.append(f"no {sorted(missing_codes)} annotations")

    n_task = sum(1 for d in raw.annotations.description if d in LABEL_MAP)
    if n_task < MIN_TASK_EVENTS_PER_RUN:
        problems.append(f"only {n_task} task events (expected >={MIN_TASK_EVENTS_PER_RUN})")

    return problems


def make_epochs(
    raw: mne.io.BaseRaw,
    tmin: float = -2.0,
    tmax: float = 4.0,
    picks: list[str] | None = None,
) -> mne.Epochs:
    """Cut task epochs around each L/R cue.

    ``baseline=None`` is deliberate -- see module docstring.

    Epochs running off either end of the recording are dropped by MNE, so read
    labels back off ``epochs.events`` rather than the annotation order.
    """
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    wanted = {code: event_id[code] for code in LABEL_MAP if code in event_id}
    if not wanted:
        raise ValueError("run contains no T1/T2 annotations")

    return mne.Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks=picks,
        preload=True,
        reject=None,
        on_missing="ignore",
        verbose="ERROR",
    )


def epoch_labels(epochs: mne.Epochs) -> np.ndarray:
    """0 = left fist, 1 = right fist, aligned to surviving epochs."""
    inverse = {code_id: LABEL_MAP[name] for name, code_id in epochs.event_id.items()}
    return np.array([inverse[code] for code in epochs.events[:, 2]], dtype=int)


def rest_intervals(raw: mne.io.BaseRaw) -> list[tuple[float, float]]:
    """(start, stop) times of the T0 rest blocks, for rest-based normalisation."""
    return [
        (onset, onset + duration)
        for onset, duration, desc in zip(
            raw.annotations.onset, raw.annotations.duration, raw.annotations.description
        )
        if desc == REST_CODE
    ]


def load_subject(
    subject: int, runs: list[int] | None = None, verbose: bool = True
) -> tuple[list[RunData], list[str]]:
    """Load and screen every run for one subject.

    Returns the usable runs plus a list of human-readable rejection reasons.
    A subject with any rejected run is still usable via its remaining runs;
    the caller decides whether to drop the subject entirely.
    """
    runs = runs or RUNS_EXECUTED
    usable: list[RunData] = []
    rejections: list[str] = []

    for run in runs:
        try:
            raw = load_run(subject, run)
        except Exception as exc:  # noqa: BLE001 - screening should never crash the sweep
            rejections.append(f"S{subject:03d}R{run:02d}: load failed ({exc!r})")
            continue

        problems = screen_run(raw, subject, run)
        if problems:
            rejections.append(f"S{subject:03d}R{run:02d}: " + "; ".join(problems))
            continue

        usable.append(RunData(subject=subject, run=run, raw=raw))

    if verbose and rejections:
        for reason in rejections:
            print(f"  REJECT {reason}", flush=True)

    return usable, rejections
