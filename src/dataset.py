"""Build and cache the feature matrix across subjects."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from . import data, features

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


@dataclass
class Dataset:
    log_power: np.ndarray  # (n_trials, n_bands, n_pairs, n_bins, 2)
    y: np.ndarray  # (n_trials,)
    subjects: np.ndarray  # (n_trials,) subject id
    runs: np.ndarray  # (n_trials,) run id
    rest_log_power: np.ndarray  # (n_rest, n_bands, n_pairs, 2) pooled
    rest_subjects: np.ndarray  # (n_rest,)
    band_names: list[str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_pairs(self) -> int:
        return self.log_power.shape[2]

    @property
    def run_groups(self) -> np.ndarray:
        """Unique id per (subject, run), for per-run normalisation."""
        return self.subjects * 100 + self.runs


def _cache_key(subject_list: list[int], bands: dict, n_pairs: int) -> str:
    payload = repr((sorted(subject_list), sorted(bands.items()), n_pairs,
                    features.TMIN, features.TMAX,
                    features.BIN_WIDTH, features.BIN_STEP))
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def build(
    subject_list: list[int],
    bands: dict[str, tuple[float, float]] | None = None,
    n_pairs: int = 1,
    use_cache: bool = True,
    verbose: bool = True,
) -> Dataset:
    bands = bands or features.BANDS_MOTOR
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"ds_{_cache_key(subject_list, bands, n_pairs)}.npz"

    if use_cache and cache_path.exists():
        if verbose:
            print(f"loading cached features from {cache_path.name}")
        z = np.load(cache_path, allow_pickle=False)
        return Dataset(
            log_power=z["log_power"],
            y=z["y"],
            subjects=z["subjects"],
            runs=z["runs"],
            rest_log_power=z["rest_log_power"],
            rest_subjects=z["rest_subjects"],
            band_names=list(bands),
        )

    mne.set_log_level("ERROR")
    t0 = time.time()
    chunks, ys, subs, runs_, rest_chunks, rest_subs = [], [], [], [], [], []
    rejections: list[str] = []

    for i, subject in enumerate(subject_list, 1):
        run_data, rej = data.load_subject(subject, verbose=False)
        rejections.extend(rej)

        for rd in run_data:
            rf = features.extract_run(rd, bands=bands, n_pairs=n_pairs)
            chunks.append(rf.log_power)
            ys.append(rf.y)
            subs.append(np.full(len(rf.y), subject))
            runs_.append(np.full(len(rf.y), rd.run))
            rest_chunks.append(rf.rest_log_power)
            rest_subs.append(np.full(len(rf.rest_log_power), subject))

        if verbose and (i % 10 == 0 or i == len(subject_list)):
            print(
                f"  [{i:3d}/{len(subject_list)}] extracted "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    if verbose and rejections:
        print(f"\n{len(rejections)} run(s) rejected during extraction:")
        for reason in rejections:
            print(f"  REJECT {reason}")

    ds = Dataset(
        log_power=np.concatenate(chunks),
        y=np.concatenate(ys),
        subjects=np.concatenate(subs),
        runs=np.concatenate(runs_),
        rest_log_power=np.concatenate(rest_chunks),
        rest_subjects=np.concatenate(rest_subs),
        band_names=list(bands),
    )

    np.savez_compressed(
        cache_path,
        log_power=ds.log_power,
        y=ds.y,
        subjects=ds.subjects,
        runs=ds.runs,
        rest_log_power=ds.rest_log_power,
        rest_subjects=ds.rest_subjects,
    )
    if verbose:
        print(f"cached -> {cache_path.name} ({time.time() - t0:.0f}s total)")
    return ds


def normalised_ds(
    ds: Dataset,
    norm_mode: str = "rest",
    baseline: bool = False,
    postcue_only: bool = True,
) -> np.ndarray:
    """(n_trials, n_bands, n_pairs, n_bins, 2) of normalised [d, s].

    Same pipeline make_xy uses, stopping before the flatten, so diagnostics can
    slice bands and bins without re-implementing the normalisation.

    ``baseline`` subtracts each trial's own pre-cue level first. Off by default:
    the pre-cue window sits on the previous trial's beta rebound, so subtracting
    it injects carryover (see run_experiments.build_arms).

    ``postcue_only`` drops the pre-cue bins. It runs AFTER baseline_correct,
    which needs those bins to compute the baseline from -- epoch wide, feed
    narrow. Pass False only for diagnostics that look at the pre-cue window.
    """
    trial_ds = features.to_ds(ds.log_power)

    if baseline:
        trial_ds = features.baseline_correct(trial_ds)

    if postcue_only:
        trial_ds = trial_ds[..., features.postcue_mask(), :]

    if norm_mode == "rest":
        # rest stats are per subject, so normalise subject by subject
        rest_ds = features.to_ds(ds.rest_log_power)
        out = np.empty_like(trial_ds)
        for subject in np.unique(ds.subjects):
            sel = ds.subjects == subject
            out[sel] = features.normalise(
                trial_ds[sel], mode="rest", rest_ds=rest_ds[ds.rest_subjects == subject]
            )
        trial_ds = out
    else:
        groups = ds.run_groups if norm_mode == "per_run" else ds.subjects
        trial_ds = features.normalise(trial_ds, mode=norm_mode, groups=groups)

    return trial_ds


def make_xy(
    ds: Dataset,
    norm_mode: str = "rest",
    use_interaction: bool = False,
    baseline: bool = False,
    postcue_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dataset -> (X, y, groups), normalisation applied per recording."""
    trial_ds = normalised_ds(
        ds, norm_mode=norm_mode, baseline=baseline, postcue_only=postcue_only
    )
    X = features.flatten(trial_ds, use_interaction=use_interaction)
    return X, ds.y, ds.subjects
