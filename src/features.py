"""Laplacian -> band-pass -> Hilbert -> time-binned log power -> d/s features.

Ordering constraints that are easy to get wrong:

* The Laplacian must be applied BEFORE the Hilbert envelope. Both the Laplacian
  and the band-pass are linear so they commute with each other, but the
  envelope is not -- we want the envelope OF the Laplacian-cleaned signal, not
  the difference of two envelopes.

* Filtering and Hilbert run on the CONTINUOUS recording, then epochs are cut
  from the resulting power timecourse. Filtering after epoching would put a
  filter edge artifact at every trial boundary.

Feature layout: for each trial we hold log power with shape
``(n_bands, n_bins, 2)`` where the last axis is [C3_lap, C4_lap], and derive

    d = log P_C3 - log P_C4     (antisymmetric: the lateralised signal)
    s = log P_C3 + log P_C4     (symmetric: common-mode nuisance covariate)
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from scipy.signal import hilbert

from .data import LAPLACIAN_PAIRS, RunData, epoch_labels, make_epochs, rest_intervals

BANDS_MOTOR = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
BANDS_EMG = {"emg": (30.0, 50.0)}  # Phase 4 control band

LINE_FREQ = 60.0

# Epoch WIDE, feed NARROW. The epoch starts at -2 s so the pre-cue window is
# available as a DIAGNOSTIC and as the trial-wise baseline, but only the
# post-cue bins reach the classifier (see postcue_mask).
#
# TMAX = 5.5 s is set by the trial structure, which is 4.1 s of movement then
# 4.1 s of rest -- the next cue lands at 8.2 s, so there is a 4.1 s clean gap
# after movement offset. The window stops at 5.5 s because:
#   * movement/ERD runs 0-4.1 s;
#   * the beta rebound peaks ~0.5-1 s after offset (~4.6-5.1 s) and typically
#     lasts 0.5-1 s, so it is spent by ~5.5 s;
#   * bins past ~6 s are resting noise, and a bin close to the next cue could
#     pick up preparation for the NEXT movement -- which is lateralised to the
#     next hand and, because hands alternate 76% of the time, anti-correlated
#     with the current label. That is the carryover confound running forward,
#     so the window deliberately stops short of it.
TMIN, TMAX = -2.0, 5.5

# 0.5 s bins, non-overlapping. The rebound is a sub-second event; a 1 s bin
# averages it together with the surrounding ERD and blurs the two into each
# other. 0-5.5 s at 0.5 s gives 11 post-cue steps.
BIN_WIDTH = 0.5
BIN_STEP = 0.5

REST_WINDOW = 1.0  # length of the windows rest power is estimated over


def bin_edges(tmin: float = TMIN, tmax: float = TMAX) -> list[tuple[float, float]]:
    """Analysis windows spanning the epoch, back to back."""
    starts = np.arange(tmin, tmax - BIN_WIDTH + 1e-9, BIN_STEP)
    return [(float(s), float(s + BIN_WIDTH)) for s in starts]


BIN_CENTERS = np.array([(a + b) / 2 for a, b in bin_edges()])


def postcue_mask(centers: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask over bins selecting the ones the classifier is allowed.

    Pre-cue bins are a poor feature source -- nothing before the cue indicates
    which hand is about to move -- but a powerful diagnostic, since that is
    where the previous trial's residue is visible. They are extracted and
    plotted, never fed to the model.
    """
    centers = BIN_CENTERS if centers is None else centers
    return centers > 0.0


@dataclass
class RunFeatures:
    """Log band power per trial, plus the rest-period power for normalisation."""

    subject: int
    run: int
    log_power: np.ndarray  # (n_trials, n_bands, n_pairs, n_bins, 2) last axis [L, R]
    y: np.ndarray  # (n_trials,) 0=left 1=right
    onsets: np.ndarray  # (n_trials,) cue time in seconds, for SURVIVING epochs
    rest_log_power: np.ndarray  # (n_rest_windows, n_bands, n_pairs, 2)
    band_names: list[str]


def laplacian_signals(raw: mne.io.BaseRaw, n_pairs: int = 1) -> np.ndarray:
    """Small surface Laplacian over ``n_pairs`` symmetric electrode pairs.

    C3_clean = C3 - mean(FC3, C1, C5, CP3), and the mirror image for C4.
    A broad, smeared source hits the centre and its four neighbours about
    equally and cancels; a focal source at C3 survives. Zero fitted
    parameters, so this behaves identically on an unseen subject.

    Returns (n_pairs, 2, n_times), last axis ordered [left, right].
    """
    out = []
    for pair in LAPLACIAN_PAIRS[:n_pairs]:
        sides = []
        for center, neighbours in pair:
            center_data = raw.get_data(picks=[center])[0]
            neighbour_data = raw.get_data(picks=neighbours)
            sides.append(center_data - neighbour_data.mean(axis=0))
        out.append(np.stack(sides))
    return np.stack(out)


def band_power_timecourse(sig: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Instantaneous power in one band, via band-pass then Hilbert envelope.

    The filtered signal swings through zero every cycle, so its raw value says
    nothing about rhythm strength. The Hilbert transform builds a quarter-cycle
    phase-shifted companion; together they give the envelope -- the smooth
    outline of how large the oscillation is at each instant.
    """
    low, high = band
    filtered = mne.filter.filter_data(
        sig, sfreq=sfreq, l_freq=low, h_freq=high, verbose="ERROR"
    )
    envelope = np.abs(hilbert(filtered, axis=-1))
    return envelope**2


def extract_run(
    rd: RunData,
    bands: dict[str, tuple[float, float]] | None = None,
    n_pairs: int = 1,
) -> RunFeatures:
    """Full feature extraction for one run."""
    bands = bands or BANDS_MOTOR
    raw = rd.raw.copy()
    sfreq = raw.info["sfreq"]

    # Remove the 60 Hz mains hum before anything else. Matters most for the
    # 30-50 Hz control band, which sits close enough for the hum to bleed in.
    raw.notch_filter(freqs=[LINE_FREQ], verbose="ERROR")

    lap = laplacian_signals(raw, n_pairs=n_pairs)  # (n_pairs, 2, n_times)

    # (n_bands, n_pairs, 2, n_times) power timecourses
    power = np.stack(
        [band_power_timecourse(lap, sfreq, band) for band in bands.values()]
    )

    # Epoch over the SAME span the bins cover. Each run ends as its last
    # movement ends -- there is no rest block after the final trial -- so with
    # the window out to 5.5 s the last trial of every run has ~1 s of missing
    # data. MNE drops those epochs here; without this the bin loop would slice
    # past the end of the recording and write NaNs for that trial.
    epochs = make_epochs(raw, tmin=TMIN, tmax=TMAX)
    y = epoch_labels(epochs)
    onsets = epochs.events[:, 0] - raw.first_samp

    windows = bin_edges()
    n_times = power.shape[-1]
    log_power = np.empty((len(y), len(bands), n_pairs, len(windows), 2), dtype=float)

    for trial_idx, onset in enumerate(onsets):
        for bin_idx, (start_t, stop_t) in enumerate(windows):
            start = onset + int(round(start_t * sfreq))
            stop = onset + int(round(stop_t * sfreq))
            if start < 0 or stop > n_times:
                raise ValueError(
                    f"S{rd.subject}R{rd.run}: trial {trial_idx} bin "
                    f"[{start_t}, {stop_t}] runs outside the recording"
                )
            # (n_bands, n_pairs, 2) mean power in the window, then log
            log_power[trial_idx, :, :, bin_idx, :] = np.log(
                power[..., start:stop].mean(axis=-1)
            )

    rest_log_power = _rest_log_power(power, raw, sfreq, len(bands), n_pairs)

    return RunFeatures(
        subject=rd.subject,
        run=rd.run,
        log_power=log_power,
        y=y,
        onsets=onsets / sfreq,
        rest_log_power=rest_log_power,
        band_names=list(bands),
    )


def _rest_log_power(
    power: np.ndarray, raw: mne.io.BaseRaw, sfreq: float, n_bands: int, n_pairs: int
) -> np.ndarray:
    """Log power over non-overlapping windows inside the T0 rest blocks."""
    chunks = []
    width = int(round(REST_WINDOW * sfreq))

    for start_t, stop_t in rest_intervals(raw):
        start = int(round(start_t * sfreq))
        stop = min(int(round(stop_t * sfreq)), power.shape[-1])
        for win_start in range(start, stop - width + 1, width):
            chunk = power[..., win_start : win_start + width].mean(axis=-1)
            chunks.append(np.log(chunk))

    if not chunks:
        return np.empty((0, n_bands, n_pairs, 2))
    return np.stack(chunks)


# --------------------------------------------------------------------------
# d / s construction and normalisation
# --------------------------------------------------------------------------


def to_ds(log_power: np.ndarray) -> np.ndarray:
    """(..., 2) log power -> (..., 2) of [d, s].

    d = log C3 - log C4 carries the lateralised signal; when the left hand is
    active C4 drops AND C3 rises, and both push d the same way.
    s = log C3 + log C4 is where the common-mode junk went -- kept as a
    nuisance covariate rather than discarded.
    """
    c3 = log_power[..., 0]
    c4 = log_power[..., 1]
    return np.stack([c3 - c4, c3 + c4], axis=-1)


def baseline_correct(
    ds: np.ndarray, window: tuple[float, float] = (-2.0, -1.0)
) -> np.ndarray:
    """Subtract each trial's OWN pre-cue level from every bin of that trial.

    This is not cosmetic -- it is the fix for a real confound in this dataset.
    Consecutive trials use different hands 76% of the time (P(switch)=0.76 vs
    0.50 for independent randomisation), and the previous trial's movement ends
    only 4.2 s before the current cue, so its post-movement beta rebound -- which
    sits contralateral to the PREVIOUS hand -- is still present in the current
    trial's pre-cue window. Absolute d therefore predicts the current label
    without any motor decoding at all.

    Subtracting the trial's own pre-cue level removes that carried-over DC
    offset, leaving the within-trial CHANGE, which is what ERD actually means.
    It also makes the pre-cue control meaningful: the baseline bins become ~0
    by construction, so any remaining pre-cue decodability is genuine leakage.
    """
    centers = BIN_CENTERS
    sel = (centers >= window[0] + BIN_WIDTH / 2 - 1e-9) & (
        centers <= window[1] - BIN_WIDTH / 2 + 1e-9
    )
    if not sel.any():
        # fall back to the single earliest bin
        sel = centers == centers.min()
    baseline = ds[..., sel, :].mean(axis=-2, keepdims=True)
    return ds - baseline


def normalise(
    ds: np.ndarray,
    mode: str = "per_subject",
    rest_ds: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Centre and scale features using the recording's own statistics.

    Applied identically at train and test time -- we never pool statistics
    across the training set and then apply them to a lone test recording.
    Uses only EEG, never labels.

    Modes:
      per_subject  z-score across all of one subject's trials (~45)  [default]
      per_run      z-score within each run (~15 trials) -- noisier estimate,
                   but what predict.py can do when handed a single file
      rest         centre/scale from the T0 rest windows instead; far more
                   data behind the estimate, and rest power is what ERD is
                   physiologically defined against
      none         no normalisation
    """
    if mode == "none":
        return ds

    if mode == "rest":
        if rest_ds is None or len(rest_ds) == 0:
            raise ValueError("rest normalisation requires rest windows")
        # rest has no trial/bin structure, so stats are per (band, d-or-s)
        mean = np.expand_dims(rest_ds.mean(axis=0), -2)  # bins axis
        std = np.expand_dims(rest_ds.std(axis=0), -2)
        std = np.where(std < 1e-12, 1.0, std)
        return (ds - mean[None]) / std[None]

    if mode in ("per_subject", "per_run"):
        if groups is None:
            raise ValueError(f"{mode} normalisation requires groups")
        out = np.empty_like(ds)
        for g in np.unique(groups):
            sel = groups == g
            block = ds[sel]
            std = block.std(axis=0)
            std = np.where(std < 1e-12, 1.0, std)
            out[sel] = (block - block.mean(axis=0)) / std
        return out

    raise ValueError(f"unknown normalisation mode {mode!r}")


def flatten(ds: np.ndarray, use_interaction: bool = False) -> np.ndarray:
    """(n_trials, n_bands, n_bins, 2) -> (n_trials, n_features).

    With ``use_interaction`` the product d*s is appended per band per bin.
    A linear model cannot otherwise express "trust d less when the
    common-mode junk s is large", since that is multiplicative.
    """
    n_trials = ds.shape[0]
    if not use_interaction:
        return ds.reshape(n_trials, -1)

    interaction = (ds[..., 0] * ds[..., 1])[..., None]
    return np.concatenate([ds, interaction], axis=-1).reshape(n_trials, -1)


def feature_names(
    band_names: list[str],
    n_pairs: int = 1,
    use_interaction: bool = False,
    postcue_only: bool = True,
) -> list[str]:
    """Labels aligned to ``flatten`` output, for reading model weights back."""
    from .data import PAIR_NAMES

    centers = BIN_CENTERS[postcue_mask()] if postcue_only else BIN_CENTERS
    kinds = ["d", "s"] + (["dxs"] if use_interaction else [])
    return [
        f"{band}|{PAIR_NAMES[p]}@{center:+.2f}s:{kind}"
        for band in band_names
        for p in range(n_pairs)
        for center in centers
        for kind in kinds
    ]
