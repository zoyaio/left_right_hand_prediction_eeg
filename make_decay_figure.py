"""The carryover figure: how long a movement stays readable on the scalp.

Measures the lateralised band power across the WHOLE 8.2 s trial -- cue,
movement, and the full rest gap up to the next cue -- aligned to the hand that
moved. Everything is measured inside one trial's own data, so nothing here is
stitched across the trial boundary.

The point of the figure: the previous trial's beta rebound is still large where
the next trial's baseline window is drawn, which is why subtracting that window
injects the previous trial's hand into the current trial's features.

Usage:  python make_decay_figure.py [n_subjects]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

import mne  # noqa: E402

from src import data, features  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
CACHE = ROOT / "results" / "decay_curve.npz"

MOVE_END = 4.1  # movement block length
NEXT_CUE = 8.2  # inter-cue interval
STEP = 0.25  # resolution of the curve
N_PAIRS = 1


def collect(n_subjects: int) -> dict:
    """Signed lateralised log power over the full trial, per band."""
    mne.set_log_level("ERROR")
    edges = np.arange(0.0, NEXT_CUE - STEP + 1e-9, STEP)
    bands = features.BANDS_MOTOR
    per_subject: list[np.ndarray] = []

    for i, subject in enumerate(range(1, n_subjects + 1), 1):
        runs, _ = data.load_subject(subject, verbose=False)
        if not runs:
            continue
        trials: list[np.ndarray] = []
        signs: list[int] = []

        for rd in runs:
            raw = rd.raw.copy()
            raw.notch_filter(freqs=[features.LINE_FREQ], verbose="ERROR")
            sfreq = raw.info["sfreq"]
            lap = features.laplacian_signals(raw, n_pairs=N_PAIRS)
            power = np.stack(
                [
                    features.band_power_timecourse(lap, sfreq, b)
                    for b in bands.values()
                ]
            )  # (n_bands, n_pairs, 2, n_times)
            n_times = power.shape[-1]

            epochs = data.make_epochs(raw, tmin=0.0, tmax=0.1)
            y = data.epoch_labels(epochs)
            onsets = epochs.events[:, 0] - raw.first_samp

            for onset, label in zip(onsets, y):
                stop = onset + int(round(NEXT_CUE * sfreq))
                if stop > n_times:
                    continue  # last trial of the run: no following gap recorded
                row = np.empty((len(bands), len(edges)))
                for k, t0 in enumerate(edges):
                    a = onset + int(round(t0 * sfreq))
                    b = onset + int(round((t0 + STEP) * sfreq))
                    lp = np.log(power[:, 0, :, a:b].mean(axis=-1))  # (n_bands, 2)
                    row[:, k] = lp[:, 0] - lp[:, 1]  # d = log C3 - log C4
                trials.append(row)
                # contralateral-positive: right hand -> C3 active -> d up
                signs.append(1 if label == 1 else -1)

        if not trials:
            continue
        arr = np.stack(trials)  # (n_trials, n_bands, n_steps)
        # z-score per subject so subjects with different absolute power are
        # comparable; uses no labels.
        arr = (arr - arr.mean(axis=0)) / np.where(
            arr.std(axis=0) < 1e-12, 1.0, arr.std(axis=0)
        )
        signed = arr * np.array(signs)[:, None, None]
        per_subject.append(signed.mean(axis=0))
        print(f"  [{i:3d}/{n_subjects}] S{subject:03d}  {len(trials)} trials", flush=True)

    stacked = np.stack(per_subject)  # (n_subjects, n_bands, n_steps)
    return {
        "curve": stacked,
        "t": edges + STEP / 2,
        "band_names": np.array(list(bands)),
    }


def draw(curve: np.ndarray, t: np.ndarray, band_names: list[str]) -> None:
    n_sub = len(curve)
    fig, ax = plt.subplots(figsize=(11, 5.2))

    lo_y = min(curve[:, b, :].mean(axis=0).min() for b in range(len(band_names)))
    hi_y = max(curve[:, b, :].mean(axis=0).max() for b in range(len(band_names)))
    pad = (hi_y - lo_y) * 0.35
    ax.set_ylim(lo_y - pad, hi_y + pad * 1.5)

    ax.axvspan(0, MOVE_END, color="#888888", alpha=0.13, zorder=0)
    ax.axhline(0, color="k", lw=0.9)
    ax.axvline(NEXT_CUE, color="k", lw=1.2)

    # Where the NEXT trial's baseline windows fall, in this trial's time frame.
    for (blo, bhi), color, name in [
        ((-2.0, -1.0), "#d62728", "baseline\n(-2,-1)"),
        ((-1.0, 0.0), "#2ca02c", "baseline\n(-1,0)"),
    ]:
        a, b = NEXT_CUE + blo, NEXT_CUE + bhi
        ax.axvspan(a, b, color=color, alpha=0.18, zorder=1)
        ax.text(
            (a + b) / 2, lo_y - pad * 0.55, name,
            ha="center", va="center", fontsize=8, color=color, fontweight="bold",
        )

    colors = {"mu": "#1f77b4", "beta": "#d62728"}
    for b, band in enumerate(band_names):
        m = curve[:, b, :].mean(axis=0)
        se = curve[:, b, :].std(axis=0) / np.sqrt(n_sub)
        ax.plot(t, m, "-", color=colors.get(band, "k"), lw=2.4, label=band, zorder=4)
        ax.fill_between(t, m - se, m + se, color=colors.get(band, "k"),
                        alpha=0.22, zorder=3)

    beta = curve[:, list(band_names).index("beta"), :].mean(axis=0)
    peak = int(np.argmax(beta))
    ax.plot([t[peak]], [beta[peak]], "o", color="#d62728", ms=9, zorder=6)
    ax.annotate(
        f"beta rebound peaks {t[peak] - MOVE_END:.2f} s after movement ends\n"
        f"(the plan assumed 0.5-1 s)",
        xy=(t[peak], beta[peak]),
        xytext=(t[peak] - 3.4, beta[peak] + pad * 0.85),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=1.1),
    )
    sel = (t >= NEXT_CUE - 2.0) & (t < NEXT_CUE - 1.0)
    ax.annotate(
        f"the current baseline window sits at\n"
        f"{100 * beta[sel].mean() / beta[peak]:.0f}% of the rebound peak",
        xy=(NEXT_CUE - 1.5, beta[sel].mean()),
        xytext=(NEXT_CUE - 3.9, lo_y - pad * 0.15),
        fontsize=9, color="#d62728", fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#d62728"),
    )

    ax.set_xlim(0, NEXT_CUE)
    ax.set_xlabel("time from cue (s)      |      grey = movement, 0-4.1 s")
    ax.set_ylabel("lateralised log power, z\n(positive = contralateral to the hand that moved)")
    ax.set_title(
        "The previous trial is still on the scalp when the next one starts\n"
        f"{n_sub} subjects, aligned to the hand that moved",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax.text(
        NEXT_CUE - 0.08, ax.get_ylim()[1] * 0.88, "next cue -->",
        ha="right", fontsize=9, style="italic",
    )
    ax.text(
        MOVE_END / 2, ax.get_ylim()[1] * 0.88, "movement",
        ha="center", fontsize=9, style="italic", color="#555555",
    )
    fig.tight_layout()
    out = FIG / "carryover_decay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")

    # numbers worth quoting
    for b, band in enumerate(band_names):
        m = curve[:, b, :].mean(axis=0)
        pk = int(np.argmax(m))
        for lo, hi in [(-2.0, -1.0), (-1.0, 0.0)]:
            sel = (t >= NEXT_CUE + lo) & (t < NEXT_CUE + hi)
            print(
                f"  {band:5s} peak {m[pk]:+.3f} at {t[pk] - MOVE_END:+.2f}s after "
                f"offset | in baseline window ({lo:+.0f},{hi:+.0f}): {m[sel].mean():+.3f} "
                f"= {100 * m[sel].mean() / m[pk]:.0f}% of peak"
            )


def main(n_subjects: int = 57) -> int:
    FIG.mkdir(exist_ok=True)
    CACHE.parent.mkdir(exist_ok=True)
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        curve, t, names = z["curve"], z["t"], [str(s) for s in z["band_names"]]
    else:
        out = collect(n_subjects)
        curve, t, names = out["curve"], out["t"], [str(s) for s in out["band_names"]]
        np.savez_compressed(CACHE, curve=curve, t=t, band_names=np.array(names))
    draw(curve, t, names)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 57))
