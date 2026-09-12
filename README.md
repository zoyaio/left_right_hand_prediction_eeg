# Left vs Right Fist Decoding — EEGMMIDB

Executed left- vs right-fist movement decoding from scalp EEG (PhysioNet
EEGMMIDB, runs 3/7/11), evaluated across subjects.

**The short version:** the obvious pipeline gets **67.7%** across subjects. Almost
all of that is an artifact. The number I actually defend is **53.4%**, and most
of this repository is the work of telling those two apart.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

```bash
python download_all.py && python train.py 57 && python run_experiments.py 57 100
```

Predict on a raw EDF the model has never seen:

```bash
python predict.py path/to/S042R03.edf
```

---

## Headline results (57 subjects, 2565 trials, leave-one-subject-out)

| Evaluation | Accuracy |
|---|---|
| Naive pipeline, no baseline correction | 0.677 ± 0.127 |
| **After removing the confound (defended figure)** | **0.534 ± 0.077** |
| Within-subject (leave-one-run-out) | 0.571 ± 0.084 |
| Label-permutation null | 0.500 ± 0.012 (p = 0.0099, 100 shuffles) |
| Chance | 0.500 |

Stability across scale, which is the reason to trust these:

| Subjects | naive | corrected | switch/stay gap (naive) |
|---|---|---|---|
| 20 | 0.656 | 0.547 | +0.201 |
| 45 | 0.677 | 0.538 | +0.201 |
| 57 | 0.677 | 0.534 | +0.216 |

The corrected figure drifts down slightly as the subject pool diversifies, as
expected. The confound reproduces at essentially **identical magnitude** at every
scale — it is a property of the experiment design, not a small-sample accident.

---

## The finding: most of the naive accuracy is the *previous* trial

I plotted the lateralisation index `d = log P(C3) − log P(C4)` over time,
expecting the classes to separate after the cue. They separated **1.5 seconds
before it** — in fact more strongly before the cue than during the movement:

| Window | mu separation | beta separation |
|---|---|---|
| Before the cue (t < −0.5 s) | +0.305 | +0.279 |
| ERD window (0.5–2.5 s) | +0.191 | +0.297 |

No amount of motor decoding explains knowing the answer before the question.
Working through it:

1. **Not filter smearing.** MNE's band-pass is zero-phase, so it does leak
   backwards — 0.83 s for mu. But a bin covering [−2,−1] s is outside that reach
   and still classified at 0.604, as well as the peak ERD window (0.620). The
   accuracy profile is *flat across the entire epoch*, which smearing cannot do.
2. **Not the label structure.** Run-level balance is 0.49–0.50, and
   position-within-run predicts the label at 0.504 — chance.
3. **It is the previous trial.** Consecutive trials use **different hands 76% of
   the time** (p ≈ 2e-53 against independent randomisation; lag-2 is 0.482, so
   it is a first-order alternation bias). The previous movement ends only 4.2 s
   before the next cue, so its **post-movement beta rebound** — which sits
   contralateral to the *previous* hand — is still present in the current
   trial's pre-cue window. Read the rebound, bet on alternation, score ~68%.

### The test that proves it

Split accuracy by whether the hand **switched** or **repeated** between
consecutive trials. A genuine decoder performs alike on both; a carryover
exploiter is good on switches and *below chance* on repeats.

| Model | switch (n=1802) | stay (n=580) | gap |
|---|---|---|---|
| Uncorrected | 0.723 | **0.507** | **+0.216** |
| Baseline-corrected | 0.548 | 0.516 | +0.032 |
| CSP+LDA (within-subject) | 0.608 | 0.586 | +0.022 |

Below chance on repeats is the signature: the uncorrected model actively
predicts the *previous* hand.

### The fix

Subtract each trial's own pre-cue level from every bin of that trial — the
standard definition of ERD as a *change* from baseline, which I had skipped in
favour of session-level normalisation only. Accuracy falls from 0.677 to 0.534,
and the switch/stay gap collapses from +0.216 to +0.032.

---

## Method

```
load EDF -> standardise channel names -> screen
  -> notch 60 Hz
  -> small Laplacian on 3 symmetric pairs (C3/C4, C1/C2, C5/C6)
  -> band-pass mu (8-13) and beta (13-30) -> Hilbert envelope -> log power
  -> 11 overlapping 1 s bins spanning [-2, +4] s
  -> d = logP(left) - logP(right),  s = logP(left) + logP(right)
  -> trial-wise baseline subtraction, then per-recording z-score
  -> L2 logistic regression
```

Two design choices worth stating:

**Features are built as `(d, s)`, not `(C3, C4)`.** These span the same space, so
a linear model can reach the same solutions either way — the difference is what
L2 shrinks. In the `(d, s)` basis the regulariser penalises the *lateralised*
direction and the *common-mode* direction independently, which is the prior I
actually hold. `d` also captures the push–pull effect in one number: when the
left hand is active C4 drops *and* C3 rises, and both move `d` the same way.
Keeping `s` as a nuisance covariate pays off visibly — its learned weights
exceed `d`'s in beta, which is suppressor-variable behaviour: it does not
discriminate alone, it cancels noise in `d`.

**Fixed Laplacian rather than learned CSP filters.** The Laplacian has zero
fitted parameters, so nothing can leak and it transfers to an unseen subject
unchanged. This is the standard pipeline (band-pass → CSP → log-variance → LDA)
with the spatial filters frozen to anatomy — trading peak accuracy for
cross-subject robustness, which is what is being graded.

---

## Controls

| Control | Result | Reading |
|---|---|---|
| Pre-cue bins only | 0.513 | Clean after correction (was 0.610 before) |
| 30–50 Hz "muscle" band | 0.527 | **See caveat below** |
| Label permutation (100 shuffles) | null 0.500 ± 0.012, p = 0.0099 | Real, but modest |
| Subject-identity probe | 0.192 vs 0.018 chance | Features identify *who* at 11× chance |
| Naive random split | inflation +0.007 | Negligible — explained below |

**The muscle-band control is my weakest result.** A 30–50 Hz classifier reaches
0.527 against a 0.500 null — uncomfortably close to the 0.534 headline. It moved
around across scales (0.532 at 20 subjects, 0.515 at 45, 0.527 at 57), so I do
not trust the exact value, but it does not go away. These are executed movements,
so real forearm muscle activity is present in the recording.

The control is also weak by construction: at 160 Hz the Nyquist limit is 80 Hz and
60 Hz is mains, so 30–50 Hz is a cramped proxy for EMG, which peaks well above it.
**I cannot rule out a muscle contribution to the 0.534, and would not claim it is
purely cortical.** The cleanest way to settle it would be the imagined runs, where
no movement occurs — which is the first thing I would do next.

**Why the random split shows no inflation** — this surprised me and the reason
matters. Random-split inflation comes from subject identity carrying label
information. Every subject here is ~50/50 left/right, so knowing *who* a trial
came from says nothing about *which hand*, and leaking subject identity buys
nothing. The dangerous leak in this dataset is temporal, not per-subject. The
identity probe confirms identity is strongly present in the features (0.192 vs
0.018 chance, 11×) — it just is not useful for this label.

*A probe I removed rather than reported:* running the identity probe on
per-subject z-scored features returns exactly 0.000, which looks dramatic and is
meaningless. Within-subject centring makes each subject's features sum to zero,
so under cross-validation a held-out sample sits opposite its own class centroid
and is never predicted. It is an artifact of combining within-group centring
with CV, not a finding.

---

## Person vs task

| | accuracy |
|---|---|
| Within-subject (leave-one-run-out) | 0.571 |
| Cross-subject (LOSO) | 0.534 |
| Gap | +0.038 |

My within-subject 0.571–0.603 closely matches an independently implemented
CSP+LDA pipeline at 0.602 on the same subjects — two different feature sets
agreeing is reasonable evidence the signal is real and that this is roughly
what is extractable here.

Per-subject LOSO accuracy ranges from 0.356 to 0.733, and only 37 of 57 subjects
clear chance at all. Reporting only the mean would hide that a substantial
minority sit at or below chance, consistent with the 15–30% "BCI illiteracy"
rate in the literature.

---

## Assumptions and limitations

1. **Per-recording normalisation uses the whole file.** Features are z-scored
   using that recording's own trials. This is label-free (EEG only), and valid
   because the deliverable takes a complete EDF. **It would not hold for a live
   online BCI** seeing one trial at a time.
2. **Trial-wise baselining assumes the pre-cue window is task-free.** It is
   contaminated by the previous trial's rebound, which is exactly what we are
   removing — but the rebound is still decaying during the trial, so subtracting
   a constant does not remove all of it. The residual +0.032 switch/stay gap
   suggests some carryover survives.
3. **Executed movements only.** Imagined runs are the more BCI-relevant and
   harder problem and are untouched here.
4. **57 of 109 subjects.** The remaining download was still running at the time
   of writing; every script takes a subject count argument, so scaling is a
   re-run of `run_experiments.py 109`. Accuracy drifted 0.547 (20) -> 0.538 (45)
   -> 0.534 (57) and would likely settle near 0.53.
5. **The defended figure is barely above chance.** 0.534 with p = 0.0099 is a
   real but weak effect. It is a far smaller claim than 0.677 would have been,
   and the honest one.

## What I would do next

- Report accuracy **conditioned on switch vs stay** as the primary metric
  rather than pooled accuracy. It is the honest number and this dataset makes
  pooled accuracy misleading.
- Model the carryover explicitly (estimate the previous trial's rebound and
  regress it out) instead of subtracting a flat baseline.
- Check whether the alternation bias affects the **imagined** runs identically,
  and whether published EEGMMIDB accuracies are exposed to it.
- Riemannian tangent-space features, the strongest simple cross-subject
  baseline, which I did not get to.

## Repository layout

```
src/data.py                  loading, screening, epoching
src/features.py              Laplacian, Hilbert, binning, d/s, baselining
src/dataset.py               feature matrix assembly + caching
src/evaluate.py              LOSO and within-subject evaluation
src/controls.py              switch/stay, permutation, identity, naive split
src/plot_lateralisation.py   the figure that exposed the confound
src/plots.py                 result figures
run_experiments.py           full sweep -> results/results.json
train.py / predict.py        trained model + CLI
design.md                    the reasoning written before any code
```

## AI use

Claude (Opus 5) was used throughout: to organise notes into the design in
`design.md`, to write most of the pipeline code, and as a sounding board while
diagnosing the pre-cue separation. The diagnosis itself was driven by the
evidence chain above — flat accuracy profile, then filter reach, then label
structure, then sequence autocorrelation — with each hypothesis checked against
the data rather than accepted on assertion. Several of its suggestions were
tested and discarded, including the initial assumption that shared stimulus
sequences across subjects explained the effect (they are 18/20 distinct).
