# Evaluation

Left vs right fist, executed movements (EEGMMIDB runs 3/7/11).
57 subjects, 2394 trials, leave-one-subject-out.
All numbers from `results/results.json` and `results/diagnostics.json`.

---

## Key metrics at a glance

| Test | Condition | Accuracy | Δ | Reading |
|---|---|---|---|---|
| *Reference* | Chance | 0.500 | — | |
| *Reference* | Permutation null (100 shuffles) | 0.500 ± 0.013 | — | observed 0.699, **p = 0.0099** |
| **Headline** | **Full model, cross-subject** | **0.699 ± 0.119** | — | |
| **Switch / stay** | Hand **switched** from previous trial (n=1714) | 0.722 | — | alternation bet pays off |
| | Hand **repeated** (n=509) | **0.646** | **−0.076** | carryover-immune floor |
| | *residual carryover gap* | | **+0.075** | reduced, not eliminated |
| **EMG control** | Full model (mu + beta) | 0.699 | — | |
| | **40–50 Hz muscle band, clean filter** | **0.537** | **−0.162** | result is not muscle-driven |
| | 35–50 Hz, clean filter | 0.544 | −0.155 | consistent |
| | 30–50 Hz, default filter | 0.601 | −0.098 | ⚠ leaks beta at 0.679 — invalid |
| **Person vs task** | Cross-subject (leave-one-subject-out) | **0.699** | — | test subjects are strangers |
| | Within-subject (leave-one-run-out) | 0.660 | **−0.039** | *lower* — not fitting individuals |
| | Per-subject range | 0.452 – 0.976 | — | 53/57 above chance, 4 below |

**How to read the three tests.**

*Switch/stay* is the carryover test. A genuine current-movement decoder is
indifferent to whether the hand switched; a model riding the previous trial is
good on switches and poor on repeats. The +0.075 gap is what survives, so 0.646
is the conservative figure.

*EMG* asks whether forearm muscle rather than cortex drives the result. The
clean band sits 0.162 below the model and only ~3 SD above the null. The 30–50 Hz
row is shown because it was the originally specified control and turned out to
be invalid — its filter passes high beta at 0.679, so it was measuring the
signal it was meant to control for.

*Person vs task* asks whether accuracy comes from recognising individuals.
Cross-subject exceeding within-subject is the wrong direction for that
explanation. But the per-subject range is wider than the margin over chance,
so which person you run on matters more than the mean suggests.

---

## 1. Data processing pipeline

### Selection and screening

Runs 3, 7 and 11 only — the executed left/right fist runs. Execution rather than
imagery is a scoping decision: the signal is spatially localised to contralateral
sensorimotor cortex, which is what justifies the narrow C3/C4 montage below.

Every run is screened before use and rejected if it fails any of: sampling rate
≠ 160 Hz, channel count ≠ 64, any required electrode missing, no T1/T2
annotations, or fewer than 10 task events. Screening is structural, never
performance-based — no run is dropped for scoring badly.

**The last trial of every run is dropped.** Recordings stop as the final movement
ends, so the 5.5 s analysis window runs past the end of the data. This costs
159 of 2553 trials (6.2%) and is why the trial count is 2394. Left unhandled it
would silently produce NaN features on one trial per run.

### The pipeline

```
load EDF → standardise channel names → screen
  → notch 60 Hz
  → small Laplacian on C3/C4
  → band-pass mu (8–13 Hz) and beta (13–30 Hz)
  → Hilbert envelope → power → log
  → 15 non-overlapping 0.5 s bins spanning [−2, +5.5] s
  → d = logP(C3) − logP(C4),  s = logP(C3) + logP(C4)
  → z-score against that subject's pooled T0 rest
  → drop pre-cue bins → 11 post-cue bins → 44 features
  → StandardScaler → L2 logistic regression
```

Two ordering constraints that are easy to get wrong: the Laplacian is applied
**before** the Hilbert envelope (we want the envelope of the cleaned signal, not
the difference of two envelopes), and filtering runs on the **continuous**
recording before epoching, so there is no filter edge artifact at every trial
boundary.

### How noise is handled

Each source of noise gets a specific mechanism rather than one generic
denoising step.

| Noise source | Mechanism |
|---|---|
| Mains hum | 60 Hz notch, applied first. Matters most for the high-frequency control band. |
| Volume conduction / spatial smearing | Small Laplacian: C3 minus the mean of FC3, C1, C5, CP3, and the mirror for C4. A broad smeared source hits centre and neighbours about equally and cancels; a focal source at C3 survives. **Zero fitted parameters** — weights are fixed by electrode geometry. |
| Broadband / non-oscillatory activity | Band-pass to mu and beta, kept separate rather than lumped, so the beta rebound is not averaged into the mu ERD. |
| Between-subject magnitude differences | Rest z-scoring (below). Skull thickness, impedance and individual alpha power vary enormously; a raw power value is meaningless until referenced to the person. |
| Common-mode drift shared by both hemispheres | Kept explicitly as `s` and handed to the model as a nuisance covariate rather than discarded. |
| Overfitting to noisy features | L2 with C chosen by inner cross-validation. |

**No trials are rejected for amplitude or artifact.** This is deliberate: an
artifact rejection threshold is a tunable knob that can be turned until the
result improves. Nothing is dropped except for the structural reasons above.

### Baselining and standardisation

Two steps, and they do different jobs.

**Baseline — per subject, from rest.** For each subject, the mean and SD of each
feature are computed over the T0 rest blocks inside their own file (~60
one-second windows per run), and their trials are z-scored against those:

```
x = (x − mean_rest) / sd_rest
```

This is what makes a feature mean *"power relative to this person's resting
level"* — the definition of an ERD — and it puts different people on a common
scale. It uses the recording's EEG and its event *timing*, never the L/R label.

**Standardisation — per feature, across the training set.** A `StandardScaler`
rescales the 44 features so L2 penalises them evenly. Fitted on training folds
only.

**There is no per-trial baseline**, and its absence is a finding rather than an
omission. See §4.

---

## 2. What the model can and cannot do

The headline is **0.699**. That figure describes one specific situation.

### The conditions under which it holds

- Executed (not imagined) left/right fist movements
- The EEGMMIDB paradigm specifically: 64 channels, 160 Hz, 4.1 s movement blocks
  with 4.1 s rest, cues 8.2 s apart
- A **complete EDF file** available at once, including its T0 rest blocks, since
  the baseline is estimated from them
- Trial times supplied by the file's own T1/T2 annotations

### The range, not the best case

| | |
|---|---|
| Mean | 0.699 |
| SD | 0.119 |
| Median | 0.714 |
| **Range** | **0.452 – 0.976** |
| Above chance | 53 / 57 |
| Below chance | **4 / 57** |

The spread is half a point of accuracy wide. Reporting 0.699 alone would hide
that the best subject is at 0.976 and the worst is *below chance*. Run on three
subjects outside the training set entirely, `predict.py` returned **0.357,
0.929 and 0.571** — the same picture from a different angle.

This is consistent with the 15–30% "BCI illiteracy" rate in the literature:
a substantial minority of people do not produce a detectable sensorimotor
rhythm modulation, and no amount of modelling recovers a signal that is not
there.

### Where it falls apart

**It decodes the rebound, not the movement.** Restricting the model to the
movement window alone collapses it:

| Bins used | Accuracy |
|---|---|
| Movement, 0–4.1 s | **0.523** |
| Post-movement, 4.1–5.5 s | **0.673** (12 features) |
| All post-cue bins | 0.699 |

Nearly everything comes from three bins *after* the hand stops. The learned
weights agree independently — 0.65 on the single beta bin at 5.25 s against
under 0.22 everywhere else. The honest description is a **post-movement beta
rebound detector**. Anything that removes or shortens the post-movement window
removes the result.

**It cannot run online.** The rest baseline needs the whole file. A live BCI
seeing one trial at a time could not compute it.

**It is not validated on imagined movement**, which is the BCI-relevant case.

**It degrades on repeat trials.** 0.722 when the hand switches from the previous
trial, 0.646 when it repeats. See §4.

---

## 3. That the result is not an artifact of how it was measured

### How train and test were chosen

**Leave-one-subject-out.** Whole subjects are held out; every subject takes a
turn as the test set; results are averaged. Nothing from a test subject appears
in training.

This is the split the task demands — the deliverable is run on people the model
has never seen. It also neutralises the most obvious leak: if trials from one
person appeared in both train and test, the model could score by recognising the
individual rather than the task.

**Hyperparameter selection is nested.** C is chosen by an inner leave-one-subject-out
*within the training folds only*. The held-out subject is never seen during
tuning. Selecting C against the outer test score would inflate the result.

### What the split assumes, and the one assumption that is not safe

A leave-one-subject-out split assumes trials are exchangeable **within** a
subject. That assumption is false here, and it is the main threat to the result:
trials are temporally adjacent, and consecutive trials use different hands 77.1%
of the time. So a model can score by reading the *previous* trial rather than the
current one. This is treated as a first-class confound in §4 rather than assumed
away.

### Checks that the measurement itself is sound

| Check | Result | What it rules out |
|---|---|---|
| Label permutation, 100 shuffles | null 0.500 ± 0.013, observed 0.699, **p = 0.0099** | The pipeline is not manufacturing accuracy from nothing |
| Trial position within run | **0.497** (best possible 0.546) | No trivial ordering artifact |
| Per-run class balance | exactly **0.500**, every run | No class imbalance to exploit |
| Naive random split vs LOSO | inflation **≈ 0.000** | Confirms the usual identity leak does not apply here |

The permutation null is the important one: shuffling labels and re-running the
entire evaluation returns 0.500 ± 0.013. If the pipeline had a structural leak,
scrambled labels would still score above chance. They do not.

The random-split result is worth explaining because it is *not* what I expected.
Random-split inflation comes from subject identity carrying label information.
Every run here is exactly 50/50 left/right, so knowing *who* a trial came from
says nothing about *which hand*. The dangerous leak in this dataset is
**temporal**, not per-subject.

---

## 4. Appropriate baselines — what else could explain the number

Chance (0.500) is the weakest reference. Four stronger ones:

### Baseline 1 — the empirical null

Shuffling labels within each subject and re-running the full evaluation gives
**0.500 ± 0.013**. Observed 0.699, p = 0.0099. This is the reference chance
*should* be measured against, because it captures any accuracy the pipeline
generates structurally.

### Baseline 2 — the previous trial (the serious one)

The stimulus sequence alternates hands **77.1%** of the time (n = 2223,
binomial p = 1e-151 against independent randomisation; lag-2 is 0.496, so it is
a first-order bias). The previous movement ends 4.1 s before the next cue, and
its beta rebound — contralateral to the *previous* hand — is still present.

So a model can decode the previous trial and bet on alternation, without
decoding the current movement at all.

**The test:** split accuracy by whether the hand switched or repeated. A genuine
current-movement decoder is indifferent; a carryover exploiter is good on
switches and poor on repeats. Attention or fatigue drift hits both equally, so a
gap here is carryover specifically.

| | switch (n=1714) | stay (n=509) | gap |
|---|---|---|---|
| Current model | 0.722 | **0.646** | **+0.075** |

A **+0.075 gap survives and is reported, not corrected.** The 0.646 on repeat
trials is the conservative figure — alternation-betting is actively penalised
there, so what survives is current-trial decoding.

**Why there is no per-trial baseline.** The textbook fix is to subtract each
trial's own pre-cue level. Measured, it makes the confound *worse*:

| Baseline scheme | accuracy | gap |
|---|---|---|
| Pooled rest (used) | 0.699 | **+0.075** |
| Per-trial pre-cue subtraction | 0.700 | **+0.108** |
| No baseline | 0.721 | +0.067 |

The reason is a timing assumption that does not hold: **the beta rebound peaks
2.28 s after movement ends, not the 0.5–1 s usually assumed**, and is still at
64% of peak 3.85 s later. So the (−2,−1) s window sits at **92% of the previous
trial's rebound peak**. Subtracting it over-corrects, and because hands
alternate, the over-correction looks like the current hand. The correction
injects the confound it was meant to remove. Measured in
`figures/carryover_decay.png`.

Pooled rest is used instead because it is one constant per subject and therefore
*cannot* carry trial-varying information about the previous hand.

What actually reduced the confound was removing the model's access to it:

| Setup | stay | gap |
|---|---|---|
| Pre-cue bins fed to the model, window to 4.1 s | 0.448 | +0.286 |
| Pre-cue bins excluded | 0.570 | +0.108 |
| Plus rebound window (current) | **0.646** | **+0.075** |

Note the first row: stay accuracy *below chance* means the model was actively
predicting the wrong hand on repeats — the signature of pure alternation-betting.

### Baseline 3 — muscle activity

Executed movement puts forearm EMG on the scalp. A classifier trained on a
high-frequency band, where cortical rhythm does not live but muscle does, is the
control.

**The first version of this control was broken.** With default filter transition
bands, a 30–50 Hz filter still responds at **0.679 to 26–28 Hz** — high beta. It
was measuring the thing it was supposed to control for.

| Band | Response at 26–28 Hz | Accuracy |
|---|---|---|
| 30–50 Hz, defaults | 0.679 | 0.601 |
| 35–50 Hz, tight | 0.001 | 0.544 |
| 40–50 Hz, tight | 0.000 | **0.537** |

Most of the apparent muscle signal was beta leaking through a soft filter edge.
Both are reported because the difference between them is itself the finding.

**What remains true:** 0.537 is still ~3 SD above the null, so a small
lateralised component does live up there. And the control is weak by
construction — at 160 Hz, Nyquist is 80 and mains is 60, so 40–50 Hz is a
cramped proxy for EMG, which peaks well above it. **A muscle contribution cannot
be excluded.** The clean test is the imagined runs, where no movement occurs.

### Baseline 4 — the bilateral component

Not every large effect is a useful one. Rest-referenced:

| | During movement | At the rebound |
|---|---|---|
| `s` = C3+C4 (both hemispheres) | **−1.07** | −0.06 |
| `d` = C3−C4 (lateralised) | −0.25 | **+0.36** |

The ERD is roughly four times larger than anything else in the trial — and it
classifies at **0.508, chance**, on its own. Both hemispheres desynchronise when
one hand moves, so an ERD says *a* hand moved, not *which*. All the
discriminative information is in `d` (0.706).

This is why the model weights the rebound: not because it is bigger, but because
it is cleaner. At 5.25 s the bilateral component is back at resting level while
the lateralised component peaks.

### Summary of references

| Reference | Accuracy |
|---|---|
| Chance | 0.500 |
| Permutation null | 0.500 ± 0.013 |
| Trial position only | 0.497 |
| Bilateral component (`s`) only | 0.508 |
| Movement window only | 0.523 |
| Muscle band, clean | 0.537 |
| **Repeat trials (carryover-immune)** | **0.646** |
| **Full model** | **0.699** |

---

## 5. How much of this is about the person rather than the task

### The individual is strongly present in the data

A probe trained to identify **which subject** a trial came from, using
un-normalised features, reaches **0.573 against a chance of 0.018** — 33× chance.
Individual identity is one of the strongest signals in this dataset.

### It is removed, and the split makes it useless

- **Leave-one-subject-out** means test subjects are strangers, so identity cannot
  be used even if present.
- **Rest z-scoring per subject** is what strips it out of the features.
- **The Laplacian has zero fitted parameters**, so nothing is calibrated to an
  individual.
- **C3/C4 are fixed anatomically**, not selected per subject.

### The quantitative check

| | Accuracy |
|---|---|
| Within-subject (leave-one-run-out) | 0.660 |
| Cross-subject (LOSO) | **0.699** |
| Gap | **−0.039** |

**Cross-subject is higher than within-subject.** If the model were fitting
individual quirks, training on a person should beat training on strangers. It
does not. (Within-subject folds also have far less training data — roughly 28
trials — which contributes to the gap.)

### But the individual still dominates the variance

| | |
|---|---|
| Per-subject range | **0.452 – 0.976** |
| SD across subjects | 0.119 |
| Below chance | 4 / 57 |

The model generalises *on average*, but which person you run it on matters far
more than anything else measured here. A half-point spread across individuals
against a 0.199 margin over chance means **individual variation is larger than
the effect itself**. Any deployment claim has to be conditioned on the person,
not on the mean.

---

## 6. Whether the model has learned or memorised

### Capacity is deliberately small

| | |
|---|---|
| Features | 44 |
| Training trials per fold | ~2352 |
| Ratio | **~53:1** |
| Regularisation | L2, C = 0.01 (heavy) |
| Fitted spatial parameters | **0** |
| Classifier | Linear |

There is very little for a 44-parameter linear model with strong L2 to memorise
across 2352 trials. The low-capacity stance is consistent across every stage:
whole-subject holdout, parameter-free Laplacian, two fixed electrodes, linear
model, heavy L2.

### The evidence it did not memorise individuals

1. **Permutation null is 0.500 ± 0.013.** Scrambled labels score at chance. A
   model memorising the training set would still be at chance out-of-fold, but a
   *pipeline* leak would show up here. It does not.
2. **Random split vs LOSO inflation ≈ 0.000.** The usual signature of identity
   leakage is absent, and §3 explains why: every run is 50/50, so identity
   carries no label information.
3. **Within-subject < cross-subject.** The wrong direction for memorisation.
4. **53/57 subjects above chance.** The result is distributed, not driven by a
   handful of subjects the model happened to fit.

### Where it *does* overfit — the honest answer

**Not to individuals. To the paradigm.**

The model has learned two things specific to this experimental design:

1. **The post-movement beta rebound**, at a latency determined by this
   paradigm's 4.1 s movement block. Change the block length and the informative
   bins move. Restrict to the movement window and accuracy falls to 0.523.
2. **Residual alternation structure.** The +0.075 switch/stay gap means part of
   the accuracy still comes from the 77% alternation bias in the BCI2000
   sequence — a property of the *stimulus program*, not of the brain. On a
   paradigm with randomised hand order, that component would disappear.

So the number would not transfer to a differently-timed experiment, even with
the same subjects. That is a real limitation of the result and is not detectable
by any within-dataset control — which is precisely why it is stated here rather
than measured.

### One thing that was tuned on the evaluation metric

The number of electrode pairs was originally chosen by comparing cross-subject
scores — selection on the evaluation metric, which makes the reported figure
optimistic. It is now **fixed a priori at C3/C4**, the textbook anatomical
choice for hand motor cortex, at a cost of roughly 0.02 accuracy. Stated openly
rather than quietly kept.

One related detail: the shipped model in `models/model.joblib` has its C chosen
by leave-one-subject-out over all 57 subjects, which is why its card reports
0.702. The **defended figure is 0.699**, from the nested procedure in
`run_experiments.py` where C is selected inside the training folds only. The
nested number is the unbiased one.

---

## Summary

The defended figure is **0.699 cross-subject**, with **0.646** as the
carryover-immune floor, against a permutation null of 0.500 ± 0.013
(p = 0.0099).

It is a post-movement beta rebound detector, not a movement decoder. It
generalises to unseen people but varies more across individuals (0.452–0.976)
than its margin over chance. A muscle contribution cannot be excluded, and a
residual +0.075 of the result still comes from the stimulus sequence's
alternation bias rather than from the brain.
