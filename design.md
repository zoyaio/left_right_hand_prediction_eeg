# L vs R Motor Imagery Decoder — Design

## 0. The one idea everything hangs off

You asked how to encode the push–pull relationship (ERD on the active side, ERS on the other). You do not need an interaction term, a conjunction detector, or two separate detectors whose outputs you compare. Work in **log power** and the push–pull relationship becomes a single subtraction.

ERD is a *multiplicative* reduction relative to a baseline, so log power is the natural unit:

```
erd_C3(t) = log P_C3(t) - log B_C3
erd_C4(t) = log P_C4(t) - log B_C4
```

Now define the two coordinates that matter:

```
d(t) = erd_C3(t) - erd_C4(t)        # ANTISYMMETRIC  -> lateralized signal
s(t) = erd_C3(t) + erd_C4(t)        # SYMMETRIC      -> common-mode nuisance
```

Substituting, the baselines collapse into a constant:

```
d(t) = [log P_C3(t) - log P_C4(t)] - [log B_C3 - log B_C4]
```

So `d` is exactly the lateralization index, baseline-corrected — and it is *already* the push–pull contrast. If the subject imagines LEFT: C4 shows ERD (goes down) and C3 shows ERS (goes up). Both effects move `d` in the same direction and **add**. You get the push–pull encoding for free, in one scalar, with no interaction term.

This is a change of basis, not a change of model. `(erd_C3, erd_C4)` and `(d, s)` span the same space, and a linear model can reach the same solutions in either. **The basis matters because of the regularizer.** L2 in the `(d, s)` basis shrinks the lateralized direction and the common-mode direction independently — which is the prior you actually hold. L2 in the raw `(C3, C4)` basis shrinks along a direction that mixes them. Choosing the basis is choosing the prior. This is your "one design decision" for the video.

**The corollary is the artifact strategy.** `s` is where the junk lives: global alertness, drowsiness, neck/jaw EMG, electrode impedance drift, overall mu amplitude, subject-level power offsets. Almost all of it is common-mode across a symmetric electrode pair, so it cancels in `d`. The difference feature is simultaneously your signal extractor and your primary artifact rejector. Keep `s` in the feature vector anyway — as an explicit nuisance covariate the model can regress against, not as signal.

Your note "get rid of whatever is the same between left hand and right hand" is literally `s`. Your Cz / foot-electrode idea is the same instinct applied to a different axis (medial vs lateral instead of left vs right); it is a good *control analysis*, not a good feature.

---

## 1. Pipeline

**Stage 0 — Load and fix the data.**

- `mne.datasets.eegbci.load_data(subject, runs)`, then `mne.datasets.eegbci.standardize(raw)`. The raw EDF channel names are `'C3..'`, `'Fc3.'`, `'Cp3.'` with trailing dots and odd casing; nothing matches a montage until you standardize. Then `raw.set_montage('standard_1005')`.
- **Verify T1/T2 yourself, per run.** Expected: in runs 3,4,7,8,11,12 T1 = left fist, T2 = right fist. In runs 5,6,9,10,13,14 T1 = both fists, T2 = both feet. The spec tells you to check this; checking it is free marks.
- **Screen subjects before you use them.** Assert `raw.info['sfreq'] == 160` and count annotations per run. Several subjects are widely reported as defective (128 Hz sampling or damaged annotations) — S088, S089, S092, S100 are the usual suspects. Do not take that list from me: write the screening loop, print what fails, and report *your* exclusion list and the criterion you used. "I ran a data integrity check and dropped N subjects for reason X" is exactly the "what you noticed that you were not told to look for" deliverable.

**Stage 1 — Temporal filtering.**

- Notch 60 Hz (US line). Sampling is 160 Hz, Nyquist 80 Hz.
- Band-pass 8–30 Hz for the main path. Keep a separate 30–50 Hz path for the EMG control (see §4).

**Stage 2 — Spatial filtering. Do it. Use the small Laplacian.**

You asked whether to jump into spatial filtering. Yes, but the answer is *Laplacian*, not CSP, and the distinction matters for this specific challenge.

```
C3_lap = C3 - mean(FC3, C1, C5, CP3)
C4_lap = C4 - mean(FC4, C2, C6, CP4)
```

- It is **fixed and subject-independent**. Nothing is fit, so nothing can leak, and it transfers to an unseen subject for free. The whole challenge is graded on generalization to people you never trained on; a zero-parameter preprocessing step is worth a lot there.
- It is a **spatial high-pass**. Broad, far-field sources — blinks, neck EMG, reference drift, the global mu rhythm — hit C3 and its neighbours nearly equally and subtract out. Focal sensorimotor activity survives.
- It makes the signal **reference-independent**, which is the single largest source of arbitrary variance in a raw C3/C4 pipeline.
- It is about five lines of code.

**CSP is a comparison arm, not the main path.** CSP is fit per-subject and does not transfer; a CSP fit across subjects is usually worse than a Laplacian. Run CSP + LDA *within-subject* to measure the ceiling you are giving up, and report that gap. Two numbers beat one.

**Stage 3 — Band power over time, via Hilbert.**

Bandpass → `hilbert` → `abs()**2` → `log` → smooth. This gives a continuous log-power envelope you can bin at any resolution, which is cleaner and more explainable than Welch/multitaper windows. Do this for mu (8–13) and beta (13–30) separately.

**Stage 4 — Epoch and bin.** See §2.

**Stage 5 — Normalize, then model.** See §3 and §5.

---

## 2. Time windows and "do I need history?"

You asked whether you need history and guessed no. **You need within-trial time structure, and here is the concrete reason it is not optional:**

The sign of `d` flips over the course of a trial.

- **0.5–2.5 s post-cue:** contralateral ERD dominates. `d` has one sign.
- **~3–5 s post-cue:** the post-movement beta rebound (ERS) is *also* contralateral-dominant. `d` flips toward the opposite sign.

If you average band power over the whole 0–4 s epoch, the ERD phase and the rebound phase partially **cancel each other**. You would be destroying your own signal and concluding the task is harder than it is. This alone justifies time bins.

**Concrete spec.** Epoch `[-2.0, +4.0]` s relative to cue. The inter-trial interval is ~4.1–4.2 s, so anything past +4.0 s is contaminated by the next trial; if you want the full rebound you must take the contamination and say so.

Do **not** hand-pick ERD/ERS windows. That is a hidden hyperparameter search, and if you tune it against your test numbers it is leakage. Instead use uniform bins and let the regularized model tell you where the information is:

```
bins: 0.5 s wide, 50% overlap, spanning [-2.0, +4.0]  ->  ~23 bins
```

Then **plot the learned L2 weights as a function of time bin**. If the profile shows a negative lobe at 0.5–2.5 s and a sign reversal around 3–4 s, your model has recovered the known ERD/ERS time course from data. That figure is worth more than an accuracy number — it is direct evidence the model learned physiology rather than an artifact. Put it in the video.

**The other kind of history — carryover between trials — you should *measure*, not feed in.** Your "trace effects of the most recent past fire" intuition is real: the beta rebound lasts up to ~5 s and the ITI is ~4.2 s, so the previous trial's rebound bleeds into the current pre-cue window.

Do not add `previous_trial_label` as a feature. It is unavailable at test time and it invites the model to learn the experiment's block structure instead of the brain. Instead:

- Include the pre-cue bins `[-2.0, 0)` in the feature vector, and
- Run the **pre-cue-only control** (§7). This is one of the most important experiments in the whole project.

Do **not** build a recurrent model over the trial sequence. Same reason.

---

## 3. Feature vector

Per symmetric electrode pair, per band, per time bin, emit two numbers: `d` and `s`.

**v1 (start here):**

```
pairs  = [(C3, C4)]                        # Laplacian-filtered
bands  = [mu 8-13, beta 13-30]
bins   = ~12  (1.0 s wide, 50% overlap, [-2, +4])
feats  = 1 x 2 x 12 x 2  =  48
```

**v1.5 (the principled way to add channels):** apply the *same* antisymmetric construction to more symmetric pairs rather than dumping in raw channels.

```
pairs = [(C3,C4), (C1,C2), (C5,C6), (FC3,FC4), (CP3,CP4)]
feats = 5 x 2 x 12 x 2 = 240
```

At ~45 trials/subject this needs real L2 and more subjects. Gate v1.5 on v1 working.

---

## 4. Artifacts — four concrete moves

Your notes said your artifact ideas "weren't really helpful." Here are four that are, all downstream of the symmetry idea.

**(a) The Laplacian.** §1, Stage 2. This is the main one. A spatial high-pass removes broad artifacts by construction.

**(b) The `d` feature itself.** Common-mode junk cancels in a difference. This is not a bonus — it is half the reason to use the difference.

**(c) The EMG-band control experiment.** This is the highest-value item in this section and it is a *result*, not a preprocessing step.

Build the entire pipeline a second time on a 30–50 Hz band instead of mu/beta. Muscle activity dominates high frequencies; the mu/beta rhythm does not live there. Then:

> **If a 30–50 Hz classifier separates L from R, your mu/beta result may be muscle, not motor cortex.**

Report that number next to your headline number. This is precisely the "what else could explain this figure" the spec demands, and almost nobody will run it.

*Be honest about the limitation:* at 160 Hz sampling, Nyquist is 80 Hz and 60 Hz is line noise, so 30–50 Hz is a cramped and weak EMG proxy. Real surface EMG peaks well above this. State that explicitly — it makes the control weaker, but stating it makes you credible.

**(d) Executed vs imagined as an artifact probe.** Your own idea, sharpened. EMG contamination should be far larger in the executed runs than the imagined ones. So:

- Run control (c) on executed and on imagined separately. If the 30–50 Hz classifier works on executed and dies on imagined, you have *localized* the contamination and shown your imagined result is cleaner.
- Train on executed, test on imagined. If it transfers, the model learned motor-command structure common to both. If it does not, it may have learned execution-specific muscle artifact.

**On ICA: consider it and reject it, on the record.** With a fixed Laplacian and two pairs of interest, ICA adds a per-subject, hard-to-validate, hard-to-automate step that cannot be fit without touching the test recording. Mention it as your "real alternative I considered" for the video.

Also add plain amplitude-based trial rejection (peak-to-peak threshold on the Laplacian channels), with the threshold fit **on training folds only**.

---

## 5. Baseline personalization — three tiers, and your worry is unfounded

You marked this v2. Mostly right, but the tiers are not equally expensive and one of them is nearly free.

**Tier 0 (free, already done).** `d` cancels any per-subject baseline offset that is *symmetric*. What it does not cancel is a per-subject **asymmetry** — skull thickness, electrode impedance mismatch, handedness. That residual is the actual problem.

**Tier 1 (cheap, do it in v1): per-recording z-scoring of `d` across all trials in that recording.** This removes the per-subject asymmetry bias and will likely be your largest single cross-subject gain.

This is transductive but **label-free**, and it is legal here because the deliverable is "raw EDF path → labels", so at test time you hold the entire recording. **You must declare this assumption in writing.** Two caveats to state yourself before anyone asks:

- If you ever use labels in the normalization, that is leakage. Do not.
- It breaks for a true online single-trial BCI, where you do not have the future. This is a legitimate candidate for your "weakest point" deliverable.

**Tier 2 (v2): explicit rest baseline.** ERD is *defined* relative to rest, so estimate `B_C3, B_C4` from actual rest — the T0 segments inside the same recording, or runs 1–2.

**Your stated worry here is wrong, and that is good news.** You wrote that for a new test subject "we only have the one sample to base the entire baseline off of." Not true: the test EDF contains many T0 rest periods. A per-subject rest baseline is fully available at test time, from unlabeled data, with no leakage. Tier 2 is cheaper than you thought.

---

## 6. Model ladder

**v0 — sanity floor.** Logistic regression on 2 features: mu `d` and beta `d`, averaged over 0.5–2.5 s. Cross-subject. If this does not clear chance on executed runs, something upstream is broken and nothing below will save you. Fix it before continuing.

**v1 — the real model.** L2 logistic regression on the §3 feature vector, per-recording z-scored, Laplacian-filtered. Cross-subject, leave-one-subject-out.

Yes, logistic regression is the right v1, and here is the argument to make rather than apologizing for it: **the canonical pipeline is bandpass → CSP → log-variance → LDA. Yours is bandpass → fixed spatial filter → log band power → logistic regression.** It is the same pipeline with the spatial filters *frozen to anatomy* instead of learned from data. Freezing them is a regularization choice, and it buys exactly the cross-subject robustness this challenge is grading. That is a positive justification, not an excuse.

**v2 — comparison arms, to bound what you gave up.**

- CSP + LDA, **within-subject** (leave-one-run-out). Your within-subject ceiling.
- Riemannian tangent space: covariance → tangent space → LR (`pyriemann`, ~10 lines). This is the strongest simple cross-subject baseline in the field. If it beats you by a lot, say so.

**v3 — not a deep net, and say why.** ~45 trials/subject and a generalize-to-new-people constraint. Braindecode's EEGNet/ShallowConvNet would be the thing to try with more subjects and time; name it as future work.

---

## 7. Evaluation — this is what you are actually graded on

The spec is explicit that the evaluation, not the accuracy, is the assignment. Budget half your remaining time here.

**Splits.**

- **Leave-one-subject-out is the headline.** Report the full per-subject distribution (strip plot), never just the mean. Expect enormous spread.
- **Within-subject, leave-one-run-out**, for contrast. Will be higher. **The LOSO-vs-within gap is your quantitative answer to "how much of this is the person rather than the task."**
- **Demonstrate the inflation deliberately.** Also run a naive random split of epochs across the whole pooled dataset, where epochs from the same run land in both train and test. Report that inflated number *next to* the LOSO number, labeled as the wrong way to do it. Showing you know why it is wrong is worth more than never running it.

**Controls — run all of these:**

1. **Pre-cue-only classifier.** Train and test using only the `[-2, 0)` s bins. Should be at chance. If it is above chance you have temporal leakage — beta-rebound carryover or block structure — and your main number is contaminated. **Run this first; it can invalidate everything else.**
2. **EMG-band classifier** (§4c).
3. **Permutation test.** Shuffle labels *within subject*, rerun the whole pipeline end to end, ~100 times. This gives an empirical null that catches pipeline leakage the analytic 50% line cannot.
4. **Subject-identity probe.** Train a classifier to predict *subject ID* from your features. It will work well. That is the concrete measurement of how much of your representation is person rather than task.
5. **Executed → imagined transfer** (§4d).

**Expect and report a bimodal per-subject distribution.** Your notes already flagged BCI illiteracy at 15–30%. If your per-subject LOSO accuracies come out bimodal — a cluster near chance and a cluster well above — that is a real finding, it matches the literature, and it is a strong candidate for the "something we did not ask about" deliverable. Do not report a mean that hides it.

---

## 8. Priority order if time is short

1. Stage 0 data loading + T1/T2 verification + subject screening
2. v0 two-feature sanity check on executed runs
3. Laplacian + Hilbert + time bins + `d`/`s` + per-recording z-score (v1)
4. LOSO with per-subject distribution plot
5. **Pre-cue control** and **permutation test**
6. EMG-band control
7. Within-subject comparison + the LOSO-vs-within gap
8. Executed → imagined transfer
9. Riemannian / CSP comparison arms
10. v1.5 multi-pair features

Cut from the bottom. An honest v1 with items 1–6 beats a v1.5 with no controls, and the spec says so in as many words.

---

# POSTSCRIPT — what survived contact with the data

This document was written before any code ran. Most of it held up. One thing did
not, and it changed the project. Recording it here rather than quietly editing
the text above, since the gap between the two is the interesting part.

**What held.** The `(d, s)` symmetry decomposition works and `s` earns its place
as a nuisance covariate (its learned weights exceed `d`'s in beta — suppressor
behaviour, exactly the intended role). The Laplacian was the right spatial
filter. Time bins were load-bearing. The pre-cue control was correctly
identified as the one that could invalidate everything — and it did.

**What was wrong.** Section 0 claims `d` is "baseline-corrected" because the
baselines collapse into a constant. That is true only for a *session-level*
baseline. It does not remove a **per-trial** offset, and this dataset has a large
one: consecutive trials use different hands 76% of the time, and the previous
trial's beta rebound is still present when the next cue arrives. So `d` carries
the previous trial's signature, and the alternation bias turns that into a
predictor of the current label.

The cost: an apparent 0.656 cross-subject accuracy that was mostly artifact. The
fix was trial-wise baseline subtraction — ERD as a *change* from each trial's own
pre-cue level. That is the textbook definition, and the design above reasoned
past it because the algebra of the session-level baseline looked like it had
already solved the problem.

**What I did not anticipate at all.** That the decisive diagnostic would be
splitting accuracy by whether the hand switched or repeated between trials. A
carryover exploiter scores above chance on switches and *below* chance on
repeats; a genuine decoder performs alike on both. That test is not in the plan
above and is now the most informative control in the repository.

**Also revised.** Section 7 predicted the naive random split would visibly
inflate the result. It does not (−0.002), because every subject is ~50/50
left/right, so subject identity carries no label information. The prediction was
right about the mechanism and wrong about whether it applies here.

See README.md for the full evidence chain and the numbers.
