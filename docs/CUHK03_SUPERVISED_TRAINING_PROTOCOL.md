# Experiment 3 — CUHK03-NP Supervised Multi-Resolution Training

## Research question

Can CUHK03-specific supervised training improve absolute CUHK03 person
re-identification performance while preserving the low-resolution robustness
observed in Experiments 1 and 2?

## Scientific status

CUHK03's official query/gallery benchmark has already been observed in
Experiment 2.

Therefore, Experiment 3 does **not** describe the CUHK03 benchmark as a
pristine unseen test.

Instead:

- the entire Experiment-3 training procedure is fixed before training;
- official CUHK03 query/gallery images are not used for gradient updates;
- official CUHK03 results are not used for epoch selection;
- official CUHK03 results are not used for hyperparameter selection;
- the post-training CUHK03 evaluation is a standardized benchmark;
- no post-benchmark tuning is allowed.

## Training data

Primary training source:

**CUHK03-NP Detected `bounding_box_train`**

Population:

- 7,365 images
- 767 identities

The Labeled variant is not mixed into gradient training.

## Identity-disjoint development split

The 767 CUHK03 training identities are split deterministically:

- Gradient training: `PID % 5 in {0,1,2}`
- Calibration: `PID % 5 == 3`
- Held-out validation: `PID % 5 == 4`

No identity appears in more than one development partition.

## Initialization

OSNet x1.0 initialized from:

`osnet_x1_0_msmt17_combineall`

This is the same original pretrained base used for the Market-1501
multi-resolution experiment.

The frozen Market-trained epoch-8 checkpoint is **not** used to initialize
Experiment 3. This isolates the effect of changing the supervised training
dataset.

## Architecture

The same OSNet x1.0 backbone processes both resolutions:

- Efficient: 160x80
- Full: 256x128

`conv1` and `conv2` remain frozen.

## Training

Fixed before training:

- 8 epochs
- batch size 16
- AdamW
- backbone LR: 5e-5
- classifier LR: 5e-4
- weight decay: 5e-4
- random seed: 42
- CPU
- four Torch threads

Joint objective:

`1.00 * low-resolution teacher cosine`

`+ 0.25 * full-resolution teacher cosine`

`+ 0.50 * relational loss`

`+ 0.25 * cross-resolution consistency`

`+ 0.50 * identity classification`

Teacher/reference embeddings come from the original pretrained
full-resolution OSNet base.

## Epoch selection

Only the calibration identities may select the epoch.

Selection rule:

1. highest calibration 160x80 Rank-1;
2. if tied, highest calibration 256x128 Rank-1;
3. if still tied, highest low/full embedding cosine.

Held-out validation is evaluated only after epoch selection and cannot change
the selected model.

## Post-training benchmark

After the checkpoint is frozen:

Primary:

**CUHK03-NP Detected**

Secondary:

**CUHK03-NP Labeled**

Each uses:

- Efficient query: 160x80
- Full query: 256x128
- Gallery: 256x128

Metrics:

- mAP
- Rank-1
- Rank-5
- Rank-10
- mINP

No post-benchmark training, threshold tuning, or architecture changes are
allowed within Experiment 3.

## Production

Experiment 3 is research-only.

The current production RAC-ReID checkpoint is not replaced by this experiment
unless a later, explicitly separate deployment decision is made.

## Experiment-3 implementation-frozen loss definitions

The original Experiment-1 repository preserved the objective names and
coefficients but did not preserve the original optimizer loop as a reusable
trainer script.

Therefore Experiment 3 freezes the exact mathematical implementation below
before the first CUHK03 gradient update.

All embeddings are L2-normalized for cosine-based terms.

### Low-resolution teacher cosine

`L_low = 1 - mean(cos(student_160x80, teacher_256x128))`

### Full-resolution teacher cosine

`L_full = 1 - mean(cos(student_256x128, teacher_256x128))`

### Relational similarity

For each minibatch, form pairwise cosine-similarity matrices for the
low-resolution student, full-resolution student, and frozen teacher.

`L_rel = 0.5*MSE(S_low, S_teacher) + 0.5*MSE(S_full, S_teacher)`

### Cross-resolution consistency

`L_cross = 1 - mean(cos(student_160x80, student_256x128))`

### Identity classification

A separate linear classifier is trained on the 512-D embeddings of the
gradient-training identities.

`L_cls = 0.5*CE(classifier(low_raw), identity) + 0.5*CE(classifier(full_raw), identity)`

### Total objective

`L = 1.00*L_low + 0.25*L_full + 0.50*L_rel + 0.25*L_cross + 0.50*L_cls`

These formulas are committed to Git before Experiment-3 training starts.
They are the frozen Experiment-3 implementation and are not claimed to be a
byte-for-byte recovery of the historical Step 29F-B optimizer loop.
