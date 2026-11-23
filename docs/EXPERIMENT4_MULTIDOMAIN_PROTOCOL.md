# Experiment 4 — Multi-Domain Generalization

## Research question

Can joint supervised training on Market-1501 and CUHK03-NP improve
generalization to a third untouched person-ReID domain while preserving
RAC-ReID's Efficient-vs-Full behavior?

## Source domains

Only these datasets influence training or checkpoint selection:

1. Market-1501
2. CUHK03-NP Detected

Identities are namespaced by dataset.

## Source development partitions

For each source domain:

- gradient training: `PID % 5 in {0,1,2}`
- calibration: `PID % 5 == 3`
- held-out validation: `PID % 5 == 4`

## Model

OSNet x1.0 initialized from:

`osnet_x1_0_msmt17_combineall`

Experiment-1 and Experiment-3 checkpoints are not used as initialization.

Paths:

- Efficient: 160x80
- Full: 256x128

`conv1` and `conv2` remain frozen.

## Training

Fixed before training:

- 8 epochs
- batch size 16
- deterministic combined-source shuffle
- AdamW
- backbone LR 5e-5
- classifier LR 5e-4
- weight decay 5e-4
- seed 42
- CPU
- four Torch threads

Objective:

`L = 1.00*L_low + 0.25*L_full + 0.50*L_rel + 0.25*L_cross + 0.50*L_cls`

The loss definitions are unchanged from Experiment 3.

## Epoch selection

SenseReID cannot influence selection.

Each epoch is evaluated separately on the Market and CUHK03 calibration
partitions.

Selection order:

1. highest macro-average Efficient Rank-1;
2. highest macro-average Full Rank-1;
3. highest macro-average low/full embedding cosine.

Held-out identities cannot influence checkpoint selection.

## Unseen target — SenseReID

SenseReID is evaluation-only and contains no training split.

Downloaded archive SHA-256:

`10f02fda87794853c174f2de9bff7343edcd12421483273329780e7150cfaa57`

Observed archive population:

- query: 1,040 images / 521 identities
- gallery: 3,388 images / 1,718 identities

Every observed query identity is represented in the gallery.

### Published metadata discrepancy

The original Spindle Net paper and the Torchreid documentation state:

- query: 522 identities / 1,040 images
- gallery: 1,717 identities / 3,388 images

The downloaded archive itself contains:

- query: 521 identities / 1,040 images
- gallery: 1,718 identities / 3,388 images

Later lifelong/generalization ReID literature also reports the 521-query /
1,718-gallery identity interpretation.

No images or identity labels are changed to force either published summary.

Experiment 4 uses the downloaded archive exactly as observed and records its
cryptographic manifests.

## Target isolation

Before checkpoint freeze, SenseReID is used only for:

- archive integrity verification;
- filename parsing;
- image and identity counts;
- cryptographic manifest generation.

No SenseReID image is passed through the model.

SenseReID cannot influence:

- gradient updates;
- epoch selection;
- hyperparameters;
- architecture.

## Final evaluation

After the Experiment-4 checkpoint is selected and frozen:

- Efficient query: 160x80
- Full query: 256x128
- Gallery: 256x128

Metrics:

- mAP
- Rank-1
- Rank-5
- Rank-10
- mINP

The target is evaluated once.

No post-target tuning is allowed.

## Production

Experiment 4 is research-only.

The production RAC-ReID model remains unchanged.
