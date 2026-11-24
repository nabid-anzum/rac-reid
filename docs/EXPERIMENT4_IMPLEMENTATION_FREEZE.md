# Experiment 4 — Implementation Freeze

This document freezes implementation details before the first Experiment-4
gradient update.

## Training implementation

Trainer:

`scripts/train_multidomain_market_cuhk_sensereid.py`

Experiment 4 reuses the Experiment-3 mathematical objective:

`L = 1.00*L_low + 0.25*L_full + 0.50*L_rel + 0.25*L_cross + 0.50*L_cls`

The classifier identity space is dataset-scoped:

- Market person IDs are Market classes.
- CUHK03 person IDs are CUHK classes.
- Numerically identical person IDs across datasets are never treated as the
  same identity.

The combined gradient population is deterministically shuffled once per epoch.

## Source-domain selection

Each epoch is independently evaluated on:

- Market-1501 calibration identities
- CUHK03 calibration identities

Selection is exactly the preregistered macro-average rule.

Source held-out identities are not evaluated until the epoch has been selected.

## SenseReID implementation clarification

The one permitted SenseReID target evaluation is a single frozen evaluation
invocation containing three models that are all fixed before target inference:

1. Experiment-1 Market-trained checkpoint
2. Experiment-3 CUHK03-trained checkpoint
3. Experiment-4 joint Market+CUHK03 checkpoint

SenseReID cannot affect any of these checkpoints.

This three-model target run allows the multi-domain model to be compared
directly with both single-source trained models.

## Ranking protocol

For every model:

- gallery embeddings: 256x128
- Efficient queries: 160x80
- Full queries: 256x128
- L2-normalized 512-D embeddings
- cosine ranking
- same-person/same-camera gallery entries removed for each query
- no reranking

Metrics:

- mAP
- Rank-1
- Rank-5
- Rank-10
- mINP

## Target isolation

No SenseReID image may pass through a model before:

1. Experiment-4 epoch selection is complete;
2. source held-out validation is complete;
3. the Experiment-4 checkpoint is written and hashed;
4. Experiment-4 training evidence is committed to GitHub.

No target-driven tuning is permitted afterward.

## Production

The production RAC-ReID encoder and epoch-8 production checkpoint are not
modified by Experiment 4.

## Pre-freeze validation correction

The first evaluator synthetic self-test used too few gallery candidates for
Rank-5/Rank-10 after same-person/same-camera removal.

Before the implementation was committed, only the synthetic self-test data
was enlarged. The actual ranking implementation and target evaluation
protocol were not changed.

The original broad textual safety check for the word `SenseReID` was also
replaced before implementation freeze by a semantic check that prohibits
actual target dataset paths such as `data/datasets/sensereid`,
`test_probe`, `test_gallery`, and `SenseReID.zip`.

Metadata/status strings mentioning SenseReID remain allowed.
