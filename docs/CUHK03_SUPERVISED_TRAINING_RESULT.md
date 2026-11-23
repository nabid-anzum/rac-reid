# Experiment 3 — CUHK03-NP Supervised Training Result

## Purpose

Experiment 2 showed strong cross-dataset domain shift: the frozen
Market-trained model retained its Efficient-vs-Full relationship but had low
absolute CUHK03 retrieval accuracy.

Experiment 3 asks whether CUHK03-specific supervised training can improve the
identity representation while preserving multi-resolution behavior.

---

## Scientific ordering

The Experiment-3 protocol was committed before training:

`78971c77998e7765fbb6fc0660a7bad6e85aa13a`

The exact Experiment-3 trainer and mathematical loss implementation were then
committed before the first gradient update:

`fd481e891293235df8b4482e4397d66a12711036`

Trainer:

`scripts/train_cuhk03_supervised_multires.py`

Trainer SHA-256:

`5e07b4574cd256042bd0401014b7c1e62c43b4d982d3814b98154f465b5e65fa`

---

## Data

Training source:

**CUHK03-NP Detected `bounding_box_train`**

Total available training population:

- 7,365 images
- 767 identities

Identity-disjoint development partitions:

- Gradient training: 452 identities / 4,352 images
- Calibration: 157 identities / 1,515 images
- Held-out validation: 158 identities / 1,498 images

Official CUHK03 query/gallery images were not used during gradient training,
epoch selection, or held-out development validation.

---

## Model

Architecture:

**OSNet x1.0**

Initialization:

`osnet_x1_0_msmt17_combineall`

The Market-trained Experiment-1 checkpoint was not used as Experiment-3
initialization.

Resolution paths:

- Efficient: 160x80
- Full: 256x128

Frozen early modules:

- `conv1`
- `conv2`

---

## Pre-training development baseline

| Split | Efficient R1 | Full R1 |
|---|---:|---:|
| Calibration | 28.713% | 42.376% |
| Held-out | 29.506% | 44.860% |

---

## Epoch history

| Epoch | Training loss | Efficient R1 | Full R1 | Agreement | Low/Full cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.59900 | 42.838% | 49.835% | 93.861% | 0.8350 |
| 2 | 0.85980 | 48.713% | 54.059% | 96.502% | 0.8463 |
| 3 | 0.50353 | 49.901% | 54.323% | 97.030% | 0.8495 | **selected**
| 4 | 0.40353 | 48.713% | 50.627% | 97.294% | 0.8527 |
| 5 | 0.35520 | 48.251% | 52.079% | 97.690% | 0.8572 |
| 6 | 0.32776 | 47.261% | 49.043% | 97.756% | 0.8579 |
| 7 | 0.30648 | 45.875% | 47.129% | 97.492% | 0.8574 |
| 8 | 0.29006 | 44.092% | 45.611% | 97.690% | 0.8620 |

Epoch **3** was selected according to the
preregistered calibration rule.

The continued fall in optimization loss after epoch 3 did not correspond to
better identity retrieval. Calibration Rank-1 began to deteriorate, making
epoch 3 the correct frozen checkpoint under the preregistered rule.

---

## Selected calibration result

Efficient 160x80 Rank-1:

**49.901%**

Full 256x128 Rank-1:

**54.323%**

Efficient gain versus pretrained calibration baseline:

**+21.188%**

Full gain versus pretrained calibration baseline:

**+11.947%**

Low/full identity agreement:

**97.030%**

Low/full embedding cosine:

**0.8495**

---

## Identity-disjoint held-out result

The held-out identities participated in neither gradient updates nor epoch
selection.

Efficient 160x80 Rank-1:

**52.870%**

Full 256x128 Rank-1:

**56.075%**

Efficient gain versus pretrained held-out baseline:

**+23.364%**

Full gain versus pretrained held-out baseline:

**+11.215%**

Low/full identity agreement:

**96.929%**

Low/full embedding cosine:

**0.8489**

---

## Frozen checkpoint

Selected epoch:

**3**

Runtime checkpoint:

`/root/rac-reid/.runtime/models/cuhk03-supervised/osnet_x1_0_cuhk03_multires.pt`

SHA-256:

`f879574a3cf21f9b106f51fbd2bef692e6044761a3bb17280c5da3a37fd1a399`

The binary checkpoint is intentionally not committed to GitHub. Its exact
SHA-256 is published for integrity verification.

---

## Loss implementation

Experiment 3 uses:

`L = 1.00*L_low + 0.25*L_full + 0.50*L_rel + 0.25*L_cross + 0.50*L_cls`

Definitions are frozen in:

`docs/CUHK03_SUPERVISED_TRAINING_PROTOCOL.md`

and implemented in:

`scripts/train_cuhk03_supervised_multires.py`

---

## Current interpretation

CUHK03-specific supervised training substantially improved development
retrieval performance over the original pretrained OSNet representation.

The calibration trajectory also shows overfitting after epoch 3: optimization
loss continued to decrease while retrieval Rank-1 declined.

This is why checkpoint selection was based on preregistered calibration
retrieval performance rather than final-epoch training loss.

---

## Official benchmark status

The standardized CUHK03-NP Detected/Labeled query-gallery benchmark has
**not yet been run with the Experiment-3 checkpoint**.

Because Experiment 2 already observed CUHK03 query/gallery, the forthcoming
Experiment-3 benchmark is explicitly described as a standardized post-training
benchmark rather than a pristine unseen test.

No benchmark-based tuning will be allowed afterward.

---

## Machine-readable evidence

Training result:

`results/cuhk03_supervised_training_result.json`

Full epoch history:

`results/cuhk03_supervised_training_history.json`

Pre-training development baseline:

`results/cuhk03_supervised_development_baseline.json`

Preregistered configuration:

`configs/cuhk03_supervised_multires_experiment.json`
