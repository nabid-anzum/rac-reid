# RAC-ReID Project Update - 22 November 2026

## Project

RAC-ReID stands for Resource-Adaptive Cascaded Person Re-Identification.

The research question is whether a ReID system can dynamically reduce
query-side computation for easier searches while preserving identification
quality and allocate more computation only when necessary.

Public application:

https://reid.nabidanzum.com

GitHub:

https://github.com/nabid-anzum/rac-reid

## Dataset

Primary dataset: Market-1501.

Development population:

- 12,936 images
- 751 identities

Identity-disjoint development split:

- gradient training: 7,811 images / 449 identities
- calibration: 2,515 images / 144 identities
- held-out development validation: 2,610 images / 158 identities

The standard Market-1501 query/gallery test split has not been used for the
new multi-resolution training or checkpoint selection.

## Established standard baselines

| Experiment | mAP | Rank-1 |
|---|---:|---:|
| OSNet x1.0 | 37.368% | 66.330% |
| OSNet x0.25 | 31.008% | 59.768% |
| Initial proxy adaptive cascade | 2.015% | 14.252% |

## Why the architecture changed

Several adaptive approaches were tested before joint training.

- Handcrafted proxy Top-K pruning failed because valid identities were removed
  before the learned OSNet stage.
- A separate OSNet x0.25 first pass duplicated too much real CPU work.
- A post-hoc x0.25-to-x1 embedding adapter damaged retrieval geometry.
- Cheap image-quality features could not safely route enough queries.
- Frozen conv2/conv3 exits were not discriminative enough for ReID.
- Progressive 224x112 to 256x128 routing had good routing quality but only
  about 2.9% best-case stable CPU saving.

These negative results motivated joint multi-resolution training.

## Training feasibility

CPU profiling showed:

- full OSNet fine-tuning: about 14.50 min/epoch
- freeze conv1+conv2: about 5.84 min/epoch
- freeze conv1+conv2+conv3: about 5.02 min/epoch

The selected configuration freezes conv1 and conv2 while allowing the
remaining 1,925,340 backbone parameters to adapt.

## Joint multi-resolution training

Low-resource resolution: 160x80.

Full resolution: 256x128.

Same OSNet x1.0 backbone is used for both.

Training configuration:

- 8 epochs
- batch size 16
- AdamW
- backbone LR 5e-5
- classifier LR 5e-4
- weight decay 5e-4
- selected checkpoint: epoch 8

Objective:

L =
1.00 * low-resolution teacher cosine loss
+ 0.25 * full-resolution teacher cosine loss
+ 0.50 * relational similarity loss
+ 0.25 * cross-resolution consistency loss
+ 0.50 * identity classification loss

## Epoch history

| Epoch | Low R1 | Full R1 | Low gain | Full delta | Agreement | Cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71.968% | 72.087% | +9.145% | -7.714% | 88.231% | 0.8944 |
| 2 | 76.103% | 74.235% | +13.280% | -5.567% | 88.469% | 0.9000 |
| 3 | 77.575% | 76.302% | +14.751% | -3.499% | 88.429% | 0.9047 |
| 4 | 80.080% | 79.960% | +17.256% | +0.159% | 89.264% | 0.9099 |
| 5 | 80.000% | 81.551% | +17.177% | +1.750% | 90.577% | 0.9217 |
| 6 | 79.722% | 78.131% | +16.899% | -1.670% | 85.686% | 0.9108 |
| 7 | 83.618% | 83.857% | +20.795% | +4.056% | 89.583% | 0.9178 |
| 8 | 85.368% | 86.362% | +22.545% | +6.561% | 92.485% | 0.9223 |

## Held-out development result

Original 160x80 Rank-1: 64.598%

Trained 160x80 Rank-1: 88.697%

Low-resolution improvement: +24.100%

Original full-resolution Rank-1: 81.456%

Trained full-resolution Rank-1: 88.314%

Full-resolution change: +6.858%

Low/full identity agreement:
93.602%

Low/full embedding cosine:
0.9186

IMPORTANT:

These are held-out development metrics using identities from the Market-1501
training population. They are not the final standard Market-1501 benchmark.

## Corrected inference timing

- 1 Torch thread(s): full 0.056818s, low 0.031487s, low-resolution saving 44.584%.
- 4 Torch thread(s): full 0.044167s, low 0.030539s, low-resolution saving 30.856%.

## Why Step 29F-B appeared to fail

The model training itself did not fail.

All eight epochs completed and epoch 8 was saved.

The command then reached a separate timing-report function. A Python loop bug
left one timing array empty, causing:

statistics.StatisticsError: no median for empty data

The checkpoint was preserved. It was reloaded and validated without retraining.

## Checkpoint

Runtime checkpoint:

/root/rac-reid/.runtime/models/multires/osnet_x1_0_multires_160x80.pt

SHA-256:

d0570ff0fec0905b75bef1b90ec5147b96ac7b51d8ba9f30de5fa3f0c0757dec

The binary model is not committed to GitHub. Its exact hash, architecture,
training configuration and results are stored here.

## What GitHub now preserves

- source code
- evaluation and recovery code
- experiment configurations
- epoch-by-epoch training history
- successful results
- failed experiment records
- methodology decisions
- dataset split information
- checkpoint SHA-256
- reproducibility protocol
- teacher-facing project update

Raw Market-1501 images, large embedding caches and temporary model caches are
not committed.

## Next step

Freeze this trained backbone.

Then train a small confidence controller using only production-observable
160x80 retrieval statistics.

Only if that passes the predetermined development gates will the complete
adaptive architecture be frozen and evaluated on the untouched standard
Market-1501 query/gallery test.
