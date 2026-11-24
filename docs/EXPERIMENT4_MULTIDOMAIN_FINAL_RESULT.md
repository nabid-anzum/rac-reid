# Experiment 4 - Multi-Domain Generalization Final Result

## Experimental ordering

Experiment 4 followed this frozen sequence:

1. SenseReID archive/manifests frozen.
2. Protocol preregistered.
3. Trainer and evaluator committed.
4. Market+CUHK03 training completed.
5. Epoch selected only from source calibration.
6. Source held-out validation completed.
7. Checkpoint and training evidence committed.
8. SenseReID evaluated exactly once.
9. Post-target tuning prohibited.

## Selected checkpoint

Epoch:

**3**

SHA-256:

`1234aa0cb5e17b3d8e8f0567096aa132db56ddeb9c01ebc084050d6acd726b60`

## Source held-out

Market:

- Efficient R1: **88.352%**
- Full R1: **88.812%**

CUHK03:

- Efficient R1: **51.402%**
- Full R1: **52.603%**

Macro:

- Efficient R1: **69.877%**
- Full R1: **70.708%**

## SenseReID unseen-target result

| Model | Efficient mAP | Efficient R1 | Full mAP | Full R1 | R1 retention |
|---|---:|---:|---:|---:|---:|
| Market-only | 43.399% | 35.039% | 44.706% | 36.390% | 96.286% |
| CUHK-only | 40.756% | 32.143% | 45.754% | 37.452% | 85.825% |
| Market+CUHK | **45.716%** | **36.680%** | **49.048%** | **40.637%** | **90.261%** |

## Joint-model deltas

Versus Market-only:

- Efficient mAP: **+2.317 pp**
- Efficient Rank-1: **+1.641 pp**
- Full mAP: **+4.342 pp**
- Full Rank-1: **+4.247 pp**

Versus CUHK-only:

- Efficient mAP: **+4.960 pp**
- Efficient Rank-1: **+4.537 pp**
- Full mAP: **+3.295 pp**
- Full Rank-1: **+3.185 pp**

Positive deltas mean the multi-domain model generalized better on the untouched
SenseReID target. Negative deltas mean it did not.

## Integrity

SenseReID:

- 1,040 query images / 521 observed IDs
- 3,388 gallery images / 1,718 observed IDs
- no target-driven training
- no target-driven selection
- no reranking
- no post-target tuning

Final SenseReID result SHA-256:

`978c88f2319827e585ef1bfd52ddd57a3b3882debc0ba7b3610bbb3242d87065`
