# Experiment 4 - Multi-Domain Training Result

## Status

Training was completed and frozen before any SenseReID model inference.

Selected epoch:

**3**

Checkpoint SHA-256:

`1234aa0cb5e17b3d8e8f0567096aa132db56ddeb9c01ebc084050d6acd726b60`

## Epoch history

| Epoch | Loss | Market Efficient R1 | CUHK Efficient R1 | Macro Efficient R1 | Macro Full R1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.99654 | 86.163% | 49.241% | 67.702% | 72.836% |
| 2 | 0.55068 | 87.913% | 47.129% | 67.521% | 70.625% |
| 3 | 0.39053 | 86.918% | 48.581% | 67.750% | 68.807% | **selected**
| 4 | 0.33382 | 86.879% | 46.865% | 66.872% | 67.649% |
| 5 | 0.30096 | 86.521% | 46.205% | 66.363% | 65.786% |
| 6 | 0.28041 | 85.129% | 41.452% | 63.291% | 63.260% |
| 7 | 0.25802 | 84.970% | 39.736% | 62.353% | 61.139% |
| 8 | 0.24453 | 84.135% | 42.046% | 63.091% | 61.493% |

## Selected calibration

Market:

- Efficient Rank-1: **86.918%**
- Full Rank-1: **87.714%**

CUHK03:

- Efficient Rank-1: **48.581%**
- Full Rank-1: **49.901%**

Macro:

- Efficient Rank-1: **67.750%**
- Full Rank-1: **68.807%**

## Source held-out validation

Market:

- Efficient Rank-1: **88.352%**
- Full Rank-1: **88.812%**

CUHK03:

- Efficient Rank-1: **51.402%**
- Full Rank-1: **52.603%**

Macro:

- Efficient Rank-1: **69.877%**
- Full Rank-1: **70.708%**

## Target isolation

At this freeze point:

- SenseReID training use: false
- SenseReID selection use: false
- SenseReID model inference: false

The target benchmark had not yet been consumed.
