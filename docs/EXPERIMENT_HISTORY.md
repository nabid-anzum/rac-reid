# RAC-ReID Experiment History

This file records the important architecture decisions and negative results.

| Step | Experiment | Main result | Decision |
|---|---|---|---|
| 27 | OSNet x1.0 baseline | 37.368% mAP, 66.330% Rank-1 | Valid full learned baseline |
| 28C | Handcrafted proxy adaptive cascade | 2.015% mAP, 14.252% Rank-1, 100% heavy invocation | Rejected: proxy shortlist recall bottleneck |
| 29B | OSNet x0.25 baseline | 31.008% mAP, 59.768% Rank-1 | Valid lightweight baseline |
| 29C3 | Learned correctness router | Initially promising but later found methodologically invalid | Superseded because routing features had ground-truth-assisted filtering |
| 29C4 | Production-observable correctness router | 3.448% held-out coverage, 84.444% precision, compute negative | Rejected |
| 29C5 | x0.25/x1 agreement router | 98.146% precision, 61.992% exits | Router worked, but independent CNN cascade had too little real CPU benefit |
| 29C6 | Teacher-distilled embedding adapter | Agreement and Rank-1 worsened | Rejected |
| 29C7 | Image-quality pre-router | 0.690% useful coverage | Rejected |
| 29D-B | Frozen shared-backbone early exit | conv3 exit 0.651%, 17.893% early-head Rank-1 | Rejected |
| 29E-A | Multi-resolution audit | 224x112 retained strong retrieval | Investigated further |
| 29E-B | 224-to-256 progressive router | 97.366% precision, 85.824% exits, -0.421 pp Rank-1 | Routing worked but CPU economics failed |
| 29E-C | Stable timing audit | Best projected saving only +2.929% | Progressive pretrained branch closed |
| 29F-A2 | Training feasibility | Full 14.50 min/epoch; freeze conv1+conv2 5.84 min/epoch | Joint training practical |
| 29F-B | Joint 160x80/256x128 OSNet training | Large held-out low-resolution recovery with preserved full-resolution quality | Current direction |

Negative experiments are intentionally retained. Gates were not lowered after
failed experiments merely to obtain a passing result.

The standard Market-1501 test query/gallery split remains reserved for the
final frozen-architecture evaluation.

| 29G | Production-observable low-resolution confidence controller | No calibration threshold passed all predefined gates | Do not deploy; redesign controller |

| 29G1 | Low/full refinement headroom audit | full rescues low on 4.3% of calibration queries; fixed-low saving 30.9%; perfect-router saving 26.6% | Use measured rescue headroom to choose the next controller target |

| 29G2 | Refinement-benefit router | adaptive R1 88.7%, refine 3.2%, CPU +27.7% | Do not deploy; retain fixed-low baseline |


## Step 29H-E — Controlled production multi-resolution benchmark

A warm alternating benchmark was executed against the current production
multi-resolution search code.

Protocol:

- 5 warm-up runs per resolution
- 30 measured runs per resolution
- 60 measured searches
- same timing image
- alternating resolution order
- 4 Torch CPU threads
- YOLO/crop/search path repeated
- official trained-model Market-1501 test not used

Results:

- 160x80 OSNet median: **0.039837 s**
- 256x128 OSNet median: **0.051718 s**
- OSNet median saving: **+22.973%**
- total-backend median saving: **+9.498%**
- software tests: **42/42 passed**

Evidence:

- `results/production_multires_benchmark_2026-11-23.json`
- `results/production_multires_benchmark_2026-11-23.csv`
- `results/project_test_summary_2026-11-23.json`
- `results/test_logs/2026-11-23_unittest.txt`
- `docs/TEACHER_RESULTS_SUMMARY.md`

The benchmark image is used for timing only and is not treated as retrieval
accuracy evidence.


## Step 29H-F — Final frozen Market-1501 evaluation

The epoch-8 architecture and checkpoint were frozen before the official
Market-1501 test was used.

Standard protocol:

- 3,368 query images
- 19,732 gallery images
- Efficient query: 160x80
- Full query: 256x128
- gallery: 256x128
- no YOLO because Market-1501 provides person crops
- same-PID/same-camera exclusion
- PID -1 junk filtering
- no re-ranking

Final Efficient 160x80:

- mAP: **55.899%**
- Rank-1: **78.088%**
- Rank-5: **89.519%**
- Rank-10: **92.488%**
- mINP: **18.160%**

Final Full 256x128:

- mAP: **59.199%**
- Rank-1: **79.543%**
- Rank-5: **89.133%**
- Rank-10: **91.746%**
- mINP: **20.953%**

Efficient relative to Full:

- Rank-1 delta: **-1.455 pp**
- mAP delta: **-3.300 pp**
- Rank-1 retention: **98.171%**
- controlled OSNet median saving: **+22.973%**

Final result SHA-256:

`a7b14a330d5dd6691635d32199aa72ace3c1389d5f00e88b78a05665da3bc8f1`

The official test set is now consumed. No post-test tuning is permitted.
