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


## Step 30B — CUHK03-NP zero-shot generalization

Experiment 2 evaluated the unchanged frozen Market-trained epoch-8 model on
CUHK03-NP.

The protocol was preregistered before testing:

- Detected = primary
- Labeled = secondary
- no CUHK03 training
- no CUHK03 adaptation
- no threshold tuning
- 160x80 Efficient query
- 256x128 Full query
- 256x128 gallery

Primary Detected:

- Efficient mAP: **20.146%**
- Efficient Rank-1: **19.786%**
- Full mAP: **19.920%**
- Full Rank-1: **19.857%**
- Rank-1 difference: **-0.071 pp**
- Rank-1 retention: **99.640%**

Secondary Labeled:

- Efficient mAP: **20.550%**
- Efficient Rank-1: **20.214%**
- Full mAP: **20.145%**
- Full Rank-1: **19.357%**

Conclusion:

Absolute zero-shot retrieval accuracy suffers substantial domain shift, while
the Efficient-vs-Full resolution relationship remains highly stable.

Result SHA-256:

`9cf940063b73af21b1b67d84b7ed590ba378f6ff5c89105ffabee806259a34d7`

Experiment 2 is frozen. Any CUHK03 training belongs to Experiment 3.


## Step 31A — Experiment 3 CUHK03 supervised preregistration

Experiment 3 was preregistered before any CUHK03-specific model training.

Research question:

Can supervised CUHK03-NP Detected training improve absolute CUHK03 retrieval
accuracy while preserving the Efficient-vs-Full multi-resolution relationship?

Fixed protocol:

- CUHK03-NP Detected train only
- 767 available training identities
- identity-disjoint gradient/calibration/held-out split
- same original MSMT17 OSNet x1.0 initialization as Experiment 1
- 160x80 Efficient path
- 256x128 Full path
- conv1 and conv2 frozen
- 8 epochs
- fixed Experiment-1 loss coefficients and optimizer settings
- calibration-only epoch selection
- held-out validation after model selection
- official CUHK03 benchmark only after freezing the selected checkpoint
- no post-benchmark tuning
- production model remains unchanged

Important limitation:

CUHK03 query/gallery was previously observed during Experiment 2, so the
Experiment-3 post-training CUHK03 evaluation is a standardized benchmark,
not a pristine blind test.

Protocol:

- `configs/cuhk03_supervised_multires_experiment.json`
- `docs/CUHK03_SUPERVISED_TRAINING_PROTOCOL.md`


## Step 31B — CUHK03 supervised training

Experiment 3 completed all eight preregistered training epochs.

Calibration selected **epoch 3**.

Selected calibration result:

- Efficient Rank-1: **49.901%**
- Full Rank-1: **54.323%**
- agreement: **97.030%**
- cosine: **0.8495**

Identity-disjoint held-out:

- Efficient Rank-1: **52.870%**
- Full Rank-1: **56.075%**
- agreement: **96.929%**
- cosine: **0.8489**

Training loss continued falling after epoch 3, but calibration retrieval
accuracy deteriorated. The preregistered model-selection rule therefore
prevented final-epoch overfitting from determining the checkpoint.

Checkpoint SHA-256:

`f879574a3cf21f9b106f51fbd2bef692e6044761a3bb17280c5da3a37fd1a399`

Official CUHK03 query/gallery images were not used during training or model
selection.

The standardized post-training benchmark remains pending.
