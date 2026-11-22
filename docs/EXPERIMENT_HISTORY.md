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
