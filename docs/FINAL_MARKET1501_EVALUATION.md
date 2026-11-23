# Final Frozen Market-1501 Evaluation

## Status

**FINAL — one-time frozen test-set evaluation**

Date: 2026-11-23

Frozen source commit:

`222cf935974e6d8c10a336eebe4eb04167f71c8b`

Frozen epoch-8 checkpoint SHA-256:

`d0570ff0fec0905b75bef1b90ec5147b96ac7b51d8ba9f30de5fa3f0c0757dec`

Final result SHA-256:

`a7b14a330d5dd6691635d32199aa72ace3c1389d5f00e88b78a05665da3bc8f1`

The standard Market-1501 test set was evaluated only after the architecture,
training procedure, selected epoch and deployment path were frozen.

No model, threshold or routing parameter may be tuned using these results.

---

## Protocol

Dataset: **Market-1501**

- query images: 3,368
- gallery images: 19,732
- Efficient query: 160x80
- Full query: 256x128
- Gallery embeddings: 256x128
- embedding dimension: 512
- backbone: OSNet x1.0, jointly trained epoch 8
- same trained backbone for both query resolutions
- L2-normalized embeddings
- cosine similarity ranking
- PID `-1` junk removed
- same-person / same-camera gallery images removed
- PID `0` distractors retained
- no re-ranking
- no YOLO in the benchmark because Market-1501 images are already person crops

For normalized embeddings, cosine ranking and squared-Euclidean ranking are
monotonic equivalents.

---

## Final results

| Metric | Efficient 160x80 | Full 256x128 | Efficient - Full |
|---|---:|---:|---:|
| mAP | **55.899%** | **59.199%** | -3.300 pp |
| Rank-1 | **78.088%** | **79.543%** | -1.455 pp |
| Rank-5 | 89.519% | 89.133% | +0.386 pp |
| Rank-10 | 92.488% | 91.746% | +0.742 pp |
| mINP | 18.160% | 20.953% | -2.793 pp |

### Retrieval retention

Efficient 160x80 retains:

- **98.171% of full-resolution Rank-1**
- **94.426% of full-resolution mAP**

The low-resolution Rank-1 penalty is only:

**1.455 percentage points**

The mAP penalty is:

**3.300 percentage points**

---

## Comparison with original pretrained baseline

Previously measured standard cross-domain OSNet x1.0 baseline:

- mAP: 37.368%
- Rank-1: 66.330%
- Rank-5: 80.285%
- Rank-10: 85.748%
- mINP: 6.079%

### Efficient 160x80 vs original baseline

- mAP improvement: **+18.531 pp**
- Rank-1 improvement: **+11.758 pp**

### Full 256x128 vs original baseline

- mAP improvement: **+21.831 pp**
- Rank-1 improvement: **+13.213 pp**

The jointly trained epoch-8 model therefore substantially exceeds the
original pretrained baseline at both query resolutions.

---

## Efficiency result

The separate controlled production timing benchmark used:

- 5 warm-up runs per mode
- 30 measured runs per mode
- 60 measured searches total
- alternating query-resolution order
- four Torch CPU threads

It measured:

- median OSNet saving at 160x80: **+22.973%**
- median complete-backend saving: **+9.498%**

This produces the central final result:

> **The 160x80 query path retains 98.171% of full-resolution Rank-1 while reducing median OSNet query inference by 22.973%.**

---

## Interpretation

The final experiment supports the project's main hypothesis.

Reducing the query resolution from 256x128 to 160x80 creates only a small
Rank-1 reduction while providing a measurable CPU inference reduction.

Full resolution remains preferable when maximum mAP and Rank-1 are the only
objectives.

Efficient resolution provides the stronger accuracy/compute trade-off.

The Efficient path also produced slightly higher Rank-5 and Rank-10 values in
this frozen evaluation. This does not imply that the lower resolution is
universally more accurate; the primary comparison remains mAP, Rank-1 and
measured compute.

---

## Scientific policy after final evaluation

The official test set is now **consumed**.

Therefore:

- no retraining based on this result
- no threshold tuning based on this result
- no architecture selection based on this result
- no selective replacement of the final result
- no repeated test runs for model selection

Any future architecture changes must be reported as a new experiment rather
than modifications to this frozen final experiment.

---

## Evidence

Machine-readable final result:

`results/market1501_multires_epoch8_final.json`

Training result:

`results/market1501_multiresolution_training.json`

Controlled timing result:

`results/production_multires_benchmark_2026-11-23.json`

Software tests:

`results/project_test_summary_2026-11-23.json`

Experiment history:

`docs/EXPERIMENT_HISTORY.md`
