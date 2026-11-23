# RAC-ReID — Teacher Results Summary

## Current research system

**RAC-ReID — Multi-Resolution Person Re-Identification**

Current real inference path:

`photo -> YOLO detection -> fixed 10% crop -> 160x80 or 256x128 -> jointly trained OSNet -> normalized 512-D embedding -> stored 256x128 gallery -> cosine ranking`

Historical proxy-pruning and failed routing experiments are retained as
research history but are not used by the live application.

## Training

Model: **OSNet x1.0**

Training setup:

- same model receives 160x80 and 256x128 views
- `conv1 + conv2` frozen
- later layers trainable
- original full-quality OSNet provides teacher/reference embeddings
- AdamW optimization
- 8 epochs
- epoch selection performed on calibration identities

Joint objective:

`L = 1.00*L_low_cos + 0.25*L_full_cos + 0.50*L_relational + 0.25*L_cross_resolution + 0.50*L_classification`

Selected epoch: **8**

Checkpoint SHA-256:

`d0570ff0fec0905b75bef1b90ec5147b96ac7b51d8ba9f30de5fa3f0c0757dec`

Machine-readable training evidence:

`results/market1501_multiresolution_training.json`

## Calibration history

| Epoch | Low R1 | Full R1 | Low gain | Full delta | Agreement | Embedding cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71.968% | 72.087% | +9.145 pp | -7.714 pp | 88.231% | 0.8944 |
| 2 | 76.103% | 74.235% | +13.280 pp | -5.567 pp | 88.469% | 0.9000 |
| 3 | 77.575% | 76.302% | +14.751 pp | -3.499 pp | 88.429% | 0.9047 |
| 4 | 80.080% | 79.960% | +17.256 pp | +0.159 pp | 89.264% | 0.9099 |
| 5 | 80.000% | 81.551% | +17.177 pp | +1.750 pp | 90.577% | 0.9217 |
| 6 | 79.722% | 78.131% | +16.899 pp | -1.670 pp | 85.686% | 0.9108 |
| 7 | 83.618% | 83.857% | +20.795 pp | +4.056 pp | 89.583% | 0.9178 |
| 8 | 85.368% | 86.362% | +22.545 pp | +6.561 pp | 92.485% | 0.9223 |

## Identity-disjoint held-out development result

| Metric | Original | Trained epoch 8 |
|---|---:|---:|
| 160x80 Rank-1 | 64.598% | **88.697%** |
| 256x128 Rank-1 | 81.456% | **88.314%** |

Additional held-out measurements:

- 160x80 Rank-1 gain: **+24.100 percentage points**
- 256x128 Rank-1 gain: **+6.858 percentage points**
- low/full Top-1 identity agreement: **93.602%**
- same-image low/full embedding cosine: **0.9186**

These are **development results**, not the final official trained-model
Market-1501 test.

## Preserved standard pretrained baseline

The repository also contains the previously completed standard Market-1501
cross-domain baseline:

`results/runtime_history/market1501_osnet_crossdomain.json`

That baseline is preserved separately from the trained multiresolution
experiment.


## Final frozen Market-1501 result

The architecture and epoch-8 checkpoint were frozen before the one-time
official Market-1501 test.

| Metric | Efficient 160x80 | Full 256x128 |
|---|---:|---:|
| mAP | **55.899%** | **59.199%** |
| Rank-1 | **78.088%** | **79.543%** |
| Rank-5 | 89.519% | 89.133% |
| Rank-10 | 92.488% | 91.746% |
| mINP | 18.160% | 20.953% |

Efficient vs Full:

- Rank-1 delta: **-1.455 pp**
- mAP delta: **-3.300 pp**
- Rank-1 retention: **98.171%**
- controlled median OSNet saving: **+22.973%**

**Final research headline:** the 160x80 query path retains
**98.171%** of full-resolution Rank-1 while reducing
median OSNet query inference by **22.973%**.

Full final-test record:

- `docs/FINAL_MARKET1501_EVALUATION.md`
- `results/market1501_multires_epoch8_final.json`

No post-test tuning is permitted.

## Controlled production benchmark — 2026-11-23

Protocol:

- 5 warm-up runs per mode
- 30 measured runs per mode
- 60 measured searches total
- same image for both modes
- alternating Efficient/Full order
- **4 Torch CPU threads**
- YOLO detection and crop repeated for every search
- timing input used only for performance measurement
- official trained-model Market-1501 test not used

| Metric | Efficient 160x80 | Full 256x128 |
|---|---:|---:|
| OSNet median | **0.039837 s** | **0.051718 s** |
| OSNet mean | 0.040986 s | 0.051388 s |
| OSNet p95 | 0.047298 s | 0.058305 s |
| Backend median | **0.126665 s** | **0.139959 s** |
| Backend p95 | 0.209029 s | 0.223040 s |

Median OSNet saving:

**+22.973%**

Median complete-backend saving:

**+9.498%**

Reproducible evidence:

- `results/production_multires_benchmark_2026-11-23.json`
- `results/production_multires_benchmark_2026-11-23.csv`
- `scripts/benchmark_production_multires.py`

## Real browser sanity test

A different real photograph of `nabid` was queried against the two-identity
live gallery.

Efficient 160x80:

- Rank 1: `nabid`
- cosine: 0.5249
- margin: 0.1121
- warm OSNet: 0.0478 s
- backend: 0.1738 s

Full 256x128:

- Rank 1: `nabid`
- cosine: 0.5962
- margin: 0.1492
- OSNet: 0.1024 s
- backend: 0.4584 s

Both modes retrieved the correct identity.

This is a functional E2E sanity test, not a statistical accuracy benchmark.

Full record:

`docs/REAL_WORLD_E2E_TEST_2026-11-23.md`

## Software verification

- complete unit tests: **42**
- status: **PASSED**
- raw log:
  `results/test_logs/2026-11-23_unittest.txt`
- JSON summary:
  `results/project_test_summary_2026-11-23.json`

## Scientific safeguards

- the frozen trained-model Market-1501 final evaluation was completed once on 2026-11-23
- no post-test tuning is allowed
- held-out development identities are separate from parameter updates and epoch selection
- failed learned routers are not deployed
- true PID/camera metadata is not used by runtime search
- cold browser timing is not used as the primary efficiency claim
- raw benchmark measurements are preserved
- historical negative experiments remain documented

## Teacher reading order

1. `docs/TEACHER_RESULTS_SUMMARY.md`
2. `results/market1501_multiresolution_training.json`
3. `docs/MULTIRESOLUTION_TRAINING_PROTOCOL.md`
4. `results/production_multires_benchmark_2026-11-23.json`
5. `results/production_multires_benchmark_2026-11-23.csv`
6. `docs/REAL_WORLD_E2E_TEST_2026-11-23.md`
7. `results/project_test_summary_2026-11-23.json`
8. `docs/EXPERIMENT_HISTORY.md`
