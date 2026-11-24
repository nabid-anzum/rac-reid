# RAC-ReID Reproducibility Evidence

## Purpose

This document tells a reviewer exactly what data, code, protocol and result
records are available for reproducing and auditing the RAC-ReID experiments.

Raw third-party research datasets are not committed to this Git repository.
Instead, the repository records the dataset source, archive SHA-256, exact
split counts, filename/size manifests, evaluator code, experiment protocol and
machine-readable results.

This keeps the repository reviewable while still making dataset integrity
verifiable.

---

## Experiment 1 — Market-1501

Final result:

`results/market1501_multires_epoch8_final.json`

Training result:

`results/market1501_multiresolution_training.json`

Final evaluation description:

`docs/FINAL_MARKET1501_EVALUATION.md`

Standard test population:

- query images: 3368
- gallery images: 19732

Final Efficient 160x80:

- mAP: 55.899%
- Rank-1: 78.088%

Final Full 256x128:

- mAP: 59.199%
- Rank-1: 79.543%

The exact Market dataset archive checksum and test manifests are recorded in:

`results/dataset_provenance_2026-11-23.json`

---

## Experiment 2 — CUHK03-NP zero-shot

Experiment 2 used the exact frozen Market-trained epoch-8 checkpoint.

No CUHK03 training or tuning occurred.

Preregistered protocol:

`results/cuhk03_zero_shot_protocol.json`

Exact evaluator executed:

`scripts/evaluate_cuhk03_zero_shot.py`

Evaluator SHA-256:

`7e689881e569204477ea0f444833420ae065c9baa406408bb57a064682e2bbc3`

Evaluation start audit record:

`results/cuhk03_zero_shot_started.json`

Evaluation completion audit record:

`results/cuhk03_zero_shot_complete.json`

Primary Detected raw metrics:

`results/cuhk03_detected_zero_shot_metrics.json`

Secondary Labeled raw metrics:

`results/cuhk03_labeled_zero_shot_metrics.json`

Combined final result:

`results/cuhk03_zero_shot_epoch8_final.json`

Human-readable analysis:

`docs/CUHK03_ZERO_SHOT_EVALUATION.md`

Primary Detected final result:

- Efficient mAP: 20.146%
- Efficient Rank-1: 19.786%
- Full mAP: 19.920%
- Full Rank-1: 19.857%
- Efficient Rank-1 retention: 99.640%

---

## Dataset integrity

Full provenance record:

`results/dataset_provenance_2026-11-23.json`

### Market-1501 archive

SHA-256:

`416bb77b5a2449b32e936f623cbee58becf1a9e7e936f36380cb8f9ab928fe96`

### CUHK03-NP archive

Archive size:

`1225247366 bytes`

SHA-256:

`9a11f21f6b3dd175bb1be41baaf88298b894dcce3b9efdbffb4ee247f019ae93`

The CUHK03 protocol file also stores filename/size SHA-256 manifests for every
Detected and Labeled train/query/gallery split.

A reviewer can therefore verify that the same research dataset was used
without the repository redistributing the raw image archive.

---

## Why raw datasets and embedding caches are not in GitHub

The following are intentionally excluded:

- raw Market-1501 image files
- raw CUHK03-NP image files
- the 1.17 GB CUHK03 ZIP archive
- temporary `.npy` embedding caches
- model cache files
- the trained binary checkpoint

These are large external/runtime artifacts.

Their hashes and provenance are stored instead.

The trained checkpoint SHA-256 is:

`d0570ff0fec0905b75bef1b90ec5147b96ac7b51d8ba9f30de5fa3f0c0757dec`

---

## Software verification

Latest post-CUHK03 verification:

- tests run: 42
- status: PASSED

Machine-readable summary:

`results/project_test_summary_after_cuhk03_2026-11-23.json`

Full test log:

`results/test_logs/2026-11-23_after_cuhk03_unittest.txt`

---

## Recommended reviewer order

1. `README.md`
2. `docs/TEACHER_RESULTS_SUMMARY.md`
3. `docs/REPRODUCIBILITY_EVIDENCE.md`
4. `docs/FINAL_MARKET1501_EVALUATION.md`
5. `docs/CUHK03_ZERO_SHOT_EVALUATION.md`
6. `results/dataset_provenance_2026-11-23.json`
7. `results/market1501_multiresolution_training.json`
8. `results/market1501_multires_epoch8_final.json`
9. `results/cuhk03_zero_shot_protocol.json`
10. `scripts/evaluate_cuhk03_zero_shot.py`
11. `results/cuhk03_zero_shot_epoch8_final.json`
12. `results/project_test_summary_after_cuhk03_2026-11-23.json`


## Experiment 3 — CUHK03 supervised training

Preregistered configuration:

`configs/cuhk03_supervised_multires_experiment.json`

Training protocol and exact loss definitions:

`docs/CUHK03_SUPERVISED_TRAINING_PROTOCOL.md`

Exact trainer:

`scripts/train_cuhk03_supervised_multires.py`

Training result:

`results/cuhk03_supervised_training_result.json`

Epoch history:

`results/cuhk03_supervised_training_history.json`

Pre-training development baseline:

`results/cuhk03_supervised_development_baseline.json`

Human-readable result:

`docs/CUHK03_SUPERVISED_TRAINING_RESULT.md`

Checkpoint binary:

Not committed.

Checkpoint SHA-256:

`f879574a3cf21f9b106f51fbd2bef692e6044761a3bb17280c5da3a37fd1a399`

The official CUHK03 query/gallery benchmark was not used during training or
model selection.


## Experiment 3 — standardized CUHK03 benchmark

Frozen epoch-3 checkpoint SHA-256:

`f879574a3cf21f9b106f51fbd2bef692e6044761a3bb17280c5da3a37fd1a399`

Fixed benchmark evaluator:

`scripts/evaluate_cuhk03_supervised_benchmark.py`

Fixed evaluator commit:

`e1dc17a30a2671aad3756433726d05080d531e67`

Evaluator SHA-256:

`d7f0941d59c0d5a35f293961522e89dace56df9feade09a63f773f0e8f71a7a2`

Benchmark start marker:

`results/cuhk03_supervised_benchmark_started.json`

Benchmark completion marker:

`results/cuhk03_supervised_benchmark_complete.json`

Primary Detected metrics:

`results/cuhk03_supervised_detected_metrics.json`

Secondary Labeled metrics:

`results/cuhk03_supervised_labeled_metrics.json`

Final benchmark result:

`results/cuhk03_supervised_benchmark_final.json`

Final result SHA-256:

`55430f6e21358abc6d80f8987558302a687e18bf3c05ac569a185b5ac0a401c5`

Human-readable analysis:

`docs/CUHK03_SUPERVISED_BENCHMARK_RESULT.md`

Cross-experiment comparison:

`results/experiment_comparison_summary_2026-11-23.json`

No post-benchmark tuning is permitted.


## Experiment 4 - SenseReID evidence

Implementation:

- `scripts/train_multidomain_market_cuhk_sensereid.py`
- `scripts/evaluate_sensereid_multidomain.py`

Training evidence:

- `results/experiment4_multidomain_training_result.json`
- `results/experiment4_multidomain_training_history.json`
- `results/experiment4_multidomain_development_baseline.json`

Target evidence:

- `results/experiment4_sensereid_started.json`
- `results/experiment4_sensereid_complete.json`
- `results/experiment4_sensereid_market_single_source.json`
- `results/experiment4_sensereid_cuhk_single_source.json`
- `results/experiment4_sensereid_multidomain.json`
- `results/experiment4_sensereid_final.json`

Checkpoint SHA-256:

`1234aa0cb5e17b3d8e8f0567096aa132db56ddeb9c01ebc084050d6acd726b60`

Target result SHA-256:

`978c88f2319827e585ef1bfd52ddd57a3b3882debc0ba7b3610bbb3242d87065`

Post-target tuning is prohibited.
