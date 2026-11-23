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
