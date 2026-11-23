# CUHK03-NP Zero-Shot Cross-Dataset Evaluation

## Experiment 2

This experiment tests whether the frozen Market-1501-trained RAC-ReID model
generalizes to a different person-ReID dataset without retraining, adaptation,
threshold tuning or model selection.

The experiment was preregistered before CUHK03 test images were evaluated.

Primary benchmark: **CUHK03-NP Detected**

Secondary benchmark: **CUHK03-NP Labeled**

---

## Frozen model

Source commit:

`f4f49db012c6c12004c0c6495fc778728aa134b9`

Checkpoint SHA-256:

`d0570ff0fec0905b75bef1b90ec5147b96ac7b51d8ba9f30de5fa3f0c0757dec`

The checkpoint was trained only through the Market-1501 development
procedure used by Experiment 1.

No CUHK03 images were used for:

- gradient training
- epoch selection
- model selection
- threshold tuning
- architecture selection

---

## CUHK03-NP protocol

New protocol:

- 767 training identities
- 700 testing identities
- 1,400 query images per variant
- Detected gallery: 5,332 images
- Labeled gallery: 5,328 images
- train/test identity overlap: 0
- Efficient query: 160x80
- Full query: 256x128
- gallery: 256x128
- 512-D normalized embeddings
- cosine similarity
- same-PID / same-camera true matches treated as junk
- no re-ranking

The Detected variant was preregistered as the primary result regardless of
which variant or resolution scored better.

---

## Primary result — Detected

| Metric | Efficient 160x80 | Full 256x128 | Efficient - Full |
|---|---:|---:|---:|
| mAP | **20.146%** | **19.920%** | +0.226 pp |
| Rank-1 | **19.786%** | **19.857%** | -0.071 pp |
| Rank-5 | 36.714% | 36.714% | +0.000 pp |
| Rank-10 | 44.214% | 44.286% | -0.071 pp |
| mINP | 12.513% | 12.683% | -0.170 pp |

Efficient Rank-1 retention relative to Full:

**99.640%**

The Rank-1 difference is only:

**-0.071 percentage points**

This shows that the low/full resolution relationship remains extremely close
under cross-dataset domain shift.

---

## Secondary result — Labeled

| Metric | Efficient 160x80 | Full 256x128 | Efficient - Full |
|---|---:|---:|---:|
| mAP | **20.550%** | **20.145%** | +0.404 pp |
| Rank-1 | **20.214%** | **19.357%** | +0.857 pp |
| Rank-5 | 35.857% | 34.571% | +1.286 pp |
| Rank-10 | 45.071% | 43.286% | +1.786 pp |
| mINP | 13.076% | 13.296% | -0.220 pp |

Efficient Rank-1 retention:

**104.428%**

Efficient slightly exceeds Full on several metrics here. Because this is a
secondary dataset variant and the differences are small, this should not be
interpreted as evidence that lower resolution is generally more accurate.

---

## Comparison with Market-1501

| Dataset / mode | mAP | Rank-1 |
|---|---:|---:|
| Market Efficient 160x80 | 55.899% | 78.088% |
| Market Full 256x128 | 59.199% | 79.543% |
| CUHK03 Detected Efficient 160x80 | 20.146% | 19.786% |
| CUHK03 Detected Full 256x128 | 19.920% | 19.857% |

Cross-dataset change, Market -> CUHK03 Detected:

- Efficient mAP: **-35.753 pp**
- Efficient Rank-1: **-58.302 pp**
- Full mAP: **-39.278 pp**
- Full Rank-1: **-59.686 pp**

This large absolute drop is evidence of a strong dataset/domain shift.

It does **not** invalidate the multi-resolution result. Instead, it identifies
a separate limitation: the frozen identity representation is not strongly
domain-general across Market-1501 and CUHK03-NP.

---

## Main finding

Experiment 2 gives two different findings:

### 1. Identity-domain generalization is limited

Absolute zero-shot retrieval performance falls substantially on CUHK03-NP.

This means that a model trained through the Market-1501 development procedure
does not automatically produce strong CUHK03 identity retrieval.

### 2. Resolution robustness generalizes

Despite the domain shift, the preregistered Detected benchmark shows:

- Efficient Rank-1: **19.786%**
- Full Rank-1: **19.857%**
- difference: **-0.071 pp**
- retention: **99.640%**

Therefore the relative robustness of the jointly trained 160x80 representation
persists across the unseen CUHK03 domain.

The separate controlled CPU benchmark previously measured a median OSNet
query-inference reduction of **22.973%** for the Efficient path. That
timing result is not a CUHK03-specific timing measurement and is reported
separately from this zero-shot accuracy experiment.

---

## Scientific interpretation

A defensible conclusion is:

> RAC-ReID's low-resolution representation remains close to its full-resolution
> counterpart under cross-dataset transfer, but the absolute identity retrieval
> accuracy reveals substantial domain shift.

This is more informative than reporting only the successful Market-1501
result because it exposes both a strength and a limitation of the system.

---

## Next experiment

Any training involving CUHK03 must be treated as a **new Experiment 3**.

Experiment 2 must remain frozen.

A suitable Experiment 3 question is:

> Can multi-domain or CUHK03-specific training improve absolute cross-domain
> retrieval accuracy while preserving the Efficient-vs-Full resolution
> robustness demonstrated in Experiments 1 and 2?

---

## Evidence

Preregistered protocol:

`results/cuhk03_zero_shot_protocol.json`

Machine-readable result:

`results/cuhk03_zero_shot_epoch8_final.json`

Result SHA-256:

`9cf940063b73af21b1b67d84b7ed590ba378f6ff5c89105ffabee806259a34d7`

Market-1501 final result:

`results/market1501_multires_epoch8_final.json`

Controlled efficiency benchmark:

`results/production_multires_benchmark_2026-11-23.json`
