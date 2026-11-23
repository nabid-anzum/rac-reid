# Experiment 3 — CUHK03-NP Supervised Post-Training Benchmark

## Research question

Can CUHK03-specific supervised multi-resolution training improve the poor
absolute CUHK03 retrieval accuracy observed during zero-shot transfer while
preserving the Efficient-vs-Full resolution relationship?

## Frozen checkpoint

Selected epoch:

**3**

Checkpoint SHA-256:

`f879574a3cf21f9b106f51fbd2bef692e6044761a3bb17280c5da3a37fd1a399`

No training or tuning occurred during this benchmark.

Post-benchmark tuning is prohibited.

---

## Benchmark implementation

Primary:

**CUHK03-NP Detected**

Secondary:

**CUHK03-NP Labeled**

The Experiment-2 split parsing and metric implementation were reused.

A benchmark-only checkpoint adapter was necessary because the production
`MultiResolutionOSNetReIDEncoder` intentionally accepts only the original
epoch-8 production checkpoint.

The production guard was not modified.

Before the benchmark, the adapter was tested against the original epoch-8
checkpoint and reproduced both Efficient and Full embeddings exactly.

Fixed benchmark evaluator commit:

`e1dc17a30a2671aad3756433726d05080d531e67`

Evaluator SHA-256:

`d7f0941d59c0d5a35f293961522e89dace56df9feade09a63f773f0e8f71a7a2`

---

## Primary result — Detected

| Metric | Experiment 2 zero-shot Efficient | Experiment 3 supervised Efficient | Experiment 2 zero-shot Full | Experiment 3 supervised Full |
|---|---:|---:|---:|---:|
| mAP | 20.146% | **28.256%** | 19.920% | **29.952%** |
| Rank-1 | 19.786% | **27.929%** | 19.857% | **29.286%** |
| Rank-5 | 36.714% | 46.000% | 36.714% | 48.357% |
| Rank-10 | 44.214% | 55.714% | 44.286% | 58.786% |
| mINP | 12.513% | 18.932% | 12.683% | 20.932% |

Supervised improvement over zero-shot:

- Efficient mAP: **+8.110 pp**
- Efficient Rank-1: **+8.143 pp**
- Full mAP: **+10.031 pp**
- Full Rank-1: **+9.429 pp**

Experiment-3 Efficient Rank-1 retention:

**95.366%**

Experiment-2 Efficient-vs-Full Rank-1 difference:

**-0.071 pp**

Experiment-3 Efficient-vs-Full Rank-1 difference:

**-1.357 pp**

The supervised model therefore improved absolute accuracy substantially, but
Full resolution benefited slightly more than Efficient on the primary
Detected benchmark.

---

## Secondary result — Labeled

| Metric | Zero-shot Efficient | Supervised Efficient | Zero-shot Full | Supervised Full |
|---|---:|---:|---:|---:|
| mAP | 20.550% | **28.646%** | 20.145% | **30.274%** |
| Rank-1 | 20.214% | **28.929%** | 19.357% | **29.071%** |

Improvement over zero-shot:

- Efficient Rank-1: **+8.714 pp**
- Full Rank-1: **+9.714 pp**

Efficient Rank-1 retention:

**99.509%**

The secondary Labeled benchmark shows almost identical Efficient and Full
Rank-1 performance after supervised adaptation.

---

## Comparison with Experiment 1

Market-1501 final:

- Efficient mAP: **55.899%**
- Efficient Rank-1: **78.088%**
- Full mAP: **59.199%**
- Full Rank-1: **79.543%**

CUHK03 supervised Detected remains substantially below these Market-1501
values.

Therefore CUHK03-specific training **partially repairs** the absolute
cross-dataset problem identified in Experiment 2, but it does not eliminate
the broader dataset/domain gap.

---

## Main findings

### Finding 1 — supervised adaptation helps

Experiment 2 Detected Rank-1:

- Efficient: **19.786%**
- Full: **19.857%**

Experiment 3 Detected Rank-1:

- Efficient: **27.929%**
- Full: **29.286%**

This is clear evidence that CUHK03-specific supervised training improves
CUHK03 identity retrieval.

### Finding 2 — resolution robustness remains useful

Efficient retains **95.366%** of
Full Rank-1 on the primary Detected benchmark.

The gap is larger than in the zero-shot experiment, meaning the supervised
adaptation benefited Full resolution somewhat more strongly.

### Finding 3 — the domain problem is not solved completely

Absolute CUHK03 accuracy remains considerably below the final Market-1501
accuracy.

Thus the experiment supports **partial domain adaptation**, not complete
domain invariance.

### Finding 4 — secondary annotation style remains robust

On Labeled, Efficient retains
**99.509%** of Full Rank-1.

---

## Important limitation

CUHK03 query/gallery images were already observed during Experiment 2.

Therefore Experiment 3 is a standardized post-training benchmark rather than
a pristine unseen final test.

The complete Experiment-3 training procedure and selected epoch were frozen
before this benchmark, and no benchmark-driven tuning is permitted.

---

## Reproducibility evidence

Experiment-3 protocol:

`configs/cuhk03_supervised_multires_experiment.json`

Trainer:

`scripts/train_cuhk03_supervised_multires.py`

Training result:

`results/cuhk03_supervised_training_result.json`

Fixed benchmark evaluator:

`scripts/evaluate_cuhk03_supervised_benchmark.py`

Benchmark start marker:

`results/cuhk03_supervised_benchmark_started.json`

Benchmark completion marker:

`results/cuhk03_supervised_benchmark_complete.json`

Primary raw metric record:

`results/cuhk03_supervised_detected_metrics.json`

Secondary raw metric record:

`results/cuhk03_supervised_labeled_metrics.json`

Final benchmark result:

`results/cuhk03_supervised_benchmark_final.json`

Final benchmark SHA-256:

`55430f6e21358abc6d80f8987558302a687e18bf3c05ac569a185b5ac0a401c5`
