# RAC-ReID

Resource-Adaptive Cascaded Person Re-Identification.

## Research Goal

Build an efficient person re-identification system that dynamically
allocates computation according to query difficulty.

Instead of running the full ReID and vision-language pipeline for every
query, the system will progressively increase computation only when needed.

## Proposed Pipeline

1. Lightweight person detection
2. Adaptive person/context crop
3. Query difficulty estimation
4. Dynamic visual-token budget
5. Identity-anchor token preservation
6. Lightweight ReID embedding and gallery search
7. Confidence-based early exit
8. Progressive token restoration for ambiguous queries
9. Conditional VLLM semantic reasoning
10. Final adaptive reranking

## Main Efficiency Idea

Easy queries should use:

- fewer visual tokens
- fewer gallery candidates
- lightweight feature extraction
- no VLLM call

Difficult queries may progressively receive:

- additional tokens
- deeper visual processing
- more candidate comparisons
- VLLM semantic reasoning

## Planned Evaluation

Primary benchmark:

- Market-1501

Secondary benchmark:

- MSMT17

Metrics:

- mAP
- Rank-1
- Rank-5
- Rank-10
- average query latency
- average tokens processed
- VLLM invocation rate
- memory usage
- compute cost per query

## Status

Initial project scaffold.
No benchmark results have been produced yet.

## Implemented Milestone 1: Resource Budget Controller

The first implementation milestone introduces a configurable controller
that allocates computation according to query difficulty.

The controller currently:

- accepts six normalized query-quality signals
- calculates a weighted difficulty score
- classifies queries as easy, medium, or hard
- assigns crop, token, candidate, and encoder-depth budgets
- exits early when retrieval confidence is sufficient
- progressively upgrades ambiguous queries
- marks unresolved hard queries as eligible for VLLM reasoning

Current default budgets:

| Difficulty | Tokens retained | Top-K | Encoder depth | VLLM |
|---|---:|---:|---:|---|
| Easy | 30% | 5 | 1 | No |
| Medium | 55% | 10 | 2 | No |
| Hard | 80% | 20 | 3 | Eligible |

Run the controller demonstration with:

    PYTHONPATH=src .venv/bin/python scripts/controller_demo.py

Run the tests with:

    PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

This milestone implements the resource-allocation logic only. Real image
signals, person detection, ReID embeddings, token extraction, and VLLM
inference will be connected in later milestones.

## Planned Public Application

The application is planned for:

    https://reid.nabidanzum.com

No public service has been deployed yet.

## Implemented Milestone 2: Real Image-Derived Signals

RAC-ReID now extracts inexpensive measurements directly from real image
pixels using NumPy and Pillow.

Current image-derived signals include:

- original image width and height
- aspect ratio
- normalized brightness
- normalized contrast
- grayscale visual entropy
- Laplacian-based blur estimate
- edge density
- approximate foreground occupancy

The image is downscaled to a maximum working dimension before signal
calculation, keeping this preprocessing stage lightweight and CPU-friendly.

The foreground occupancy value is intentionally treated as an estimate.
It compares pixels against a border-derived background color and is not a
replacement for person segmentation.

Detector confidence, true person occlusion, and crowding are not inferred
from these pixel statistics. They remain separate detector-level signals
for a later milestone.

Project-local runtime dependencies are pinned in requirements.txt:

    numpy==2.5.5
    pillow==13.0.0

Run the image-signal demonstration with:

    PYTHONPATH=src .venv/bin/python scripts/image_signal_demo.py

## Implemented Milestones 3-4: Dynamic Tokens and Adaptive Retrieval

RAC-ReID now contains an executable resource-adaptive visual retrieval
pipeline.

The implementation performs the following stages:

1. Extract low-cost image quality signals.
2. Combine image signals with detector-provided evidence.
3. Normalize the query to a person-ReID image canvas.
4. Split the image into lightweight visual patch tokens.
5. Estimate patch saliency using texture, edge activity, and a center prior.
6. Preserve identity anchors from upper, middle, and lower body regions.
7. Retain a query-specific percentage of visual tokens.
8. Build a fixed-size descriptor from the retained tokens.
9. Search against precomputed full-budget gallery descriptors.
10. Exit early when retrieval confidence is sufficient.
11. Restore additional tokens for ambiguous queries.
12. Route unresolved hard queries toward the future VLLM stage.

The default 128 x 256 canvas with 16 x 16 patches produces 128 visual
tokens.

Current default token budgets are:

| Difficulty | Retention | Tokens |
|---|---:|---:|
| Easy | 30% | 39 / 128 |
| Medium | 55% | 71 / 128 |
| Hard | 80% | 103 / 128 |

The query image is tokenized once. Progressive refinement reuses the same
prepared token set and increases the retained subset instead of repeatedly
decoding and tokenizing the image.

Gallery descriptors are precomputed at the full lightweight budget, moving
gallery feature generation outside the query-time retrieval path.

Run the full pipeline demonstration with:

    PYTHONPATH=src .venv/bin/python scripts/pipeline_demo.py

The current descriptor is a lightweight engineering baseline used to test
the adaptive routing and token-budget methodology. A learned ReID backbone
will be introduced in a later milestone.

## Implemented Milestone 5: True Incremental Token Restoration

Progressive refinement now processes only newly restored visual tokens.

For the default 128-token query:

| Stage | Retained | Newly processed | Cumulative unique |
|---|---:|---:|---:|
| Easy | 39 | 39 | 39 |
| Medium | 71 | 32 | 71 |
| Hard | 103 | 32 | 103 |

The earlier prototype logically reused the token set but recomputed all
retained descriptor contributions at each stage.

That produced:

    39 + 71 + 103 = 213 repeated token operations

The incremental descriptor state now maintains running sums, squared sums,
regional sums, and counts.

The complete easy-to-hard route therefore processes:

    39 + 32 + 32 = 103 unique tokens

instead of 213 repeated token operations.

The final incremental descriptor is tested against a one-shot descriptor at
the same 80 percent budget.

## Implemented Milestone 6: Real Person Detection and Adaptive Crop

RAC-ReID now accepts full-scene images rather than requiring a manually
cropped person image.

A CPU-only Ultralytics YOLO nano detector finds person-class bounding boxes.
The detector is restricted to the person class and uses a maximum of two
PyTorch CPU threads to reduce interference with other VPS workloads.

The primary person is selected using detector confidence together with
relative bounding-box size.

Detector-derived query evidence now includes:

- detector confidence
- person bounding-box occupancy
- overlap-based occlusion proxy
- crowd-count and overlap-based crowding proxy

The occlusion and crowding values are explicitly treated as inexpensive
proxies rather than ground-truth annotations.

RAC-ReID first measures a tight person crop, calculates the initial resource
budget, then expands the final person crop using the budget-specific adaptive
context margin.

The current detector model is:

    yolo26n.pt

The model runs on CPU only.

## Implemented Milestone 7: Local Research Web Application

A functional upload/search interface is available through:

    scripts/run_web.sh

The application supports:

1. adding labeled gallery identities
2. automatic person detection and crop generation
3. uploading a full-scene query image
4. adaptive token processing
5. gallery ranking
6. display of token budgets, confidence, margins, and routing decisions

The application currently binds only to localhost. Public DNS and HTTPS will
be configured only after the existing VPS reverse-proxy routing is inspected.

## Implemented Milestone 8: Conditional Learned Person ReID

RAC-ReID now includes a pretrained OSNet x1.0 appearance encoder:

    osnet_x1_0_msmt17_combineall

The learned model runs on CPU.

The cascade is now:

    YOLO person detection
      -> adaptive context crop
      -> cheap adaptive visual-patch retrieval
      -> early exit when sufficient
      -> conditional OSNet reranking when ambiguous
      -> future semantic VLLM stage

OSNet is not loaded during normal server startup.

It is loaded lazily only when learned embedding work is required.

Gallery OSNet embeddings are precomputed when an administrator adds an
identity. Query-time learned reranking therefore computes only the query
embedding and reuses stored gallery embeddings.

The web interface reports whether OSNet was invoked, learned inference
latency, embedding dimensionality, cosine ranking, and top-1/top-2 margin.

OSNet cosine similarity is not treated as a calibrated probability.
Threshold calibration will be performed during benchmark evaluation.

### Headless server dependency

The upstream `reid` package declares `opencv-python`.

RAC-ReID runs on a headless VPS and uses `opencv-python-headless` instead.
Both provide the `cv2` Python API required by the project, while the
headless build avoids GUI/OpenGL dependencies.

Only one OpenCV wheel is installed in the RAC-ReID virtual environment.

### Evaluation boundary

`osnet_x1_0_msmt17_combineall` was trained with all MSMT17 identities,
including test identities.

It must not be reported as an unbiased MSMT17 evaluation checkpoint.

It can be used as a cross-domain pretrained baseline on Market-1501.

## Gallery administration

Permanent gallery modification requires a random server-side token stored at:

    .runtime/admin.token

The token is ignored by Git and has mode 600.

Public search remains available without the token.


## Implemented Milestone 9: Market-1501 Learned Baseline

The pretrained OSNet stage was evaluated using the complete standard
Market-1501 query/gallery protocol.

This is the learned cross-domain baseline, not the final adaptive RAC-ReID
experiment.

Observed results:

- mAP: 37.368%
- Rank-1: 66.330%
- Rank-5: 80.285%
- Rank-10: 85.748%
- mINP: 6.079%
- queries evaluated: 3,368
- gallery images: 19,732
- evaluation wall time: 1360.025 seconds
- distance: Euclidean
- device: CPU

Dataset archive SHA-256:

    416bb77b5a2449b32e936f623cbee58becf1a9e7e936f36380cb8f9ab928fe96

The result is committed at:

    results/market1501_osnet_crossdomain.json

The observed Rank-1 and mAP reproduce the independent documented
cross-domain baseline to within a small fraction of one percentage point.

This experiment establishes the full-compute learned OSNet baseline.

It does not yet demonstrate an RAC-ReID compute/accuracy advantage.
That requires the adaptive ablation experiment.

## Implemented Milestone 10: Teacher-Facing Adaptive Trace

The public research demo now exposes the actual backend resource-adaptive
execution path for each query.

The interface reports:

- YOLO/person preprocessing time
- detector confidence
- foreground ratio
- blur
- visual entropy
- occlusion proxy
- crowding proxy
- normalized difficulty score
- initial easy/medium/hard classification
- adaptive crop margin
- each progressive visual-patch proxy stage
- requested patch retention
- new and cumulative patches processed
- Top-K
- Top-1 score
- Top-1/Top-2 margin
- real per-stage processing time
- the resource-controller decision reason
- early exit or progressive refinement
- proxy patch saving versus full proxy processing
- conditional OSNet invocation
- OSNet query inference and cold-load timing
- complete backend query timing
- proxy and learned rankings

The current 30/55/80 percent resource budgets are handcrafted visual-patch
proxy budgets. They are not transformer or VLM token counts.

The controller action historically named `invoke_vllm` currently causes the
web application to escalate the ambiguous candidate set to OSNet learned
person-ReID reranking. A semantic VLLM is not connected yet.

The verified full-compute Market-1501 baseline is displayed on the public
research interface. The adaptive Market-1501 comparison remains pending the
next experiment.

## Implemented Milestone 11: Live Backend Progress Dashboard

RAC-ReID now exposes the actual server-side processing sequence while a query
is executing.

A search is started as an in-memory job and the browser polls a lightweight
progress endpoint. Backend components emit progress events as their operations
occur.

Visible operations include:

- upload acceptance
- worker queue/start
- YOLO person detection
- person localization
- tight person crop
- image-quality signal extraction
- difficulty scoring
- initial resource allocation
- adaptive context crop
- gallery-index loading/building
- visual-patch proxy tokenization
- identity-aware proxy patch pruning
- upper/middle/lower identity-anchor preservation
- progressive patch restoration
- gallery candidate search
- proxy candidate ranking
- Top-1/margin confidence routing
- early exit
- conditional OSNet query embedding
- learned embedding comparison
- final ranking
- completion or failure

The frontend reports the current backend operation, progress percentage,
elapsed time, Running/Completed/Failed state, and a chronological event log.

The final search result retains the complete execution timeline so the process
can be explained after the query has finished.

Progress is generated by backend callbacks and is not a simulated frontend
animation.

Expensive searches are serialized with a process-local lock because the
deployed YOLO/Torch models are shared inside the CPU web process.


## Implemented Milestone 12: Market-1501 Adaptive Ablation

The fixed RAC-ReID adaptive retrieval policy was evaluated on all 3,368
Market-1501 standard queries against the 19,732-image gallery.

Because Market-1501 supplies pedestrian crops, YOLO/person detection is
excluded from this retrieval-layer experiment so the scope matches the
full-compute OSNet retrieval baseline. Detector-derived difficulty values
are neutralized rather than fabricated.

No adaptive thresholds were tuned on the Market-1501 test queries.

### Accuracy

| Metric | Full OSNet | Adaptive RAC-ReID | Delta |
| --- | ---: | ---: | ---: |
| mAP | 37.368% | 2.015% | -35.353 pp |
| Rank-1 | 66.330% | 14.252% | -52.078 pp |
| Rank-5 | 80.285% | 16.776% | -63.510 pp |
| Rank-10 | 85.748% | 17.221% | -68.527 pp |
| mINP | 6.079% | 0.182% | -5.897 pp |

### Efficiency

- OSNet query invocation rate: 100.0%
- Query-side OSNet forward-pass reduction: 0.0%
- Early-exit rate: 0.0%
- Average proxy patches processed: 103.00
- Average proxy patch saving: 19.5%
- Average adaptive stages: 3.000
- Average proxy query processing: 0.0225 seconds
- Average OSNet query inference when invoked: 0.0429 seconds

Final routing distribution:

    {"invoke_vllm": 3368}

Initial-difficulty distribution:

    {"easy": 3368}

This experiment evaluates the currently implemented fixed policy as-is.
The result must be interpreted from the measured accuracy delta and compute
reduction; no efficiency advantage is claimed merely because fewer learned
forward passes were used.

The current visual-patch budgets remain handcrafted proxy budgets rather
than learned transformer token pruning.


## Implemented Milestone 13: Lightweight Learned ReID Baseline

The official MSMT17-combineall OSNet x0.25 checkpoint was added as the
candidate coarse learned stage for the redesigned RAC-ReID cascade.

The checkpoint is stored project-locally at runtime and is not committed as
a binary Git artifact.

Observed complete Market-1501 cross-domain results:

- mAP: 31.008%
- Rank-1: 59.768%
- Rank-5: 76.900%
- Rank-10: 83.106%
- mINP: 4.266%
- queries: 3,368
- gallery images: 19,732
- CPU evaluation wall time: 169.481 seconds

Model reference complexity:

- architecture: OSNet x0.25
- parameters: approximately 0.2 million
- reference compute: approximately 0.08 GFLOPs

This establishes a real learned coarse ReID stage before production routing
is changed.

The failed handcrafted proxy-first candidate shortlist remains historical
experimental evidence and will not be used as the identity candidate gate in
the redesigned production cascade.


## Implemented Milestone 14: Learned Confidence Controller

Simple Top-1/margin thresholds were rejected because they failed to
generalize across held-out Market-1501 training identities.

A learned confidence controller is now trained to predict whether the
OSNet x0.25 Top-1 identity can safely exit without invoking OSNet x1.0.

The controller is a compact logistic model over 18 statistics derived from
the x0.25 Top-10 similarity distribution.

All controller development uses Market-1501 training identities only.

Identity-disjoint partitions:

- controller training: PID modulo 5 in {0, 1, 2}
- threshold calibration: PID modulo 5 == 3
- final held-out validation: PID modulo 5 == 4

The Market-1501 test query/gallery split is untouched.

Selected controller:

- probability threshold: 0.929637
- calibration precision: 97.163%
- calibration coverage: 11.213%
- held-out validation precision: 95.968%
- held-out validation coverage: 14.253%
- projected learned-compute reduction: 6.090%

The projected reduction is based on the reference complexity values of
approximately 0.08 GFLOPs for OSNet x0.25 and 0.98 GFLOPs for OSNet x1.0.

This is still a training-data controller result. The complete learned
x0.25 -> x1.0 cascade must next be evaluated on the untouched Market-1501
test query/gallery protocol before production routing is changed.


## Current Research Update - 22 November 2026

The current research phase uses a jointly trained multi-resolution OSNet
backbone at 160x80 and 256x128.

The latest training completed 8 epochs and selected epoch 8.

Detailed teacher-facing update:

- `docs/PROJECT_UPDATE_2026-11-22.md`

Complete experiment history:

- `docs/EXPERIMENT_HISTORY.md`

Training protocol:

- `docs/MULTIRESOLUTION_TRAINING_PROTOCOL.md`

Machine-readable result:

- `results/market1501_multiresolution_training.json`

The standard Market-1501 query/gallery test remains untouched for the final
frozen-architecture evaluation.


## Teacher / evaluation entry point

For current training, held-out development, controlled benchmark,
real-browser sanity testing, and software verification, start here:

**[`docs/TEACHER_RESULTS_SUMMARY.md`](docs/TEACHER_RESULTS_SUMMARY.md)**

Key machine-readable evidence:

- [`results/market1501_multiresolution_training.json`](results/market1501_multiresolution_training.json)
- [`results/production_multires_benchmark_2026-11-23.json`](results/production_multires_benchmark_2026-11-23.json)
- [`results/project_test_summary_2026-11-23.json`](results/project_test_summary_2026-11-23.json)

The frozen epoch-8 model has now completed its one-time final standard
Market-1501 evaluation. See
[`docs/FINAL_MARKET1501_EVALUATION.md`](docs/FINAL_MARKET1501_EVALUATION.md)
and
[`results/market1501_multires_epoch8_final.json`](results/market1501_multires_epoch8_final.json).
No post-test model or threshold tuning is permitted.
