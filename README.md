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
