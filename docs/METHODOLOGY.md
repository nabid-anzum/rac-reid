# Proposed Methodology

## RAC-ReID

RAC-ReID is a resource-adaptive person re-identification pipeline.

The central hypothesis is that every query does not require the same
amount of computation.

The system begins with inexpensive processing and progressively allocates
more resources only when retrieval confidence is insufficient.

### Stages

1. Detect and crop the target pedestrian using a lightweight detector.
2. Estimate query difficulty from inexpensive visual signals.
3. Assign a dynamic visual-token budget.
4. Preserve important identity-related body regions while pruning
   redundant visual information.
5. Generate a lightweight visual embedding.
6. Search precomputed gallery embeddings.
7. Exit immediately when retrieval confidence is high.
8. Restore additional tokens and perform deeper processing when uncertain.
9. Invoke a VLLM only for difficult cases.
10. Fuse visual and semantic evidence for the final ranking.

### Intended Contribution

The proposed contribution is a unified resource controller that dynamically
adjusts:

- crop context
- token budget
- candidate count
- visual-processing depth
- VLLM invocation

according to query difficulty and retrieval confidence.

## Milestone 1 Algorithm Specification

The first implemented component is the Resource Budget Controller.

For each query, the controller calculates:

    D =
    0.20 * (1 - detector confidence)
    + 0.15 * (1 - foreground ratio)
    + 0.15 * blur
    + 0.15 * visual entropy
    + 0.25 * occlusion
    + 0.10 * crowding

The initial difficulty policy is:

- D less than or equal to 0.35: easy query
- D greater than 0.35 and less than or equal to 0.65: medium query
- D greater than 0.65: hard query

The resulting resource plans are:

| Level | Crop margin | Token retention | Top-K | Depth | VLLM eligible |
|---|---:|---:|---:|---:|---|
| Easy | 5% | 30% | 5 | 1 | No |
| Medium | 10% | 55% | 10 | 2 | No |
| Hard | 15% | 80% | 20 | 3 | Yes |

After visual retrieval, the controller exits early when:

    Top-1 score >= 0.80

and:

    Top-1 score - Top-2 score >= 0.12

Otherwise, computation is increased progressively:

    Easy -> Medium -> Hard -> Conditional VLLM

These values are initial research parameters. They will later be calibrated
through experiments on Market-1501 rather than treated as final constants.

## Milestone 2: Image-Derived Query Evidence

The second implementation milestone introduces a CPU-friendly image signal
extractor.

Before expensive ReID inference, a query image can now be reduced to a
small set of measurable visual properties:

1. Brightness
2. Contrast
3. Grayscale entropy
4. Blur estimate
5. Edge density
6. Approximate foreground occupancy
7. Image geometry

For efficiency, the analysis copy is resized so its longest dimension is
at most 512 pixels. The original image dimensions are preserved separately.

Blur is represented as a normalized inverse function of Laplacian variance.
A larger value therefore indicates a blurrier or less locally detailed
image, while a smaller value indicates stronger high-frequency detail.

Visual entropy is calculated from the normalized 256-bin grayscale
histogram.

Edge density uses neighboring grayscale intensity differences and measures
the fraction that exceed a configurable threshold.

Foreground occupancy is currently a lightweight heuristic. The median
color of the image border is treated as an approximate background color,
and pixels sufficiently different from that color are counted as probable
foreground.

This foreground estimate must not be interpreted as person segmentation.

The following signals still require detector-level evidence and are not
fabricated from raw pixel statistics:

- person detector confidence
- true person bounding-box occupancy
- person occlusion
- number of nearby people or crowding

The next integration stage will combine image-derived evidence with
detector-derived evidence before the Resource Budget Controller chooses
the query compute budget.

## Milestones 3-4: Identity-Aware Dynamic Token Retrieval

The resource controller is now connected to executable visual token
processing and gallery retrieval.

### Patch Tokenization

Each normalized 128 x 256 query image is divided into 16 x 16 patches,
producing 128 initial visual tokens.

Each patch currently stores lightweight visual statistics:

- mean RGB values
- RGB standard deviation
- luminance
- local edge activity

### Patch Saliency

Patch importance is approximated from:

- local texture
- edge activity
- a weak horizontal center prior

This gives the adaptive controller an executable token-pruning mechanism
without requiring a GPU.

### Identity Anchors

Pure saliency pruning can remove visually quiet but identity-relevant body
areas.

RAC-ReID therefore forces representation from three vertical regions:

- upper region
- middle region
- lower region

At least one token from each region is retained before the remaining token
budget is filled according to saliency.

### Progressive Token Restoration

The complete token set is generated once for a query.

Depending on retrieval difficulty, RAC-ReID progressively increases the
retained subset:

    30 percent -> 55 percent -> 80 percent

This allows easy queries to consume fewer visual tokens while ambiguous
queries progressively receive additional visual evidence.

### Budgeted Descriptor

The current CPU baseline aggregates retained patch features into a
fixed-dimensional visual descriptor using:

- global retained-token mean
- global retained-token standard deviation
- upper-region mean
- middle-region mean
- lower-region mean

Gallery descriptors are generated at full lightweight budget and can be
precomputed before query-time search.

### Retrieval and Early Exit

Gallery candidates are ranked using a normalized exponential similarity
derived from descriptor RMS distance.

The Resource Budget Controller then uses the existing Top-1 score and
Top-1/Top-2 margin rules.

A confident result exits immediately.

An ambiguous result receives a larger token budget.

A hard result that remains ambiguous becomes eligible for the future VLLM
semantic reranking stage.

### Current Research Boundary

The handcrafted visual descriptor is not presented as the final learned
person ReID model.

Its role is to make the proposed resource-adaptive orchestration fully
executable and measurable before integrating a learned ReID backbone such
as OSNet.

## Milestone 5: True Incremental Progressive Computation

Progressive token restoration now reuses computation, not only token IDs.

A naive staged implementation would calculate:

    39 + 71 + 103 = 213 token operations

when processing easy, medium, and hard stages.

RAC-ReID instead maintains an incremental descriptor state.

Stage 1:

    39 new tokens
    cumulative = 39

Stage 2:

    32 new tokens
    cumulative = 71

Stage 3:

    32 new tokens
    cumulative = 103

Therefore the complete progressive route evaluates 103 unique tokens rather
than 213 repeated token operations.

The state maintains feature sums, squared sums, vertical-region sums, and
counts so that previously processed token contributions do not need to be
recalculated.

Automated tests verify that the final incremental descriptor matches the
equivalent one-shot descriptor at the same token budget.

## Milestone 6: Lightweight Person Detection and Adaptive Context Crop

The prototype now includes an actual person-detection stage.

A nano YOLO detector is executed on CPU and restricted to the person class.

For every detected person, RAC-ReID records its bounding box and detection
confidence.

The current primary-person selection score combines confidence with relative
bounding-box area, favoring detections that are both reliable and visually
prominent.

### Detector evidence

Detector confidence is used directly.

Foreground occupancy is the selected person's bounding-box area divided by
the full image area.

Occlusion is represented by a low-cost proxy combining:

- overlap between the selected person and other detected people
- whether the selected bounding box touches image boundaries

Crowding is represented by a proxy combining:

- number of additional detected people
- overlap with the selected person

These two values are research heuristics, not ground-truth occlusion or
crowding labels.

### Adaptive crop

RAC-ReID initially generates a tight person crop.

Cheap image-quality measurements and detector evidence are then passed to
the Resource Budget Controller.

The selected difficulty plan determines the context margin:

- easy: 5 percent
- medium: 10 percent
- hard: 15 percent

A second crop retains the person plus the selected amount of surrounding
context.

This implements the adaptive-context-crop stage of the proposed methodology.

## Milestone 7: Interactive Retrieval Demo

The current system includes a local web interface that can build a gallery
from uploaded person images and search it using a full-scene query image.

Gallery images and uploaded query images are automatically passed through
the person detector and crop stage.

The web interface exposes the resource-adaptive process, including:

- detected person count
- detector confidence
- initial difficulty
- adaptive crop margin
- per-stage token budget
- newly processed tokens
- cumulative unique tokens
- retrieval score
- confidence margin
- final routing action

The current server is intentionally bound to localhost until reverse-proxy
and HTTPS deployment are completed.

## Milestone 8: Conditional Learned ReID

A real learned appearance encoder is now present in the resource-adaptive
cascade.

The lightweight patch descriptor remains the cheapest retrieval stage.

When the cheap stage reaches its ambiguous escalation path, RAC-ReID invokes
OSNet x1.0 and reranks the selected candidates using normalized
512-dimensional appearance embeddings.

Easy queries that exit earlier avoid the OSNet query forward pass.

### Gallery precomputation

Learned gallery embeddings are generated when identities are added and are
persisted for reuse.

At conditional query time only the query embedding is computed.

This separates offline gallery computation from online query computation.

### Learned metrics recorded

The system records:

- whether OSNet was invoked
- cold model-loading latency
- query embedding latency
- embedding dimensionality
- learned cosine ranking
- learned top-1 score
- learned top-1/top-2 margin
- cheap retrieval score for comparison

No learned similarity threshold is claimed as final before experimental
calibration.

### Scientific boundary

The adaptive visual "tokens" in the current prototype are pre-encoder image
patches.

They are not OSNet internal tokens and are not VLM transformer tokens.

Conditional OSNet demonstrates adaptive backbone invocation.

Actual model-level visual-token pruning remains a later experimental stage.

### Dataset boundary

The pretrained `osnet_x1_0_msmt17_combineall` checkpoint contains all
MSMT17 identities.

It is therefore unsuitable for unbiased evaluation on MSMT17.

Market-1501 can be used for cross-domain evaluation of this checkpoint.


## Milestone 9: Standard Market-1501 Baseline

The complete Market-1501 retrieval protocol was executed using the
cross-domain pretrained OSNet x1.0 encoder.

Results:

- mAP: 37.368%
- Rank-1: 66.330%
- Rank-5: 80.285%
- Rank-10: 85.748%
- mINP: 6.079%
- scored queries: 3,368
- gallery images: 19,732
- evaluation wall time: 1360.025 seconds

This measurement is designated the full learned baseline.

The next experiment must compare this full-compute baseline against the
resource-adaptive RAC-ReID cascade using accuracy, learned-model invocation
rate, patch budget, and latency.

No adaptive-compute improvement is claimed from this baseline alone.
