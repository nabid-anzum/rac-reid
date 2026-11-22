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
