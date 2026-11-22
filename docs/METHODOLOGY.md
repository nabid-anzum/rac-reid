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

## Milestone 10: Observable Adaptive Decision Trace

The online RAC-ReID query path is now instrumented so that the resource
controller can be inspected during a live demonstration.

Each adaptive stage records its actual controller decision reason and measured
elapsed time in addition to the existing difficulty class, visual-patch proxy
budget, number of newly restored patches, cumulative patch count, Top-K,
Top-1 score, and Top-1/Top-2 margin.

The interface also reports the six query-difficulty inputs and component-level
timing for person preprocessing, gallery-index access, the adaptive proxy
pipeline, optional OSNet reranking, and total request processing.

The visual-patch budget remains an engineering proxy implemented by the
handcrafted patch encoder. It must not be represented as transformer-token
pruning or learned-backbone FLOP reduction.

The internal historical route name `invoke_vllm` currently represents an
escalation request; the deployed web application satisfies that request using
OSNet learned person-ReID reranking. No semantic VLLM is connected at this
milestone.

This observability milestone does not establish an efficiency or accuracy
advantage by itself. The next Market-1501 experiment must compare the adaptive
method against the verified full-compute OSNet baseline.

## Milestone 11: Live Execution Observability

The online research prototype now exposes a backend-generated execution trace
while a query is running.

Progress events are emitted at the boundaries of real processing operations,
including detector execution, person localization, crop preparation, signal
extraction, difficulty estimation, budget assignment, proxy tokenization,
identity-aware pruning, gallery retrieval, confidence routing, progressive
restoration, optional learned OSNet reranking, and final ranking.

The browser polls job state and displays the actual server event sequence,
current operation, elapsed time, and completion state.

The displayed progress percentage is an execution-stage indicator rather than
a measurement of neural-network completion percentage.

The patch-pruning terminology continues to refer only to the handcrafted
visual-patch proxy stage. It is not reported as transformer-token pruning.

Search workers are serialized within the web process to avoid concurrent use
of shared CPU YOLO/Torch model state.


## Milestone 12: Full Market-1501 Adaptive Retrieval Ablation

The current fixed resource-adaptive retrieval policy was evaluated on the
complete standard Market-1501 test split.

The full-compute learned reference remains:

- mAP: 37.368%
- Rank-1: 66.330%

The adaptive result is:

- mAP: 2.015% (-35.353 percentage points)
- Rank-1: 14.252% (-52.078 percentage points)
- Rank-5: 16.776% (-63.510 percentage points)
- Rank-10: 17.221% (-68.527 percentage points)
- mINP: 0.182% (-5.897 percentage points)

Measured resource behavior:

- OSNet invocation rate: 100.0%
- Query-side learned-forward reduction: 0.0%
- Early-exit rate: 0.0%
- Average proxy patch saving: 19.5%
- Average adaptive stages: 3.000

Market-1501 already supplies cropped pedestrians. Therefore detector execution
is excluded and detector-derived controller evidence is neutralized for this
retrieval-layer comparison.

OSNet gallery embeddings are treated as offline gallery precomputation, which
matches the deployed system's gallery workflow. Only adaptively invoked query
embeddings count as query-side learned forward passes.

For learned escalation, the current production design reranks the proxy
candidate set with normalized OSNet cosine similarity and keeps the remaining
gallery in proxy order.

This experiment uses the existing policy without tuning thresholds on the
Market-1501 test queries.

The visual-patch savings are proxy-compute measurements and must not be
reported as transformer-token or neural-FLOP savings.


## Milestone 13: Lightweight Learned Coarse Stage

The official OSNet x0.25 MSMT17-combineall checkpoint was validated locally
and evaluated on the same complete Market-1501 protocol used by the existing
OSNet x1.0 reference.

Observed results:

- mAP: 31.008%
- Rank-1: 59.768%
- Rank-5: 76.900%
- Rank-10: 83.106%
- mINP: 4.266%

The intended redesigned cascade is:

1. OSNet x0.25 performs the inexpensive learned full-gallery retrieval.
2. A confidence controller decides whether the x0.25 result is sufficient.
3. Confident queries exit immediately.
4. Ambiguous queries invoke OSNet x1.0.
5. The heavy model searches the full learned gallery rather than a
   handcrafted proxy shortlist.

This avoids the failure mode measured in the previous proxy-first ablation,
where the handcrafted candidate restriction reduced accuracy while still
invoking the heavy learned model for every test query.

Thresholds for learned early exit must be calibrated on training data rather
than tuned on Market-1501 test queries.


## Milestone 14: Learned Confidence Controller

The rectangular Top-1/margin confidence rule did not generalize sufficiently
to unseen training identities, so it was rejected.

The replacement confidence controller uses logistic regression over 18
features derived from the lightweight OSNet x0.25 Top-10 similarity
distribution.

The controller is not a ReID model. It predicts whether the current x0.25
Top-1 identity is reliable enough to skip OSNet x1.0.

Development uses three identity-disjoint partitions of Market-1501 training
data:

- model fitting: PID modulo 5 in {0, 1, 2}
- probability-threshold calibration: PID modulo 5 == 3
- held-out validation: PID modulo 5 == 4

The selected operating point achieved:

- calibration precision: 97.163%
- calibration early-exit coverage: 11.213%
- held-out validation precision: 95.968%
- held-out validation early-exit coverage: 14.253%

The theoretical learned-model compute reduction at this validation exit rate
is 6.090%, before measured runtime
overheads.

No Market-1501 test query or gallery label was used to train the controller
or choose its threshold.

The next required experiment is a complete untouched-test evaluation of the
learned coarse-to-fine cascade.


## Research Traceability - 22 November 2026

The repository now retains both successful and unsuccessful development
experiments.

Negative results are preserved because they justify architecture changes and
prevent post-hoc threshold manipulation.

The current direction is joint cross-resolution OSNet training at 160x80 and
256x128 with conv1 and conv2 frozen.

See:

- `docs/PROJECT_UPDATE_2026-11-22.md`
- `docs/EXPERIMENT_HISTORY.md`
- `docs/MULTIRESOLUTION_TRAINING_PROTOCOL.md`
- `results/market1501_multiresolution_training.json`
- `results/runtime_history/`

Large datasets, pretrained-model caches, learned checkpoint binaries and
temporary NPY/NPZ feature caches remain outside GitHub, while the repository
stores their scientific role, configuration and hashes.


## Production multi-resolution inference architecture

The production-oriented learned ReID design uses one jointly trained OSNet x1.0
backbone.

Gallery enrollment:

1. detect and crop the person
2. preprocess the crop
3. run the trained backbone at 256x128
4. store one normalized 512-D embedding

Efficient query:

1. detect and crop the person
2. run the trained backbone at 160x80
3. compare the resulting 512-D embedding against all stored 256x128 gallery
   embeddings

Full-quality query:

1. use the same person crop
2. run the same trained backbone at 256x128
3. compare against the same stored gallery embeddings

The previously evaluated learned routers are not part of production because
they failed the predetermined held-out validation gates.

The full learned gallery is searched directly; proxy Top-K pruning is not used
to restrict the learned candidate population.
