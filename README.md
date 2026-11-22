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
