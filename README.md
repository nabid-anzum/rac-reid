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
