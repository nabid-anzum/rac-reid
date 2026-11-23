# Real-world browser E2E sanity test — 2026-11-23

## Purpose

This records a live browser query using a photograph different from the
enrolled gallery photograph.

The live gallery contained two identities: `nabid` and `shetu`.

This is therefore a functional end-to-end sanity test, not a statistical
accuracy benchmark.

## Live pipeline

`upload -> YOLO person detection -> fixed 10% crop -> resize -> trained OSNet -> normalized 512-D embedding -> complete gallery comparison -> cosine ranking`

No historical proxy pruning, difficulty scoring, or adaptive resource
allocation was used.

## Warm Efficient — 160x80

| Metric | Result |
|---|---:|
| Rank 1 | `nabid` |
| `nabid` cosine | 0.5249 |
| `shetu` cosine | 0.4128 |
| Top-1 / Top-2 margin | 0.1121 |
| OSNet query inference | 0.0478 s |
| Total backend search | 0.1738 s |

## Warm Full quality — 256x128

| Metric | Result |
|---|---:|
| Rank 1 | `nabid` |
| `nabid` cosine | 0.5962 |
| `shetu` cosine | 0.4470 |
| Top-1 / Top-2 margin | 0.1492 |
| OSNet query inference | 0.1024 s |
| Total backend search | 0.4584 s |

Both modes correctly ranked `nabid` first.

The first Efficient browser request was much slower because of cold/runtime
effects. Individual browser requests are therefore not used as the primary
efficiency claim.

The reproducible performance evidence is stored in:

`results/production_multires_benchmark_2026-11-23.json`
