# Multi-Resolution Training Protocol

## Model

Base model:

osnet_x1_0_msmt17_combineall

Spatial budgets:

- low: 160x80
- full: 256x128

Frozen modules:

- conv1
- conv2

Trainable backbone parameters:

1,925,340

## Development data

Market-1501 training population:

- 12,936 images
- 751 identities

Split:

- PID % 5 in {0,1,2}: gradient training
- PID % 5 == 3: checkpoint selection
- PID % 5 == 4: held-out development validation

The standard Market-1501 query/gallery test split is not used.

## Optimization

- epochs: 8
- batch size: 16
- optimizer: AdamW
- backbone learning rate: 5e-5
- classifier learning rate: 5e-4
- weight decay: 5e-4
- seed: 29601

Loss:

L =
1.00 * low teacher cosine
+ 0.25 * full teacher cosine
+ 0.50 * relational similarity
+ 0.25 * cross-resolution consistency
+ 0.50 * identity classification

## Selected checkpoint

Epoch: 8

SHA-256:

d0570ff0fec0905b75bef1b90ec5147b96ac7b51d8ba9f30de5fa3f0c0757dec

## Scientific rule

Development identities may be used to train, calibrate and validate the
architecture according to the fixed identity-disjoint split.

The standard Market-1501 query/gallery test split must remain untouched until
the architecture and adaptive controller are frozen.
