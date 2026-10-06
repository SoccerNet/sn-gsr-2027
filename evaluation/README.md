# Evaluation — awaiting 2027 specification

No official 2027 scorer is released in this repository yet. The task leads must confirm the protocol before implementation.

## Required decisions

- Confirm sequence durations, frame timestamps and pitch coordinate convention.
- Define identity persistence across camera cuts, substitutions and replays.
- Specify required player attributes and initialization rules.
- Choose metric version and aggregation across sequences and matches.
- Confirm offline versus causal processing and runtime limits.

## Planned deliverables

A versioned local scoring command, environment, synthetic format example, malformed-submission checks, baseline score, and CodaBench integration if selected. Earlier-edition scores and formats are background only.

See [baseline references](../baselines/README.md) and [draft rules](../docs/rules.md).
