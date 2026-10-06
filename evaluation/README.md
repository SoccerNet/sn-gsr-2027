# Evaluation — awaiting 2027 specification

No official 2027 scorer is released in this repository yet. The task leads must confirm the protocol before implementation.

The [prior SoccerNet GSR repository](https://github.com/SoccerNet/sn-gamestate#evaluation) uses the [SoccerNet TrackEval fork](https://github.com/SoccerNet/sn-trackeval) and GS-HOTA, which matches pitch positions and identity attributes. This is the reference implementation to adapt once the long-form sequence and identity-continuity rules are fixed. Previous scores and tracker-state files are **not** 2027 baseline results.

## Required decisions

- Confirm sequence durations, frame timestamps and pitch coordinate convention.
- Define identity persistence across camera cuts, substitutions and replays.
- Specify required player attributes and initialization rules.
- Choose metric version and aggregation across sequences and matches.
- Confirm offline versus causal processing and runtime limits.

## Planned deliverables

A versioned local scoring command, environment, format example, malformed-submission checks, baseline predictions and score, and CodaBench integration if selected. Use the same [train/valid/test/challenge layout](../data/README.md) as the other 2027 repositories, with a challenge input manifest but no challenge GT in Git. Earlier-edition scores and formats are background only.

See [baseline references](../baselines/README.md) and [draft rules](../docs/rules.md).
