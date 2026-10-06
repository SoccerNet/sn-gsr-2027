# Long Form Game State Reconstruction task specification

Status: draft; technical choices below are not final rules.

Understanding a match requires following the game over time. This challenge focuses on reconstructing player positions and identities on the pitch across extended football sequences. Participants will tackle the challenge of maintaining a coherent representation of the game as players move, interact and become occluded.

## Decisions before launch

- [ ] Confirm sequence durations, frame timestamps and pitch coordinate convention.
- [ ] Define identity persistence across camera cuts, substitutions and replays.
- [ ] Specify required player attributes and initialization rules.
- [ ] Choose metric version and aggregation across sequences and matches.
- [ ] Confirm offline versus causal processing and runtime limits.
