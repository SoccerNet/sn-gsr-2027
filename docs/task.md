# Long Form Game State Reconstruction task specification

Status: draft; technical choices below are not final rules.

The 2027 task uses **one fully annotated match half, continuous for 45 minutes plus added time**. The earlier SoccerNet-Tracking and SoccerNet-GSR releases used 200 separate 30-second sequences across 11 games. The long-half annotation was created at that time and is now intended as the 2027 benchmark. GS-HOTA and its existing evaluation implementation remain the scoring basis.

Understanding a match requires following the game over time. This challenge focuses on reconstructing player positions and identities on the pitch across extended football sequences. Participants will tackle the challenge of maintaining a coherent representation of the game as players move, interact and become occluded.

## Decisions before launch

- [x] Evaluation unit: one continuously annotated match half (45 minutes plus added time).
- [x] Metric: prior SoccerNet GS-HOTA implementation.
- [ ] Confirm exact frame rate, timestamps, pitch coordinates, sequence ID and annotation packaging.
- [ ] Define identity persistence across camera cuts, substitutions and replays.
- [ ] Specify required player attributes and initialization rules.
- [ ] Verify the previous metric version and its behavior on this full-half sequence; no cross-sequence aggregation is needed if the final set is one half.
- [ ] Confirm offline versus causal processing and runtime limits.
