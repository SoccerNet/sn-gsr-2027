# Baseline

Existing SoccerNet Game State Reconstruction code, provided as background. Compatibility with long sequences and the 2027 protocol remains to be confirmed. The selected baseline version and runtime requirements will be published here.

Repository: [https://github.com/SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate).

1. Request access to [SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) after the task leads publish it. It is currently a private, empty placeholder. Then run `hf download SoccerNet/SN-GSR-2027 --repo-type dataset --local-dir ./SN-GSR-2027`.
2. Clone [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate), follow its environment and weight instructions, and run the previous-edition entry point `uv run tracklab -cn soccernet` from that repository root.
3. When the full-half data arrives, adapt the input/config to its frame and timestamp schema, retain identities throughout the half, and export the required JSON for the [evaluation adapter](../evaluation/README.md). This adaptation is pending and the previous command does not yet produce a 2027 submission.

That repository also links to prior validation, test, and challenge [tracker-state predictions on Zenodo](https://github.com/SoccerNet/sn-gamestate#downloading-the-baselines-tracker-states). These are useful for understanding the legacy output shape, but the 2027 long-form prediction files remain TBD and are not copied here.

Before the challenge release, record a tested commit or tag, environment, weights, inference command, output conversion, expected scores and compute requirements. External code remains in its upstream repository and under its own license.
