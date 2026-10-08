# Long Form Game State Reconstruction challenge rules

📅 **Submission deadline:** April 25, 2027, 23:59 Anywhere on Earth (AoE; UTC−12).

Reconstruct athlete positions and identities over one continuously annotated match half, including added time. Keep each athlete's `track_id` coherent across the full sequence. Predict pitch position, role, team affiliation, and jersey number using the SoccerNet Game State Reconstruction format. The ranking metric is **GS-HOTA**, using the same evaluator as earlier editions.

Use the [2025 GSR training, validation, and test splits](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) for development. The new half in [SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) is the sole 2027 challenge set.

Submit a ZIP containing one `<sequence>.json` prediction file for the challenge half. The JSON has a top-level `predictions` list; each detection carries its frame and track IDs, pitch position, role, team and jersey attributes, and confidence. Use JSON `null` for attributes that do not apply under the SoccerNet GSR annotation convention. The [evaluation guide](evaluation/README.md) gives the field names and local scoring command.

Respect the [dataset access terms](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027). Challenge annotations are withheld for evaluation.
