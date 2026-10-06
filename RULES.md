# Long Form Game State Reconstruction challenge rules

📅 **Submission deadline:** April 25, 2027, 23:59 Anywhere on Earth (AoE; UTC−12).

Reconstruct athlete positions and identities over one continuously annotated match half, including added time. Maintain coherent identities over the full sequence. The ranking metric is **GS-HOTA**, using the same SoccerNet Game State Reconstruction metric as earlier editions.

Use the [2025 GSR training, validation, and test splits](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) for development. The new half in [SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) is the sole 2027 challenge set.

Respect the [dataset access terms](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027). Challenge annotations are withheld for evaluation. The [evaluation guide](evaluation/README.md) documents the SoccerNet metric implementation and its input format.
