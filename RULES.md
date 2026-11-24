# Long Form Game State Reconstruction challenge rules

📅 **Submission deadline:** April 25, 2027, 23:59 Anywhere on Earth (AoE; UTC−12).

Reconstruct athlete positions and identities over one continuously annotated match half, including added time. Keep each athlete's `track_id` coherent across the full sequence. Predict pitch position, role, team affiliation, and jersey number using the SoccerNet Game State Reconstruction format. The ranking metric is **GS-HOTA**, using the same evaluator as earlier editions.

Use the [2025 GSR training, validation, and test splits](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) for development. The new half in [SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) is the sole 2027 challenge set.

Submit a ZIP containing one `<sequence>.json` prediction file for the challenge half. The JSON has a top-level `predictions` list; each detection carries its frame and track IDs, pitch position, role, team and jersey attributes, and an optional confidence (1 when omitted). Use JSON `null` for attributes that do not apply under the SoccerNet GSR annotation convention. The [evaluation guide](evaluation/README.md) gives the field names and local scoring command.

Each team may submit **once per day**. A team's best valid submission appears on the leaderboard.

## Who can participate

- Any individual can participate in the challenge, except the organizers.
- Participants are encouraged to form teams; a team can have one or more members.
- An individual associated with multiple teams, or a team with multiple accounts, will be disqualified.

## Permitted inputs

For each half, use only the frames of that half. A method may use every frame of the half, in any order (offline processing is allowed): the long-form setting precisely rewards consistent reasoning over the whole half.

Do not use the annotations, predictions or frames of other halves (including the other half of the same match) to make a prediction. Do not use match metadata (team names, line-ups, rosters, kit descriptions, event data) or look up information about the challenge match in external sources.

## Automatic processing

Predictions must be produced by a fully automatic method. Manual annotation, correction or selection of the outputs of the challenge half is not allowed, including the manual tuning of parameters per half.

## Training data

Any external training data, pretrained model, and local or remote model/API service may be used, provided every resource is fully disclosed. External datasets, model weights, and model/API implementations must be publicly available under open licenses, with the code and configuration needed to reproduce the submitted predictions. Teams must report the source URLs, licenses, versions or revisions, checkpoints, prompts where applicable, and inference settings. Closed-source or API-only systems that cannot be independently reproduced are not eligible.

**Private datasets and additional manual annotations are not allowed.** Teams using any kind of custom dataset, including additional manual annotations, will be disqualified. A dataset is *private* if it is not accessible to all participants under the same conditions: older SoccerNet data are not private, but manually adding or modifying annotations (e.g. bounding boxes or corrected labels) on them during the challenge creates a disparity and makes them private, unless those modifications are shared with the community in advance.

Any new data must be published or shared with all participants on [Discord](https://discord.gg/cPbqf2mAwF) **at least one month before the submission deadline**, so that all teams can retrain their methods on equal footing.

Public data may be used in creative and fully automatic ways (for instance an algorithm deriving additional features or labels from existing data), as long as no manual annotation is involved and the method is described in the technical report. Label corrections of SoccerNet datasets may be proposed before or after the challenge, for inclusion in future releases; they must not be made during the challenge.

## Final submissions and ranking

The winner is the team with the highest **GS-HOTA** on the **challenge** set. To be eligible, a team must provide:

- a short **report** describing the details of the methodology, the data and models used, and the compute required to process one half (CVPR format, **2 pages maximum**);
- a short **demo video** on the challenge half showcasing the solution;
- the code, weights and configuration needed to reproduce the predictions.

Organizers will review the full inference pipeline before confirming the final rankings.

Respect the [dataset access terms](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) and do not redistribute the video frames or the annotations. Challenge annotations are withheld for evaluation.
