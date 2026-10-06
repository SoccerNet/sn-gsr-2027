# Data

The planned dataset repository is [SoccerNet/SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027). It is intended to be private during preparation; creation and 2027 content are pending. Access conditions and split manifests are not finalized. Do not assume earlier-edition data is the official 2027 release.

The organizers have a continuously annotated half of one game, covering 45 minutes plus added time. Earlier SoccerNet-Tracking/GSR editions centered on 200 separate 30-second sequences drawn from 11 games. The new release must preserve contiguous frame IDs and player identities for the full half. The final allocation of train, validation, test, and challenge material has not been specified; the [split directories](../data/README.md) remain placeholders.

See [the explicit split layout](../data/README.md).

## Release requirements

Before opening submissions, freeze the release version, stable sample IDs, split boundaries and public versus private fields. Verify that final-ranking labels have not already been released. Keep ground truth outside public Git history.
