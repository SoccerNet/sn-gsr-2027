# SoccerNet Challenge 2027 — Long Form Game State Reconstruction

📢 **Long Form Game State Reconstruction Challenge!** 🚀

Understanding a match requires following the game over time. This challenge focuses on reconstructing player positions and identities on the pitch across extended football sequences. Participants will tackle the challenge of maintaining a coherent representation of the game as players move, interact and become occluded.

⏳ **Status:** In preparation — submissions are not open.  
📅 **Submission deadline:** April 25, 2027; cutoff time and timezone to be announced.  
👥 **Task lead(s):** Anthony  
🤝 **Sponsor:** To be announced

🏠 [Challenge website](https://www.soccer-net.org/challenges/2027) / 🗂️ [Planned private dataset](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) / 💻 [Prior baseline repository](https://github.com/SoccerNet/sn-gamestate) / 📚 [Rules](docs/rules.md) / 📊 [Evaluation](evaluation/README.md)

## Task

[Task specification and open decisions](docs/task.md) describes the current scope. This repository will hold the versioned evaluation code and submission instructions for the 2027 challenge. Baselines may live here or in an external repository linked below.

## Data

The 2027 dataset release has not been announced. The data guide will be updated when the task leads confirm the release.

See [the data guide](docs/data.md) for release status and split details. Dataset availability does not mean that the challenge is open. Private evaluation labels are not included in this repository.

## Baseline

Existing SoccerNet Game State Reconstruction code, provided as background. Compatibility with long sequences and the 2027 protocol remains to be confirmed. The selected baseline version and runtime requirements will be published here.

See [baseline instructions](baselines/README.md). Existing research baselines must be checked against the final challenge format before their scores are advertised as 2027 reference results.

## Evaluation

The 2027 evaluation code and official protocol are awaiting confirmation from the task leads.

See [evaluation instructions](evaluation/README.md) for current code, submission formats and remaining decisions. CodaBench test and challenge links will be added after the servers have been validated.

## Release readiness

| Item | Current state |
| --- | --- |
| Task definition | Long-form protocol pending lead confirmation |
| Data and splits | 2027 private dataset and splits pending |
| Baseline | Prior code linked; 2027 predictions pending |
| Evaluation | Sequence metric and adapter pending |
| Benchmark | CodaBench server pending |
| Sponsor | Not confirmed |

Each task launches independently after its rules, data, baseline, evaluator and benchmark are verified. See the [task specification](docs/task.md) for remaining technical decisions.

## Getting started

1. Read the [task specification](docs/task.md) and [draft rules](docs/rules.md).
2. Follow the [data access guide](docs/data.md).
3. Consult the [baseline instructions](baselines/README.md).
4. Check the [evaluation status](evaluation/README.md) before generating submissions.

## Updates and support

Each task launches independently once ready. Follow the [challenge website](https://www.soccer-net.org/challenges/2027) and [SoccerNet Discord](https://discord.gg/cPbqf2mAwF). Use repository issues for reproducible code problems; do not post access tokens, restricted data or private labels.

## Citation and licensing

Please cite [SoccerNet-Tracking (CVPR Workshops 2022)](https://mlanthology.org/cvprw/2022/cioppa2022cvprw-soccernettracking/) and [SoccerNet Game State Reconstruction (CVPR Workshops 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.html). Use `\cite{Cioppa_2022_CVPR,Somers_2024_CVPR}` in LaTeX; conference BibTeX is in [CITATION.bib](CITATION.bib). The 2027 long-form challenge citation and code license will be confirmed before release.
