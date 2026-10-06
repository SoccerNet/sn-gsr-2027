# SoccerNet Challenge 2027 — Long Form Game State Reconstruction (GSR)

📢 **Long Form Game State Reconstruction (GSR) Challenge!** 🚀

⏳ **Status:** In preparation — submissions are not open.\
📅 **Submission deadline:** April 25, 2027, 23:59 Anywhere on Earth (AoE; UTC−12).\
👥 **Task leads:** Anthony Cioppa and Cedric Gavage\
🤝 **Sponsor:** To be announced

🏠 [Challenge website](https://www.soccer-net.org/challenges/2027) · 🗂️ [Data on Hugging Face](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) · 💻 [Baseline](baselines/README.md) · 📚 [Rules](RULES.md) · 📊 [Evaluation](evaluation/README.md) · 🏆 [Evaluation server](https://www.codabench.org/) (task URL TBD)

## Task

Reconstruct athlete positions and identities on the pitch from a **continuously annotated match half**, covering 45 minutes plus added time. Previous SoccerNet-Tracking and GSR benchmarks used 200 separate 30-second clips across 11 games. This edition tests identity continuity over the full half. It retains the prior **GS-HOTA** metric, which combines pitch localization with role, team, jersey number, detection, and association quality.

## Data

The [SN-GSR-2027 Hugging Face repository](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) exists as a **private, empty placeholder**. The annotated half exists, but its 2027 package, frame IDs, JSON schema, and train/valid/test/challenge allocation have not been released. The [split inventory](data/README.md) tracks those decisions. Challenge ground truth stays outside Git.

## Baseline

The [baseline folder](baselines/README.md) links the [SoccerNet Game State Reconstruction/TrackLab pipeline](https://github.com/SoccerNet/sn-gamestate), explains where to get the Hugging Face dataset, and gives the previous baseline run command. Existing 30-second tracker states are background material; a full-half 2027 prediction set is pending.

## Evaluation

The [evaluation folder](evaluation/README.md) documents the **SoccerNet Python package's GameStateReconstruction evaluator**, backed by the SoccerNet TrackEval fork. It contains the adapter for scoring ZIP packages. The full-half annotation and prediction format must be checked before that adapter is used for official CodaBench scoring.

## Citation

Please cite the SoccerNet-Tracking and SoccerNet GSR papers in their CVPR Workshops versions:

```bibtext
@InProceedings{Cioppa_2022_CVPR,
  author = {Cioppa, Anthony and Giancola, Silvio and Deliege, Adrien and Kang, Le and Zhou, Xin and Cheng, Zhiyu and Ghanem, Bernard and Van Droogenbroeck, Marc},
  title = {SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year = {2022},
  pages = {3490--3501},
  doi = {10.1109/CVPRW56347.2022.00393},
  url = {https://mlanthology.org/cvprw/2022/cioppa2022cvprw-soccernettracking/}
}

@InProceedings{Somers_2024_CVPR,
  author = {Somers, Vladimir and Joos, Victor and Cioppa, Anthony and Giancola, Silvio and Ghasemzadeh, Seyed Abolfazl and Magera, Floriane and Standaert, Baptiste and Mansourian, Amir M. and Zhou, Xin and Kasaei, Shohreh and Ghanem, Bernard and Alahi, Alexandre and Van Droogenbroeck, Marc and De Vleeschouwer, Christophe},
  title = {SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2024},
  pages = {3293--3305},
  url = {https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.html}
}
```

## Updates and support

Each task launches independently when ready. Follow the [challenge website](https://www.soccer-net.org/challenges/2027) and [SoccerNet Discord](https://discord.gg/cPbqf2mAwF) for announcements. Use repository issues for reproducible code problems; do not post access tokens or restricted data.

## Release readiness

| Item | Current state |
| --- | --- |
| Task definition | One fully annotated match half and GS-HOTA confirmed; detailed continuity rules pending |
| Data | Private Hugging Face placeholder created; full-half package and splits pending |
| Baseline | Prior TrackLab pipeline linked; full-half predictions pending |
| Evaluation | SoccerNet package metric entry point identified; full-half compatibility pending |
| Benchmark | CodaBench server TBD |
| Sponsor | TBD |

This section is for preparation and will be removed when this task is ready for release.
