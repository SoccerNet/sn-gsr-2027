# SoccerNet Challenge 2027 — Long Form Game State Reconstruction (GSR)

📢 **Long Form Game State Reconstruction (GSR) Challenge!** 🚀

📅 **Submission deadline:** April 25, 2027, 23:59 Anywhere on Earth (AoE; UTC−12).\
👥 **Task leads:** Anthony Cioppa and Cedric Gavage

🏠 [Challenge website](https://www.soccer-net.org/challenges/2027) · 🗂️ [Data on Hugging Face](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) · 💻 [Baseline](baselines/README.md) · 📚 [Rules](RULES.md) · 📊 [Evaluation](evaluation/README.md)

## Task

Given the broadcast video of one continuously annotated match half, including added time, reconstruct the game state frame by frame. For each athlete, predict the position in pitch coordinates, a consistent track identity, the role (player, goalkeeper, referee, or other), and, for players and goalkeepers, team affiliation and jersey number when visible. The earlier benchmark used 30-second clips; this edition measures reconstruction and identity continuity across the full half. The ranking metric remains SoccerNet **GS-HOTA**.

## Data

The new match half is the **2027 challenge set** in [SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027). Training, validation, and test data are the unchanged splits from [SN-GSR-2025](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025). The [data guide](data/README.md) lists the corresponding files.

## Baseline

The [baseline guide](baselines/README.md) explains how to install and run the SoccerNet Game State Reconstruction [TrackLab pipeline](https://github.com/SoccerNet/sn-gamestate) on the existing validation data, then use its prediction format for the full-half challenge. The prior GSR and Tracking papers are cited below.

## Evaluation

The [evaluation guide](evaluation/README.md) documents the prediction JSON and ZIP format, the SoccerNet package's GS-HOTA evaluator, and the local scoring adapter. GS-HOTA determines the ranking.

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

## Community

Follow the [SoccerNet challenge website](https://www.soccer-net.org/challenges/2027) and [Discord](https://discord.gg/cPbqf2mAwF) for challenge announcements and discussion.
