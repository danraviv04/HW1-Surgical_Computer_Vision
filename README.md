# 🔬 Surgical Tool Detection with Semi-Supervised Learning
This HW implements an object detection system for surgical tools in leg suture surgeries using YOLO11 and Semi-Supervised Learning (SSL) with pseudo-labeling. The system is designed to generalize from a small labeled dataset to unseen (out-of-distribution) videos.

<details>
<summary><strong>Project Structure</strong></summary>
.
  
├── predict.py # The main code of the HW trains the ID and OOD models + Predicts the tools on the Videos.

├── hw1.yaml # Dataset config for initial training

#├── refined_train/

#├── runs/

├── requirements.txt # Python environment dependencies

└── README.md # You are here

</details>

## 🚀 Quick Start
### Environment Setup
<pre>
  git clone https://github.com/danraviv04/HW1-Surgical_Computer_Vision.git
  cd surgical-tool-detection
  pip install -r requirements.txt
</pre>

### Training and Predictiction
To train and predict on all models, run the `predict.py` file

## 🏋️‍♂️ Pretrained Weights
| Model Stage     | Download |
|-----------------|----------|
| Initial YOLO11  | 📥 [yolo11n_trained.pt](models/yolo11n_trained.pt) |
| Refind YOLO11 (ID)  | 📥 [yolo11n_refined.pt](models/yolo11n_refined.pt) |
| Refined YOLO11 (OOD)  | 📥 [yolo11n_ood_refined.pt](models/yolo11n_ood_refined.pt) |





