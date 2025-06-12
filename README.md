# 🔬 Surgical Tool Detection with Semi-Supervised Learning
This HW implements an object detection system for surgical tools in leg suture surgeries using YOLO11 and Semi-Supervised Learning (SSL) with pseudo-labeling. The system is designed to generalize from a small labeled dataset to unseen (out-of-distribution) videos.

<details>
<summary><strong>📁 Project Structure</strong></summary>
.

├── models / # Including The models' weights in different stages
  
├── train.py # The main code of the HW trains the ID and OOD models + Predicts the tools on the Videos.

├── predict.py # Performs inference on a single image

├── video.py # Performs inference on a full video

├── hw1.yaml # Dataset config for initial training

├── requirements.txt # Python environment dependencies

└── README.md # You are here

</details>

## 🚀 Quick Start
### ⚙️ Environment Setup
```bash
  git clone https://github.com/danraviv04/HW1-Surgical_Computer_Vision.git
  cd surgical-tool-detection
  pip install -r requirements.txt
```

### Training and Predictiction (Combined)
To train and predict on all models, run the `train.py` file

## 🏋️‍♂️ Pretrained Weights
| Model Stage     | Download |
|-----------------|----------|
| Initial YOLO11  | 📥 [yolo11n_trained.pt](models/yolo11n_trained.pt) |
| Refind YOLO11 (ID)  | 📥 [yolo11n_refined.pt](models/yolo11n_refined.pt) |
| Refined YOLO11 (OOD)  | 📥 [yolo11n_ood_refined.pt](models/yolo11n_ood_refined.pt) |

### Using video.py and predict.py
those parts use a Parser:
<pre>
  parser = argparse.ArgumentParser(description="YOLO11 video inference with optional label export")
  parser.add_argument("--source", default="/datashare/HW1/ood_video_data/surg_1.mp4", help="Path to input video file")
  parser.add_argument("--weights", default="/home/student/HW1/yolo11n_ood_refined.pt", help="Path to YOLO model weights")
  parser.add_argument("--output", default="output_video.mp4", help="Path to output annotated video")
  parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
  parser.add_argument("--save-labels", action="store_true", help="Save YOLO-format label files per frame")
</pre>

In order to use a different path for the video or model run in terminal:
<pre>
  python code(predict or video).py \
  --source /path/to/video_or_image \
  --weights path/to/model_weights.pt \
  --output path/to/output \
  --conf number \
  --save-labels
</pre>

or you can use the default paths mentioned above




