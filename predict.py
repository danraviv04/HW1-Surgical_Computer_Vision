from multiprocessing.pool import worker
from ultralytics import YOLO
import os
import shutil
import torch
from glob import glob
import cv2
from natsort import natsorted
from collections import defaultdict
import gc

pseudo_labels_dir = "runs/test"
video_dir = "/datashare/HW1/id_video_data"
model = YOLO("/home/student/HW1/yolov8_refined.pt")


os.makedirs(pseudo_labels_dir, exist_ok=True)
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
print(f"🎬 Found {len(video_files)} videos to process.")

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    video_name = os.path.splitext(video_file)[0]

    print(f"\n▶️ Predicting and saving video for: {video_name}")

    try:
        model.predict(
            source=video_path,
            conf=0.5,
            save=True,
            save_txt=False,
            save_conf=False,
            project=pseudo_labels_dir,
            name=video_name
        )
        print(f"✅ Saved predicted video in: {os.path.join(pseudo_labels_dir, video_name)}")
    except Exception as e:
        print(f"❌ Error processing {video_name}: {e}")

print(f"\n🏁 All predicted videos saved in: {pseudo_labels_dir}")