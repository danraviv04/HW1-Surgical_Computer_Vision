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
from tqdm import tqdm


def train_yolo_model(data_path="hw1.yaml", epochs=50, batch_size=16, img_size=640):
    print('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO('yolov8n.pt')
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        hsv_h=0.005,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=2.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        flipud=0.0,
        fliplr=0.2,
        mosaic=0.5,
        mixup=0.0,
        workers=4,
    )
    return model


# def extract_pseudo_labels(results, output_dir="pseudo_labels"):
#     os.makedirs(output_dir, exist_ok=True)
#     for result in results:
#         h, w = result.orig_shape
#         boxes = result.boxes
#         img_name = os.path.splitext(os.path.basename(result.path))[0]
#         label_file = os.path.join(output_dir, f"{img_name}.txt")
#
#         with open(label_file, "w") as f:
#             for box in boxes:
#                 conf = box.conf.item()
#                 cls = int(box.cls.item())
#                 xywh = box.xywh[0]
#                 x_center = xywh[0].item() / w
#                 y_center = xywh[1].item() / h
#                 width = xywh[2].item() / w
#                 height = xywh[3].item() / h
#                 f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.4f}\n")
#
#     print(f"✅ Pseudo labels (unfiltered) saved in '{output_dir}'")

def extract_pseudo_labels_from_video(original_label_dir="runs/pseudo_labels",
                                     output_dir="runs/pseudo_labels_extracted"):
    """
    Copies pseudo labels and their corresponding images from an existing directory structure.
    Each video's frames are saved under:
        output_dir/{video_name}/Images/
        output_dir/{video_name}/Labels/
    """
    frame_counters = defaultdict(int)

    video_dirs = [d for d in os.listdir(original_label_dir) if os.path.isdir(os.path.join(original_label_dir, d))]

    for video_name in video_dirs:
        label_dir = os.path.join(original_label_dir, video_name, "Labels")
        image_dir = os.path.join(original_label_dir, video_name, "Images")

        label_files = sorted(glob(os.path.join(label_dir, "*.txt")))

        for label_file in label_files:
            frame_idx = frame_counters[video_name]
            frame_counters[video_name] += 1

            # Create output dirs
            out_img_dir = os.path.join(output_dir, video_name, "Images")
            out_lbl_dir = os.path.join(output_dir, video_name, "Labels")
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)

            # Image path from corresponding label name
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            img_path = os.path.join(image_dir, base_name + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, base_name + ".png")
            if not os.path.exists(img_path):
                print(f"⚠️ Skipping {base_name}, image not found.")
                continue

            # Copy image
            out_img_path = os.path.join(out_img_dir, f"{video_name}_frame{frame_idx:05d}.jpg")
            shutil.copy(img_path, out_img_path)

            # Read and copy label
            out_lbl_path = os.path.join(out_lbl_dir, f"{video_name}_frame{frame_idx:05d}.txt")
            with open(label_file, "r") as fin, open(out_lbl_path, "w") as fout:
                for line in fin:
                    fout.write(line)

    print(f"✅ Pseudo labels and frames saved under '{output_dir}' (grouped by video)")


# def filter_pseudo_labels(input_dir="pseudo_labels", output_dir="pseudo_labels_filtered", conf_thresh=0.85):
#     os.makedirs(output_dir, exist_ok=True)
#
#     for file in os.listdir(input_dir):
#         if not file.endswith(".txt"):
#             continue
#         in_path = os.path.join(input_dir, file)
#         out_path = os.path.join(output_dir, file)
#         with open(in_path, "r") as fin, open(out_path, "w") as fout:
#             for line in fin:
#                 parts = line.strip().split()
#                 if len(parts) != 6:
#                     continue
#                 x, y, w, h, conf, cls = parts
#                 if float(conf) >= conf_thresh:
#                     fout.write(f"{x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f} {cls}\n")
#     print(f"✅ Filtered pseudo labels saved in '{output_dir}'")

# def filter_pseudo_labels(input_dir="runs/pseudo_labels_extracted",
#                          output_dir="runs/pseudo_labels_filtered",
#                          conf_thresh=0.85):
#     """
#     Filters labels by confidence and saves only frames with high-confidence boxes.
#     Directory structure:
#     - input_dir/videoX/Labels/*.txt
#     - input_dir/videoX/Images/*.jpg
#     Will create:
#     - output_dir/videoX/Labels/*.txt
#     - output_dir/videoX/Images/*.jpg
#     """
#     videos = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
#     total_kept = 0
#     total_skipped = 0
#
#     for video_name in videos:
#         input_label_dir = os.path.join(input_dir, video_name, "Labels")
#         input_image_dir = os.path.join(input_dir, video_name, "Images")
#
#         output_label_dir = os.path.join(output_dir, video_name, "Labels")
#         output_image_dir = os.path.join(output_dir, video_name, "Images")
#         os.makedirs(output_label_dir, exist_ok=True)
#         os.makedirs(output_image_dir, exist_ok=True)
#
#         label_files = [f for f in os.listdir(input_label_dir) if f.endswith(".txt")]
#
#         for file in label_files:
#             input_label_path = os.path.join(input_label_dir, file)
#             output_label_path = os.path.join(output_label_dir, file)
#
#             with open(input_label_path, "r") as fin:
#                 lines = fin.readlines()
#
#             filtered_lines = []
#             for line in lines:
#                 parts = line.strip().split()
#                 if len(parts) != 6:
#                     continue
#                 cls, x, y, w, h, conf = parts
#                 if float(conf) >= conf_thresh:
#                     filtered_lines.append(f"{cls} {x} {y} {w} {h}\n")
#
#             if filtered_lines:
#                 # Save label
#                 with open(output_label_path, "w") as fout:
#                     fout.writelines(filtered_lines)
#
#                 # Copy corresponding image
#                 base = os.path.splitext(file)[0]
#                 for ext in [".jpg", ".png"]:
#                     image_path = os.path.join(input_image_dir, base + ext)
#                     if os.path.exists(image_path):
#                         shutil.copy(image_path, os.path.join(output_image_dir, os.path.basename(image_path)))
#                         break
#
#                 total_kept += 1
#             else:
#                 total_skipped += 1
#
#     print(f"✅ Filtered pseudo labels saved in '{output_dir}'")
#     print(f"✅ {total_kept} frames kept | {total_skipped} skipped (no high-confidence detections)")

def filter_pseudo_labels(input_dir="runs/pseudo_labels_extracted",
                         output_dir="runs/pseudo_labels_filtered",
                         conf_thresh=0.85):
    """
    Filters labels by confidence and saves only frames with high-confidence boxes.
    After copying, deletes the original labels and images from the input directory.
    Directory structure:
    - input_dir/videoX/Labels/*.txt
    - input_dir/videoX/Images/*.jpg
    Will create:
    - output_dir/videoX/Labels/*.txt
    - output_dir/videoX/Images/*.jpg
    """
    videos = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    total_kept = 0
    total_skipped = 0

    for video_name in videos:
        input_label_dir = os.path.join(input_dir, video_name, "Labels")
        input_image_dir = os.path.join(input_dir, video_name, "Images")

        output_label_dir = os.path.join(output_dir, video_name, "Labels")
        output_image_dir = os.path.join(output_dir, video_name, "Images")
        os.makedirs(output_label_dir, exist_ok=True)
        os.makedirs(output_image_dir, exist_ok=True)

        label_files = [f for f in os.listdir(input_label_dir) if f.endswith(".txt")]

        for file in label_files:
            input_label_path = os.path.join(input_label_dir, file)
            output_label_path = os.path.join(output_label_dir, file)
            base = os.path.splitext(file)[0]

            # Read and filter lines
            with open(input_label_path, "r") as fin:
                lines = fin.readlines()

            filtered_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                cls, x, y, w, h, conf = parts
                if float(conf) >= conf_thresh:
                    filtered_lines.append(f"{cls} {x} {y} {w} {h}\n")

            if filtered_lines:
                # Save filtered label
                with open(output_label_path, "w") as fout:
                    fout.writelines(filtered_lines)

                # Copy and delete image
                for ext in [".jpg", ".png"]:
                    image_path = os.path.join(input_image_dir, base + ext)
                    if os.path.exists(image_path):
                        dst_image_path = os.path.join(output_image_dir, os.path.basename(image_path))
                        shutil.copy(image_path, dst_image_path)
                        try:
                            os.remove(image_path)
                        except Exception as e:
                            print(f"⚠️ Could not delete image {image_path}: {e}")
                        break

                total_kept += 1
            else:
                # Delete skipped label and image
                try:
                    os.remove(input_label_path)
                except Exception as e:
                    print(f"⚠️ Could not delete low-confidence label {input_label_path}: {e}")

                for ext in [".jpg", ".png"]:
                    image_path = os.path.join(input_image_dir, base + ext)
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except Exception as e:
                            print(f"⚠️ Could not delete low-confidence image {image_path}: {e}")
                        break

                total_skipped += 1

    print(f"✅ Filtered pseudo labels saved in '{output_dir}'")
    print(f"✅ {total_kept} frames kept | {total_skipped} skipped (no high-confidence detections)")


# def create_combined_training_dataset(orig_train_img_dir, orig_train_lbl_dir,
#                                      pseudo_img_dir, pseudo_lbl_dir,
#                                      output_root="refined_train", step=1):
#     """
#         Creates a new combined training directory like:
#         refined_train/step1/images/
#         refined_train/step1/labels/
#     """
#
#     output_img_dir = os.path.join(output_root, f"step{step}", "images")
#     output_lbl_dir = os.path.join(output_root, f"step{step}", "labels")
#     os.makedirs(output_img_dir, exist_ok=True)
#     os.makedirs(output_lbl_dir, exist_ok=True)
#
#     train_imgs = glob(os.path.join(orig_train_img_dir, '*.[jp][pn]g'))
#     train_lbls = glob(os.path.join(orig_train_lbl_dir, '*.txt'))
#
#     print(f"🔍 Copying {len(train_imgs)} original images and {len(train_lbls)} labels")
#     for img_path in train_imgs:
#         shutil.copy(img_path, os.path.join(output_img_dir, os.path.basename(img_path)))
#     for lbl_path in train_lbls:
#         shutil.copy(lbl_path, os.path.join(output_lbl_dir, os.path.basename(lbl_path)))
#
#     pseudo_lbls = glob(os.path.join(pseudo_lbl_dir, '*.txt'))
#     print(f"🔍 Adding {len(pseudo_lbls)} pseudo-labeled samples")
#     for label_path in pseudo_lbls:
#         base = os.path.splitext(os.path.basename(label_path))[0]
#         img_path = os.path.join(pseudo_img_dir, base + '.jpg')
#         if os.path.exists(img_path):
#             shutil.copy(img_path, os.path.join(output_img_dir, os.path.basename(img_path)))
#             shutil.copy(label_path, os.path.join(output_lbl_dir, os.path.basename(label_path)))
#
#     print(f"✅ Final train set: {len(os.listdir(output_img_dir))} images, {len(os.listdir(output_lbl_dir))} labels")
#     return output_img_dir, output_lbl_dir

def create_combined_training_dataset(
        orig_train_img_dir,
        orig_train_lbl_dir,
        pseudo_root_dir,  # e.g., "runs/pseudo_labels_filtered"
        output_root="refined_train",
        step=1
):
    """
    Combines original train set and all pseudo-labeled folders (with Images/Labels subfolders)
    into:
        refined_train/step{step}/images/train/
        refined_train/step{step}/labels/train/
    """

    output_img_dir = os.path.join(output_root, f"step{step}", "images", "train")
    output_lbl_dir = os.path.join(output_root, f"step{step}", "labels", "train")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    # Original training data
    train_imgs = glob(os.path.join(orig_train_img_dir, '*.[jp][pn]g'))
    train_lbls = glob(os.path.join(orig_train_lbl_dir, '*.txt'))
    print(f"🔁 Copying {len(train_imgs)} original train images and {len(train_lbls)} labels")
    for img_path in train_imgs:
        shutil.copy(img_path, os.path.join(output_img_dir, os.path.basename(img_path)))
    for lbl_path in train_lbls:
        shutil.copy(lbl_path, os.path.join(output_lbl_dir, os.path.basename(lbl_path)))

    # Go through each video folder in pseudo_root_dir
    pseudo_dirs = [d for d in os.listdir(pseudo_root_dir) if os.path.isdir(os.path.join(pseudo_root_dir, d))]
    added = 0
    for video_dir in pseudo_dirs:
        img_dir = os.path.join(pseudo_root_dir, video_dir, "Images")
        lbl_dir = os.path.join(pseudo_root_dir, video_dir, "Labels")
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        for label_path in glob(os.path.join(lbl_dir, '*.txt')):
            base = os.path.splitext(os.path.basename(label_path))[0]
            img_path = os.path.join(img_dir, base + '.jpg')
            if os.path.exists(img_path):
                img_dst = os.path.join(output_img_dir, os.path.basename(img_path))
                lbl_dst = os.path.join(output_lbl_dir, os.path.basename(label_path))

                # Copy files
                shutil.copy(img_path, img_dst)
                shutil.copy(label_path, lbl_dst)
                added += 1

                # Delete original pseudo image and label to free up space
                try:
                    os.remove(img_path)
                    os.remove(label_path)
                except Exception as e:
                    print(f"⚠️ Could not delete {img_path} or {label_path}: {e}")

    print(f"➕ Added {added} pseudo-labeled image-label pairs")
    print(f"✅ Final train set: {len(os.listdir(output_img_dir))} images, {len(os.listdir(output_lbl_dir))} labels")

    return output_img_dir, output_lbl_dir


def create_data_yaml(img_dir, val_dir, class_names, output_path):
    with open(output_path, "w") as f:
        f.write(f"train: {img_dir}\n")
        f.write(f"val: {val_dir}\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: " + str(class_names) + "\n")
    print(f"📄 Data config written to: {output_path}")
    return output_path


# def predict_and_export_video_frames(pseudo_labels_dir="runs/pseudo_labels", video_dir="/datashare/HW1"):
#     for video_name in os.listdir(video_dir):
#         video_path = os.path.join(pseudo_labels_dir, os.path.splitext(video_name)[0])
#         os.makedirs(os.path.join(video_path, "Labels"), exist_ok=True)
#         os.makedirs(os.path.join(video_path, "Images"), exist_ok=True)
#
#     frame_counters = defaultdict(int)
#
#     # Stream inference
#     prev_path, prev_video_name = None, None
#     for result in model.predict(source=video_dir, stream=True, conf=0.5, vid_stride=1):
#         if result.path != prev_path:
#             print(f"🆕 New video detected: {result.path}")
#             prev_path = result.path
#
#         frame = result.orig_img
#         H, W = frame.shape[:2]
#
#         video_name = os.path.splitext(os.path.basename(result.path))[0]
#
#         if video_name != prev_video_name:
#             frame_counters[video_name] = 0
#             prev_video_name = video_name
#
#         frame_idx = frame_counters[video_name]
#         frame_counters[video_name] += 1
#
#         # Save frame image with predictions
#         result_image_path = os.path.join(pseudo_labels_dir, f"{video_name}/Images",
#                                          f"{video_name}_frame{frame_idx:05d}.jpg")
#         result.save(filename=result_image_path)
#
#         # Save YOLO-style .txt annotation
#         txt_path = os.path.join(pseudo_labels_dir, f"{video_name}/Labels", f"{video_name}_frame{frame_idx:05d}.txt")
#         with open(txt_path, "w") as f:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 conf = float(box.conf[0])
#                 cls = int(box.cls[0])
#
#                 # Convert to YOLO format
#                 box_w = x2 - x1
#                 box_h = y2 - y1
#                 x_center = x1 + box_w / 2
#                 y_center = y1 + box_h / 2
#
#                 # Normalize to [0, 1]
#                 x_c = x_center / W
#                 y_c = y_center / H
#                 w = box_w / W
#                 h = box_h / H
#
#                 f.write(f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.4f} {cls}\n")
#
#     print(f"✅ Predictions on videos saved to: {pseudo_labels_dir}")

def predict_and_export_video_frames(pseudo_labels_dir="runs/pseudo_labels", video_dir="/datashare/HW1/id_video_data"):
    os.makedirs(pseudo_labels_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"🎬 Found {len(video_files)} videos to process.")

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        print(f"\n▶️ Processing: {video_name}")

        save_path = os.path.join(pseudo_labels_dir, video_name)
        os.makedirs(os.path.join(save_path, "Images"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "Labels"), exist_ok=True)

        frame_counter = 0

        try:
            for result in model.predict(source=video_path, stream=True, conf=0.5, vid_stride=1):
                frame = result.orig_img
                H, W = frame.shape[:2]

                frame_filename = f"{video_name}_frame{frame_counter:05d}"
                result_image_path = os.path.join(save_path, "Images", frame_filename + ".jpg")
                result.save(filename=result_image_path)

                txt_path = os.path.join(save_path, "Labels", frame_filename + ".txt")
                with open(txt_path, "w") as f:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        # Convert to YOLO format
                        box_w = x2 - x1
                        box_h = y2 - y1
                        x_center = x1 + box_w / 2
                        y_center = y1 + box_h / 2

                        # Normalize to [0, 1]
                        x_c = x_center / W
                        y_c = y_center / H
                        w = box_w / W
                        h = box_h / H

                        f.write(f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.4f} {cls}\n")

                if frame_counter % 500 == 0:
                    print(f"🖼️ Processed {frame_counter} frames for {video_name}...")
                    gc.collect()  # clean up memory

                frame_counter += 1

            print(f"✅ Finished: {video_name} ({frame_counter} frames processed)")

        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")

        gc.collect()  # full cleanup per video

    print(f"\n🏁 All predictions saved to: {pseudo_labels_dir}")

def combine_frames_to_video(frames_dir="runs/id_video_eval/id_video_preds",
                            output_video_dir="runs/id_video_eval/id_video_preds"):
    """
    Combines saved frames from each video subfolder into an output .mp4 video.
    Shows progress bar while writing frames.
    """
    video_folders = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]

    for video_name in video_folders:
        video_frames_dir = os.path.join(frames_dir, video_name, "Images")
        output_path = os.path.join(output_video_dir, f"{video_name}.mp4")
        original_video_path = os.path.join("/datashare/HW1/id_video_data", f"{video_name}.mp4")

        # Get frame paths
        image_paths = natsorted(glob(os.path.join(video_frames_dir, "*.jpg")))
        if not image_paths:
            print(f"⚠️ No frames found for '{video_name}'. Skipping.")
            continue

        # Load first frame to get dimensions
        first_frame = cv2.imread(image_paths[0])
        if first_frame is None:
            print(f"⚠️ Could not read first frame for '{video_name}'. Skipping.")
            continue
        height, width, _ = first_frame.shape

        # Try to get original FPS
        if os.path.exists(original_video_path):
            cap = cv2.VideoCapture(original_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            print(f"⚠️ Original video not found for '{video_name}', using default FPS = 29")
            fps = 29

        print(f"🎞 Creating '{video_name}.mp4' from {len(image_paths)} frames at {fps:.2f} FPS")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames with progress bar
        for img_path in tqdm(image_paths, desc=f"🧵 Writing {video_name}", unit="frame"):
            frame = cv2.imread(img_path)
            if frame is not None:
                out.write(frame)

        out.release()
        print(f"✅ Saved video: {output_path}\n")

    print("🏁 All videos created.")


if __name__ == "__main__":
    if not os.path.exists("/home/student/HW1/yolov8_refined.pt"):
        print("🚀 Starting YOLOv8 training pipeline...")
        if not os.path.exists("hw1.yaml"):
            raise FileNotFoundError("Dataset configuration file 'hw1.yaml' not found.")

        if os.path.exists("yolov8_trained.pt"):
            retrain = input("Model already exists. Retrain? (yes/no): ").strip().lower()
            # retrain = "no"  # For testing purposes, set to 'no' to skip retraining
            model = train_yolo_model() if retrain in ['yes', 'y', '1'] else YOLO("yolov8_trained.pt")
        else:
            model = train_yolo_model()

        model.save("yolov8_trained.pt")
        print("📦 Model saved as 'yolov8_trained.pt'")

        # --- Step 1: Predict on validation set ---
        print("Step 1: Predicting on validation set...")
        if not os.path.exists("runs/pseudo_labels") or len(os.listdir("runs/pseudo_labels")) == 0:
            predict_and_export_video_frames(pseudo_labels_dir="runs/pseudo_labels",
                                            video_dir="/datashare/HW1/id_video_data")  # "/home/student/HW1/temp"
        else:
            print("🚀 Skipping Step 1: Pseudo labels already exist in 'runs/pseudo_labels'")

        # predict_and_export_video_frames(pseudo_labels_dir="runs/pseudo_labels",
        #                                 video_dir="/datashare/HW1/id_video_data")

        # --- Step 2: Save all pseudo labels with confidence ---
        # print("Step 2: Extracting pseudo labels...")
        # if not os.path.exists("/home/student/HW1/runs/pseudo_labels_extracted") or len(os.listdir("/home/student/HW1/runs/pseudo_labels_extracted")) == 0:
        #     extract_pseudo_labels_from_video()
        # else:
        #     print("🚀 Skipping Step 2: Pseudo labels already extracted in 'runs/pseudo_labels'")

        # --- Step 3: Filter by confidence threshold ---
        print("Step 3: Filtering pseudo labels by confidence threshold...")
        if not os.path.exists("runs/pseudo_labels_filtered") or len(os.listdir("runs/pseudo_labels_filtered")) == 0:
            filter_pseudo_labels(
                input_dir="runs/pseudo_labels",
                output_dir="runs/pseudo_labels_filtered",
                conf_thresh=0.85
            )
        else:
            print("🚀 Skipping Step 3: Pseudo labels already filtered in 'runs/pseudo_labels_filtered'")

        # --- Step 4: Create combined training dataset ---
        print("Step 4: Creating combined training dataset...")
        if not os.path.exists("refined_train/step1/images/train") or len(os.listdir("refined_train/step1/images/train")) == 0:
            img_dir, lbl_dir = create_combined_training_dataset(
                orig_train_img_dir="/datashare/HW1/labeled_image_data/images/train",
                orig_train_lbl_dir="/datashare/HW1/labeled_image_data/labels/train",
                pseudo_root_dir="runs/pseudo_labels_filtered",
                step=1
            )
        else:
            img_dir, lbl_dir = "refined_train/step1/images/train", "refined_train/step1/labels/train"
            print("🚀 Skipping Step 4: Combined training dataset already exists in 'refined_train/step1'")

        # --- Step 4.5: Cleanup intermediate pseudo label directories ---
        print("🧹 Cleaning up intermediate pseudo-label directories to free up space...")
        for folder in ["runs/pseudo_labels", "runs/pseudo_labels_extracted", "runs/pseudo_labels_filtered"]:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    print(f"🗑️ Deleted: {folder}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {folder}: {e}")
        print("✅ Cleanup done.")

        # --- Step 5: Create data.yaml for the new training set ---
        print("Step 5: Creating data.yaml for the new training set...")
        data_yaml_path = create_data_yaml(
            img_dir=os.path.join("/home/student/HW1", img_dir),
            val_dir="/datashare/HW1/labeled_image_data/images/val",
            class_names=["Empty", "Tweezers", "Needle_driver"],
            output_path="hw1_refined.yaml"
        )

        # --- Step 6: Train the refined model ---
        print("Step 6: Training the refined model...")

        print(data_yaml_path)

        refined_model = train_yolo_model(data_path=data_yaml_path, epochs=20)
        refined_model.save("yolov8_refined.pt")
    else:
        print("🚀 YOLOv8 refined model already exists. Skipping training.")
        refined_model = YOLO("/home/student/HW1/yolov8_refined.pt")
        print("📦 Loaded existing refined model from 'yolov8_refined.pt'")

    # print("🎥 Step 7: Predicting on In-Distribution Videos...")
    # id_video_dir = "/datashare/HW1/id_video_data"
    # refined_model.predict(
    #     source=id_video_dir,
    #     conf=0.5,
    #     save=True,
    #     save_txt=True,
    #     save_conf=True,
    #     stream=True,
    #     project="runs/id_video_eval",
    #     name="id_video_preds"
    # )
    # print("✅ Predictions on ID videos saved to 'runs/id_video_eval/id_video_preds'")

    # check if predictions already exist
    if not os.path.exists("runs/id_video_eval/id_video_preds") or len(os.listdir("runs/id_video_eval/id_video_preds")) == 0:
        print("🎥 Step 7: Predicting on In-Distribution Videos...")
        predict_and_export_video_frames(pseudo_labels_dir="runs/id_video_eval/id_video_preds",
                                        video_dir="/datashare/HW1/id_video_data")
    else:
        print("🚀 Skipping Step 7: Predictions on ID videos already exist in 'runs/id_video_eval/id_video_preds'")

    # -- Combine frames into a video --
    combine_frames_to_video()
