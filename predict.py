import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import shutil
import torch
from glob import glob
import pandas as pd
import cv2
from natsort import natsorted
from collections import defaultdict
import gc
from tqdm import tqdm


def train_yolo_model(model, data_path="hw1.yaml", epochs=50, batch_size=16, img_size=640, name="initial_train"):
    """
    Train a YOLO model with specified parameters.
    :param model: the YOLO model to train
    :param data_path: Path to the dataset configuration file (YAML).
    :param epochs: number of training epochs
    :param batch_size: the batch size for training
    :param img_size: image size for training (default 640)
    :param name: name for the training run, used for saving results
    :return: the trained YOLO model
    """
    print('cuda' if torch.cuda.is_available() else 'cpu')
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        name=name,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.05,
        scale=0.5,
        shear=2.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
        workers=4,
    )
    return model

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
                print(f"Skipping {base_name}, image not found.")
                continue

            # Copy image
            out_img_path = os.path.join(out_img_dir, f"{video_name}_frame{frame_idx:05d}.jpg")
            shutil.copy(img_path, out_img_path)

            # Read and copy label
            out_lbl_path = os.path.join(out_lbl_dir, f"{video_name}_frame{frame_idx:05d}.txt")
            with open(label_file, "r") as fin, open(out_lbl_path, "w") as fout:
                for line in fin:
                    fout.write(line)

    print(f"Pseudo labels and frames saved under '{output_dir}' (grouped by video)")


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
                x, y, w, h, conf, cls = parts
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
                            print(f"Could not delete image {image_path}: {e}")
                        break

                total_kept += 1
            else:
                # Delete skipped label and image
                try:
                    os.remove(input_label_path)
                except Exception as e:
                    print(f"Could not delete low-confidence label {input_label_path}: {e}")

                for ext in [".jpg", ".png"]:
                    image_path = os.path.join(input_image_dir, base + ext)
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except Exception as e:
                            print(f"Could not delete low-confidence image {image_path}: {e}")
                        break

                total_skipped += 1

    print(f"Filtered pseudo labels saved in '{output_dir}'")
    print(f"{total_kept} frames kept | {total_skipped} skipped (no high-confidence detections)")


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
    print(f"Copying {len(train_imgs)} original train images and {len(train_lbls)} labels")
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
                    print(f"Could not delete {img_path} or {label_path}: {e}")

    print(f"Added {added} pseudo-labeled image-label pairs")
    print(f"Final train set: {len(os.listdir(output_img_dir))} images, {len(os.listdir(output_lbl_dir))} labels")

    return output_img_dir, output_lbl_dir


def create_data_yaml(img_dir, val_dir, class_names, output_path):
    """
    Creates a data.yaml file for YOLO training.
    :param img_dir: Path to the training images directory.
    :param val_dir: Path to the validation images directory.
    :param class_names: List of class names for the dataset.
    :param output_path: Path to save the data.yaml file.
    :return: Path to the created data.yaml file.
    """
    with open(output_path, "w") as f:
        f.write(f"train: {img_dir}\n")
        f.write(f"val: {val_dir}\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: " + str(class_names) + "\n")
    print(f"Data config written to: {output_path}")
    return output_path


def predict_and_export_video_frames(model_use, pseudo_labels_dir="runs/pseudo_labels",
                                    video_dir="/datashare/HW1/id_video_data"):
    """
    Predicts on video files and exports frames with pseudo labels.
    :param model_use: The YOLO model to use for predictions.
    :param pseudo_labels_dir: Directory to save the pseudo labels and frames.
    :param video_dir: Directory containing video files to process.
    :return: None
    """

    os.makedirs(pseudo_labels_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} videos to process.")

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        print(f"\nProcessing: {video_name}")

        save_path = os.path.join(pseudo_labels_dir, video_name)
        os.makedirs(os.path.join(save_path, "Images"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "Labels"), exist_ok=True)

        frame_counter = 0

        try:
            for result in model_use.predict(source=video_path, stream=True, conf=0.5, vid_stride=1):
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
                    print(f"Processed {frame_counter} frames for {video_name}...")
                    gc.collect()  # clean up memory

                frame_counter += 1

            print(f"Finished: {video_name} ({frame_counter} frames processed)")

        except Exception as e:
            print(f"Error processing {video_name}: {e}")

        gc.collect()  # full cleanup per video

    print(f"\nAll predictions saved to: {pseudo_labels_dir}")


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
            print(f"No frames found for '{video_name}'. Skipping.")
            continue

        # Load first frame to get dimensions
        first_frame = cv2.imread(image_paths[0])
        if first_frame is None:
            print(f"Could not read first frame for '{video_name}'. Skipping.")
            continue
        height, width, _ = first_frame.shape

        # Try to get original FPS
        if os.path.exists(original_video_path):
            cap = cv2.VideoCapture(original_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            print(f"Original video not found for '{video_name}', using default FPS = 29")
            fps = 29

        print(f"Creating '{video_name}.mp4' from {len(image_paths)} frames at {fps:.2f} FPS")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames with progress bar
        for img_path in tqdm(image_paths, desc=f"Writing {video_name}", unit="frame"):
            frame = cv2.imread(img_path)
            if frame is not None:
                out.write(frame)

        out.release()
        print(f"Saved video: {output_path}\n")

    print("All videos created.")


if __name__ == "__main__": # Main entry point for the script
    if not os.path.exists("/home/student/HW1/yolo11n_refined.pt"):
        print("------Starting YOLO11 training pipeline...------")
        if not os.path.exists("hw1.yaml"):
            raise FileNotFoundError("Dataset configuration file 'hw1.yaml' not found.")
        model = YOLO("yolo11n.pt")
        if os.path.exists("yolo11n_trained.pt"):
            retrain = input("Model already exists. Retrain? (yes/no): ").strip().lower()
            # retrain = "no"  # For testing purposes, set to 'no' to skip retraining
            trained_model = train_yolo_model(model, epochs=70) if retrain in ['yes', 'y', '1'] else YOLO(
                "yolo11n_trained.pt")
        else:
            trained_model = train_yolo_model(model, epochs=70)

        trained_model.save("yolo11n_trained.pt")
        print("Model saved as 'yolo11n_trained.pt'!")

        # --- Step 5: Train the refined model ---
        print("Steps 1-5: Training the refined model...")
        model = trained_model
        for step in range(5):
            print(f"\n=== Refinement Iteration {step + 1} ===")

            # Step A: Generate pseudo labels
            print("Step A: Generating Pseudo Labels...")
            predict_and_export_video_frames(model, pseudo_labels_dir=f"runs/pseudo_labels_step{step}",
                                            video_dir="/datashare/HW1/id_video_data")

            # Step B: Filter
            print("Step B: Filtering Pseudo Labels...")
            filter_pseudo_labels(
                input_dir=f"runs/pseudo_labels_step{step}",
                output_dir=f"runs/pseudo_labels_filtered_step{step}",
                conf_thresh=0.85
            )

            # Step C: Combine with original
            print("Step C: Creating Combined Training Dataset...")
            img_dir, lbl_dir = create_combined_training_dataset(
                orig_train_img_dir="/datashare/HW1/labeled_image_data/images/train",
                orig_train_lbl_dir="/datashare/HW1/labeled_image_data/labels/train",
                pseudo_root_dir=f"runs/pseudo_labels_filtered_step{step}",
                output_root="refined_train",
                step=step + 1
            )

            # Step D: Create data.yaml
            print("Step D: Creating data.yaml for the new training set...")
            data_yaml_path = create_data_yaml(
                img_dir=os.path.join("/home/student/HW1", img_dir),
                val_dir="/datashare/HW1/labeled_image_data/images/val",
                class_names=["Empty", "Tweezers", "Needle_driver"],
                output_path=f"hw1_refined_step{step + 1}.yaml"
            )

            # Step E: Train for 10 epochs
            print("Step E: Training the refined model for 10 epochs...")
            model = train_yolo_model(model, data_path=data_yaml_path, epochs=10, name=f"refined_train_step{step + 1}")

            # Clean pseudo label folders
            print("Cleaning up intermediate pseudo-label directories to free up space...")
            for folder in [f"runs/pseudo_labels_step{step}", f"runs/pseudo_labels_filtered_step{step}"]:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder)
                        print(f"Deleted: {folder}")
                    except Exception as e:
                        print(f"Failed to delete {folder}: {e}")

            # Clean combined training data
            train_folder = os.path.join("refined_train", f"step{step + 1}")
            if os.path.exists(train_folder):
                try:
                    shutil.rmtree(train_folder)
                    print(f"Deleted: {train_folder}")
                except Exception as e:
                    print(f"Failed to delete {train_folder}: {e}")

        model.save("yolo11n_refined.pt")
        refined_model = model
    else:
        print("Skipping Parts 1-5: YOLOv8 refined model already exists.")
        refined_model = YOLO("/home/student/HW1/yolo11n_refined.pt")
        print("Loaded existing refined model from 'yolo11n_refined.pt'")

    # check if predictions already exist
    if not os.path.exists("runs/id_video_eval/id_video_preds") or len(
            os.listdir("runs/id_video_eval/id_video_preds")) == 0:
        # --- Step 6: Predict on in-distribution videos ---
        print("Step 6: Predicting on In-Distribution Videos...")
        predict_and_export_video_frames(refined_model, pseudo_labels_dir="runs/id_video_eval/id_video_preds",
                                        video_dir="/datashare/HW1/id_video_data")

        # -- Step 6.5: Combine frames into a video --
        combine_frames_to_video(frames_dir="runs/id_video_eval/id_video_preds",
                                output_video_dir="runs/id_video_eval/id_video_preds")
    else:
        print("Skipping Step 6: Predictions on ID videos already exist in 'runs/id_video_eval/id_video_preds'")

    # ------------ Part 4 ------------

    if not os.path.exists("/home/student/HW1/yolo11n_ood_refined.pt"):
        # --- Step 11: Train the OOD refined model ---
        print("Steps 7-11: Training the OOD refined model...")
        model = refined_model  # start from refined ID model
        for step in range(5):
            print(f"\n=== OOD Refinement Iteration {step + 1} ===")

            # Step A: Predict on OOD video frames
            print("Step A: Generating Pseudo Labels for OOD Videos...")
            predict_and_export_video_frames(model, pseudo_labels_dir=f"runs/pseudo_labels_ood_step{step}",
                                            video_dir="/datashare/HW1/ood_video_data")

            # Step B: Filter OOD pseudo-labels
            print("Step B: Filtering OOD Pseudo Labels...")
            filter_pseudo_labels(
                input_dir=f"runs/pseudo_labels_ood_step{step}",
                output_dir=f"runs/pseudo_labels_ood_filtered_step{step}",
                conf_thresh=0.85
            )

            # Step C: Combine with original ID data
            print("Step C: Creating Combined OOD Training Dataset...")
            img_dir, lbl_dir = create_combined_training_dataset(
                orig_train_img_dir="/datashare/HW1/labeled_image_data/images/train",
                orig_train_lbl_dir="/datashare/HW1/labeled_image_data/labels/train",
                pseudo_root_dir=f"runs/pseudo_labels_ood_filtered_step{step}",
                output_root="refined_train_ood",
                step=step + 1
            )

            # Step D: Create new data.yaml
            print("Step D: Creating OOD data.yaml for the new training set...")
            ood_data_yaml_path = create_data_yaml(
                img_dir=os.path.join("/home/student/HW1", img_dir),
                val_dir="/datashare/HW1/labeled_image_data/images/val",
                class_names=["Empty", "Tweezers", "Needle_driver"],
                output_path=f"hw1_ood_refined_step{step + 1}.yaml"
            )

            # Step E: Train
            print("Step E: Training the OOD refined model for 10 epochs...")
            model = train_yolo_model(model, data_path=ood_data_yaml_path, epochs=10,
                                     name=f"ood_refined_train_step{step + 1}")

            # Clean pseudo label folders
            print("Cleaning up intermediate OOD pseudo-label directories to free up space...")
            for folder in [f"runs/pseudo_labels_ood_step{step}", f"runs/pseudo_labels_ood_filtered_step{step}"]:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder)
                        print(f"Deleted: {folder}")
                    except Exception as e:
                        print(f"Failed to delete {folder}: {e}")

            # Clean combined training data
            train_folder = os.path.join("refined_train", f"step{step + 1}")
            if os.path.exists(train_folder):
                try:
                    shutil.rmtree(train_folder)
                    print(f"Deleted: {train_folder}")
                except Exception as e:
                    print(f"Failed to delete {train_folder}: {e}")

        # Save final OOD refined model
        model.save("yolo11n_ood_refined.pt")
        ood_refined_model = model
    else:
        print("Skipping Parts 7-11: YOLOv8 OOD refined model already exists.")
        ood_refined_model = YOLO("/home/student/HW1/yolo11n_ood_refined.pt")
        print("Loaded existing OOD refined model from 'yolo11n_ood_refined.pt'")

    if os.path.exists("refined_train/step1"):
        # --- Step 11.5: Cleanup refined training directories ---
        print("Cleaning up refined training directories to free up space...")
        for folder in ["refined_train/step1"]:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    print(f"Deleted: {folder}")
                except Exception as e:
                    print(f"Failed to delete {folder}: {e}")
        print("Cleanup done!")

    # check if predictions already exist
    if not os.path.exists("runs/ood_video_eval/ood_video_preds") or len(
            os.listdir("runs/ood_video_preds/ood_video_preds")) == 0:
        # --- Step 12: Predict on out-of-distribution videos ---
        print("Step 12: Predicting on out-of-distribution Videos...")
        predict_and_export_video_frames(ood_refined_model, pseudo_labels_dir="runs/ood_video_preds/ood_video_preds",
                                        video_dir="/datashare/HW1/ood_video_data")

        # -- Step 12.5: Combine frames into a video --
        combine_frames_to_video(frames_dir="runs/ood_video_preds/ood_video_preds",
                                output_video_dir="runs/ood_video_preds/ood_video_preds")
    else:
        print("Skipping Step 12: Predictions on OOD videos already exist in 'runs/ood_video_preds/ood_video_preds'")
