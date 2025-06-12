import cv2
import argparse
import os
from ultralytics import YOLO
from tqdm import tqdm


def run_video_inference(source, weights, output="output_video.mp4", conf=0.25, save_labels=False):
    model = YOLO(weights)

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video: {source}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    print(f"Running inference on: {source}")
    base_name = os.path.splitext(os.path.basename(source))[0]

    # Optional label saving
    if save_labels:
        label_dir = os.path.join("runs", "labels", base_name)
        os.makedirs(label_dir, exist_ok=True)

    frame_count = 0
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 prediction (no RGB conversion needed)
            results = model.predict(source=frame, conf=conf, imgsz=640, verbose=False)[0]
            H, W = frame.shape[:2]
            label_lines = []

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls = int(box.cls[0])

                # Convert to YOLO normalized format
                box_w = x2 - x1
                box_h = y2 - y1
                x_center = x1 + box_w / 2
                y_center = y1 + box_h / 2

                x_c = x_center / W
                y_c = y_center / H
                w = box_w / W
                h = box_h / H

                # Draw box and label
                label_str = f"{cls} {model.names[cls]} {score:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label_str, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if save_labels:
                    label_lines.append(f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {score:.4f} {cls}\n")

            # Save YOLO label file
            if save_labels:
                frame_filename = f"{base_name}_frame{frame_count:05d}.txt"
                with open(os.path.join(label_dir, frame_filename), "w") as f:
                    f.writelines(label_lines)

            out.write(frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Saved annotated video to: {output}")
    if save_labels:
        print(f"Labels saved to: {label_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 video inference with optional label export")
    parser.add_argument("--source", default="/datashare/HW1/ood_video_data/surg_1.mp4", help="Path to input video file")
    parser.add_argument("--weights", default="/home/student/HW1/yolo11n_ood_refined.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--output", default="output_video.mp4", help="Path to output annotated video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-labels", action="store_true", help="Save YOLO-format label files per frame")

    args = parser.parse_args()
    run_video_inference(args.source, args.weights, args.output, args.conf, args.save_labels)
