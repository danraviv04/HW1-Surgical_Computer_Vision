import cv2
import argparse
import os
from ultralytics import YOLO


def run_image_inference(source, weights, conf=0.25, output="predicted_image.jpg", save_labels=False):
    # Load the YOLO model
    model = YOLO(weights)

    # Load the image
    image = cv2.imread(source)
    if image is None:
        print(f"Could not load image: {source}")
        return

    # Run inference
    results = model(image, conf=conf)[0]

    label_lines = []
    H, W = image.shape[:2]

    # Draw predictions
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{cls} {model.names[cls]} {box_conf:.2f}"

        # Draw on image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if save_labels:
            box_w = x2 - x1
            box_h = y2 - y1
            x_center = x1 + box_w / 2
            y_center = y1 + box_h / 2

            x_c = x_center / W
            y_c = y_center / H
            w = box_w / W
            h = box_h / H

            label_lines.append(f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {box_conf:.4f} {cls}\n")

    # Save annotated image
    cv2.imwrite(output, image)
    print(f"Saved image to: {output}")

    # Save labels if requested
    if save_labels:
        label_path = os.path.splitext(output)[0] + ".txt"
        with open(label_path, "w") as f:
            f.writelines(label_lines)
        print(f"Labels saved to: {label_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 image inference")
    parser.add_argument("--source", default="/datashare/HW1/labeled_image_data/images/val/de6b6a6c-frame_2683.jpg", help="Path to input image")
    parser.add_argument("--weights", default="/home/student/HW1/yolo11n_ood_refined.pt", help="Path to YOLO11 model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", default="predicted_image.jpg", help="Path to save annotated image")
    parser.add_argument("--save-labels", action="store_true", help="Save YOLO-format label file")

    args = parser.parse_args()
    run_image_inference(args.source, args.weights, args.conf, args.output, args.save_labels)