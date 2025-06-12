import cv2
import argparse
from ultralytics import YOLO

def run_image_inference(source, weights, conf=0.25):
    # Load the YOLO model
    model = YOLO(weights)

    # Load the image
    image = cv2.imread(source)
    if image is None:
        print(f"Could not load image: {source}")
        return

    # Run inference
    results = model.predict(source=image, conf=conf, imgsz=640, verbose=False)[0]

    # Draw predictions
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{cls} {model.names[cls]} {box_conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save result to file
    output_path = "predicted_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Saved prediction to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 image inference")
    parser.add_argument("--source", default="/datashare/HW1/labeled_image_data/images/val/de6b6a6c-frame_2683.jpg", help="Path to input image")
    parser.add_argument("--weights", default="/home/student/HW1/yolo11n_ood_refined.pt", help="Path to YOLO11 model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run_image_inference(args.source, args.weights, args.conf)
