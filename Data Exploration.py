import os
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# CONFIG
base_path = '/datashare/HW1/labeled_image_data'
label_dir = os.path.join(base_path, 'labels')
classes_file = os.path.join(base_path, 'classes.txt')
splits = ['train', 'val']
colors = {'train': 'steelblue', 'val': 'darkorange'}

# Load class names
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Stats containers
stats = {
    split: {
        'class_counts': defaultdict(int),
        'bbox_sizes': [],
        'image_object_counts': [],
        'centers': []
    }
    for split in splits
}

# Read and process label files
for split in splits:
    split_label_dir = os.path.join(label_dir, split)
    for label_file in os.listdir(split_label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(split_label_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        object_count = 0
        for line in lines:
            if not line.strip():
                continue
            cls_id, x_center, y_center, w, h = map(float, line.strip().split())
            cls_id = int(cls_id)
            stats[split]['class_counts'][classes[cls_id]] += 1
            stats[split]['bbox_sizes'].append(w * h)
            stats[split]['centers'].append((x_center, y_center))
            object_count += 1
        stats[split]['image_object_counts'].append(object_count)

# --- Class Distribution ---
plt.figure(figsize=(10, 5))
x = range(len(classes))
width = 0.35

for i, split in enumerate(splits):
    counts = [stats[split]['class_counts'][cls] for cls in classes]
    offset = (i - 0.5) * width
    plt.bar([xi + offset for xi in x], counts, width=width, label=split, color=colors[split])

plt.xticks(ticks=x, labels=classes)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution: Train vs Val")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- BBox Size Distribution ---
plt.figure(figsize=(10, 5))
for split in splits:
    plt.hist(stats[split]['bbox_sizes'], bins=30, alpha=0.5, label=split, color=colors[split])
plt.title("BBox Area Distribution")
plt.xlabel("Area (w × h)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Object Count per Image ---
plt.figure(figsize=(10, 5))
for split in splits:
    plt.hist(stats[split]['image_object_counts'], bins=range(max(stats[split]['image_object_counts']) + 2),
             alpha=0.5, label=split, color=colors[split])
plt.title("Objects per Image")
plt.xlabel("Number of Objects")
plt.ylabel("Number of Images")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Scatter Plot of Box Centers ---
plt.figure(figsize=(6, 6))
for split in splits:
    if stats[split]['centers']:
        x_coords, y_coords = zip(*stats[split]['centers'])
        plt.scatter(x_coords, y_coords, alpha=0.3, label=split, color=colors[split], s=10)

plt.title("Scatter of Bounding Box Centers")
plt.xlabel("x_center (normalized)")
plt.ylabel("y_center (normalized)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')
plt.legend()
plt.tight_layout()
plt.show()

# --- Summary Stats ---
for split in splits:
    s = stats[split]
    summary_df = pd.DataFrame({
        'Class': list(s['class_counts'].keys()),
        'Count': list(s['class_counts'].values())
    }).sort_values(by='Count', ascending=False)

    print(f"\n=== {split.upper()} Class Summary ===")
    print(summary_df.to_string(index=False))
    print(f"Total Images: {len(s['image_object_counts'])}")
    print(f"Images with No Objects: {s['image_object_counts'].count(0)}")
    print(f"Average Objects per Image: {sum(s['image_object_counts']) / len(s['image_object_counts']):.2f}")
