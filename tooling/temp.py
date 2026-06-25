import json
import cv2

# ==========================
# CHANGE THESE PATHS
# ==========================

IMAGE_ID = 0

image_path = "output/aws_instance/000000.png"

gt_path = "src/blenderproc_proj/output/bop/train_pbr/000000/scene_gt_info.json"

pred_path = "coco_instances_results.json"

# ==========================

img = cv2.imread(image_path)

if img is None:
    raise RuntimeError(f"Could not load {image_path}")

# --------------------------
# Ground Truth
# --------------------------

with open(gt_path) as f:
    gt = json.load(f)

for ann in gt[str(IMAGE_ID)]:
    x, y, w, h = ann["bbox_visib"]

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),   # Green
        2,
    )

# --------------------------
# Predictions
# --------------------------

with open(pred_path) as f:
    preds = json.load(f)

for pred in preds:
    if pred["image_id"] != IMAGE_ID:
        break
    print("Image ID:", pred["image_id"]
          
          )
    # continue
    x, y, w, h = pred["bbox"]

    x = int(round(x))
    y = int(round(y))
    w = int(round(w))
    h = int(round(h))

    score = pred["score"]
    cls = pred["category_id"]

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),   # Red
        2,
    )
    
    
    cv2.rectangle(
        img,
        (2*x, 2*y),
        (2*(x + w), 2*(y + h)),
        (255, 0, 0),   # Blue
        2,
    )

    cv2.putText(
        img,
        f"{cls}:{score:.2f}",
        (2*x, 2*y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),   # Blue
        3,
    )

# --------------------------
# Save
# --------------------------

out_path = "compare_image0.png"

# display = cv2.resize(img, (960, 600))

cv2.imwrite(out_path, img)
# 
print("Saved:", out_path)