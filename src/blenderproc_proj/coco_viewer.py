import cv2
import numpy as np
from pycocotools import mask as maskUtils
import json

# --- 1. Load your COCO annotation file ---
json_file_path = 'src/blenderproc_proj/output/coco_annotations.json'  # <-- Change this!
with open(json_file_path, 'r') as f:
    coco_data = json.load(f)

# --- 2. Get the first annotation as an example ---
# Each annotation in the list represents one object instance.
first_annotation = coco_data['annotations'][1]

# The 'segmentation' field can be a polygon (list) or RLE (dict).
# Here we assume it's the RLE format you showed.
segmentation = first_annotation['segmentation']

# The image height and width are usually stored in the 'images' list.
# We can find the corresponding image info using the 'image_id'.
image_id = first_annotation['image_id']
image_info = None
for img in coco_data['images']:
    if img['id'] == image_id:
        image_info = img
        break

if image_info is None:
    # Fallback: if image info is not found, you might know the dimensions.
    # For this example, we'll use dummy values, but you should replace them.
    height, width = 800, 1200
    print("Warning: Image info not found. Using dummy dimensions.")
else:
    height, width = image_info['height'], image_info['width']

# --- 3. Decode the RLE mask ---
# The frPyObjects function ensures the RLE is in the correct format for decoding.
rle = maskUtils.frPyObjects(segmentation, height, width)
mask = maskUtils.decode(rle)  # This returns a 2D numpy array (H, W) with values 0 or 1

# --- 4. Visualize the mask with OpenCV ---
# Convert the binary mask to an 8-bit image (0-255) for display.
mask_image = (mask * 255).astype(np.uint8)

cv2.imshow('Segmentation Mask', mask_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the mask to a file.
# cv2.imwrite('decoded_mask.png', mask_image)