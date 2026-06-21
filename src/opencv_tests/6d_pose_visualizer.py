#!/usr/bin/env python3
"""
6D pose Visualizer: DATA + image → pose projected on RGB image.
Dataset Structure 
/├── /    
    ├──train_pbr/
        ├──000000/
            ├──depth/00000*_00000*.png
            ├──mask/00000*_00000*.png
            ├──mask_visib/00000*_00000*.png
            ├──rgb/00000*.png
            ├──scene_camera.json
            ├──scene_gt_coco.json
            ├──scene_gt_info.json
            ├──scene_gt.json
    ├──camera.json


"""

import numpy as np
import open3d as o3d
import os 
import cv2
import copy
from pathlib import Path
import json 

# ROI x and y of the Image 
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y 



# taking entire image as ROI was for point cloud in IPC_test.py
SCENE_ROI_MIN = Point(0, 0) # Image
SCENE_ROI_MAX = Point(1920, 1200) 


DATASET_PATH = Path(__file__).parent.parent / "blenderproc_proj/output/bop/"
# print(f"DATASET_PATH: {(DATASET_PATH.exists(), DATASET_PATH)}")

# camera intrinsic 
K = np.array([[2481.9412514178307, 0.0, 978.95936559694314],
            [0.0, 2482.3917472975795, 629.72289542481894],
            [0.0, 0.0, 1.0]],dtype=np.float64)

dist =   np.array([[
    -0.091539129459748417,
     1.6518788910916924,
     -0.00096826424151305102,
     -0.0023115236516727399,
     -7.1086932137755738
     ]]) 



# Draw Bounding box around the object 
def draw_bounding_box(image, bbox:np.ndarray=None):
    pass
# Draw the axis from the origin usnig rvec and tvec 
def draw_pose(image, rvec, tvec, K, dist):
    cv2.drawFrameAxes(image, K, dist, rvec, tvec, 50)

    # Draw the pose on the image using OpenCV's projectPoints function
    # axis_length = 0.05  # Length of the axes to be drawn
    # axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
    # imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist)

    # # Draw the axes on the image
    # corner = tuple(imgpts[0].ravel())
    # image = cv2.line(image, corner, tuple(imgpts[1].ravel()), (255, 0, 0), 5)  # X-axis in blue
    # image = cv2.line(image, corner, tuple(imgpts[2].ravel()), (0, 255, 0), 5)  # Y-axis in green
    # image = cv2.line(image, corner, tuple(imgpts[3].ravel()), (0, 0, 255), 5)  # Z-axis in red

    # return image


def process_json(dataset_type:str="bop"):
    if dataset_type == "bop":
        scene_gt_path = DATASET_PATH / "train_pbr/000000/scene_gt.json"
        scene_gt_data = json.decoder.JSONDecoder().decode(open(scene_gt_path).read())
    
    return scene_gt_data



    

def bop_rvec_tvec(scene_gt_data, instance_id=0):
    # Extract the rotation vector (rvec) and translation vector (tvec) for the specified instance ID
    instance_data = scene_gt_data[str(instance_id)]

    # for all objects in the img
    rvec_arr=[]
    tvec_arr=[]
    # for instance_id in scene_gt_data:

        # rvec = np.array(instance_data[0]['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
        # tvec = np.array(instance_data[0]['cam_t_m2c'], dtype=np.float64).reshape(3, 1)
    for item in instance_data:
        rvec = np.array(item['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
        tvec = np.array(item['cam_t_m2c'], dtype=np.float64).reshape(3, 1)
        rvec_arr.append(rvec)
        tvec_arr.append(tvec)
    size = len(instance_data)
    return rvec_arr, tvec_arr, size


def main(dataset:str="bop"):
    if dataset=="bop":
        json_obj = process_json(dataset)
        i = 41

        img_path = DATASET_PATH / f"train_pbr/000000/rgb/0000{i}.png"
        rvec_arr, tvec_arr, size = bop_rvec_tvec(json_obj, instance_id=i)
        
        #reading the image 
        img = cv2.imread(str(img_path))
        for i in range(size):
            draw_pose(img, rvec_arr[i], tvec_arr[i], K, dist)
        cv2.imshow("Pose Visualization", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main("bop")