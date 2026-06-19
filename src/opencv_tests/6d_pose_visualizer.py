#!/usr/bin/env python3
"""
6D pose Visualizer: DATA + image → pose projected on RGB image.
"""

import numpy as np
import open3d as o3d

import cv2
import copy
from pathlib import Path



# ROI x and y of the Image 
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y 



# taking entire image as ROI was for point cloud in IPC_test.py
SCENE_ROI_MIN = Point(0, 0) # Image
SCENE_ROI_MAX = Point(1920, 1200) 


# ICP params  (units = same as your point cloud — usually mm)
VOXEL_SIZE = 2.0          # downsample voxel size
ICP_MAX_ITER = 100
AXIS_DRAW_LEN = None       # None → auto (30% of longest bbox edge)


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

