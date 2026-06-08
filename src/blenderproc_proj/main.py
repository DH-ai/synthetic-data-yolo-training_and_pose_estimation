import blenderproc as bproc
# from blenderproc import 
import os 
import numpy as np
bproc.init()

ASSET = os.path.join("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation", "assets", "models")


SAVEASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)),"assets")


OUTPUTPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# Load objects
objs = bproc.loader.load_obj(os.path.join(ASSET, "triangle_shape.obj"))

# Add light
light = bproc.types.Light()

# Set camera
bproc.camera.set_resolution(640, 480)

# Add camera poses


cam_pose = bproc.math.build_transformation_mat([0, 0, 1], [0, 0])
bproc.camera.add_camera_pose(cam_pose)
# Render
data = bproc.renderer.render()

# Write annotations
bproc.writer.write_hdf5(os.path.join(OUTPUTPATH, "annotations.hdf5"), data)
# bproc.writer.write_coco_annotations(os.path.join("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation", "assets", "annotations.json"), data)
