import blenderproc as bproc


# from blenderproc import 
import os 
import numpy as np
bproc.init()
# from implmentation.camera import set_camera
# from src.camera import set_camera
# import camera
import blenderproc.api as bproc_api



GAMMA = 0.712
CONTRAST = 0.513
K = np.array([[2481.9412514178307, 0.0, 978.95936559694314],
        [0.0, 2482.3917472975795, 629.72289542481894],
        [0.0, 0.0, 1.0]],dtype=np.float64)

dist =   np.array([[ -0.091539129459748417, 1.6518788910916924,
    -0.00096826424151305102, -0.0023115236516727399,
    -7.1086932137755738]]) 
WIDTH = 1920
HEIGHT = 1200







def set_camera():
    """Set BlenderProc camera intrinsics from the calibrated K matrix."""
    bproc.camera.set_intrinsics_from_K_matrix(K, WIDTH, HEIGHT)
    bproc.camera.set_lens_distortion(dist)
    # bproc.camera.add_camera_pose()







scene = bproc.loader.load_blend("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/blender_files/scene_final_v2.blend",
                                data_blocks="objects",
    obj_types=["mesh", "light"])


# Set the background environment using an HDRI file
bproc.world.set_world_background_hdr_img("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/hdri_hugin/hdri/frames_0001 - frames_0111.tif", strength=1.3)


set_camera()




























# ==========================================
# 5. Render (The heavy lifting)
# ==========================================
#data = bproc.renderer.render()

# Filter for just your target mesh objects
#target_objs = [obj for obj in objs if isinstance(obj, bproc.types.MeshObject)]

# ==========================================
# 6. Export Data (The fast part)
# ==========================================

# Writer 1: BOP Format (For your 6D Pose Model)
# bproc.writer.write_bop(
#     output_dir=args.output_dir,
#     target_objects=target_objs,
#     dataset_name="aeroforge_dataset",
#     depth_scale=1.0,
#     depth_type=np.uint16,
#     append_to_existing_output=True,
#     save_world2cam=True
# )

# Writer 2: COCO Format (For YOLO Bounding Boxes & Segmentation)
# bproc.writer.write_coco_annotations(
#     output_dir=args.output_dir,
#     instance_segmaps=data["instance_segmaps"],
#     instance_attribute_maps=data["instance_attribute_maps"],
#     colors=data["colors"],
#     color_file_format="JPEG", # or PNG
#     append_to_existing_output=True
# )