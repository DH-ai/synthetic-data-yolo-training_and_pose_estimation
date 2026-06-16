import blenderproc as bproc
import os 
import numpy as np
# import blenderproc.api as bproc_api



bproc.init()
GAMMA = 0.712
CONTRAST = 0.513
K = np.array([[2481.9412514178307, 0.0, 978.95936559694314],
        [0.0, 2482.3917472975795, 629.72289542481894],
        [0.0, 0.0, 1.0]],dtype=np.float64)

dist =   np.array([ -0.091539129459748417, 1.6518788910916924,
    -0.00096826424151305102, -0.0023115236516727399,
    -7.1086932137755738]) 
WIDTH = 1920
HEIGHT = 1200

R = np.array([[ 7.20424879e-04, -9.99956100e-01,  9.34230221e-03],
 [ 9.95211168e-01, -1.96226122e-04, -9.77481110e-02],
 [ 9.77456531e-02,  9.36798366e-03,  9.95167337e-01]],dtype=np.float64)
t = np.array([ 0.07441451, -0.08793114,  1.0063877 ],dtype=np.float64)


def set_camera():
    """Set BlenderProc camera intrinsics from the calibrated K matrix."""
    bproc.camera.set_intrinsics_from_K_matrix(K, HEIGHT, WIDTH)
    # bproc.camera.set_lens_distortion(dist[0],dist[1],dist[2],dist[3],dist[4])
    # bproc.camera.set_lens_distortion(*dist )
    # transformation_matrix = np.zeros((4,4))
    # transformation_matrix[:3, :3] = R
    # transformation_matrix[:3, 3] = t
    # transformation_matrix[3, 3] = 1


    T_board_to_cam = np.eye(4)
    T_board_to_cam[:3, :3] = R
    T_board_to_cam[:3, 3]  = t

    T_cw_opencv = np.linalg.inv(T_board_to_cam)   # this is what BlenderProc needs

    M = np.diag([1, 1, 1, 1]).astype(np.float64)
    T_cw_blender = T_cw_opencv @ M

    bproc.camera.add_camera_pose(T_cw_blender)

scene = bproc.loader.load_blend("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/blender_files/moved_v1.blend",
                                data_blocks="objects",
    obj_types=["mesh", "light"])


# Set the background environment using an HDRI file
bproc.world.set_world_background_hdr_img("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/hdri_hugin/hdri/frames_0001 - frames_0111.tif", strength=1.3)


set_camera()




print(bproc.camera.get_camera_pose())


