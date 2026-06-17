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


Test_pose = [
        [-7.20306474e-04,  9.99605238e-01,  2.80854404e-02,  0.00000000e+00],
        [ 9.99956012e-01,  4.57608112e-04,  9.35884845e-03,  0.00000000e+00],
        [-9.34230164e-03, -2.80909501e-02,  9.99561667e-01,  1.14627576e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ]


def set_camera():
    """Set BlenderProc camera intrinsics from the calibrated K matrix."""
    bproc.camera.set_intrinsics_from_K_matrix(K, WIDTH,HEIGHT)
    
    # first we will have world to cam matrix Tw->c

    # T_wtoc = np.eye(4, dtype=np.float64)
    # T_wtoc[:3, :3] = R
    # T_wtoc[:3, 3] = t
  
    # # T_ctow = inverse (t_wtoc)

    # T_ctow = np.linalg.inv(T_wtoc)
    # # print(np.dot(T_c) T_wtoc)
    # # But its assuming final rotation in the opencv convention of camera, so need to rotate it by 180 degrees about x axis to get the blender convention of camera
    # T_ctow = T_ctow @ np.array([[-1, 0, 0, 0],
    #                     [0, 1, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, -1]], dtype=np.float64)
    # T_ctow[:2,3]= [0,0]
    bproc.camera.add_camera_pose(Test_pose)

scene = bproc.loader.load_blend("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/blender_files/moved_v3.blend",
                                data_blocks="objects",
    obj_types=["mesh", "light"])


# Set the background environment using an HDRI file
bproc.world.set_world_background_hdr_img("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/hdri_hugin/hdri/frames_0001 - frames_0111.tif", strength=1.3)


set_camera()




normal_obj = []
for obj in scene:
    if type(obj) ==bproc.types.MeshObject: normal_obj.append(obj)
    
    
    
category = {
    "T1":0,
    "T2":1,
    "table":2,
    "SC1":3,
    "SC2":4,
    "H1":5,
    "H2":6
    
}

#print(normal_obj)

# 2 -> table 
#print(normal_obj[6].edit_mode())


map_obj = [
    "T1",
    "T2",
    "table",
    "SC1",
    "SC2",
    "H1",
    "H2"

]
for obj in normal_obj:
    obj.set_cp("category_id", category[map_obj[normal_obj.index(obj)]])


# bproc.renderer.enable_segmentation_output(map_by=map_obj)

seg_data = bproc.renderer.render_segmap(map_by=["name", "instance", "class"])
print(seg_data["instance_attribute_maps"])
print(np.unique(seg_data["instance_segmaps"][0]))
data = bproc.renderer.render()


bproc.writer.write_coco_annotations(
    output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
    instance_segmaps=seg_data["instance_segmaps"],
    instance_attribute_maps=seg_data["instance_attribute_maps"],
    colors=data["colors"],
    color_file_format="JPEG"
)

