# from email.mime import image

import blenderproc as bproc
import os 
import numpy as np
# import blenderproc.api as bproc_api

import cv2

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







   
    


#print(normal_obj)

# 2 -> table 
#print(normal_obj[6].edit_mode())




# bproc.renderer.enable_segmentation_output(map_by=map_obj)





def main():
    # bproc.renderer.enable_segmentation_output(map_by=map_obj)
    # seg_data = bproc.renderer.render_segmap(map_by=["name", "instance", "class"])
    # print(seg_data["instance_attribute_maps"])
    # print(np.unique(seg_data["instance_segmaps"][0]))
    # data = bproc.renderer.render()
    # bproc.writer.write_coco_annotations(
    #     output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
    #     instance_segmaps=seg_data["instance_segmaps"],
    #     instance_attribute_maps=seg_data["instance_attribute_maps"],
    #     colors=data["colors"],
    #     color_file_format="JPEG"
    # )


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
    "Triangle":1,
    "SemiC":2,
    "Heart":3,

    
    }
    # map_obj = [

    # "table",
    # "SemiC",
    # "Heart",
    # "Triangle"

    # ]
    triangle = normal_obj[0:2]
    table = normal_obj[2:3]
    semiC = normal_obj[3:5]
    heart = normal_obj[5:7]
  
    for obj in triangle:
        obj.set_cp("category_id", category["Triangle"])
        print(obj.get_name())
    for obj in semiC:
        obj.set_cp("category_id", category["SemiC"])
        print(obj.get_name())


    for obj in heart:
        obj.set_cp("category_id", category["Heart"])
        print(obj.get_name())
        



    # print(10*"=")
    
    # for i in seg_data["instance_attribute_maps"][0]:
    #     print(i)
    # print(10*"=")
    
    # # print(seg_data.get("instance_segmaps"))
    # print(10*"=")
    
    # print(seg_data.get("class_segmaps"))
    # print(seg_data["instance_attribute_maps"])
    # print(np.unique(seg_data["instance_segmaps"]))
    # return None
    bproc.renderer.set_max_amount_of_samples(1)
    bproc.renderer.enable_depth_output(activate_antialiasing=True)
    bproc.renderer.enable_normals_output()
    # bproc.
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"],default_values={"category_id": 0})
    # seg_data = bproc.renderer.render_segmap(map_by=[
    #     "name",
    #     "instance",
    #     "class",
    data = bproc.renderer.render()
    print(data.keys())
    # exit()

    # bproc.writer.write_coco_annotations(
    #     output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
    #     instance_segmaps=data["instance_segmaps"],
    #     instance_attribute_maps=data["instance_attribute_maps"],
    #     colors=data["colors"],
    #     color_file_format="PNG"
    # )

    bproc.writer.write_bop(
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
        target_objects=triangle + semiC + heart,
        depths = data["depth"],
        colors = data["colors"],
        color_file_format="PNG",



    )


if __name__=='__main__':
    main()
    
 

    