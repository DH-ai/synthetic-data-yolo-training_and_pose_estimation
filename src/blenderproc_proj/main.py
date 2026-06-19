# from email.mime import image

import blenderproc as bproc
import os 
import numpy as np
# import blenderproc.api as bproc_api

import cv2

bproc.init()
GAMMA = 0.712
CONTRAST = 0.513
NOISE_STD = 0.02  # std-dev of additive Gaussian image noise, in normalized [0, 1] range

# --- Data generation config ---
NUM_ITERATIONS = 1        # number of scene/render iterations (number of data points)
INWARD_FRACTION = 0.8       # drop objects only within the inner 90% of the table top
SPAWN_HEIGHT_OFFSET = 0.02  # meters above the table top to spawn objects before the (flat) drop
SPAWN_HEIGHT_STAGGER = 0.0  # extra random height per object so overlapping footprints don't collide at spawn
CAMERA_SAMPLE_PROB = 0.0    # 20% of the time sample the camera, 80% use the fixed pose
CIRCLE_TOP_CONST = 0.4919   # y-threshold for the top 20% area of a unit circle
HDRI_BASE_STRENGTH = 1.3    # base HDRI strength before randomization
RANDOM_RANGE = 0.3          # +-30% randomization range for HDRI strength and light energy

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


def apply_image_adjustments(colors, gamma_contrast: bool = True)->np.ndarray:
    """Apply the calibrated gamma/contrast and add Gaussian noise to rendered RGB images.

    Blender's display post-processing is disabled (view_transform="Standard"), so these are
    the only adjustments applied. Operates on a list of HxWx3 uint8 arrays from the renderer.
    """
    adjusted = []
    for img in colors:
        x = np.asarray(img, dtype=np.float32) / 255.0


        
        if gamma_contrast:
            # Calibrated gamma correction
            x = np.power(np.clip(x, 0.0, 1.0), GAMMA)
            # Calibrated contrast around mid-gray
            x = (x - 0.5) * CONTRAST + 0.5
    
    
        # Additive Gaussian sensor noise
        x = x + np.random.normal(0.0, NOISE_STD, x.shape)
        x = np.clip(x, 0.0, 1.0)
        adjusted.append((x * 255.0).astype(np.uint8))
    return adjusted


def set_camera()->None:
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


def sample_camera_pose(targets, table_center)->None:
    """Sample a camera pose on the top part of a dome centered at the table.

    The dome radius equals the current camera distance from the world origin and
    the cap is restricted using the provided top-of-circle constant. The camera is
    oriented to look at the dropped objects.
    """
    bproc.camera.set_intrinsics_from_K_matrix(K, WIDTH, HEIGHT)

    radius = float(np.linalg.norm(np.array(Test_pose)[:3, 3]))

    location = bproc.sampler.part_sphere(
        center=table_center,
        radius=radius,
        mode="SURFACE",
        dist_above_center=CIRCLE_TOP_CONST * radius,
        part_sphere_dir_vector=[0, 0, 1],
    )

    poi = bproc.object.compute_poi(targets)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        poi - location, inplane_rot=np.random.uniform(-0.349, 0.349)
    )
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


def main():


    scene = bproc.loader.load_blend("/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/blender_files/moved_v3.blend",
                                data_blocks="objects",
    obj_types=["mesh", "light"])


    normal_obj = []
    for obj in scene:
        if type(obj) ==bproc.types.MeshObject: normal_obj.append(obj)


    category = {
    "Triangle":1,
    "SemiC":2,
    "Heart":3,

    
    }

    triangle = normal_obj[0:2]
    table = normal_obj[2:3]
    semiC = normal_obj[3:5]
    heart = normal_obj[5:7]

    # Rename objects to the heart_1 / heart_2 convention and assign category ids
    for i, obj in enumerate(triangle):
        obj.set_name(f"triangle_{i + 1}")
        obj.set_cp("category_id", category["Triangle"])
        print(obj.get_name())
    for i, obj in enumerate(semiC):
        obj.set_name(f"semicircle_{i + 1}")
        obj.set_cp("category_id", category["SemiC"])
        print(obj.get_name())
    for i, obj in enumerate(heart):
        obj.set_name(f"heart_{i + 1}")
        obj.set_cp("category_id", category["Heart"])
        print(obj.get_name())
    table[0].set_name("table")

    target_objects = triangle + semiC + heart

    # Collect lights already in the scene and remember their base energy for randomization
    lights = [obj for obj in scene if isinstance(obj, bproc.types.Light)]
    light_base_energies = [light.get_energy() for light in lights]

    # --- Physics drop setup ---
    # Compute the table top height and the inner-90% XY region from the table bound box
    table_bb = np.array(table[0].get_bound_box())
    table_top_z = float(table_bb[:, 2].max())
    xy_min = table_bb[:, :2].min(axis=0)
    xy_max = table_bb[:, :2].max(axis=0)
    xy_center = (xy_min + xy_max) / 2.0
    xy_half = (xy_max - xy_min) / 2.0 * INWARD_FRACTION
    inner_min = xy_center - xy_half
    inner_max = xy_center + xy_half

    spawn_z = table_top_z + SPAWN_HEIGHT_OFFSET
    table_center = [float(xy_center[0]), float(xy_center[1]), table_top_z]

    # Remember each object's original (face-up) orientation so we only randomize the yaw
    base_rotation_by_name = {obj.get_name(): np.array(obj.get_rotation_euler()) for obj in target_objects}

    # Enable rigid bodies: targets are active (they fall), the table is passive
    for obj in target_objects:
        obj.enable_rigidbody(active=True)
    table[0].enable_rigidbody(active=False, collision_shape="MESH")

    def sample_pose_func(obj: bproc.types.MeshObject):
        x = np.random.uniform(inner_min[0], inner_max[0])
        y = np.random.uniform(inner_min[1], inner_max[1])
        # Stagger the spawn height so objects with overlapping XY footprints don't collide at spawn
        z = spawn_z + np.random.uniform(0, SPAWN_HEIGHT_STAGGER)
        obj.set_location([x, y, z])

        # Keep the object's original face-up orientation, randomize only the in-plane (yaw) rotation
        base_rot = base_rotation_by_name[obj.get_name()]
        obj.set_rotation_euler([base_rot[0], base_rot[1], np.random.uniform(0, 2 * np.pi)])

    # --- Renderer config (set once) ---
    # Remove Blender's display post-processing (Filmic/AgX look) so only our gamma/contrast apply
    # bproc.renderer.set_output_format(view_transform="Standard")
    # bproc.renderer.set_render_devices(["GPU"])
    bproc.renderer.enable_depth_output(activate_antialiasing=False)  # for perfect depth maps without interpolation artifacts
    bproc.renderer.set_max_amount_of_samples(256)
    bproc.renderer.engine = "CYCLES"
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"],default_values={"category_id": 0})

    for it in range(NUM_ITERATIONS):
        # Reset keyframes so camera poses do not accumulate across iterations
        bproc.utility.reset_keyframes()

        # Randomize HDRI strength (+-30%) and re-apply the background
        hdri_strength = HDRI_BASE_STRENGTH * np.random.uniform(1 - RANDOM_RANGE, 1 + RANDOM_RANGE)
        bproc.world.set_world_background_hdr_img(
            "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/hdri_hugin/hdri/frames_0001 - frames_0111.tif",
            strength=hdri_strength,
        )

        # Randomize the energy of the existing scene lights (sun / point) by +-30%
        for light, base_energy in zip(lights, light_base_energies):
            light.set_energy(base_energy * np.random.uniform(1 - RANDOM_RANGE, 1 + RANDOM_RANGE))

        # Randomize object positions by dropping them onto the table with physics
        bproc.object.sample_poses(
            objects_to_sample=target_objects,
            sample_pose_func=sample_pose_func,
            max_tries=10,
        )
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=2,
            max_simulation_time=10,
            check_object_interval=1,
        )

        # Camera: 20% sampled on the dome, 80% the fixed calibrated pose
        if np.random.rand() < CAMERA_SAMPLE_PROB:
            sample_camera_pose(target_objects, table_center)
        else:
            set_camera()

        data = bproc.renderer.render()

        # Apply calibrated gamma/contrast and add noise (Blender post-processing already disabled)
        data["colors"] = apply_image_adjustments(data["colors"], gamma_contrast=True)

        coco = False
        if coco:
            bproc.writer.write_coco_annotations(
                output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/coco"),
                instance_segmaps=data["instance_segmaps"],
                instance_attribute_maps=data["instance_attribute_maps"],
                colors=data["colors"],
                color_file_format="PNG",
            )
        else:
            bproc.writer.write_bop(
                output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/bop"),
                target_objects=target_objects,
                colors = data["colors"],
                color_file_format="PNG",
                depth = data["depth"],
                annotation_unit="mm",
            )



if __name__=='__main__':
    main()
    
 

    