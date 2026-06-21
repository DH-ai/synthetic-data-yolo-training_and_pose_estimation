import blenderproc as bproc
import time
import os
import bpy
import numpy as np
import cv2
import logging
import sys

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "errors.log")


def setup_file_logger(path: str = LOG_PATH) -> None:
    """Configure root logger to write only to a file (no console output)."""
    root = logging.getLogger()
    root.setLevel(logging.ERROR)

    # Remove any existing handlers (avoid console handlers)
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)

    root.addHandler(fh)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """Handle uncaught exceptions by logging them to file and exiting silently."""
    logging.getLogger().error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    sys.exit(1)


# Configure logging and suppress console output
try:
    setup_file_logger()
    sys.excepthook = _excepthook
    # Redirect stdout/stderr to devnull so nothing is printed to console
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
except Exception:
    # If setup fails, ensure at least the excepthook is set
    try:
        sys.excepthook = _excepthook
    except Exception:
        pass


bproc.init()
GAMMA = 0.712
CONTRAST = 0.513
NOISE_STD_MAX = 0.03   # upper bound; each iteration samples sigma ~ uniform(0, NOISE_STD_MAX)
BLUR_PROB = 0.2        # probability of applying a slight Gaussian blur to an image
BASE_EXPOSURE = 0.0    # Blender exposure offset base; jittered +-0.5 each iteration

# --- Data generation config ---
# Number of scene/render iterations (data points). Overridable via the NUM_ITERATIONS env var (used by Docker).
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "1"))
INWARD_FRACTION = 0.8       # drop objects only within the inner 80% of the table top
SPAWN_HEIGHT_OFFSET = 0.02  # meters above the table top to spawn objects before the (flat) drop
SPAWN_HEIGHT_STAGGER = 0.024  # extra random height per object so overlapping footprints don't collide at spawn
CAMERA_SAMPLE_PROB = 0    # 10% of the time sample the camera, 80% use the fixed pose
CIRCLE_TOP_CONST = 0.999   # y-threshold for the top 0.2% area of a unit circle
HDRI_BASE_STRENGTH = 1.3    # base HDRI strength before randomization
RANDOM_RANGE = 0.5          # +-50% randomization range for HDRI strength and light energy

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

def kelvin_to_rgb(temp_k: float):
    """Convert a color temperature in Kelvin to a normalized RGBA color.

    Uses the Tanner Helland piecewise approximation, valid for 1000–40000 K.
    Returns [r, g, b, 1.0] with values in [0, 1].
    """
    t = temp_k / 100.0
    # Red
    r = 1.0 if t <= 66 else float(np.clip(329.698727446 * ((t - 60) ** -0.1332047592) / 255.0, 0, 1))
    # Green
    if t <= 66:
        g = float(np.clip((99.4708025861 * np.log(t) - 161.1195681661) / 255.0, 0, 1))
    else:
        g = float(np.clip(288.1221695283 * ((t - 60) ** -0.0755148492) / 255.0, 0, 1))
    # Blue
    if t >= 66:
        b = 1.0
    elif t <= 19:
        b = 0.0
    else:
        b = float(np.clip((138.5177312231 * np.log(t - 10) - 305.0447927307) / 255.0, 0, 1))
    return [r, g, b]


def apply_image_adjustments(colors, noise_sigma: float = 0.0, exposure: float = 0.0, gamma_contrast: bool = True):
    """Apply exposure, gamma/contrast, per-iteration Gaussian noise, and occasional blur.

    :param colors:         List of HxWx3 uint8 arrays from the renderer.
    :param noise_sigma:    Std-dev of additive Gaussian noise for this iteration (sampled externally).
    :param exposure:       EV stop offset: image is multiplied by 2^exposure before other ops.
    :param gamma_contrast: Apply the calibrated GAMMA / CONTRAST correction.
    """
    adjusted = []
    for img in colors:
        x = np.asarray(img, dtype=np.float32) / 255.0

        # Exposure in EV stops: image * 2^exposure (identical to Blender's exposure slider)
        if exposure != 0.0:
            x = x * (2.0 ** exposure)

        if gamma_contrast:
            x = np.power(np.clip(x, 0.0, 1.0), GAMMA)
            x = (x - 0.5) * CONTRAST + 0.5

        # Additive Gaussian sensor noise with per-iteration sigma
        if noise_sigma > 0.0:
            x = x + np.random.normal(0.0, noise_sigma, x.shape)

        x = np.clip(x, 0.0, 1.0)
        img_u8 = (x * 255.0).astype(np.uint8)

        # 20% chance of a slight Gaussian blur (simulates focus softness / motion)
        if np.random.rand() < BLUR_PROB:
            img_u8 = cv2.GaussianBlur(img_u8, (5, 5), sigmaX=0)

        adjusted.append(img_u8)
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
    # print(f"Sampling camera pose on a dome of radius {radius:.3f} m centered at {table_center}")
    # print(f"Dist above center: {CIRCLE_TOP_CONST * radius:.3f} m (top {100 * (1 - CIRCLE_TOP_CONST ** 2):.1f}% of the dome)")
    location = bproc.sampler.part_sphere(
        center=table_center,
        radius=radius,
        mode="SURFACE",
        dist_above_center=CIRCLE_TOP_CONST * radius,
        part_sphere_dir_vector=[0, 0, 1],
    )

    poi = bproc.object.compute_poi(targets)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        poi - location, inplane_rot=np.random.uniform(0.1, -0.1)
    )
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


def main():


    scene = bproc.loader.load_blend("blender_files/moved_v9.blend",
                                data_blocks="objects",
    obj_types=["mesh", "light"])

    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.use_compositing = False
    view_layer = bpy.context.view_layer
    view_layer.use_pass_normal = False
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
    # Not renaiming the names as the new blend file already has the correct names set
    for i, obj in enumerate(triangle):
        print(obj.get_name())
        # obj.set_name(f"triangle_{i + 1}")
        # print("setting name",obj.get_name())
        obj.set_cp("category_id", category["Triangle"])

    for i, obj in enumerate(semiC):
        print(obj.get_name())
        # obj.set_name(f"semicircle_{i + 1}")
        # print("setting name",obj.get_name())
        obj.set_cp("category_id", category["SemiC"])

    for i, obj in enumerate(heart):
        print(obj.get_name())
        # obj.set_name(f"heart_{i + 1}")
        # print("setting name",obj.get_name())

        obj.set_cp("category_id", category["Heart"])
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
    bproc.renderer.set_max_amount_of_samples(128)
    bproc.renderer.engine = "EEVEE"  # faster than Cycles and with good enough quality for our purposes
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"],default_values={"category_id": 0})
    avge_time = 0.0

    for it in range(NUM_ITERATIONS):
        # Reset keyframes so camera poses do not accumulate across iterations
        # continue
        bproc.utility.reset_keyframes()

        # Randomize HDRI strength (+-30%) and re-apply the background
        hdri_strength = HDRI_BASE_STRENGTH * np.random.uniform(1 - RANDOM_RANGE, 1 + RANDOM_RANGE)
        bproc.world.set_world_background_hdr_img(
            "assets/hdri_hugin/hdri/frames_0001 - frames_0111.tif",
            strength=hdri_strength,
        )

        # Randomize the energy and color temperature of the existing scene lights
        light_temp_k = np.random.uniform(3500, 6500)
        light_color = kelvin_to_rgb(light_temp_k)
        for light, base_energy in zip(lights, light_base_energies):
            light.set_energy(base_energy * np.random.uniform(1 - RANDOM_RANGE, 1 + RANDOM_RANGE))
            light.set_color(light_color)

        # Randomize camera exposure +-0.5 stops around the base (applied in post as 2^exposure)
        exposure = BASE_EXPOSURE + np.random.uniform(-0.7, 0.7)
        
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
        



        t_render = time.time()
        data = bproc.renderer.render()
        t_render = time.time() - t_render
        # Per-iteration noise sigma sampled from uniform(0, NOISE_STD_MAX)
        noise_sigma = np.random.uniform(0.0, NOISE_STD_MAX)
        data["colors"] = apply_image_adjustments(data["colors"], noise_sigma=noise_sigma, exposure=exposure, gamma_contrast=False)



        coco = False
        t_writer = time.time()
        if coco:
            print("Writing COCO annotations...")
            bproc.writer.write_coco_annotations(
                output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/coco"),
                instance_segmaps=data["instance_segmaps"],
                instance_attribute_maps=data["instance_attribute_maps"],
                colors=data["colors"],
                color_file_format="PNG",
            )
        else:
            
            print("Writing BOP annotations...")
            bproc.writer.write_bop(
                output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/bop"),
                target_objects=target_objects,
                colors = data["colors"],
                depths = data["depth"],
                color_file_format="PNG",
                annotation_unit="mm",
                
            )
        
        t_writer = time.time() - t_writer
        # exopusure value, noise sigma, temperature value, gamma_contrast
        if coco:
            writer = "COCO"
        else:            
            writer = "BOP"

        avge_time += t_render + t_writer
        print(f"Iteration {i}: Render time: {t_render:.2f} s, {writer} write time: {t_writer:.2f} s")
        print(f"Average time per iteration: {avge_time / i:.2f} s")
        print(f"Iteration {i}: Exposure: {exposure:.2f} EV")
        print(f"Iteration {i}: Noise sigma: {noise_sigma:.4f}")
        print(f"Iteration {i}: Light temp: {light_temp_k:.0f} K")
        print(f"Iteration {i}/{NUM_ITERATIONS} complete.......")
        i += 1




if __name__=='__main__':
    main()
    
 

    