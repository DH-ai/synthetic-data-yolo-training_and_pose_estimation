import blenderproc as bproc
import time
import os

import bpy # type: ignore
import numpy as np
import cv2
import logging
import sys
import warnings
import re

#TODO; TURN OF ALL LOGGING AND HAVE A PROGRESS BAR VIA TQDM

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "errors.log")
LAST_RUN_STATE = None

# TODO: still not robust enought to catch all the errors
def setup_file_logger(path: str = LOG_PATH) -> None:
    """Configure logging to write warnings and errors to a file only."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any existing handlers (avoid console handlers)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    # FILE HANDLER -> Only captures ERROR and CRITICAL
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # CONSOLE (TERMINAL) HANDLER -> Captures INFO, WARNING, and ERROR
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)   # Lets you see progress in the terminal
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logging.captureWarnings(True)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """Handle uncaught exceptions by logging them to file and exiting silently."""
    logging.getLogger().error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    sys.exit(1)


def append_run_summary(state=None, path: str = LOG_PATH) -> None:
    """Append the last known run state to the log file once at exit."""
    if state is None:
        state = LAST_RUN_STATE

    if not state:
        return

    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO: Final run state\n")
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO: Iteration {state['iteration']}/{NUM_ITERATIONS} complete.......\n")
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO: Average time per iteration: {state['average_time']:.2f} s\n")
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO: Writer: {state['writer']}\n")


try:
    setup_file_logger()
    sys.excepthook = _excepthook
    warnings.simplefilter("default")
except Exception:
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
SPAWN_HEIGHT_STAGGER = 0.004  # extra random height per object so overlapping footprints don't collide at spawn
CAMERA_SAMPLE_PROB = 0    # 10% of the time sample the camera, 80% use the fixed pose
CIRCLE_TOP_CONST = 0.999   # y-threshold for the top 0.2% area of a unit circle
HDRI_BASE_STRENGTH = 1.3    # base HDRI strength before randomization
RANDOM_RANGE = 0.5          # +-50% randomization range for HDRI strength and light energy
DISTRACTOR_CATEGORY_ID = 0  # rendered in the image, ignored by BOP/COCO target labels
OUTPUT_DIR =  os.environ.get("OUTPUT_DIR_BPROC", "src/output/bop")

# Keep these IDs aligned with src/gdrnpp/ref/mydataset.py and existing BOP models.
TARGET_CLASSES = {
    "heart": {
        "id": 1,
        "patterns": ("heart",),
    },
    "semi_circle": {
        "id": 2,
        "patterns": ("semi circle", "semicircle", "semi-circle", "semic"),
    },
    "triangle": {
        "id": 3,
        "patterns": ("triangle",),
    },
}

SUPPORT_NAME_PATTERNS = ("table",) # Table for most cases, then drop the cavity plate, then randomize everything and then drop the objects on the table
PRIMARY_SUPPORT_PRIORITY = ("plate","plates")

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


def _normalized_name(obj: bproc.types.MeshObject) -> str:
    """
    Normalizes a MeshObject's name by:
    1. Converting to lowercase.
    2. Replacing '_' and '-' with spaces.
    3. Removing periods (.) and all numeric digits (0-9).
    4. Stripping and collapsing any extra duplicate spaces.
    """
    name = obj.get_name().lower()
    
    # 1. Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")
    
    # 2. Remove periods
    name = name.replace(".", "")
    
    # 3. Remove all numeric digits (0 through 9)
    name = re.sub(r"\d+", "", name)
    
    # 4. Collapse extra consecutive whitespace into single spaces and trim
    return " ".join(name.split())

def _matches_any(name: str, patterns) -> bool:
    """Check if the normalized object name contains any of the given patterns."""

    return any(pattern in name for pattern in patterns)



def _sort_by_name(objects):
    return sorted(objects, key=lambda obj: obj.get_name())

# TODO: Optimize this function, its running too mmany, for each n^2 complexity and space n
def split_scene_objects(mesh_objects):
    """Split meshes into labelled targets, passive support, and unlabelled moving negatives."""


    target_objects_by_class = {class_name: [] for class_name in TARGET_CLASSES} # created a dictionary with the class name 
    support_objects = [] # In our case a table
    distractor_objects = [] # everything wich is not target object and plate object and support_objects
    plate_obj = None  # Initialize plate_obj to None

    for obj in mesh_objects:
        name = _normalized_name(obj)
        matched_target_class = None

        # its to create class id for the target objects, and assign category_id to the object, so that it can be used to get final output
        for class_name, class_cfg in TARGET_CLASSES.items(): # this loop is not necessary _match_any can have class name patterns and we can remove this 
            # logging.info(f"  Checking against target class '{class_name}' with patterns: {class_cfg['patterns']} and config ")
            if _matches_any(name, class_cfg["patterns"]):
                matched_target_class = class_name
                # logging.info(f"  Object '{obj.get_name()}' matched target class '{class_name}' with patterns: {class_cfg['patterns']}")
                break
        # set the category_id for the object
        if matched_target_class is not None:
            class_id = TARGET_CLASSES[matched_target_class]["id"]
            obj.set_cp("category_id", class_id)
            target_objects_by_class[matched_target_class].append(obj)

        
        # I have set SUPPORT_NAME_PATTERNS to only "table" so that it can be used to identify the table object, 
        
        elif _matches_any(name, SUPPORT_NAME_PATTERNS):
            # logging.info(f"  Object'{obj.get_name()}' matched support patterns: {SUPPORT_NAME_PATTERNS}, name: {name}")
            obj.set_cp("category_id", DISTRACTOR_CATEGORY_ID)
            support_objects.append(obj)
        
        # for distractor obj also set category id to 0
        elif _matches_any(name, PRIMARY_SUPPORT_PRIORITY):
            # logging.info(f"  Object '{obj.get_name()}' matched plate pattern, name: {name}")
            obj.set_cp("category_id", DISTRACTOR_CATEGORY_ID)
            plate_obj = obj
        else:
            obj.set_cp("category_id", DISTRACTOR_CATEGORY_ID)
            # logging.info(f"  Object '{obj.get_name()}' did not match any target or support patterns, assigned category_id {DISTRACTOR_CATEGORY_ID} and added to distractors.")
            distractor_objects.append(obj)

    target_objects = []
    for class_name in sorted(TARGET_CLASSES, key=lambda key: TARGET_CLASSES[key]["id"]):
        target_objects.extend(_sort_by_name(target_objects_by_class[class_name]))

    missing_classes = [
        class_name
        for class_name, objects in target_objects_by_class.items()
        if not objects
    ]
    if missing_classes:
        raise RuntimeError(
            "Could not find target meshes for classes: "
            + ", ".join(missing_classes)
            + ". Check object names in the .blend file."
        )

    if not support_objects:
        raise RuntimeError(
            "Could not find a passive support mesh. Expected a name containing one of: "
            + ", ".join(SUPPORT_NAME_PATTERNS)
        )
    if plate_obj is None:
        raise RuntimeError(
            "Could not find the moving cavity/plate mesh. Expected a name containing one of: "
            + ", ".join(PRIMARY_SUPPORT_PRIORITY)
        )

    return target_objects_by_class, target_objects, support_objects, distractor_objects, plate_obj


    
# TODO: Optimize this function, its running too mmany, for each n^2 complexity and space n
def place_obj(moving_objects, plate_obj,max_tries=10, boundary=None):
    
    inner_min, inner_max, spawn_z, base_rotation_by_name = boundary

    def sample_pose_func(obj: bproc.types.MeshObject):
        x = np.random.uniform(inner_min[0], inner_max[0])
        y = np.random.uniform(inner_min[1], inner_max[1])
        # Stagger the spawn height so objects with overlapping XY footprints don't collide at spawn
        z = spawn_z + np.random.uniform(0, SPAWN_HEIGHT_STAGGER)
        obj.set_location([x, y, z])

        # Keep the object's original face-up orientation, randomize only the in-plane (yaw) rotation
        base_rot = base_rotation_by_name[obj.get_name()]
        obj.set_rotation_euler([base_rot[0], base_rot[1], np.random.uniform(0, 2 * np.pi)])


    # plate object placement 


    bproc.object.sample_poses(
        objects_to_sample=[plate_obj],
        sample_pose_func=sample_pose_func,
        max_tries=max_tries,
    )
    # bproc.object.simulate_physics_and_fix_final_poses(
    #     min_simulation_time=2,
    #     max_simulation_time=5,
    #     check_object_interval=1,
    # )
    # normal Object 
    bproc.object.sample_poses(
        objects_to_sample=moving_objects,
        sample_pose_func=sample_pose_func,
        max_tries=max_tries,
        )
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=2,
        max_simulation_time=10,
        check_object_interval=1,
    )
    
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

def eta(avg_t:float, iteration:int, num_iterations:int)->str:
    # time the estimated time remaining
    time_total = avg_t * (num_iterations)
    time_remaining = time_total - (avg_t * iteration)

    # Convert time remaining to a more readable format
    if time_remaining < 60:
        return f"{time_remaining:.1f} seconds"
    elif time_remaining < 3600:
        minutes = int(time_remaining // 60)
        seconds = int(time_remaining % 60)
        return f"{minutes} minutes, {seconds} seconds"
    else:
        hours = int(time_remaining // 3600)
        minutes = int((time_remaining % 3600) // 60)
        return f"{hours} hours, {minutes} minutes"
    

def main():
    global LAST_RUN_STATE
    logging.info("Running the generation loop for {}".format(NUM_ITERATIONS))

    scene = bproc.loader.load_blend("blender_files/moved_v11.blend",
                                data_blocks="objects",
    obj_types=["mesh", "light"])

    view_layer = bpy.context.view_layer
    view_layer.use_pass_normal = False
    
    
    # scene1 = bpy.context.scene
    # print("use_nodes:", scene1.use_nodes)
    # if scene1.node_tree:
    #     for node in scene1.node_tree.nodes:
    #         print(node.name, node.bl_idname)

    # if scene1.node_tree:
    #     for node in scene1.node_tree.nodes:
    #         if node.bl_idname == "CompositorNodeDenoise":
    #             scene1.node_tree.nodes.remove(node)
    #             print("Removed Denoise node from compositor.")
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.use_compositing = False
    # scene1.node_tree.nodes.clear()





    tree = bpy.context.scene.node_tree

    rlayers = next(n for n in tree.nodes
                if n.bl_idname == "CompositorNodeRLayers")

    comp = next(n for n in tree.nodes
                if n.bl_idname == "CompositorNodeComposite")

    denoise = next(n for n in tree.nodes
                if n.bl_idname == "CompositorNodeDenoise")

    tree.links.new(
        rlayers.outputs["Image"],
        comp.inputs["Image"]
    )

    tree.nodes.remove(denoise)





    normal_obj = [obj for obj in scene if isinstance(obj, bproc.types.MeshObject)]

    (
        target_objects_by_class,
        target_objects,
        support_objects,
        distractor_objects,
        plate_obj
    ) = split_scene_objects(normal_obj)


    moving_objects = target_objects + distractor_objects 


    # primary_support = choose_table_support(support_objects) No longer needed as only one supper objet is there
    primary_support = support_objects[0]
    logging.info(f"Primary support object: {primary_support.get_name()}") 


    logging.info("Scene object roles:")
    
    for class_name in sorted(TARGET_CLASSES, key=lambda key: TARGET_CLASSES[key]["id"]):
        class_id = TARGET_CLASSES[class_name]["id"]
        names = ", ".join(obj.get_name() for obj in target_objects_by_class[class_name])
        logging.info(f"  target {class_id} ({class_name}): {names}")

    
    logging.info(f"  support: {', '.join(obj.get_name() for obj in support_objects)}")
    logging.info(f"  primary support: {primary_support.get_name()}")
    logging.info(
        "  distractors: "
        + (", ".join(obj.get_name() for obj in distractor_objects) if distractor_objects else "none")
    )



    # Collect lights already in the scene and remember their base energy for randomization
    lights = [obj for obj in scene if isinstance(obj, bproc.types.Light)]
    light_base_energies = [light.get_energy() for light in lights]

    # --- Physics drop setup --

    # TODO: to implment a probability system to drop the objects over discs too, need to think about classifying them is in cavity
    table_bb = np.array(primary_support.get_bound_box())
    table_top_z = float(table_bb[:, 2].max())
    xy_min = table_bb[:, :2].min(axis=0)
    xy_max = table_bb[:, :2].max(axis=0)
    xy_center = (xy_min + xy_max) / 2.0
    xy_half = (xy_max - xy_min) / 2.0 * INWARD_FRACTION
    inner_min = xy_center - xy_half
    inner_max = xy_center + xy_half

    spawn_z = table_top_z + SPAWN_HEIGHT_OFFSET
    table_center = [float(xy_center[0]), float(xy_center[1]), table_top_z]

    # Remember each moving object's original (face-up) orientation so we only randomize the yaw.
    base_rotation_by_name = {obj.get_name(): np.array(obj.get_rotation_euler()) for obj in moving_objects}
    base_rotation_by_name.update({plate_obj.get_name(): np.array(plate_obj.get_rotation_euler())})  # Include plate object
    # Enable rigid bodies: labelled targets and negative distractors are active; supports are passive.
    for obj in moving_objects:
        obj.enable_rigidbody(active=True)
    for obj in support_objects:
        obj.enable_rigidbody(active=False, collision_shape="MESH") # our table
    plate_obj.enable_rigidbody(active=True)  # Enable the plate object as active rigid body
    



    # --- Renderer config (set once) ---
    # Remove Blender's display post-processing (Filmic/AgX look) so only our gamma/contrast apply
    # bproc.renderer.set_output_format(view_transform="Standard")
    # bproc.renderer.set_render_devices(["GPU"])
    bproc.renderer.enable_depth_output(activate_antialiasing=False)  # for perfect depth maps without interpolation artifacts
    bproc.renderer.set_max_amount_of_samples(128)
    bproc.renderer.engine = "EEVEE"  # faster than Cycles and with good enough quality for our purposes
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"],default_values={"category_id": 0}) 

    avge_time = 0.0
    overall_time = 0.0
    for i in range(1, NUM_ITERATIONS +1):
        # Reset keyframes so camera poses do not accumulate across iterations
        # continue
        t_start = time.time()
        logging.info(f"Starting iteration {i}/{NUM_ITERATIONS}...")
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
        place_obj(moving_objects, plate_obj, max_tries=10, boundary=[inner_min, inner_max, spawn_z, base_rotation_by_name])


        # Camera: 20% sampled on the dome, 80% the fixed calibrated pose
        if np.random.rand() < CAMERA_SAMPLE_PROB:
            sample_camera_pose(target_objects, table_center)
        else:
            set_camera()
        

        # continue
        t_render = time.time()
        data = bproc.renderer.render()

        t_render = time.time() - t_render
        # Per-iteration noise sigma sampled from uniform(0, NOISE_STD_MAX)
        noise_sigma = np.random.uniform(0.0, NOISE_STD_MAX)
        data["colors"] = apply_image_adjustments(data["colors"], noise_sigma=noise_sigma, exposure=exposure, gamma_contrast=False)



        coco = False
        t_writer = time.time()
        if coco:
            logging.info("Writing COCO annotations...")
            bproc.writer.write_coco_annotations(
                output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/coco"),
                instance_segmaps=data["instance_segmaps"],
                instance_attribute_maps=data["instance_attribute_maps"],
                colors=data["colors"],
                color_file_format="PNG",
            )
        else:
          
            bproc.writer.write_bop(
                output_dir=OUTPUT_DIR,
                target_objects=target_objects,
                colors = data["colors"],
                depths = data["depth"],
                color_file_format="PNG",
                annotation_unit="mm",
                calc_mask_info_coco=True
                
            )
        
        t_writer = time.time() - t_writer
        # exopusure value, noise sigma, temperature value, gamma_contrast
        if coco:
            writer = "COCO"
        else:            
            writer = "BOP"


        total_time_render = t_render + t_writer
        avge_time = total_time_render/i
        overall_time += time.time()-t_start
        avg_overall_time = overall_time/i
        _eta = eta(avg_overall_time, i, NUM_ITERATIONS)
        logging.info(f"Iteration {i}: Render time: {t_render:.2f} s, {writer} write time: {t_writer:.2f} s")
        logging.info(f"Average time per rendering_sim: {avge_time :.2f} s")
        logging.info(f"Estimated time remainuing: {_eta} s")
        logging.info(f"Iteration {i}: Exposure: {exposure:.2f} EV")
        logging.info(f"Iteration {i}: Noise sigma: {noise_sigma:.4f}")
        logging.info(f"Iteration {i}: Light temp: {light_temp_k:.0f} K")
        logging.info(f"Iteration {i}/{NUM_ITERATIONS} complete.......")

        LAST_RUN_STATE = {
            "iteration": i,
            "average_time": avge_time / i,
            "writer": writer,
        }




if __name__=='__main__':
    try:
        main()
    except Exception:
        logging.exception("Unhandled exception in application")
        append_run_summary()
        sys.exit(1)
    else:
        append_run_summary()