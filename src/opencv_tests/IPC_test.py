#!/usr/bin/env python3
"""
ICP pose estimation: CAD + scene point cloud → pose on RGB image.

Reference frame convention:
  - Origin  : bottom-left corner of CAD axis-aligned bounding box
  - Rotation: zero (CAD as-loaded, no extra rotation)

Symmetry:
  Object is symmetric about its mid-plane (left-right).
  Only top face visible → two valid ICP solutions differing by 180° about Z.
  Both are returned. Disambiguation: see notes at bottom.
"""

import numpy as np
import open3d as o3d

import cv2
import copy
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  —  fill these in
#   Importal Values
#   2D Image: 1920 x 1200
#   ROI 2d to PCD (0,97), (1920, 1050) 
# ─────────────────────────────────────────────────────────────────────────────
CAD_PATH        = "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/models/heart_shape.obj"      # .stl / .obj / .ply
SCENE_PCD_PATH  = "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/test_ref_frame_cam/point_cloud_20260610_200731_521.ply"
SCENE_IMG_PATH  = "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/test_ref_frame_cam/rgb_image_20260610_200731_521.png"
# ROI x and y of the Image 
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y 

SCENE_ROI_MIN = Point(0, 97) # Image
SCENE_ROI_MAX = Point(1920, 1050) # Image

# ICP params  (units = same as your point cloud — usually mm)
VOXEL_SIZE = 2.0          # downsample voxel size
ICP_MAX_ITER = 100
AXIS_DRAW_LEN = None       # None → auto (30% of longest bbox edge)

# Camera intrinsics  (get from your scanner SDK / calibration)

#changed or new intrinsic parameter

# K and dist: from mech eye sdk
K = np.array([[2430.38, 0.0, 969.89],
            [0.0, 2431.72, 619.58],
            [0.0, 0.0, 1.0]],dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)   # distortion coeffs

# K and dist: from OpenCV calibration using chessboard pattern, AFTER TEST IT DIDN'T WORK :CRY

# K = np.array([[2.87094566e+03, 0.00000000e+00, 1.04248232e+03],
#               [0.00000000e+00, 2.88403367e+03, 7.04362990e+02],
#               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)




# DIST =   np.array([[ 2.03276531e-01, -3.47445013e+00,  6.88249751e-03,  1.74410119e-02,
#    3.35986406e+01]])    


# ── Two-scale voxel sizes ─────────────────────────────────────────────────────
VOXEL_COARSE = 5.0    # for RANSAC global reg  (fast, approximate)
VOXEL_FINE   = 1.0    # for ICP refinement     (slow, accurate)












#



# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD CAD  →  surface point cloud, origin at bottom-left corner
# ─────────────────────────────────────────────────────────────────────────────
def load_cad_as_pcd(path: str, n_points: int = 30_000):
    """
    Returns:
        pcd        : PointCloud in a frame where BL corner = origin
        bl_offset  : translation that was subtracted (for reference)
        bbox_extent: [W, D, H] of the CAD bounding box
    """
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(number_of_points=n_points)

    # Bottom-left = min corner of AABB
    aabb       = pcd.get_axis_aligned_bounding_box()
    bl_corner  = aabb.get_min_bound()           # [min_x, min_y, min_z]
    pcd.translate(-bl_corner)                   # shift: BL → origin

    extent = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())
    print(f"[CAD]  {len(pcd.points):,} pts | extent (W×D×H): {np.round(extent,2)}")
    return pcd, bl_corner, extent


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD SCENE POINT CLOUD
# ─────────────────────────────────────────────────────────────────────────────


## Error 1.1
# python src/IPC_test.py
# [CAD]  30,000 pts | extent (WÃDÃH): [31.03 35.22 24.02]
# [Scene] 2,304,000 pts  |  extent: [nan nan nan]
#     â³ 1,174 pts after voxel=2.0 downsample  |  extent: [30.93 35.17 24.02]
#     â³ 1 pts after voxel=2.0 downsample  |  extent: [nan nan nan]
# Traceback (most recent call last):
#   File "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/src/IPC_test.py", line 368, in <module>
#     main()
#   File "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/src/IPC_test.py", line 344, in main
#     scene_d, scene_f = preprocess(scene_pcd, VOXEL_SIZE)
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/src/IPC_test.py", line 123, in preprocess
#     raise ValueError(
# ValueError: Only 1 pts after downsampling â almost certainly a unit mismatch.
# CAD extent is ~[31,35,24] (mm). If scene is in metres, set VOXEL_SIZE=0.002






def load_scene_pcd(path: str, scale: float = 1.0    ) -> o3d.geometry.PointCloud:
    # pcd = o3d.io.read_point_cloud(path)
    # print(f"[Scene] {len(pcd.points):,} pts")
    # return pcd
    pcd = o3d.io.read_point_cloud(path)
    n_raw = len(pcd.points)
    
    
    
    # ── Strip NaN / Inf (Mech-Eye outputs these for invalid depth pixels) ──── workaround ERORR 1.1, NaN coordinates baked in by camera
    try:
        # Open3D >= 0.13
        pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    except AttributeError:
        # Fallback for older Open3D
        pts = np.asarray(pcd.points)
        valid = np.isfinite(pts).all(axis=1)
        pcd = pcd.select_by_index(np.where(valid)[0])

    n_clean = len(pcd.points)
    print(f"[Scene] {n_raw:,} raw  →  {n_clean:,} finite pts  "
          f"({n_raw - n_clean:,} NaN removed)")



    if scale != 1.0:
        pcd.scale(scale, center=pcd.get_center())
    print(f"[Scene] {len(pcd.points):,} pts  |  "
          f"extent: {np.round(np.asarray(pcd.get_axis_aligned_bounding_box().get_extent()), 2)}")
    return pcd

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESS  →  downsample + normals + FPFH
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(pcd: o3d.geometry.PointCloud, voxel: float):
    down = pcd.voxel_down_sample(voxel)

    n = len(down.points)
    print(f"    ↳ {n:,} pts after voxel={voxel} downsample  |  "
          f"extent: {np.round(np.asarray(down.get_axis_aligned_bounding_box().get_extent()), 2)}")

    if n < 50:
        raise ValueError(
            f"Only {n} pts after downsampling — almost certainly a unit mismatch.\n"
            f"CAD extent is ~[31,35,24] (mm). If scene is in metres, set VOXEL_SIZE=0.002"
        )

    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
    )

    # orient_normals_consistent_tangent_plane → builds tetra mesh internally → dies on
    # planar / one-sided scans.
    # orient_towards_camera is strictly better here: scanner is at origin in camera
    # frame, so every surface normal genuinely should point toward [0,0,0].
    down.orient_normals_towards_camera_location(np.array([0., 0., 0.]))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
    )
    return down, fpfh


###  Old preprocess function, kept for reference

# def preprocess(pcd: o3d.geometry.PointCloud, voxel: float):
#     down = pcd.voxel_down_sample(voxel)

#     # Normals (needed for point-to-plane ICP and FPFH)
#     down.estimate_normals(
#         o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
#     )
#     # Consistently orient normals (important for point-to-plane)
#     down.orient_normals_consistent_tangent_plane(k=15)

#     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
#     )
#     return down, fpfh


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GLOBAL REGISTRATION  (RANSAC + FPFH)
#     Gives a coarse initial alignment before ICP.
# ─────────────────────────────────────────────────────────────────────────────
def global_reg(src_d, src_f, tgt_d, tgt_f, voxel: float):
    dist_thr = voxel * 1.5
    # o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength.set_default(0.9)  # global default
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching( 
        src_d, tgt_d, src_f, tgt_f,
        mutual_filter=True,
        max_correspondence_distance=dist_thr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thr),
        ],
        # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 500),

    )
    print(f"[GlobalReg] fitness={result.fitness:.4f}  rmse={result.inlier_rmse:.4f}")
    return result.transformation


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ICP REFINEMENT  (point-to-plane)
# ─────────────────────────────────────────────────────────────────────────────
def icp(src, tgt, init_T, max_dist, max_iter=ICP_MAX_ITER):
    res = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_dist,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=max_iter
        ),
    )
    print(f"[ICP]  fitness={res.fitness:.4f}  rmse={res.inlier_rmse:.4f}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SYMMETRY  —  build the 180° alternative initial guess
#
#     The object is symmetric about the mid-plane perpendicular to its X-axis.
#     So the alternate pose is: rotate 180° about Z, pivoting at the bbox center.
#
#     In CAD local frame (after BL-corner shift), the center is at extent/2.
#     We apply this symmetry BEFORE T_init takes the CAD into camera frame,
#     i.e.:  T_alt = T_init  @  T_sym_local
# ─────────────────────────────────────────────────────────────────────────────
def symmetric_T(T_init: np.ndarray, bbox_extent: np.ndarray) -> np.ndarray:
    """
    Returns the pose that's geometrically equivalent under left-right symmetry.
    Assumes symmetry axis = Z, pivot = bbox center in CAD local frame.
    """
    cx, cy = bbox_extent[0] / 2, bbox_extent[1] / 2   # center XY in local frame

    # Translate to center → flip 180° about Z → translate back
    T_pivot = np.eye(4);  T_pivot[:3, 3]  = [-cx, -cy, 0]
    T_unpiv = np.eye(4);  T_unpiv[:3, 3]  = [ cx,  cy, 0]
    T_flip  = np.eye(4)
    T_flip[:3, :3] = np.diag([-1., -1., 1.])   # 180° about Z

    T_sym_local = T_unpiv @ T_flip @ T_pivot
    return T_init @ T_sym_local


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RUN BOTH SOLUTIONS, RETURN BEST
# ─────────────────────────────────────────────────────────────────────────────
def run_symmetric_icp(cad_pcd, scene_pcd, coarse_T, bbox_extent, voxel):
    max_dist = voxel * 1.5

    T_alt = symmetric_T(coarse_T, bbox_extent)

    print("[Solution A]")
    res_a = icp(cad_pcd, scene_pcd, coarse_T, max_dist)
    print("[Solution B (symmetric)]")
    res_b = icp(cad_pcd, scene_pcd, T_alt, max_dist)

    if res_a.fitness >= res_b.fitness:
        print("[Symmetry] → Solution A wins")
        return res_a.transformation, res_b.transformation, "A"
    else:
        print("[Symmetry] → Solution B wins")
    return res_b.transformation, res_a.transformation, "B"

    # NOTE: for a perfectly symmetric flat-top object, res_a.fitness ≈ res_b.fitness
    # — see disambiguation notes at the bottom of this file.


# ─────────────────────────────────────────────────────────────────────────────
# 8.  OPENCV VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def project(pts_3d: np.ndarray, T: np.ndarray, K, dist,dx=0, dy=0 , S=1):
    """
    Project Nx3 points using pose T (obj→cam frame) through camera K.
    dx, dy: optional pixel translation added after projection.
    """
    R = T[:3, :3];  t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec    = t.reshape(3, 1)
    pts2d, _ = cv2.projectPoints(pts_3d.astype(np.float32), rvec, tvec, K, dist)
    pts2d = pts2d.reshape(-1, 2) + np.array([dx, dy])  # apply optional translation
    pts2d = pts2d * S  # apply optional scaling
    return pts2d.reshape(-1, 2).astype(int)


def draw_axes(img, T, K, dist, length):
    """Draw XYZ axes at the CAD origin (bottom-left corner of object)."""
    pts3d = np.float32([[0, 0, 0],
                        [length, 0, 0],       # +X  red
                        [0, length, 0],       # +Y  green
                        [0, 0, length]])      # +Z  blue
    p = project(pts3d, T, K, dist)
    o, px, py, pz = p

    img = cv2.arrowedLine(img, tuple(o), tuple(px), (0,   0, 255), 2, tipLength=0.25)
    img = cv2.arrowedLine(img, tuple(o), tuple(py), (0, 255,   0), 2, tipLength=0.25)
    img = cv2.arrowedLine(img, tuple(o), tuple(pz), (255,  0,   0), 2, tipLength=0.25)

    for pt, lbl, col in [(px, "X", (0,0,255)), (py, "Y", (0,255,0)), (pz, "Z", (255,0,0))]:
        cv2.putText(img, lbl, tuple(pt + [4, 4]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return img


def draw_bbox_3d(img, T, K, dist, extent):
    """Project and draw the full 3D bounding box of the object."""
    W, D, H = extent

    
    # 8 corners: origin = BL corner = (0,0,0) in CAD frame
    corners = np.float32([
        [0, 0, 0], [W, 0, 0], [W, D, 0], [0, D, 0],  # bottom face
        [0, 0, H], [W, 0, H], [W, D, H], [0, D, H],  # top face
    ])
    p = project(corners, T, K, dist, 0.01 )  # scale down for better visualization

    edges = [(0,1),(1,2),(2,3),(3,0),    # bottom ring
             (4,5),(5,6),(6,7),(7,4),    # top ring
             (0,4),(1,5),(2,6),(3,7)]    # verticals
    # edges = [(a, b) for a, b in edges if not (a == b)]  # remove self-edges
    # edges = [(a, b) for a, b in edges if np.linalg.norm(p[a] - p[b]) > 0]  # remove zero-length edges
    # edges =/ [(a, b) for a, b in edges if np.linalg.norm(p[a] - p[b]) < 1000.0]  # convert from mm to metres for better visualization (thinner lines)
    # edges =np.array(edges)  # convert to numpy array for vectorized operations
    # edges = edges*np.full(edges.shape, dtype=np.float32, fill_value=0.001)
    for a, b in edges:
        cv2.line(img, tuple(p[a]), tuple(p[b]), (0, 220, 255), 1, cv2.LINE_AA)
        print(f"({p[a]},{[b]}):")

    return img


def visualize_pose(img_path, T_best, T_alt, K, dist, extent, label="A"):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    axis_len = AXIS_DRAW_LEN or (max(extent) * 0.3)

    # Draw best pose
    img = draw_bbox_3d(img, T_best, K, dist, extent)
    img = draw_axes(img, T_best, K, dist, axis_len)

    # Optionally draw the alternate symmetric pose in faded cyan
    # Uncomment to see both solutions overlaid
    # img = draw_bbox_3d(img, T_alt, K, dist, extent)   # dim overlay

    cv2.putText(img, f"Pose: Solution {label}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    out = Path(img_path).stem + "_pose.png"
    cv2.imwrite(out, img)
    print(f"[Saved] {out}")

    cv2.imshow("Pose Estimation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def boundingBox(scene_pcd) :
    """
        returns: (x, y, w, h) of the selected bounding box in the image
    """
    img = cv2.imread(SCENE_IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load image, image shape: {img.shape if img is not None else 'Unknown'}")
    
    roi = (SCENE_ROI_MIN.x, SCENE_ROI_MIN.y, SCENE_ROI_MAX.x,SCENE_ROI_MAX.y  )   # (x_min, y_min, x_max, y_max)
    cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)  

    overlay = img.copy()
    # img = cv2.imread(SCENE_IMG_PATH)
    pts_2d = project(np.asarray(scene_pcd.points), np.eye(5), K, DIST)
    for uv in pts_2d:
        cv2.circle(img, tuple(uv), 1, (0,255,0), thickness=-1,)

    # cv2.imshow("Projected points", img)
    # cv2.waitKey(0)
    alpha = 0.2 

    # 6. Blend the overlay layer back onto the original image
    # Equation applied: output = (overlay * alpha) + (image * beta) + gamma
    beta = 1.0 - alpha
    gamma = 0
    output_image = cv2.addWeighted(overlay, alpha, img, beta, gamma)
    bbox = cv2.selectROI("Select ROI", output_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    print(f"Selected ROI: {bbox}")
    return bbox



def pcd_overlayed_image(scene_pcd, img_path, K, dist,head):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image, image shape: {img.shape if img is not None else 'Unknown'}")
    pts_2d = project(np.asarray(scene_pcd.points), np.eye(5), K, dist)
    for uv in pts_2d:
        cv2.circle(img, tuple(uv), 1, (0,255,0), thickness=-1,)
    
    
    cv2.imshow(head, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    
## 

# Projects every 3D point from the scene onto the RGB image using the camera intrinsics.
# Keeps only points whose projected (u,v) fall inside the selected 2D ROI.

def crop_pcd_by_roi(pcd: o3d.geometry.PointCloud, roi_xywh, K, dist_coeffs=np.zeros(4)):
    """
    roi_xywh = (x, y, w, h)  from cv2.selectROI
    Returns a new point cloud containing only points that project into that rectangle.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd

    # Project all 3D points onto the image
    rvec = np.zeros(3)          # no rotation, points are already in camera frame
    tvec = np.zeros(3)          # no translation
    img_pts, _ = cv2.projectPoints(pts.astype(np.float32),
                                   rvec, tvec, K, dist_coeffs)
    uv = img_pts.reshape(-1, 2).astype(int)

    x, y, w, h = roi_xywh
    mask = (uv[:, 0] >= x) & (uv[:, 0] < x + w) & \
           (uv[:, 1] >= y) & (uv[:, 1] < y + h)

    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])

    pcd_overlayed_image(cropped, SCENE_IMG_PATH, K, dist_coeffs,"Cropped PCD Overlay")
    return cropped







def main():
    # Load

    ## T_table→cam
    #   T_table_to_cam = np.eye(4, dtype=np.float64)
    #   T_table_to_cam[0, 3] =  227.5   # X translation
    #   T_table_to_cam[1, 3] =  407.55   # Y translation
    #   T_table_to_cam[2, 3] = -965.0    # Z translation
    cad_pcd, _bl_off, extent = load_cad_as_pcd(CAD_PATH)
    # scene_pcd = load_scene_pcd(SCENE_PCD_PATH)



    scene_pcd = load_scene_pcd(SCENE_PCD_PATH, scale=1)   # metres → mm
    # scene_pcd.transform(T_table_to_cam)
    pts = np.asarray(scene_pcd.points)  
    print(f"Z range: {np.min(pts[:,2]):.2f} to {np.max(pts[:,2]):.2f} (mm before scaling)")
    # pts = np.asarray(scene_pcd.points)
    # print(f"Z range after load: {np.min(pts[:,2]):.1f} to {np.max(pts[:,2]):.1f} mm")

    bbox = boundingBox(scene_pcd)

    # exit()
    # img = cv2.imread(SCENE_IMG_PATH)
    # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    # cv2.imshow("ROI Selection", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    # ── Crop scene to object ROI 
    # roi = o3d.geometry.AxisAlignedBoundingBox(
    #     min_bound=np.array([bbox[0], bbox[1], -2000]),  # (x_min, y_min, z_min)
    #     max_bound=np.array([bbox[0] + bbox[2], bbox[1] + bbox[3], 2000])  # (x_max, y_max, z_max)
    # )
    # scene_pcd = scene_pcd.crop(roi)

    # ── Crop scene using projection and the selected ROI

    scene_pcd = crop_pcd_by_roi(scene_pcd, bbox, K, DIST)
    print(f"[Crop]  {len(scene_pcd.points):,} pts remain in ROI")


    import time
    t0 = time.time()

    # preprocess
    cad_c, cad_fc     = preprocess(cad_pcd,   VOXEL_COARSE)
    scene_c, scene_fc = preprocess(scene_pcd, VOXEL_COARSE)

    # Coarse alignment

    coarse_T = global_reg(cad_c, cad_fc, scene_c, scene_fc, VOXEL_COARSE)
    print(f"[Timing] RANSAC: {time.time()-t0:.2f}s")

    # ── Fine: ICP on dense clouds 
    t0 = time.time()
    cad_f, _     = preprocess(cad_pcd,   VOXEL_FINE)
    scene_f, _   = preprocess(scene_pcd, VOXEL_FINE)

    # ICP with symmetry

    T_best, T_alt, which = run_symmetric_icp(
        cad_f, scene_f, coarse_T, extent, VOXEL_FINE
    )
    print(f"[Timing] ICP: {time.time()-t0:.2f}s")


    print("\n── Final Pose  T_cam_obj (CAD frame → Camera frame) ──")
    print(np.round(T_best, 4))

    R = T_best[:3, :3]
    t = T_best[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    print(f"   Translation : {np.round(t, 3)}")
    print(f"   Rotation vec: {np.round(rvec.ravel(), 4)}")

    # Visualize
    visualize_pose(SCENE_IMG_PATH, T_best, T_alt, K, DIST, extent, label=which)


if __name__ == "__main__":
    main()