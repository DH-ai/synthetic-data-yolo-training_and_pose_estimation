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
# ─────────────────────────────────────────────────────────────────────────────
CAD_PATH        = "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/models/heart_shape.obj"      # .stl / .obj / .ply
SCENE_PCD_PATH  = "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/image_data_win/image_set/point_cloud_20260605_154148_725.ply"       # point cloud from your scanner
SCENE_IMG_PATH  = "/home/dhruv/obscureP/synthetic-data-yolo-training_and_pose_estimation/assets/image_data_win/image_set/rgb_20260605_154148_725.png"        # RGB image corresponding to the scan
















# Camera intrinsics  (get from your scanner SDK / calibration)
K = np.array([[1727.4641025602748,    0.0, 655.82],
              [   0.0, 1727.46, 516.63],
              [   0.0,    0.0,   1.0]], dtype=np.float64)
DIST = np.zeros(5, dtype=np.float64)   # distortion coeffs















# ICP params  (units = same as your point cloud — usually mm)
VOXEL_SIZE = 2.0          # downsample voxel size
ICP_MAX_ITER = 100
AXIS_DRAW_LEN = None       # None → auto (30% of longest bbox edge)




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
def load_scene_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    print(f"[Scene] {len(pcd.points):,} pts")
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
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
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
def project(pts_3d: np.ndarray, T: np.ndarray, K, dist):
    """Project Nx3 points using pose T (obj→cam frame) through camera K."""
    R = T[:3, :3];  t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    tvec    = t.reshape(3, 1)
    pts2d, _ = cv2.projectPoints(pts_3d.astype(np.float32), rvec, tvec, K, dist)
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
    p = project(corners, T, K, dist)

    edges = [(0,1),(1,2),(2,3),(3,0),    # bottom ring
             (4,5),(5,6),(6,7),(7,4),    # top ring
             (0,4),(1,5),(2,6),(3,7)]    # verticals

    for a, b in edges:
        cv2.line(img, tuple(p[a]), tuple(p[b]), (0, 220, 255), 1, cv2.LINE_AA)

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
def main():
    # Load
    cad_pcd, _bl_off, extent = load_cad_as_pcd(CAD_PATH)
    scene_pcd = load_scene_pcd(SCENE_PCD_PATH)

    # Preprocess
    cad_d,   cad_f   = preprocess(cad_pcd,   VOXEL_SIZE)
    scene_d, scene_f = preprocess(scene_pcd, VOXEL_SIZE)

    # Coarse alignment
    coarse_T = global_reg(cad_d, cad_f, scene_d, scene_f, VOXEL_SIZE)

    # ICP with symmetry
    T_best, T_alt, which = run_symmetric_icp(
        cad_pcd, scene_pcd, coarse_T, extent, VOXEL_SIZE
    )

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