#!/usr/bin/env python3
"""
Hand-eye calibration (EYE-TO-HAND configuration).

Setup
-----
- Camera is mounted overhead on the ceiling and is STATIONARY.
- A ChArUco board is rigidly attached to the robot end-effector (gripper).
- The robot is moved to N poses; one RGB image is captured per pose.

What it solves
--------------
For each pose we know two transforms:
  * target2cam   : ChArUco board -> camera        (from solvePnP on the image)
  * gripper2base : end-effector  -> robot base     (read from the robot/TF)

Eye-to-hand solves for the FIXED camera-to-base transform `cam2base`
(the pose of the static camera in the robot base frame).

OpenCV trick for eye-to-hand
----------------------------
`cv2.calibrateHandEye` natively solves the eye-IN-hand problem
(returns cam2gripper). For eye-TO-hand we feed it the *inverse* robot
transforms (base2gripper) together with target2cam; the returned transform
is then `cam2base`. See:
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b

Robot pose file format (YOU fill this in)
-----------------------------------------
Create a JSON file next to the images named `robot_poses.json`.
It maps each image filename -> the base2gripper pose reported by the robot at
capture time (pose of the base in the gripper/tool frame). The loader inverts
it to gripper2base internally, so paste the RAW robot values here.
Three layouts are accepted per entry (pick one). Translation is in METERS.

  1) Translation + Fanuc-style W/P/R Euler angles in DEGREES (likely yours):
       "rgb_image_20260626_181413_164.png": {
           "position": [tx, ty, tz],             # meters
           "wpr_deg":  [W, P, R]                 # deg, rotations about X, Y, Z
       }

  2) Translation + quaternion (qx,qy,qz,qw):
       "rgb_image_20260626_181413_164.png": {
           "position":   [tx, ty, tz],
           "quaternion": [qx, qy, qz, qw]
       }

  3) Homogeneous 4x4 matrix (nested rows):
       "rgb_image_20260626_181413_164.png": {
           "matrix": [[r11,r12,r13,tx], [r21,r22,r23,ty],
                      [r31,r32,r33,tz], [0,0,0,1]]
       }

Run `python hand_eye_calibration.py --make-template` to auto-generate a
`robot_poses.json` stub listing every image in which the board was detected,
with identity poses you can overwrite.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).parent
IMAGES_DIR = HERE / "media_robot_charuca_calib"
POSES_FILE = IMAGES_DIR / "robot_poses.json"
OUTPUT_FILE = HERE / "hand_eye_result.json"

# ChArUco board geometry (matches calibration.py in this folder)
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.022   # meters
MARKER_LENGTH = 0.011   # meters
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# Camera intrinsics (provided)
K = np.array([[2481.9412514178307, 0.0, 978.95936559694314],
              [0.0, 2482.3917472975795, 629.72289542481894],
              [0.0, 0.0, 1.0]], dtype=np.float64)

DIST = np.array([[-0.091539129459748417,
                  1.6518788910916924,
                  -0.00096826424151305102,
                  -0.0023115236516727399,
                  -7.1086932137755738]], dtype=np.float64)

# Methods to compare. Tsai is the classic baseline; Park/Horaud/Andreff/Daniilidis
# often do better. We run all available and report each.
HANDEYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# Robot-world / hand-eye (AX = ZB). Solves cam2base AND the board->flange
# offset simultaneously, as an independent cross-check. Only two methods exist.
ROBOT_WORLD_METHODS = {
    "SHAH": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    "LI": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
}


# --------------------------------------------------------------------------- #
# Small SE(3) helpers
# --------------------------------------------------------------------------- #
def quat_to_R(q):
    """Quaternion (qx, qy, qz, qw) -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def wpr_to_R(w, p, r, degrees=True):
    """
    Fanuc-style W/P/R Euler angles -> 3x3 rotation matrix.

    W, P, R are rotations about the X, Y, Z axes respectively. The composed
    orientation is  R = Rz(R) @ Ry(P) @ Rx(W)  (extrinsic XYZ / intrinsic ZYX),
    which is the convention Fanuc (and most "WPR" robot reports) use.
    Angles default to DEGREES.
    """
    if degrees:
        w, p, r = np.radians([w, p, r])
    cw, sw = np.cos(w), np.sin(w)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R, t):
    """Build 4x4 homogeneous transform from R (3x3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T):
    """Inverse of a homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_angle_deg(R):
    """Geodesic angle of a rotation matrix, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def average_transforms(Ts):
    """
    Average a list of homogeneous transforms. Translation is the arithmetic
    mean; rotation is the orthonormalized mean rotation (SVD projection back
    onto SO(3)). Good enough when the inputs are already close (as they should
    be for a single fixed cam2base computed per view).
    """
    Ts = [T for T in Ts if T is not None]
    t_mean = np.mean([T[:3, 3] for T in Ts], axis=0)
    R_sum = np.sum([T[:3, :3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:          # guard against reflection
        U[:, -1] *= -1
        R_mean = U @ Vt
    return make_T(R_mean, t_mean)


# --------------------------------------------------------------------------- #
# ChArUco detection -> target2cam pose per image
# --------------------------------------------------------------------------- #
def build_board_and_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def detect_target2cam(image_paths, board, detector, min_corners=6, debug_dir=None):
    """
    For each image, detect the ChArUco board and estimate the board pose in the
    camera frame via solvePnP.

    Returns
    -------
    names         : list[str]   image filenames that yielded a valid pose
    R_target2cam  : list[3x3]
    t_target2cam  : list[3x1]   meters
    """
    names, R_t2c, t_t2c = [], [], []

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [skip] cannot read {path.name}")
            continue

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(img)
        if charuco_ids is None or len(charuco_ids) < min_corners:
            n = 0 if charuco_ids is None else len(charuco_ids)
            print(f"  [skip] {path.name}: only {n} corners (< {min_corners})")
            continue

        # Match detected corners to 3D board points, then solvePnP.
        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is None or len(obj_pts) < 4:
            print(f"  [skip] {path.name}: matchImagePoints failed")
            continue

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, DIST, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            print(f"  [skip] {path.name}: solvePnP failed")
            continue

        # Reprojection error as a quality gate / sanity print.
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, DIST)
        err = float(np.mean(np.linalg.norm(img_pts - proj, axis=2)))

        R, _ = cv2.Rodrigues(rvec)
        names.append(path.name)
        R_t2c.append(R)
        t_t2c.append(tvec.reshape(3, 1))
        print(f"  [ok]   {path.name}: {len(charuco_ids)} corners, "
              f"reproj={err:.3f}px, dist={np.linalg.norm(tvec):.3f}m")

        if debug_dir is not None:
            vis = img.copy()
            cv2.drawFrameAxes(vis, K, DIST, rvec, tvec, SQUARE_LENGTH * 2)
            cv2.imwrite(str(debug_dir / f"axes_{path.name}"), vis)

    return names, R_t2c, t_t2c


# --------------------------------------------------------------------------- #
# Robot poses
# --------------------------------------------------------------------------- #
def load_gripper2base(names, invert=False):
    """
    Load robot poses for the given image names from POSES_FILE and return
    gripper2base (the pose of the flange/tool expressed in the robot base frame).

    Most controllers (e.g. Fanuc "current position") already report the TCP/flange
    pose in the base frame == gripper2base, so by default we use the values as-is.
    Set invert=True only if your controller reports base2gripper ("base <- tool").
    A convention mismatch shows up as a large method-disagreement / consistency
    spread; use --invert-poses to flip if needed.

    Returns R_gripper2base (list of 3x3), t_gripper2base (list of 3x1),
    and the filtered list of names that had a usable pose.
    """
    if not POSES_FILE.exists():
        raise FileNotFoundError(
            f"Robot pose file not found: {POSES_FILE}\n"
            f"Run `python {Path(__file__).name} --make-template` to create a stub, "
            f"then fill in the gripper2base poses."
        )

    with open(POSES_FILE) as f:
        data = json.load(f)

    R_g2b, t_g2b, kept = [], [], []
    for name in names:
        if name not in data:
            print(f"  [skip] no robot pose for {name}")
            continue
        entry = data[name]

        if "matrix" in entry:
            T_b2g = np.asarray(entry["matrix"], dtype=np.float64).reshape(4, 4)
        elif "position" in entry and "wpr_deg" in entry:
            w, p, r = entry["wpr_deg"]
            T_b2g = make_T(wpr_to_R(w, p, r, degrees=True), entry["position"])
        elif "position" in entry and "quaternion" in entry:
            T_b2g = make_T(quat_to_R(entry["quaternion"]), entry["position"])
        else:
            print(f"  [skip] {name}: entry needs 'matrix', "
                  f"'position'+'wpr_deg', or 'position'+'quaternion'")
            continue

        # T_b2g here is the raw file value. By default it already IS gripper2base.
        T_g2b = invert_T(T_b2g) if invert else T_b2g
        R_g2b.append(T_g2b[:3, :3])
        t_g2b.append(T_g2b[:3, 3].reshape(3, 1))
        kept.append(name)

    return R_g2b, t_g2b, kept


def make_template(names):
    """Write a robot_poses.json stub (identity poses) for the detected images."""
    template = {
        name: {
            "position": [0.0, 0.0, 0.0],
            "wpr_deg": [0.0, 0.0, 0.0],
            "_comment": "RAW base2gripper. position in METERS, wpr in DEGREES. "
                        "Loader inverts to gripper2base."
        }
        for name in names
    }
    with open(POSES_FILE, "w") as f:
        json.dump(template, f, indent=2)
    print(f"\nWrote template with {len(names)} entries -> {POSES_FILE}")
    print("Fill in each gripper2base pose, then re-run without --make-template.")


# --------------------------------------------------------------------------- #
# Hand-eye solve (eye-to-hand)
# --------------------------------------------------------------------------- #
def solve_eye_to_hand(R_g2b, t_g2b, R_t2c, t_t2c, method):
    """
    Eye-to-hand: feed the INVERSE robot transforms (base2gripper) so the result
    is cam2base instead of cam2gripper.
    """
    R_b2g, t_b2g = [], []
    for R, t in zip(R_g2b, t_g2b):
        T_b2g = invert_T(make_T(R, t))
        R_b2g.append(T_b2g[:3, :3])
        t_b2g.append(T_b2g[:3, 3].reshape(3, 1))

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_b2g, t_b2g, R_t2c, t_t2c, method=method
    )
    return make_T(R_cam2base, t_cam2base)


def solve_robot_world(R_g2b, t_g2b, R_t2c, t_t2c, method):
    """
    Eye-to-hand via cv2.calibrateRobotWorldHandEye, as an independent cross-check.

    That function natively assumes the eye-IN-hand robot-world geometry, whose
    two rigid unknowns are (base<->world) and (gripper<->camera). Our eye-TO-hand
    geometry has the rigid pairs swapped: (base<->camera) and (gripper<->board).
    We bridge the two by RELABELING the frames passed to OpenCV:

        OpenCV "world"  :=  our camera
        OpenCV "camera" :=  our board (target)

    With that relabel OpenCV's inputs/outputs become:
        input  world2cam      = our cam2target   = inv(target2cam)
        input  base2gripper   = raw robot pose   (invert our gripper2base)
        output base2world     = base2cam         -> cam2base   = inv(output)
        output gripper2cam    = gripper2target    -> target2gripper = inv(output)

    This was verified to recover a known cam2base and board->flange offset
    exactly on synthetic noiseless data.

    Returns
    -------
    T_cam2base       : 4x4   (comparable to calibrateHandEye output)
    T_base2cam       : 4x4   (raw OpenCV output X, = inv(cam2base))
    T_gripper2target : 4x4   (raw OpenCV output Z)
    T_target2gripper : 4x4   board->flange "bolt offset" (sanity check)
    """
    # OpenCV "world2cam" = our cam2target = inv(target2cam)
    R_c2t, t_c2t = [], []
    for R, t in zip(R_t2c, t_t2c):
        T = invert_T(make_T(R, t))
        R_c2t.append(T[:3, :3])
        t_c2t.append(T[:3, 3].reshape(3, 1))

    # OpenCV "base2gripper" = raw robot pose = inv(gripper2base)
    R_b2g, t_b2g = [], []
    for R, t in zip(R_g2b, t_g2b):
        T = invert_T(make_T(R, t))
        R_b2g.append(T[:3, :3])
        t_b2g.append(T[:3, 3].reshape(3, 1))

    R_X, t_X, R_Z, t_Z = cv2.calibrateRobotWorldHandEye(
        R_c2t, t_c2t, R_b2g, t_b2g, method=method
    )
    T_base2cam = make_T(R_X, t_X)             # OpenCV X (base in "world"=cam)
    T_gripper2target = make_T(R_Z, t_Z)       # OpenCV Z (gripper in "cam"=board)

    T_cam2base = invert_T(T_base2cam)
    T_target2gripper = invert_T(T_gripper2target)
    return T_cam2base, T_base2cam, T_gripper2target, T_target2gripper


def consistency_error(T_cam2base, R_g2b, t_g2b, R_t2c, t_t2c):
    """
    Eye-to-hand self-consistency check.

    The board is bolted to the gripper, so the board->flange transform is a
    fixed rigid offset that MUST be identical for every view:
        T_target2gripper_i = base2gripper_i @ cam2base @ target2cam_i
                           = inv(gripper2base_i) @ T_cam2base @ T_target2cam_i
    (The board-in-base pose is NOT constant here because the gripper moves.)

    We recompute target2gripper for each view from the candidate cam2base and
    report how much it scatters. Lower spread = better calibration.

    Returns (translation_std_m, rotation_spread_deg).
    """
    pts, Rs = [], []
    for (Rg, tg), (Rc, tc) in zip(zip(R_g2b, t_g2b), zip(R_t2c, t_t2c)):
        T_base2gripper = invert_T(make_T(Rg, tg))
        T_t2g = T_base2gripper @ T_cam2base @ make_T(Rc, tc)
        pts.append(T_t2g[:3, 3])
        Rs.append(T_t2g[:3, :3])

    trans_std = float(np.linalg.norm(np.array(pts).std(axis=0)))

    # Rotation spread: mean geodesic distance to the orthonormalized mean.
    R_mean = average_transforms([make_T(R, np.zeros(3)) for R in Rs])[:3, :3]
    rot_spread = float(np.mean([rot_angle_deg(R_mean.T @ R) for R in Rs]))
    return trans_std, rot_spread


def per_view_residuals(T_cam2base, R_g2b, t_g2b, R_t2c, t_t2c):
    """
    Per-view board->flange translation residual (meters) vs the mean. A single
    bad robot pose (or mis-synced image) shows up here as a large spike.
    """
    t2g = []
    for (Rg, tg), (Rc, tc) in zip(zip(R_g2b, t_g2b), zip(R_t2c, t_t2c)):
        T = invert_T(make_T(Rg, tg)) @ T_cam2base @ make_T(Rc, tc)
        t2g.append(T[:3, 3])
    t2g = np.array(t2g)
    return np.linalg.norm(t2g - t2g.mean(axis=0), axis=1)


def reject_outliers(R_g2b, t_g2b, R_t2c, t_t2c, names, thresh_mm=80.0):
    """
    Fit once (PARK), drop views whose board->flange residual exceeds thresh_mm
    OR exceeds median + 3*MAD, then return the cleaned, index-aligned lists.
    Robust to the common case of one or two mis-recorded robot poses.
    """
    T = solve_eye_to_hand(R_g2b, t_g2b, R_t2c, t_t2c, cv2.CALIB_HAND_EYE_PARK)
    res = per_view_residuals(T, R_g2b, t_g2b, R_t2c, t_t2c) * 1000  # mm
    med = np.median(res)
    mad = np.median(np.abs(res - med)) + 1e-9
    keep = [i for i, r in enumerate(res) if r <= thresh_mm and r <= med + 3 * mad]
    dropped = [(names[i], float(res[i])) for i in range(len(names)) if i not in keep]
    for n, r in sorted(dropped, key=lambda x: -x[1]):
        print(f"  [outlier] dropping {n}: residual {r:.0f} mm")
    sel = lambda L: [L[i] for i in keep]
    return sel(R_g2b), sel(t_g2b), sel(R_t2c), sel(t_t2c), sel(names)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Eye-to-hand calibration (ceiling camera).")
    ap.add_argument("--make-template", action="store_true",
                    help="Detect boards and write a robot_poses.json stub, then exit.")
    ap.add_argument("--min-corners", type=int, default=6,
                    help="Minimum ChArUco corners required per image.")
    ap.add_argument("--debug-axes", action="store_true",
                    help="Save images with drawn board axes for visual QC.")
    ap.add_argument("--invert-poses", action="store_true",
                    help="Treat robot poses as base2gripper and invert them. "
                         "Default assumes gripper2base (flange-in-base).")
    ap.add_argument("--no-reject", action="store_true",
                    help="Disable automatic outlier rejection.")
    ap.add_argument("--reject-mm", type=float, default=80.0,
                    help="Outlier threshold on board->flange residual (mm).")
    args = ap.parse_args()

    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images in {IMAGES_DIR}")
    print(f"Found {len(image_paths)} images in {IMAGES_DIR.name}")

    debug_dir = None
    if args.debug_axes:
        debug_dir = IMAGES_DIR / "debug_axes"
        debug_dir.mkdir(exist_ok=True)

    board, detector = build_board_and_detector()

    print("\n== Detecting ChArUco board (target2cam) ==")
    names, R_t2c, t_t2c = detect_target2cam(
        image_paths, board, detector, args.min_corners, debug_dir
    )
    print(f"Detected board in {len(names)}/{len(image_paths)} images.")

    if args.make_template:
        make_template(names)
        return

    conv = "base2gripper (inverting)" if args.invert_poses else "gripper2base (as-is)"
    print(f"\n== Loading robot poses [{conv}] ==")
    R_g2b, t_g2b, kept = load_gripper2base(names, invert=args.invert_poses)

    # Keep only views that have BOTH a board detection and a robot pose.
    idx = [names.index(n) for n in kept]
    R_t2c = [R_t2c[i] for i in idx]
    t_t2c = [t_t2c[i] for i in idx]
    print(f"Paired {len(kept)} views (board detection + robot pose).")

    if len(kept) < 3:
        raise RuntimeError(
            "Need at least 3 paired views (more, with varied rotation, is better). "
            "Hand-eye is poorly conditioned with too few or near-parallel motions."
        )

    if not args.no_reject:
        print("\n== Outlier rejection (board->flange residual) ==")
        R_g2b, t_g2b, R_t2c, t_t2c, kept = reject_outliers(
            R_g2b, t_g2b, R_t2c, t_t2c, kept, thresh_mm=args.reject_mm
        )
        print(f"Using {len(kept)} views after rejection.")

    results = {}

    print("\n== Eye-to-hand: cv2.calibrateHandEye (cam2base) ==")
    for label, method in HANDEYE_METHODS.items():
        try:
            T = solve_eye_to_hand(R_g2b, t_g2b, R_t2c, t_t2c, method)
        except cv2.error as e:
            print(f"  {label:11s}: failed ({e})")
            continue
        t_std, r_spread = consistency_error(T, R_g2b, t_g2b, R_t2c, t_t2c)
        results[label] = {
            "solver": "calibrateHandEye",
            "cam2base": T.tolist(),
            "translation_m": T[:3, 3].tolist(),
            "trans_consistency_std_m": t_std,
            "rot_consistency_deg": r_spread,
        }
        tx, ty, tz = T[:3, 3]
        print(f"  {label:11s}: t=({tx:+.4f}, {ty:+.4f}, {tz:+.4f}) m  "
              f"| consistency: trans_std={t_std*1000:.2f}mm rot={r_spread:.3f}deg")

    print("\n== Robot-world: cv2.calibrateRobotWorldHandEye (cross-check) ==")
    for label, method in ROBOT_WORLD_METHODS.items():
        try:
            T, T_base2cam, T_grip2tgt, T_t2g = solve_robot_world(
                R_g2b, t_g2b, R_t2c, t_t2c, method
            )
        except cv2.error as e:
            print(f"  RW-{label:8s}: failed ({e})")
            continue
        t_std, r_spread = consistency_error(T, R_g2b, t_g2b, R_t2c, t_t2c)
        results[f"RW-{label}"] = {
            "solver": "calibrateRobotWorldHandEye",
            "cam2base": T.tolist(),
            "translation_m": T[:3, 3].tolist(),
            "trans_consistency_std_m": t_std,
            "rot_consistency_deg": r_spread,
            "board2flange_offset": T_t2g.tolist(),
            "board2flange_offset_m": T_t2g[:3, 3].tolist(),
        }
        tx, ty, tz = T[:3, 3]
        bx, by, bz = T_t2g[:3, 3]
        print(f"  RW-{label:8s}: t=({tx:+.4f}, {ty:+.4f}, {tz:+.4f}) m  "
              f"| consistency: trans_std={t_std*1000:.2f}mm rot={r_spread:.3f}deg")
        print(f"  {'':11s}  board->flange offset = "
              f"({bx*1000:+.1f}, {by*1000:+.1f}, {bz*1000:+.1f}) mm")

    if not results:
        raise RuntimeError("All calibration methods failed.")

    # --- Cross-method comparison against the consensus (mean) cam2base ----- #
    consensus = average_transforms([np.array(r["cam2base"]) for r in results.values()])
    print("\n== Cross-method comparison (deviation from consensus cam2base) ==")
    header = f"  {'method':12s} {'tx':>8s} {'ty':>8s} {'tz':>8s}  " \
             f"{'Δpos(mm)':>9s} {'Δrot(deg)':>9s}  {'consist(mm)':>11s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, r in results.items():
        T = np.array(r["cam2base"])
        dpos = float(np.linalg.norm(T[:3, 3] - consensus[:3, 3])) * 1000
        drot = rot_angle_deg(consensus[:3, :3].T @ T[:3, :3])
        r["dev_from_consensus_mm"] = dpos
        r["dev_from_consensus_deg"] = drot
        tx, ty, tz = T[:3, 3]
        print(f"  {label:12s} {tx:8.4f} {ty:8.4f} {tz:8.4f}  "
              f"{dpos:9.2f} {drot:9.3f}  "
              f"{r['trans_consistency_std_m']*1000:11.2f}")

    # Best = lowest per-view consistency spread (most internally self-consistent).
    best = min(results, key=lambda k: results[k]["trans_consistency_std_m"])
    spread_mm = float(np.linalg.norm(
        np.std([np.array(r["cam2base"])[:3, 3] for r in results.values()], axis=0))) * 1000
    print(f"\nMethod agreement (std of base position across methods): {spread_mm:.2f} mm")
    print(f"Best by consistency: {best}")
    print("cam2base (camera pose in robot base frame):")
    print(np.array(results[best]["cam2base"]))

    out = {
        "config": "eye-to-hand (ceiling-mounted static camera, board on gripper)",
        "units": "meters",
        "num_views": len(kept),
        "views": kept,
        "best_method": best,
        "consensus_cam2base": consensus.tolist(),
        "method_agreement_std_mm": spread_mm,
        "results": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
