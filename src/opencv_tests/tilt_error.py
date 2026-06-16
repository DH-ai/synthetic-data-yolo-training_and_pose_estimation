import argparse
from dataclasses import dataclass
from pathlib import Path
import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np


K = np.array(
    [
        [2481.9412514178307, 0.0, 978.95936559694314],
        [0.0, 2482.3917472975795, 629.72289542481894],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

dist = np.array(
    [
        [
            -0.091539129459748417,
            1.6518788910916924,
            -0.00096826424151305102,
            -0.0023115236516727399,
            -7.1086932137755738,
        ]
    ],
    dtype=np.float64,
)

ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.022
MARKER_LENGTH = 0.011


@dataclass
class TiltResult:
    image_name: str
    corner_count: int
    tvec_norm_mm: float
    tilt_angle_deg: float
    perpendicular_distance_mm: float
    tz_distance_mm: float
    tilt_depth_error_mm: float


def create_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict,
    )
    return board, cv2.aruco.CharucoDetector(board)


def calculate_tilt_error(image_path: Path, board, detector, min_corners: int) -> TiltResult:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("could not read image")

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(img)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_corners:
        found = 0 if charuco_ids is None else len(charuco_ids)
        raise ValueError(f"not enough ChArUco corners detected ({found} < {min_corners})")

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not success:
        raise ValueError("solvePnP failed")

    rotation_matrix = cv2.Rodrigues(rvec)[0]
    translation = tvec.flatten()

    board_normal = rotation_matrix[:, 2]
    perpendicular_distance_m = float(board_normal @ translation) # dot product syntax intersting
    # print(f"perpendicular_distance_m: {perpendicular_distance_m}")
    tilt_angle_rad = np.arccos(np.clip(rotation_matrix[2, 2], -1.0, 1.0))
    tilt_depth_error_m = translation[2] - perpendicular_distance_m

    return TiltResult(
        image_name=image_path.name,
        corner_count=len(charuco_ids),
        tvec_norm_mm=float(np.linalg.norm(translation) * 1000.0),
        tilt_angle_deg=float(np.degrees(tilt_angle_rad)),
        perpendicular_distance_mm=perpendicular_distance_m * 1000.0,
        tz_distance_mm=float(translation[2] * 1000.0),
        tilt_depth_error_mm=float(tilt_depth_error_m * 1000.0),
    )


def iter_images(image_dir: Path, pattern: str):
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for image_path in image_dir.glob(pattern):
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            yield image_path


def print_result(result: TiltResult):
    print(
        f"{result.image_name}: "
        f"corners={result.corner_count:2d}, "
        f"|tvec|={result.tvec_norm_mm:9.2f} mm, "
        f"tilt={result.tilt_angle_deg:8.3f} deg, "
        f"d_perp={result.perpendicular_distance_mm:9.2f} mm, "
        f"t_z={result.tz_distance_mm:9.2f} mm, "
        f"tilt_depth_error={result.tilt_depth_error_mm:8.3f} mm"
    )


def summarize(results: list[TiltResult]):
    tilt_angles = np.array([result.tilt_angle_deg for result in results], dtype=np.float64)
    tilt_errors = np.array([result.tilt_depth_error_mm for result in results], dtype=np.float64)
    perpendicular_distances = np.array([result.perpendicular_distance_mm for result in results], dtype=np.float64)
    print("\nSummary")
    print(f"Images processed              : {len(results)}")
    print(f"Average tilt angle            : {tilt_angles.mean():.3f} deg")
    print(f"Average signed tilt error     : {tilt_errors.mean():.3f} mm")
    print(f"Average absolute tilt error   : {np.abs(tilt_errors).mean():.3f} mm")
    print(f"Max absolute tilt error       : {np.abs(tilt_errors).max():.3f} mm")
    print(f"Std dev tilt error            : {tilt_errors.std(ddof=0):.3f} mm")
    print(f"Average perpendicular distance: {perpendicular_distances.mean():.3f} mm")
    print(f"Max perpendicular distance    : {perpendicular_distances.max():.3f} mm")
    print(f"Min perpendicular distance    : {perpendicular_distances.min():.3f} mm")
    print(f"Std dev perpendicular distance: {perpendicular_distances.std(ddof=0):.3f} mm")
    return perpendicular_distances.mean(), tilt_angles.mean(), np.abs(tilt_errors).mean()


def binned_std_by_distance(results: list[TiltResult], bin_count: int):
    distances = np.array([result.tvec_norm_mm for result in results], dtype=np.float64)
    errors = np.array([result.tilt_depth_error_mm for result in results], dtype=np.float64)

    if len(results) < 2:
        return np.array([]), np.array([]), np.array([])

    bin_count = max(1, min(bin_count, len(results)))
    edges = np.linspace(distances.min(), distances.max(), bin_count + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    std_values = []
    sample_counts = []

    for index in range(bin_count):
        if index == bin_count - 1:
            mask = (distances >= edges[index]) & (distances <= edges[index + 1])
        else:
            mask = (distances >= edges[index]) & (distances < edges[index + 1])

        bin_errors = errors[mask]
        sample_counts.append(len(bin_errors))
        std_values.append(float(bin_errors.std(ddof=0)) if len(bin_errors) else np.nan)

    return centers, np.array(std_values), np.array(sample_counts)


def plot_std_vs_tvec_norm(results: list[TiltResult], output_path: Path, bin_count: int):
    distances = np.array([result.tvec_norm_mm for result in results], dtype=np.float64)
    abs_errors = np.abs(
        np.array([result.tilt_depth_error_mm for result in results], dtype=np.float64)
    )
    centers, std_values, sample_counts = binned_std_by_distance(results, bin_count)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(distances, abs_errors, color="tab:blue", alpha=0.75, label="Absolute tilt error")
    ax.plot(
        centers,
        std_values,
        color="tab:red",
        marker="o",
        linewidth=2,
        label="Std dev of signed tilt error by |tvec| bin",
    )

    for center, std_value, count in zip(centers, std_values, sample_counts):
        if not np.isnan(std_value):
            ax.annotate(
                f"n={count}",
                (center, std_value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )

    ax.set_title("Tilt Error Standard Deviation vs Absolute Translation Distance")
    ax.set_xlabel("|tvec| distance (mm)")
    ax.set_ylabel("Tilt error / std dev (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    default_image_dir = Path(__file__).resolve().parent / "media_charucoBoard"
    default_plot_path = Path(__file__).resolve().parent / "tilt_error_std_vs_tvec_norm.png"

    parser = argparse.ArgumentParser(
        description="Calculate average ChArUco tilt error and plot std dev vs |tvec| distance."
    )
    parser.add_argument("--image-dir", type=Path, default=default_image_dir)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--min-corners", type=int, default=4)
    parser.add_argument("--bins", type=int, default=5, help="Distance bins for std-dev plot.")
    parser.add_argument("--plot", type=Path, default=default_plot_path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {args.image_dir}")

    board, detector = create_detector()
    results = []
    skipped = 0
    i = 0
    for image_path in iter_images(args.image_dir, args.pattern):

        try:
            result = calculate_tilt_error(image_path, board, detector, args.min_corners)
            i += 1
            if i > 13:
                break
        except ValueError as exc:
            skipped += 1
            print(f"{image_path.name}: skipped ({exc})")
            continue

        results.append(result)
        print_result(result)

    if not results:
        raise RuntimeError(f"No usable images found in {args.image_dir} matching {args.pattern}")

    d_perp_avg, avg_tilt_angle, avg_tilt_error = summarize(results)
    if skipped:
        print(f"Skipped images                : {skipped}")

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plot_std_vs_tvec_norm(results, args.plot, args.bins)
    print(f"Saved std-dev plot            : {args.plot}")

    data = {
        "d_perp_avg": float(d_perp_avg),
        "avg_tilt_angle": float(avg_tilt_angle),
        "avg_tilt_error": float(avg_tilt_error),
    }
    with open("data_for_pose_estimation.yaml", "w") as f:
        yaml.dump(data, f,default_flow_style=False, sort_keys=False)
    print(f"Saved pose data to data_for_pose_estimation.yaml")
if __name__ == "__main__":
    main()
