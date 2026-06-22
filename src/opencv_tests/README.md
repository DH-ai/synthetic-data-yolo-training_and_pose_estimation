# opencv_tests

OpenCV-based experiments and tooling used to support the synthetic-data and pose-estimation pipeline: camera calibration, calibration-target generation, classical 6D pose estimation (ICP / PnP), and error analysis. These are working/exploratory scripts rather than a packaged module.

## Files

- `calibration.py` — camera calibration from chessboard / ChArUco images; estimates the intrinsics matrix `K` and distortion coefficients.
- `calibration_charuco.yaml` — saved ChArUco calibration parameters / results.
- `gen_pattern.py` — generates printable calibration patterns (checkerboard / circles / ChArUco) as SVG. Sourced from the OpenCV project.
- `svgfig.py` — third-party SVG helper library (GPL, by Jim Pivarski) that `gen_pattern.py` depends on.
- `IPC_test.py` — ICP-based pose estimation: aligns a CAD model to a scene point cloud and projects the resulting 6D pose onto the RGB image. (Filename is a typo for "ICP".)
- `6d_pose_visualizer.py` — visualizes ground-truth 6D poses from a BOP-format dataset by projecting them onto the RGB images.
- `vector_TR.py` — experiments with translation/rotation (tvec/rvec) vectors across different calibration sources (checkerboard, Mech-Eye SDK, ChArUco), with 3D plotting.
- `tilt_error.py` — analyzes pose/tilt error vs. translation-vector norm; produces `tilt_error_std_vs_tvec_norm.png`.
- `tilt_error_std_vs_tvec_norm.png` — plot output from `tilt_error.py`.

## Folders

- `charuco/` — generated ChArUco board image(s).
- `media_charucoBoard/` — ChArUco calibration capture images.
- `media_chessBoard/` — chessboard calibration capture images.
