# AGENTS.md

## Cursor Cloud specific instructions

This repo is a synthetic-data generation + 6D pose-estimation pipeline
(BlenderProc → BOP/COCO datasets → YOLO/GDRNPP training). See `README.md` and the
per-folder READMEs (`src/blenderproc_proj/README.md`, `src/opencv_tests/README.md`,
`tooling/FILE_SERVER_SETUP.md`) for the standard commands.

### Environment
- Python deps live in a **`.venv`** at the repo root (gitignored). Activate with
  `. .venv/bin/activate`. Deps are pinned in `docker/requirements.txt` (the update
  script installs these); `requirmentv14.txt` is a pip-freeze snapshot, not directly installable.
- The Cloud VM is **CPU-only** (no GPU, no Docker, no NVIDIA driver). GL/EGL/Mesa system
  libraries required by the render pipeline are already baked into the VM snapshot; the
  authoritative list is `docker/Dockerfile` (plus `libegl-mesa0`/`libgl1-mesa-dri` for
  software EGL). They are NOT reinstalled by the update script.

### BlenderProc render pipeline (`src/blenderproc_proj/main.py`) — the core product
- Run from the **repo root** (paths to `blender_files/` and `assets/` are relative):
  `NUM_ITERATIONS=1 blenderproc run src/blenderproc_proj/main.py`.
- On first run BlenderProc downloads Blender 4.2.1 into `~/blender/` (cached in the
  snapshot afterwards, so subsequent runs skip the download).
- No GPU: prefix with `LIBGL_ALWAYS_SOFTWARE=1` so the BOP writer's `pyrender` mask step
  uses Mesa software EGL. Without it you get `ImportError: Unable to load EGL library`.
  CPU render is ~70 s/frame at 1920×1200.
- Output is written to `src/output/bop` (override with `OUTPUT_DIR_BPROC`); `NUM_ITERATIONS`
  controls the number of rendered frames. `src/output` and `output/` are gitignored.
- Gotcha: the BOP writer *appends* to existing chunk dirs. A crashed/partial run leaves
  `src/output/bop/...` inconsistent and the next run fails with
  `FileNotFoundError: .../scene_gt_info.json`. **Delete `src/output` before re-running.**

### Other runnable components (CPU-only)
- `tooling/file_server.py` — stdlib HTTP file server: `python tooling/file_server.py --dir <dir> --port 8000`.
  Pair with `tooling/file_client.py` (`list`/`upload`/`download`, needs `requests`).
- `src/opencv_tests/gen_pattern.py` — generates calibration-pattern SVGs (needs only `opencv-python`).
- Most other `opencv_tests` scripts need input images that are **not** in the repo (they
  reference a LAN file server) and some need `open3d`/`trimesh` (not in `docker/requirements.txt`)
  or a GUI display — so they are not runnable out-of-the-box here.

### Submodules (out of scope on the CPU VM)
- `src/gdrnpp` and `src/detectron2` are git submodules that are **empty/uninitialized** and
  require a GPU + CUDA toolchain (PyTorch/Detectron2/MMCV) to build and run.

### Lint / test
- There is no configured linter or test framework. Use `python -m py_compile <files>` for a
  syntax check (exclude the empty submodule dirs).
