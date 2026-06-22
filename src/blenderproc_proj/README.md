# blenderproc_proj

BlenderProc-based synthetic scene generation. This folder renders the synthetic scenes (RGB + depth + segmentation masks + 6D pose labels) and exports them in BOP and COCO formats for downstream YOLO and GDRNPP training.

## Files

- `main.py` — main BlenderProc pipeline: builds the scene, applies domain randomization, renders RGB/depth/masks, and writes the BOP + COCO output. Includes file-based logging and run-state summaries on crash/exit. Run it through BlenderProc, e.g. `blenderproc run main.py`.
- `coco_viewer.py` — quick standalone viewer to sanity-check the generated COCO annotations (decodes RLE segmentation masks and overlays them).
- `camera.py` — ⚠️ **dead file, will be removed during repo cleanup.** Holds the calibrated camera intrinsics (`K`, distortion) and a `set_camera()` helper. It was meant to be a reusable camera module, but BlenderProc would not import it as a local module (it kept asking to `pip install` it as a package), so the camera setup was inlined into `main.py` instead. Kept only for reference.

## Folders
- `output/` — generated datasets (`bop/`, `coco/`). # Ignored in .gitignore as the sized can be 20+Gbs
