# 03 — GDRNPP submodule (install & compile)

GDRNPP lives at `src/gdrnpp` and provides YOLOX detection + GDRN 6D pose. This page covers **CUDA / PyTorch / detectron2** install for this parent repo. Dataset wiring is [04](04_custom_dataset_mydataset.md).

**DeepWiki:** [GDRNPP overview](https://deepwiki.com/DH-ai/GDRNPP/1-gdrnpp-modernized-project-overview) · [Installation](https://deepwiki.com/DH-ai/GDRNPP/1.1-installation-and-environment-setup)  
**Also:** [`src/gdrnpp/docs/INSTALL.md`](../src/gdrnpp/docs/INSTALL.md) · [`src/gdrnpp/troubleshoot.md`](../src/gdrnpp/troubleshoot.md)

## Requirements

- NVIDIA GPU + **CUDA ≥ 10.1** (driver/toolkit matching the PyTorch wheel you install)
- Ubuntu ≥ 16.04 (18.04+ preferred)
- Python ≥ 3.6 (this guide uses **3.10** / env name `gdrnpp_env`)
- PyTorch ≥ 1.9 + torchvision

## Init submodule

From the parent repo root:

```bash
git submodule update --init src/gdrnpp src/detectron2
```

Confirm `src/gdrnpp` is not empty (`ls src/gdrnpp/configs`).

## 1. Conda env + PyTorch (CUDA)

```bash
conda create -n gdrnpp_env python=3.10 -y
conda activate gdrnpp_env

# Install PyTorch + torchvision for YOUR CUDA version:
# https://pytorch.org/get-started/locally/
# Example only — replace cuXXX with the build that matches `nvidia-smi` / toolkit:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expect: True for cuda.is_available() on the training machine
```

If `cuda.is_available()` is `False`, fix the driver / CUDA / wheel mismatch before continuing.

## 2. Detectron2 (from this repo’s submodule)

Use the sibling submodule at `src/detectron2` (already initialized above) — do not need a second clone:

```bash
# From parent repo root, with gdrnpp_env active:
pip install ninja
cd src/detectron2
pip install -e .
cd ../gdrnpp
```

Upstream notes: [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2). Full options also in [`INSTALL.md`](../src/gdrnpp/docs/INSTALL.md).

## 3. GDRNPP system + Python deps

```bash
cd src/gdrnpp   # if not already there

# apt + python extras (needs sudo for apt section)
sh scripts/install_deps.sh

# Python-only if you are not a sudoer:
sh scripts/install_deps.sh python
```

## 4. Compile CUDA / C++ extensions

```bash
cd src/gdrnpp
sh scripts/compile_all.sh
```

This builds FPS, EGL renderer (needed for `XYZ_ONLINE=True`), and related ops. If Ceres / uncertainty PnP / EGL / detectron2 build fail, follow [`troubleshoot.md`](../src/gdrnpp/troubleshoot.md) rather than inventing flags. Ubuntu 18.04 `libassimp` note: see `INSTALL.md`.

## Sanity check

```bash
conda activate gdrnpp_env
cd src/gdrnpp
python -c "import torch, detectron2; print(torch.cuda.is_available(), torch.__version__)"
python -c "import ref; print(ref.mydataset.obj_num, ref.mydataset.width, ref.mydataset.height)"
```

## Path model (read this before configs)

| Layer | What it stores |
|-------|----------------|
| Config `DATASETS.TRAIN` / `TEST` | **Dataset names only** (e.g. `"mydataset_pbr_train"`) |
| `ref/mydataset.py` | Filesystem roots (`train_dir`, `model_dir`), camera, `id2obj` |
| `core/gdrn_modeling/datasets/mydataset_pbr.py` | Detectron2 registration; builds paths from `ref` + local `PROJ_ROOT` |
| YOLOX twin | `det/yolox/data/datasets/mydataset_pbr.py` (+ test variant) |

There is **no** requirement that data live under `datasets/BOP_DATASETS/` for `mydataset`. Stock BOP datasets (hb, ycbv, …) use that layout; custom synthetic data from this parent repo typically lives at `src/output/bop`.

## Next

- Wire `mydataset` → [04_custom_dataset_mydataset.md](04_custom_dataset_mydataset.md)
- Train → [05_train_yolox.md](05_train_yolox.md), [06_train_gdrn.md](06_train_gdrn.md)
