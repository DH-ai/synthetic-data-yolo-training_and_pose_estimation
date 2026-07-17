# 03 — GDRNPP submodule (install & compile)

GDRNPP lives at `src/gdrnpp` and provides YOLOX detection + GDRN 6D pose. This page is install-only; dataset wiring is [04](04_custom_dataset_mydataset.md).

**DeepWiki:** [GDRNPP overview](https://deepwiki.com/DH-ai/GDRNPP/1-gdrnpp-modernized-project-overview) · [Installation](https://deepwiki.com/DH-ai/GDRNPP/1.1-installation-and-environment-setup)  
**Canonical install doc:** [`src/gdrnpp/docs/INSTALL.md`](../src/gdrnpp/docs/INSTALL.md)  
**Build failures:** [`src/gdrnpp/troubleshoot.md`](../src/gdrnpp/troubleshoot.md)

## Init submodule

From the parent repo root:

```bash
git submodule update --init src/gdrnpp src/detectron2
```

Confirm `src/gdrnpp` is not empty (`ls src/gdrnpp/configs`).

## Create the training env

On a **GPU** host (CUDA matching your PyTorch wheel):

```bash
conda create -n gdrnpp_env python=3.10 -y
conda activate gdrnpp_env
# Install PyTorch + torchvision for your CUDA version (pytorch.org)
```

Install Detectron2 from the sibling submodule (or as documented in `INSTALL.md`):

```bash
cd src/detectron2
pip install -e .
cd ../gdrnpp
```

## System + Python deps

From `src/gdrnpp`:

```bash
# apt + python extras (needs sudo for apt section)
sh scripts/install_deps.sh

# Python-only if you are not a sudoer:
sh scripts/install_deps.sh python
```

## Compile CUDA / C++ extensions

```bash
cd src/gdrnpp
sh scripts/compile_all.sh
```

If Ceres / uncertainty PnP / EGL renderer fail, follow [`troubleshoot.md`](../src/gdrnpp/troubleshoot.md) rather than inventing flags.

Useful pieces compiled here include FPS sampling, EGL renderer (needed for `XYZ_ONLINE=True` in GDRN), and related CUDA ops.

## Sanity check

```bash
conda activate gdrnpp_env
cd src/gdrnpp
python -c "import torch, detectron2; print(torch.cuda.is_available())"
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
