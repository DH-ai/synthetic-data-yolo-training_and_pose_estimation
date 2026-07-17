# 03 — GDRNPP submodule

GDRNPP lives at `src/gdrnpp` and provides YOLOX detection + GDRN 6D pose. Dataset wiring is [04](04_custom_dataset_mydataset.md).

**DeepWiki:** [GDRNPP overview](https://deepwiki.com/DH-ai/GDRNPP/1-gdrnpp-modernized-project-overview) · [Installation](https://deepwiki.com/DH-ai/GDRNPP/1.1-installation-and-environment-setup)

## Init submodule

From the parent repo root:

```bash
git submodule update --init src/gdrnpp src/detectron2
```

Confirm `src/gdrnpp` is not empty (`ls src/gdrnpp/configs`).

## Install (do not duplicate here)

PyTorch, CUDA matching, detectron2, `scripts/install_deps.sh`, and `scripts/compile_all.sh` are documented in the submodule — follow those, not a second copy in this parent repo:

1. [`src/gdrnpp/docs/INSTALL.md`](../src/gdrnpp/docs/INSTALL.md) — full install
2. [`src/gdrnpp/troubleshoot.md`](../src/gdrnpp/troubleshoot.md) — Ceres / CUDA / detectron2 / EGL build failures

Typical env name used elsewhere in these docs: `gdrnpp_env` on a GPU host.

After install, a quick check from `src/gdrnpp`:

```bash
conda activate gdrnpp_env
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
