# 01 — Setup

Clone the parent repo, init submodules, and pick the right environment for generation vs training.

**DeepWiki:** [project overview](https://deepwiki.com/DH-ai/synthetic-data-yolo-training_and_pose_estimation/1-project-overview) · [GDRNPP install](https://deepwiki.com/DH-ai/GDRNPP/1.1-installation-and-environment-setup)

## Clone

```bash
git clone --recurse-submodules https://github.com/DH-ai/synthetic-data-yolo-training_and_pose_estimation.git
cd synthetic-data-yolo-training_and_pose_estimation

# Or, if already cloned:
git submodule update --init src/gdrnpp src/detectron2
```

Layout that matters:

| Path | Role |
|------|------|
| `src/blenderproc_proj/` | BlenderProc generation (`main.py`) |
| `docker/` | Generation container + `requirements.txt` |
| `src/gdrnpp/` | YOLOX + GDRN training (submodule) |
| `src/detectron2/` | Detectron2 (submodule; GDRNPP dependency) |
| `src/output/bop/` | Default BOP output (gitignored) |

## Environment A — data generation (CPU or GPU)

Used for BlenderProc only. Does **not** need CUDA for a smoke test (software EGL works; slow).

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r docker/requirements.txt
```

System GL/EGL libraries are required for the BOP writer’s pyrender step. The authoritative package list is in [`docker/Dockerfile`](../docker/Dockerfile). On a headless CPU host without NVIDIA:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

Optional Docker path (GPU host):

```bash
./docker/build.sh
./docker/run.sh 100   # NUM_ITERATIONS
```

See [02_generate_data.md](02_generate_data.md).

## Environment B — training (GPU + CUDA)

Train on a GPU host. **Do not install PyTorch / CUDA / detectron2 from these parent docs** — use the submodule’s install guide so versions stay in one place:

1. Init submodules (above), then follow [`src/gdrnpp/docs/INSTALL.md`](../src/gdrnpp/docs/INSTALL.md)
2. Build failures: [`src/gdrnpp/troubleshoot.md`](../src/gdrnpp/troubleshoot.md)
3. Parent-specific notes (paths, `mydataset`): [03_gdrnpp_submodule.md](03_gdrnpp_submodule.md)

Point `ref/mydataset.py` `root_dir` at the parent `src/` that contains `output/bop` (see [04_custom_dataset_mydataset.md](04_custom_dataset_mydataset.md)).

## Quick sanity checks

```bash
# Generation env
. .venv/bin/activate
blenderproc --version

# Training env (on GPU host, from src/gdrnpp)
conda activate gdrnpp_env
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

## Next

- Generate data → [02_generate_data.md](02_generate_data.md)
- Install GDRNPP → [03_gdrnpp_submodule.md](03_gdrnpp_submodule.md)
