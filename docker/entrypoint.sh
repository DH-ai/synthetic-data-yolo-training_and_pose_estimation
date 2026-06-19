#!/usr/bin/env bash
set -euo pipefail

# The repo is mounted at this absolute path so the hardcoded paths inside
# main.py (blend file, HDRI, output dir) resolve unchanged.
: "${PROJECT_DIR:=/synthetic-data-yolo-training_and_pose_estimation}"
: "${NUM_ITERATIONS:=1000}"
: "${BLENDER_INSTALL_PATH:=/opt/blenderproc/blender}"

export NUM_ITERATIONS

SCRIPT="${PROJECT_DIR}/src/blenderproc_proj/main.py"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: ${SCRIPT} not found. Did you mount the repo to ${PROJECT_DIR}?" >&2
    exit 1
fi

echo "=============================================================="
echo " BlenderProc data generation"
echo "   project    : ${PROJECT_DIR}"
echo "   iterations : ${NUM_ITERATIONS}"
echo "=============================================================="

# bproc.init() auto-selects OPTIX/CUDA GPUs when present.
exec blenderproc run \
    --blender-install-path "${BLENDER_INSTALL_PATH}" \
    "${SCRIPT}"
