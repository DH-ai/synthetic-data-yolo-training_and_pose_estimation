#!/usr/bin/env bash
set -euo pipefail

# Run the BlenderProc data generation on all available GPUs.
#
# Usage:
#   ./run.sh [NUM_ITERATIONS]
#
# Env overrides:
#   IMAGE       docker image tag           (default: blenderproc-datagen:latest)
#   REPO_DIR    path to this repo on host  (default: auto-detected, 3 dirs up)
#   GPUS        docker --gpus value        (default: all)

IMAGE="${IMAGE:-blenderproc-datagen:latest}"
GPUS="${GPUS:-all}"
NUM_ITERATIONS="${1:-1000}"

# Repo root = .../docker -> blenderproc_proj -> src -> <repo>
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# Mount the repo at the absolute path main.py expects, so its hardcoded paths work
CONTAINER_PROJECT="/synthetic-data-yolo-training_and_pose_estimation"

echo "Repo (host)     : ${REPO_DIR}"
echo "Mounted to      : ${CONTAINER_PROJECT}"
echo "Iterations      : ${NUM_ITERATIONS}"
echo "GPUs            : ${GPUS}"

docker run --rm \
    --gpus "${GPUS}" \
    --shm-size=8g \
    -e NUM_ITERATIONS="${NUM_ITERATIONS}" \
    -e PROJECT_DIR="${CONTAINER_PROJECT}" \
    -v "${REPO_DIR}:${CONTAINER_PROJECT}" \
    "${IMAGE}"
