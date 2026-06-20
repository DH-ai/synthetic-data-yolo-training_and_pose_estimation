#!/usr/bin/env bash
set -euo pipefail

# Build the BlenderProc data-generation image.
cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE="${IMAGE:-blenderproc-datagen:latest}"

docker build -t "${IMAGE}" .
echo "Built ${IMAGE}"

