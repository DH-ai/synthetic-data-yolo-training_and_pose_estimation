# GPU-enabled image for BlenderProc synthetic data generation.
# Base provides CUDA runtime; the NVIDIA driver is injected at run time via
# `--gpus all` (nvidia-container-toolkit). Blender ships its own OPTIX/CUDA kernels.
FROM nvidia/cuda:12.6.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    BLENDER_INSTALL_PATH=/opt/blenderproc/blender \
    # Make all GPUs and the capabilities Blender/Cycles needs visible to the container
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# System libraries required by headless Blender (Cycles + EGL/GL) plus Python tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        wget xz-utils ca-certificates \
        libxrender1 libxxf86vm1 libxfixes3 libxi6 libxkbcommon0 libsm6 libxext6 \
        libgl1 libglib2.0-0 libxrandr2 libxinerama1 libxcursor1 \
        libegl1 libgles2 libgomp1 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Recreate the project's environment inside a venv (the ".venv" equivalent)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /opt/app
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download Blender into the image and install OpenCV into Blender's bundled
# Python. This bakes the Blender install so containers start instantly (no
# download at run time) and `import cv2` works inside the render script.
RUN blenderproc pip install opencv-python==4.13.0.92 \
        --blender-install-path "${BLENDER_INSTALL_PATH}"

COPY entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

ENTRYPOINT ["/opt/app/entrypoint.sh"]
