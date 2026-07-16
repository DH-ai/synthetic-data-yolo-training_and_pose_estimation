# Synthetic Data YOLO Training & Pose Estimation

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DH-ai/synthetic-data-yolo-training_and_pose_estimation)

This repository contains the pipeline for generating synthetic datasets for object detection and 6D pose estimation. It uses **BlenderProc** to render photorealistic synthetic scenes (RGB + depth + segmentation masks + pose labels) and **GDRNPP** for 6D pose estimation, alongside YOLO for object detection. Datasets are produced in BOP format with simulation-based domain randomization for sim-to-real transfer.

It's a part of a larger robot automation project to follow; please refer to [Fanuc Pick n Place](https://github.com/dh-ai/fanuc_pickn_place)

## Features

- Synthetic RGB image generation with BlenderProc
- Instance & semantic segmentation masks
- YOLO dataset generation
- BOP dataset generation
- 6D pose ground-truth export
- Domain randomization for sim-to-real transfer
- GDRNPP-based 6D pose estimation training & fine-tuning
- FoundationPose-compatible dataset creation

## Workflow

```text
BlenderProc
    │
    ▼
Synthetic Scene Generation (Domain Randomization)
    │
    ▼
RGB + Depth + Segmentation Masks + 6D Pose Labels (BOP format)
    │
    ├──────────────┐
    ▼              ▼
YOLO Training   GDRNPP Training / Fine-Tuning
(Detection)     (6D Pose Estimation)
    │              │
    └──────┬───────┘
           ▼
   Trained Weights
   (YOLO detection + GDRNPP 6D pose)
           │
           ▼
   Consumed by the ROS implementation in
   Fanuc Pick n Place (separate repo)
```

## Scope

This repository ends at the **trained weights**. It is intentionally focused on synthetic data generation and model training/fine-tuning only.

The ROS implementation — robot control, perception nodes, motion planning, and the pick-and-place execution that *consumes* these weights — lives in the separate [Fanuc Pick n Place](https://github.com/dh-ai/fanuc_pickn_place) repository. Keeping it there avoids unnecessarily coupling the data/training pipeline with the runtime ROS stack and keeps each repo simple and single-purpose.

## Getting Started

> **TODO:** Add setup and usage instructions here — environment setup, BlenderProc scene generation, dataset export, YOLO training, and GDRNPP training/fine-tuning steps.