# Synthetic Data YOLO Training & Pose Estimation

This repository contains the pipeline for generating synthetic datasets for object detection and 6D pose estimation using BlenderProc, BOP format datasets, segmentation masks, and simulation-based domain randomization.

## Features

- Synthetic RGB image generation
- Instance & semantic segmentation masks
- YOLO dataset generation
- BOP dataset generation
- 6D pose ground-truth export
- Domain randomization for sim-to-real transfer
- FoundationPose-compatible dataset creation

## Workflow

```text
BlenderProc
    ↓
Synthetic Scene Generation
    ↓
RGB + Depth + Masks + Pose Labels
    ↓
├── YOLO Training
└── FoundationPose Fine Tuning / Training