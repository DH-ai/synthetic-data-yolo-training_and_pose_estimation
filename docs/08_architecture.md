# 08 — Architecture and data contracts

This page is the local architecture map for the complete vision pipeline. The
[parent DeepWiki](https://deepwiki.com/DH-ai/synthetic-data-yolo-training_and_pose_estimation/1-project-overview)
and [GDRNPP DeepWiki](https://deepwiki.com/DH-ai/GDRNPP/1-gdrnpp-modernized-project-overview)
contain deeper code-level navigation.

## System context

```mermaid
flowchart LR
    subgraph offline [This repository: offline preparation and training]
        CAD[Measured CAD assets] --> BP[BlenderProc]
        Calibration[Camera calibration] --> BP
        BP --> BOP[BOP dataset]
        BOP --> YOLOX[YOLOX training]
        BOP --> GDRN[GDRN training]
        YOLOX --> DetectorWeights[Detector checkpoint]
        GDRN --> PoseWeights[Pose checkpoint]
    end
    subgraph runtime [fanuc_pickn_place: runtime]
        Camera[Mech-Eye RGB-D] --> Detect[Object detection]
        DetectorWeights --> Detect
        Detect --> PoseEstimate[6D pose estimation]
        PoseWeights --> PoseEstimate
        PoseEstimate --> Robot[ROS 2 and FANUC M-20iD/35]
    end
```

The repository ends at trained weights. Robot control, MoveIt, gripper, force
sensing, and camera runtime integration belong in
[`fanuc_pickn_place`](https://github.com/DH-ai/fanuc_pickn_place).

## End-to-end artifact flow

```mermaid
flowchart TD
    Mesh[Triangulated PLY meshes] --> Models[models and models_info.json]
    Scene[Blender scene and HDRI] --> Main[blenderproc_proj/main.py]
    CameraK[Camera intrinsics K] --> Main
    Models --> Main
    Main --> TrainPBR[train_pbr scenes]
    TrainPBR --> Integrity[Dataset integrity gate]
    Integrity --> Split[Create independent test_pbr scenes]
    Split --> Registration[mydataset registration]
    Registration --> YTrain[YOLOX train and eval]
    Registration --> GTrain[GDRN train and eval]
    YTrain --> RawDet[COCO or BOP detection list]
    RawDet --> Convert[Convert detection schema]
    Convert --> DetDict[DET_FILES_TEST dictionary]
    DetDict --> GTrain
    Models --> FPS[fps_points.pkl]
    FPS --> GTrain
```

`main.py` generates `train_pbr`; it does not create a held-out `test_pbr`
automatically. Evaluation requires scenes generated or moved into a genuine
held-out split. Do not evaluate on copied training frames and report the result
as validation.

## Contract between stages

| Artifact | Producer | Consumer | Contract |
|----------|----------|----------|----------|
| PLY meshes | CAD / Blender | BlenderProc, GDRN renderer, FPS tools | Correct units, triangular faces, supported PLY types, vertex normals |
| `models_info.json` | BOP model tooling | Dataset loader / evaluator | Object IDs aligned with filenames; valid diameter |
| Camera calibration | OpenCV / ROS camera calibration | BlenderProc and pose loader | Correct `K`, distortion handling, frame convention and depth scale |
| `train_pbr` / `test_pbr` | BlenderProc + deliberate split step | YOLOX / GDRN registrations | Complete RGB, depth, masks and scene JSON for every image ID |
| Dataset names | GDRNPP registration modules | Config `DATASETS.*` | Name maps to the intended filesystem root and object mapping |
| `fps_points.pkl` | FPS preprocessing | GDRN region supervision | Generated after final mesh scaling |
| Detection JSON | YOLOX evaluation + converter | GDRN `DET_FILES_TEST` | Dict keyed by `scene_id/im_id`, `bbox_est` in original-image `xywh` |
| Checkpoint | Trainer | Resume / inference | Same model architecture, class count and compatible config |

## BOP scene files

```mermaid
flowchart LR
    Frame[One rendered frame] --> RGB[rgb image]
    Frame --> Depth[depth image]
    Frame --> FullMask[mask]
    Frame --> VisibleMask[mask_visib]
    Frame --> GT[scene_gt.json]
    Frame --> GTInfo[scene_gt_info.json]
    Frame --> SceneCamera[scene_camera.json]
    GT --> PoseData[Object ID, rotation and translation]
    GTInfo --> BoxData[bbox_obj, bbox_visib and visibility]
    SceneCamera --> CameraData[cam_K and depth_scale]
```

Bounding boxes do **not** come from `scene_gt.json`. A loader error involving a
box should be traced through `scene_gt_info.json`; camera and depth conversion
errors should be traced through `scene_camera.json`.

## Camera calibration contract

OpenCV pose estimates and BlenderProc camera poses may describe opposite
transform directions and use different axis conventions. A plausible axis
overlay is not proof of a correct calibration.

```mermaid
flowchart TD
    Board[Chessboard or Charuco observations] --> Calibrate[OpenCV calibration]
    Calibrate --> K[Intrinsics and distortion]
    Calibrate --> RvecTvec[rvec and tvec]
    RvecTvec --> Rodrigues[Rodrigues rotation]
    Rodrigues --> Transform[Homogeneous transform]
    Transform --> Convention{Which frames does it connect?}
    Convention -->|Camera to world| CameraPose[BlenderProc camera pose]
    Convention -->|World to camera| Invert[Invert transform]
    Invert --> CameraPose
    K --> Project[Project known 3D points with OpenCV]
    CameraPose --> Render[Render reference scene]
    Project --> Compare[Compare projected and observed pixels]
    Render --> Compare
```

Verify marker dimensions, `K`, distortion, transform direction, and OpenCV ↔
Blender axes by projecting known 3D points. Record hand-eye calibration as
complete only when this geometric test and the robot-frame transform are
repeatable.

## GDRN training internals

```mermaid
flowchart LR
    Dataset[BOP dataset dictionary] --> Loader[Data loader]
    Loader --> ROI[ROI crop and transforms]
    Loader --> PoseGT[Pose, camera and masks]
    PoseGT --> Renderer[EGL online XYZ renderer]
    Renderer --> XYZ[XYZ tensor]
    ROI --> ROIXYZ[ROI XYZ and object mask]
    XYZ --> ROIXYZ
    FPS[fps_points.pkl] --> Regions[Region assignment]
    ROIXYZ --> Regions
    ROI --> Backbone[ConvNeXt backbone]
    Backbone --> GeoHead[Mask, XYZ and region heads]
    GeoHead --> PnP[PnP network]
    Regions --> Losses[Geometric losses]
    PnP --> Losses
    Losses --> Checkpoint[Pose checkpoint]
```

YOLOX success proves only the detector path. GDRN additionally depends on mesh
scale, renderer output, FPS points, ROI transforms, masks, region labels, and
PnP supervision.

## Configuration and path resolution

```mermaid
flowchart TD
    Config[Config DATASETS.TRAIN and TEST] --> Names[Registered names only]
    Names --> Factory[Dataset factory]
    Factory --> Ref[ref/mydataset.py]
    Factory --> Module[mydataset_pbr.py]
    Ref --> Roots[train_dir, test_dir and model_dir]
    Module --> Records[Detectron2 dataset records]
    Roots --> Disk[src/output/bop]
    Disk --> Records
```

Current `mydataset` files contain machine-specific absolute paths. Before
training, reconcile `ref/mydataset.py`, the GDRN registration module, and the
YOLOX registration modules. Relative paths derived from the current working
directory are not a robust substitute; use one explicit dataset root.

## Validation gates

```mermaid
flowchart TD
    Generated[Generation completed] --> Files{Every frame has RGB, depth and masks?}
    Files -->|No| Repair[Regenerate or repair the scene]
    Files -->|Yes| JSON{GT, GT info and camera JSON parse?}
    JSON -->|No| Repair
    JSON -->|Yes| Boxes{Boxes and masks have non-zero area?}
    Boxes -->|No| Repair
    Boxes -->|Yes| Geometry{Mesh units, normals, triangles and diameter valid?}
    Geometry -->|No| FixMesh[Fix mesh and regenerate FPS]
    Geometry -->|Yes| Overlay{GT boxes align on real loaded pixels?}
    Overlay -->|No| FixRegistration[Fix dimensions, paths or stale cache]
    Overlay -->|Yes| Train[Start training]
```

Run the gates before a long GPU job. Assertions, `KeyError`, and
`JSONDecodeError` during loading often indicate an interrupted render rather
than a model defect.

## Historical incident status

The root [`documentation_final.md`](../documentation_final.md) is a project
incident draft, not the source of truth for current commands. In particular:

- its generic “registration/base-class mismatch” explanation for YOLOX
  `loss_l1=inf` is superseded by the verified 960×600 declared-size versus
  1920×1200 image-size failure fixed in `Base_DatasetFromList.load_anno`;
- its hand-eye sections contain conflicting “must retake” and “completed”
  statements, so completion must be confirmed from geometric evidence;
- environment and parser fixes are historical compatibility notes and may not
  apply to the current submodule revision.

Use [07_troubleshooting.md](07_troubleshooting.md) for the maintained incident
matrix.
