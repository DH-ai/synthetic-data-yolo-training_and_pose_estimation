# Mid-Internship Review Report

**Project:** Vision-Guided Autonomous Pick-and-Place on a FANUC Industrial Robot
**Duration:** 8-week internship · Started 21 May 2026 · This review covers Weeks 1–6
**Intern:** Dhruv
**Repositories:**
- Vision & synthetic data pipeline — [synthetic-data-yolo-training_and_pose_estimation](https://github.com/DH-ai/synthetic-data-yolo-training_and_pose_estimation)
- Robot control (ROS 2) — [fanuc_pickn_place](https://github.com/DH-ai/fanuc_pickn_place)
- Modernized 6D pose model — [gdrnpp_bop2022 (patched fork)](https://github.com/DH-ai/gdrnpp_bop2022)

---

## 1. The Story (Elevator Pitch)

Today, if you want an industrial robot to pick up a part, an engineer has to program every single movement by hand. The robot is blind — it repeats a memorized routine. If the part moves two centimeters, the robot fails. If the part changes, someone has to reprogram everything.

**My project gives the robot eyes and a brain.**

I am building a system where a robot looks at a table through a 3D camera, *recognizes* the parts lying on it, *figures out* exactly where each part is and how it is oriented in 3D space, and then *picks it up and places it* into the correct location — completely on its own, with no pre-programmed positions.

The demonstration task: a table with heart, triangle, and semicircle shaped parts scattered on it, and a board with matching cavities. The robot must see each shape, pick it, and slot it into the right cavity — like a child's shape-sorter toy, but performed by a 25 kg-payload FANUC industrial robot with millimeter precision.

**The key innovation is *how* the robot learns to see.** Normally, teaching an AI to recognize objects requires thousands of hand-photographed, hand-labeled images — weeks of tedious human effort *per part*. Instead, I generate all training data **synthetically**: I take the CAD model of a part, place it in a physically accurate virtual copy of our real workcell, and render thousands of photorealistic training images automatically overnight. The AI trains entirely on simulated images and then works on the real robot.

**Why that matters:** Bajaj already has a CAD model for every component it manufactures. This means a robot can be taught to handle a *brand-new part* without anyone ever photographing or labeling anything — you go from a CAD file to a working robot vision model in about a day, at near-zero incremental cost.

---

## 2. Exact Project Definition

> **Build a complete, autonomous vision-guided pick-and-place system:** integrate a FANUC M-20iD/25 robot with ROS 2, a Mech-Eye 3D stereo camera, a Tesollo 3-finger gripper, and a FANUC FS-40iA force sensor, so that the robot can detect known objects (heart, semicircle, triangle), estimate their full 6D pose (position + orientation), and autonomously pick and place them into matching cavities — with all vision models trained purely on synthetic data generated from CAD models.

The project splits cleanly into two halves:

| Half | What it does | Repo |
|---|---|---|
| **Perception & training pipeline** | CAD → simulated scenes → synthetic dataset → trained detection (YOLOX) + 6D pose (GDRNPP) models | `synthetic-data-yolo-training_and_pose_estimation` + `gdrnpp_bop2022` fork |
| **Robot execution stack** | ROS 2 drivers, motion planning (MoveIt 2), gripper & force-sensor integration, camera calibration, pick-and-place logic that *consumes* the trained models | `fanuc_pickn_place` |

**System pipeline (one line):**
Camera image → YOLOX finds the object → GDRNPP computes its exact 3D position & orientation → coordinates transformed into the robot's frame (hand-eye calibration) → MoveIt 2 plans a collision-free path → gripper picks → robot places into cavity.

---

## 3. Why This Project Is Important for Bajaj

### The problem with today's automation
Conventional industrial robots at any manufacturing plant are "record and replay" machines. Every application needs:
- Precise fixturing so parts always arrive in the exact same position
- Weeks of manual robot programming per task
- Complete reprogramming whenever a part or layout changes

This makes automation **expensive, rigid, and slow to redeploy** — economical only for very high-volume, never-changing tasks.

### What this project unlocks
1. **Flexible automation.** A robot that sees doesn't need fixtures or exact part positioning. Parts can arrive loosely placed, in trays, or in bins.
2. **Near-zero cost of teaching new parts.** Because training data is synthetic (generated from CAD), adding a new component to the system costs GPU-hours, not engineer-weeks. Bajaj's entire parts catalog already exists as CAD models — the raw material for this pipeline is already sitting on company servers.
3. **In-house capability, not vendor lock-in.** Commercial vision-guided-robotics solutions are closed black boxes with per-seat licensing. This project builds the capability internally, on open frameworks (ROS 2, PyTorch), fully owned and modifiable by Bajaj.
4. **Foundation for the next decade.** 6D pose estimation + synthetic data + force sensing is the base layer for machine tending, bin picking, assembly, and eventually VLA-style (vision-language-action) robots. This project is Bajaj's first working end-to-end instance of that stack.

### Where it can be applied at Bajaj (example applications)
- **Bin picking / part singulation** — picking castings, gears, or fasteners from unsorted bins for machine loading
- **Machine tending** — loading/unloading CNC machines, presses, and injection-molding machines without fixtures
- **Assembly line kitting** — recognizing and gathering the correct set of parts for a vehicle variant
- **Engine/transmission sub-assembly** — placing components that require orientation-aware insertion (the cavity-insertion demo is exactly this, in miniature)
- **Quality inspection** — the same detection + pose models can verify a part's presence, identity, and correct placement
- **Warehouse & intralogistics** — depalletizing and sorting incoming components

The shape-sorter demo was deliberately chosen because "detect an arbitrary part, estimate its orientation, and insert it into a matching cavity" is the *generic template* of most factory handling tasks.

---

## 4. Approach — What I Considered and What I Chose

### Approaches evaluated

| Approach | Verdict | Why |
|---|---|---|
| **Classical vision (ICP on point clouds)** | Studied deeply, kept as fallback/refinement | Works without training, but fragile: needs good initialization, fails with occlusion and symmetric shapes. I implemented and studied ICP + RANSAC to understand the math. |
| **FoundationPose / MegaPose (zero-shot pose models)** | Evaluated | Impressive generalization, but heavy at runtime and less accurate on textureless machined parts; kept on the comparison roadmap. |
| **VLAs (Vision-Language-Action end-to-end robot models)** | Explored, deferred | The long-term future, but not yet reliable enough for millimeter-precision industrial insertion. Noted as the natural evolution of this stack. |
| **Manual data collection + labeling** | Rejected | Weeks of human labeling per part; completely defeats scalability. |
| **✅ Synthetic data (BlenderProc) + YOLOX detection + GDRNPP 6D pose** | **Chosen** | Best accuracy-per-effort for known parts with CAD models; scalable to any new part; GDRNPP is a top performer on the BOP industrial pose benchmark. |

### The chosen pipeline, step by step

1. **CAD modeling** — Measured the real parts with vernier calipers, photographed them, and modeled them in FreeCAD (image-calibrated). Exported meshes to the assets library.
2. **Digital twin scene** — Rebuilt the real workcell (table, lighting, parts) in Blender. Calibrated the *real* camera (checkerboard + ChArUco, ~2% reprojection error) and gave the *virtual* camera the exact same intrinsics and mounting pose, so simulated images match real ones geometrically.
3. **Synthetic dataset generation (BlenderProc)** — Scripted scene randomization: object positions, orientations, lighting, backgrounds (domain randomization, so the model doesn't overfit to the simulation). Outputs RGB + depth + segmentation masks + exact 6D pose labels in the standard **BOP format**, plus COCO annotations for YOLO.
4. **Detection training (YOLOX)** — Trained on the synthetic images to find and classify each shape in the camera view.
5. **6D pose training (GDRNPP)** — Trained to output the full position + rotation of each detected object, precise enough for insertion.
6. **Robot execution (ROS 2)** — Custom FANUC hardware interface + MoveIt 2 config for motion planning; ROS 2 packages for the Tesollo gripper, force sensor, and Mech-Eye camera; hand-eye calibration to convert camera coordinates to robot coordinates. All running on an NVIDIA Jetson.

### Hardware & software stack

| Component | Item |
|---|---|
| Robot | FANUC M-20iD/25 — 6 axes, 25 kg payload, 1831 mm reach |
| Gripper | Tesollo DG-3F (3-finger adaptive) |
| 3D Camera | Mech-Eye stereo camera (ROS 2 interface) |
| Force sensor | FANUC FS-40iA |
| Compute | NVIDIA Jetson (on-cell) + AWS GPU instances (training) |
| Software | ROS 2, MoveIt 2, BlenderProc, PyTorch, Detectron2, YOLOX, GDRNPP, OpenCV, FreeCAD, Docker |

---

## 5. Journey So Far — Week by Week

**Week 1–2 · Bringing the robot cell to life (21 May – 29 May)**
Started from an unconnected pile of hardware. Set up the network infrastructure connecting Jetson ↔ robot ↔ camera ↔ gripper ↔ force sensor. Hit the first real-world lesson: the public FANUC ROS 2 driver **does not support the M-20 series**, and our controller runs an older software version (V9.40) requiring an older driver line. **Solution: rewrote the FANUC hardware interface and built a custom MoveIt 2 configuration from scratch** — after which the robot moved under ROS 2 control. Reading the force sensor also required patching the driver's C++ hardware interface (`hardware_interface.hpp` + conditional CMake linking). Wrote a custom ROS 2 driver package for the Tesollo gripper, produced troubleshooting documentation for the whole bring-up, and authored the vision-pipeline architecture document (v3) that has guided the project since. By end of Week 2: robot controllable, gripper controllable, force sensor streaming, camera capturing — *full workcell integration complete (Phase 1 done)*.

**Week 3 · Choosing the brain (1 Jun – 5 Jun)**
Resolved robot joint-limit faults (MOTN-017). Researched the perception approach seriously: studied YOLO-family detectors, 6D pose estimation methods (ICP, MegaPose, FoundationPose, GDR-Net), and even VLAs. Converged on the synthetic-data + YOLOX + GDRNPP architecture and began the BlenderProc dataset code.

**Week 4 · The mathematics of seeing (8 Jun – 12 Jun)**
Measured and CAD-modeled the physical parts. Performed rigorous camera calibration (checkerboard, then ChArUco board, 30+ poses) — the recovered intrinsics matched the Mech-Eye factory SDK values to within ±1.7, an independent confirmation the calibration is physically correct, not just visually plausible. Went deep on the coordinate-frame mathematics — camera-to-world transforms, OpenCV vs Blender rotation conventions, when to invert a transformation matrix — because the synthetic camera must exactly replicate the real one. Implemented and studied ICP (via Open3D) and RANSAC at the algorithm level. This was the hardest conceptual week and the most valuable one.

**Week 5 · The digital twin produces data (15 Jun – 19 Jun)**
Solved the OpenCV↔Blender convention mismatch and completed the BlenderProc pipeline: correct intrinsics, correct extrinsics, randomized scenes, RGB + depth + segmentation output in BOP and COCO formats. Before committing to generating thousands of images, I wrote a dedicated **6D-pose visualizer** that re-projects the ground-truth labels back onto the rendered images — a verify-first discipline that caught convention errors early. Then hardened the generator for long unattended runs: rendering-engine optimization, per-frame timing, and crash-safe logging that records the run state on any failure. **The pipeline now generates unlimited, verified, perfectly labeled training data for both YOLO and GDRNPP.**

**Week 6 · Training in the cloud & resurrecting a legacy codebase (22 Jun – 29 Jun)**
Set up AWS GPU training. The GDRNPP repository is 2022-era research code, and getting it to run on a modern stack meant fixing an avalanche of version-skew issues: CUDA/glibc header conflicts, compiler incompatibilities, broken vendored Eigen headers, deprecated PyTorch/Detectron2 APIs (fixes committed for Python 3.10, NumPy 2.x, Blender 4.2 compatibility), missing pretrained weights whose official links are dead. I systematically modernized the codebase rather than hacking around it — this became the **patched GDRNPP fork**, complete with committed modernization documentation (vision, architecture, design decisions, roadmap). Registered the custom dataset into the Detectron2/GDRNPP framework, fixed the data-loading path bugs, wrote a mesh/model-info generator to adapt Blender-exported meshes to BOP assumptions, and **completed the first successful YOLOX training run**. Built a standalone inference script, `infer.py` (the repo had no clean inference API), and **verified the detector on real camera images: it correctly recognizes the heart, triangle, and semicircle** — a synthetic-to-real transfer success. Identified remaining issue: false positives on the calibration board (fix: add it as an explicit class in the synthetic data). Closed the week by **collecting hand-eye calibration data on the physical robot** — calibration images plus recorded robot poses are committed, ready for solving the camera-to-robot transform.

---

## 6. Current Status

### ✅ Done and working
- Full workcell integration: robot, gripper, force sensor, 3D camera, all controlled from ROS 2 on the Jetson
- Custom FANUC hardware interface + MoveIt 2 configuration (the public driver didn't support our robot — I wrote our own), including a patched C++ interface to stream force-sensor data and a custom ROS 2 gripper driver
- Camera calibrated with ChArUco (30+ poses); recovered intrinsics match the Mech-Eye factory SDK values to within ±1.7 — hand-eye transformation mathematics worked out
- CAD models of all workpieces created from physical measurements
- BlenderProc synthetic data pipeline generating verified, correctly-labeled datasets (BOP + COCO format), with a 6D-pose visualizer for label verification and crash-safe logging for long unattended runs
- GDRNPP/YOLOX codebase modernized to run on current Python/PyTorch/CUDA (patched fork published, with documented modernization roadmap)
- Standalone inference script (`infer.py`) for running the trained models outside the training framework
- **First YOLOX detector trained purely on synthetic data and validated on real camera images — it recognizes all three shapes in the real world**
- Hand-eye calibration data collected on the physical robot (calibration images + recorded robot poses committed)

### 🔄 In progress
- GDRNPP 6D pose model training on the synthetic dataset
- Eliminating detector false positives (adding calibration board as a 4th class)
- Solving the hand-eye calibration from the collected robot poses + images

### ⏭ Plan for Weeks 7–8 — a de-risked, two-track plan

The risky research problems are solved; what remains is integration. To guarantee a working demonstration by the end of the internship, the remaining work runs on two parallel tracks:

**Track A — Guaranteed demo path (primary).** For flat parts on a flat table, a top-down grasp only needs *(x, y, rotation)* — and every ingredient for that already works: YOLOX detection + segmentation mask + depth from the Mech-Eye camera + known table height. This track assembles the complete pick-and-place loop from proven components:
- Solve hand-eye calibration from the already-collected robot poses + images (standard OpenCV routine)
- Pose from detection mask + depth; predefined grasp pose per shape (3 known shapes — no learned grasping needed)
- Cavity board localized via fiducial marker; place into cavity region
- ROS 2 loop: detect → pose → transform → MoveIt plan → pick → place

**Track B — The full 6D version (parallel, in the cloud).** GDRNPP pose training continues on AWS in parallel. If it validates on real images in time, it swaps in as the pose source — the ROS interface stays identical either way. If not, it is documented as the immediate next step, and the demo still works via Track A.

**Milestones:**
- **End of Week 7: first vision-guided pick of a real part** (Track A) — the milestone that determines everything downstream
- Week 8: cavity placement, reliability passes, swap in GDRNPP if validated, then code freeze ~3 days before the end for demo video, documentation, and final presentation

**Deliberate scope decisions:** learned grasp planners (XGrasp / AnyDexGrasp) were evaluated and deliberately deferred — hardcoded grasps suffice for three known shapes; force-guided tight insertion is documented as future work, with the force sensor already integrated and streaming, ready for it.

---

## 7. Challenges Overcome (the honest version)

These are worth telling in the review — they show real engineering, not tutorial-following:

1. **The driver that didn't exist.** The official FANUC ROS 2 driver doesn't support the M-20 series, and our controller firmware needed an older protocol version. Rather than being blocked, I rewrote the hardware interface and built a custom MoveIt configuration. *Lesson: in industrial robotics, you often have to build the bridge yourself.*
2. **Coordinate-frame hell.** OpenCV, Blender, ROS, and the robot each use different mathematical conventions for describing position and rotation. A single wrong inversion silently corrupts every dataset image. I derived and verified the full transform chain by re-projecting ground truth back onto images until it matched pixel-perfectly.
3. **"It looks right" isn't proof.** I discovered that a camera calibration can *look* visually correct while being wrong — changing physically incorrect parameters still produced plausible-looking axes. This taught me to build quantitative verification: I cross-checked my recovered intrinsics against the camera manufacturer's factory SDK values (they matched to within ±1.7) and built a 6D-pose visualizer that re-projects ground-truth labels onto rendered images before generating any large dataset.
4. **Resurrecting 2022 research code in 2026.** GDRNPP's toolchain collided with a modern OS at every level — glibc, GCC, CUDA, setuptools, PyTorch APIs, dead download links. I fixed it layer by layer and published the modernized fork instead of keeping hacks local.
5. **Low-resource compilation.** The Jetson would crash building MoveIt; solved with sequential, RAM-limited builds. Small thing, real-world thing.

---

## 8. What I Have Learned (and Will Learn)

### Learned so far — technical
- **Industrial robotics integration:** ROS 2, MoveIt 2, hardware interfaces, industrial networking, real robot safety and fault handling (joint limits, TP operation)
- **3D vision mathematics:** camera models, intrinsics/extrinsics, calibration (checkerboard/ChArUco), rigid-body transforms, rotation representations, PnP, ICP, RANSAC
- **Deep learning for perception:** YOLOX architecture and training, GDRNPP/GDR-Net 6D pose estimation, transfer learning, the BOP benchmark ecosystem, Detectron2 internals (LazyConfig, dataset catalogs, trainer/evaluator architecture)
- **Synthetic data & sim-to-real:** BlenderProc, domain randomization, digital-twin scene construction, why and how simulation-trained models transfer to reality
- **ML infrastructure:** Docker, AWS GPU training, CUDA toolchain debugging, dependency management for legacy research code
- **CAD & metrology:** FreeCAD modeling from vernier measurements and calibrated photos

### Learned so far — professional
- Breaking an ambiguous 8-week goal into milestones and tracking daily progress
- Reading research papers (GDR-Net, T-PAMI'25) and translating them into working systems
- Debugging systematically through five-layer-deep software stacks instead of guessing
- Knowing when to fix legacy code vs. rewrite it — and defending that decision
- Documenting for the next engineer, not just for myself

### Will learn in the remaining weeks
- 6D pose model training, evaluation, and accuracy tuning on real hardware
- Hand-eye calibration in practice, including drift correction with fiducial markers
- Grasp planning for multi-finger grippers
- Full-system integration: making perception, planning, and control work *together* reliably — the hardest and most valuable skill in robotics
- How a research prototype becomes something a factory could actually depend on

---

## 9. Where This Takes Bajaj — the Long View

By the end of this internship, Bajaj will have:
1. A **working demonstrator**: an autonomous, vision-guided FANUC cell that picks unfixtured parts and places them with orientation-aware precision.
2. A **reusable synthetic-data pipeline**: point it at any CAD model in Bajaj's catalog, and it produces training data for that part overnight. This is the scalable asset — the demo is just its first customer.
3. **Three documented, open repositories** covering robot integration, data generation, and a modernized state-of-the-art pose estimator — institutional knowledge, not intern-leaves-and-it's-gone knowledge.
4. A **validated technology roadmap**: which pose estimation approaches work for machined parts, what accuracy is achievable from pure simulation training, and where the pitfalls are (calibration drift, domain gap, legacy tooling) — de-risking any future investment in flexible automation.

The strategic point for the review: **the shape-sorter is a stand-in for every part Bajaj makes.** The same pipeline that slots a heart into a cavity can load a gear into a fixture, pick a casting from a bin, or verify an assembly — because the underlying capability is "see any known part, know exactly where it is, handle it."

---

## Appendix A — Suggested Presentation Flow (slide-by-slide)

1. **Title** — Project name, photo of the actual robot cell
2. **The problem** — "Industrial robots are blind" (30-second story from §1)
3. **The idea** — Robot + eyes + brain; the shape-sorter demo (photo/video)
4. **The clever part** — Synthetic data: CAD → simulation → trained AI, no manual labeling (side-by-side: rendered image vs real camera image)
5. **Why Bajaj should care** — 4 points from §3 + applications list
6. **How it works** — The one-line pipeline diagram from §2 (keep it to 6 boxes)
7. **Journey** — Week-by-week timeline graphic from §5
8. **Live results** — Real-image detections from the synthetic-trained model (your strongest slide: "trained entirely in simulation, works on the real camera")
9. **Challenges** — Pick 2 stories from §7 (the driver rewrite + sim-to-real math)
10. **Status & plan** — Done / in-progress from §6, then the two-track Weeks 7–8 plan ("guaranteed demo path + full 6D version in parallel" — shows risk management, reviewers love this)
11. **What I'm learning** — §8, condensed to one slide
12. **The long view** — §9: "this pipeline scales to any part with a CAD model"
13. **Thank you / demo invitation**

## Appendix B — Quick Facts (for Q&A)

- Robot: FANUC M-20iD/25 · 6 axes · 25 kg payload · 1831 mm reach
- Camera calibration: ChArUco, 30+ poses; recovered intrinsics within ±1.7 of Mech-Eye factory SDK values
- Dataset format: BOP (industry-standard 6D pose benchmark format) + COCO
- Detector: YOLOX, trained 100% on synthetic images, validated on real camera images
- Pose model: GDRNPP — top performer of the BOP Challenge 2022, modernized by me for the current software stack
- Training infra: AWS GPU instances, Dockerized
- Runtime compute: NVIDIA Jetson mounted at the cell
- All code in 3 public GitHub repos (linked at top)

### Commit-history evidence (as of 29 Jun)
- `synthetic-data-yolo-training_and_pose_estimation`: ~130 commits, 4 Jun – 29 Jun — CAD assets → calibration → ICP experiments → BlenderProc pipeline → dataset tooling → hand-eye calibration data
- `fanuc_pickn_place`: 23 commits, 21 May – 5 Jun — driver build, force-sensor C++ patch, gripper ROS 2 driver, troubleshooting docs, vision-pipeline architecture doc
- `gdrnpp_bop2022` fork: ~30 commits on top of the unmaintained 2022 upstream, 22 – 29 Jun — build fixes, Python 3.10 / NumPy 2.x compatibility, custom dataset registration, modernization docs, standalone `infer.py`
