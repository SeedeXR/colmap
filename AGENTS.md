# AGENTS.md — COLMAP Guide

## Project Overview

COLMAP is a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline that reconstructs 3D models from 2D image collections. Written in C++17 with optional CUDA support. Single binary (colmap) with many subcommands, a Qt GUI, and Python bindings (pycolmap).

## macOS / Apple Silicon additions (this fork)

This fork adds Metal (Apple GPU) acceleration and a production CLI/process contract on top of upstream COLMAP. All Apple-specific code is behind `__APPLE__` / `COLMAP_METAL_ENABLED` (CMake `-DMETAL_ENABLED=ON`, Apple-only, default OFF) and is a no-op elsewhere — Linux/Windows builds are unaffected (verified by a `METAL_ENABLED=OFF` build). Full history + rationale live in `memory/` (`todo.md`, `future_enhancements.md`, `testing_mechanism.md`). RAM discipline for this fork: 3–4 GB ideal, 6 GB hard cap.

**Implemented / shipping:**
- **Metal PatchMatch MVS** — `colmap patch_match_stereo` runs on macOS via Metal (no CUDA needed): `mvs/patch_match_metal.{h,mm,_stub.cc}` + `patch_match_metal_controller.cc`. Depth equivalent to the CUDA golden (median ~0.14%), peak RSS ~0.8 GB on south-building.
- **Resource-aware feature extraction** — CPU SIFT threads auto-sized to a RAM budget (`--FeatureExtraction.max_memory_gb`); large-image `first_octave=0` hint. CPU SIFT is the only macOS extractor (SiftGPU needs OpenGL/CUDA).
- **Process contract** — structured `--progress_format jsonl` events (`util/progress`), cooperative SIGINT/SIGTERM stop with **partial reconstruction save** + exit codes 130/143 (`util/signal_handler`, `BaseController::CheckIfStopped` defaults to the interrupt flag, `exe/sfm.cc`), Ceres + vocab-tree **heartbeats** (`HeartbeatThrottle` in `util/progress`), grouped `--help`.
- **Perf harnesses** — `scripts/macos/pipeline_regression.sh` (absolute RAM/quality budget) + `scripts/macos/perf_baseline.sh` (relative drift vs a stored baseline); shared helpers in `scripts/macos/_metrics_lib.sh`.

**Deliberate DECISIONS (do not re-litigate without new data):**
- **Feature matching stays on CPU.** A Metal MPS matcher was implemented, benchmarked, then **REMOVED** — on Apple Silicon the CPU matcher is faster (per-pair GPU dispatch overhead) with identical output. `--FeatureMatching.use_gpu` defaults to 0; do NOT re-add a Metal matcher.
- **SIFT-on-Metal skipped** (CPU already fast/in-budget, not the bottleneck); **GPU vocab-tree skipped** (FAISS has no Metal backend); **CoreML EP stays OFF** (~50× slower / OOM when benchmarked).
- **Caspar-style Metal bundle adjustment = EXPERIMENTAL, branch `caspar-style` only, NOT for production.** Output equivalent to Ceres but no end-to-end speed win demonstrated; Ceres remains the BA. `estimators/bundle_adjustment_metal.*` is research-only and not wired into the BA pipeline.

## Directory Structure

| Path | Description |
|------|-------------|
| CMakeLists.txt | Root build config |
| cmake/CMakeHelper.cmake | COLMAP_ADD_LIBRARY / _EXECUTABLE / _TEST macros |
| cmake/FindDependencies.cmake | All dependency discovery (Eigen, Ceres, CUDA, Qt, etc.) |
| cmake/Find*.cmake | Custom find modules |
| src/colmap/ | Primary C++ source (see Architecture below) |
| src/pycolmap/ | pybind11 C++ bindings |
| src/thirdparty/ | Bundled (VLFeat, SiftGPU, PoissonRecon, LSD) and fetched (PoseLib, faiss, ONNX Runtime) |
| python/pycolmap/ | Python package (__init__.py, utilities) |
| python/CMakeLists.txt | scikit-build-core build for pycolmap |
| doc/ | Sphinx/RST documentation |
| docker/ | Dockerfile + build/run scripts |
| scripts/format/ | c++.sh (clang-format), python.sh (ruff) |
| benchmark/ | Reconstruction + runtime benchmarks |
| .github/workflows/ | CI: Ubuntu, macOS, Windows, Docker, pycolmap |
| vcpkg.json | vcpkg manifest (Windows/macOS deps) |
| pyproject.toml | Python build config (scikit-build-core, cibuildwheel) |
| .clang-format | C++ formatting style |
| ruff.toml | Python linting/formatting config |

## Module Dependency Layers (bottom → top)

| Module | Description |
|--------|-------------|
| util/ | Threading, logging, caching, PLY I/O, CUDA/OpenGL helpers |
| math/ | Random, polynomials, graph algorithms (cuts, union-find, spanning trees) |
| geometry/ | Rigid3d, Sim3d, essential/homography matrices, triangulation, GPS |
| sensor/ | Camera distortion models, Bitmap (image I/O), Rig, sensor specs DB |
| feature/ | SIFT (CPU/GPU), ALIKED (ONNX), LightGlue, descriptor indexing (FAISS) |
| optim/ | RANSAC, LO-RANSAC, SPRT, samplers, support measurers |
| scene/ | Camera, Image, Frame, Point2D/3D, Track, Reconstruction, Database (SQLite), CorrespondenceGraph |
| estimators/ | Bundle adjustment (Ceres), absolute/relative pose, two-view geometry, triangulation, alignment |
| estimators/solvers/ | Minimal solvers: P3P, 5-pt essential, 7/8-pt fundamental, homography (via PoseLib) |
| estimators/cost_functions/ | Ceres cost functors: reprojection, Sampson, alignment, pose prior |
| sfm/ | IncrementalMapper, GlobalMapper, IncrementalTriangulator, ObservationManager |
| mvs/ | PatchMatch stereo (CUDA), depth/normal maps, fusion, meshing |
| image/ | Image undistortion, warping, line detection |
| retrieval/ | Vocabulary tree (VisualIndex), inverted index, vote-and-verify |
| controllers/ | AutomaticReconstruction, IncrementalPipeline, GlobalPipeline, HierarchicalPipeline, OptionManager |
| exe/ | CLI command implementations (colmap.cc dispatcher + per-domain .cc files) |
| ui/ | Qt GUI: MainWindow, ModelViewerWidget, OpenGL painters, config dialogs |

## Key Classes & Files

| Class/File | Location | Purpose |
|------------|----------|---------|
| Reconstruction | scene/reconstruction.h | Top-level container: cameras, rigs, images, frames, points, tracks |
| Camera | scene/camera.h | Intrinsics (focal, principal pt, distortion model) |
| Rig | sensor/rig.h | Multi-sensor rig with sensor_from_rig transforms |
| Image | scene/image.h | Exposure: name, Point2D observations, camera_id, frame_id |
| Frame | scene/frame.h | Posed rig instantiation: rig_from_world + sensor data |
| Point3D | scene/point3d.h | Triangulated 3D point: xyz, color, error, Track |
| Track | scene/track.h | List of (image_id, point2D_idx) observations |
| Database | scene/database.h | Abstract DB interface (SQLite impl in database_sqlite.h) |
| DatabaseCache | scene/database_cache.h | In-memory cache + CorrespondenceGraph |
| CorrespondenceGraph | scene/correspondence_graph.h | Feature-to-feature correspondences across images |
| Rigid3d | geometry/rigid3.h | 6-DOF rigid transform (quaternion + translation) |
| Sim3d | geometry/sim3.h | 7-DOF similarity transform (Rigid3d + scale) |
| FeatureExtractor | feature/extractor.h | Abstract extractor (SIFT, ALIKED); factory Create() |
| FeatureMatcher | feature/matcher.h | Abstract matcher; supports Match() and MatchGuided() |
| FeatureKeypoint | feature/types.h | x, y + affine shape (a11, a12, a21, a22) |
| BundleAdjuster | estimators/bundle_adjustment.h | Abstract BA; Ceres impl via CreateDefaultBundleAdjuster() |
| BundleAdjustmentConfig | estimators/bundle_adjustment.h | What to optimize vs. hold constant |
| EstimateAbsolutePose() | estimators/pose.h | P3P RANSAC from 2D-3D correspondences |
| EstimateTwoViewGeometry() | estimators/two_view_geometry.h | Essential/fundamental/homography estimation |
| EstimateTriangulation() | estimators/triangulation.h | Robust multi-view triangulation |
| IncrementalMapper | sfm/incremental_mapper.h | Core incremental SfM engine |
| GlobalMapper | sfm/global_mapper.h | Global SfM (rotation averaging + global positioning) |
| IncrementalTriangulator | sfm/incremental_triangulator.h | Point creation, track merging/completion |
| ObservationManager | sfm/observation_manager.h | Per-image visibility stats, filtering |
| PatchMatch | mvs/patch_match.h | CPU wrapper for CUDA PatchMatch stereo |
| PatchMatchController | mvs/patch_match.h | Orchestrates multi-GPU depth estimation |
| StereoFusion | mvs/fusion.h | Fuses depth maps into 3D point cloud |
| AutomaticReconstructionController | controllers/automatic_reconstruction.h | End-to-end pipeline (extract, match, SfM, MVS) |
| IncrementalPipeline | controllers/incremental_pipeline.h | Manages incremental SfM loop + multi-model |
| OptionManager | controllers/option_manager.h | Centralized CLI option parsing |
| Camera models | sensor/models.h | SimplePinhole, Radial, OpenCV, Fisheye, etc. |
| Bitmap | sensor/bitmap.h | Image I/O wrapper (OpenImageIO), EXIF extraction |

CLI entry point: exe/colmap.cc (subcommand dispatcher).

## Build Instructions

```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
ninja

# No GUI, no CUDA (minimal)
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DGUI_ENABLED=OFF -DCUDA_ENABLED=OFF

# With tests
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DTESTS_ENABLED=ON
```

### Build pycolmap

First build and install the C++ code:

```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=../install
ninja install
```

Then build pycolmap (from the repo root):

```bash
colmap_DIR=./install ./python/incremental_build.sh  # Fast incremental build
colmap_DIR=./install ./python/build.sh              # Clean build (slower)
```

If there is a local `.python-version` file, use pyenv/uv for Python commands, `pip install`, and building pycolmap.

## Testing

Follow C++ and Python build instructions above.

Run ctest from the build directory:

```bash
cd build
ctest --output-on-failure          # All C++ tests
ctest -R "util/cache_test"         # Specific test
ctest -E "(feature/sift_test)"     # Exclude GPU tests
```

Run Python tests from the repo root:

```bash
pytest                             # All Python tests (config in pyproject.toml)
```

- Test files (`*_test.cc`) across all modules, created via `COLMAP_ADD_TEST()` macro
- Framework: GTest/GMock with custom main (`util/gtest_main.cc`)
- Test utilities: `util/testing.h`, Eigen matchers: `util/eigen_matchers.h`, transform matchers: `geometry/rigid3_matchers.h`, `geometry/sim3_matchers.h`
- CTest names: `module/test_name` (e.g., `estimators/alignment_test`)

## Code Style & Conventions

### Naming

- Classes: `PascalCase`
- Methods/functions: `PascalCase` (e.g., `FindNextImages()`)
- Member variables: `snake_case_` (trailing underscore)
- Local variables: `snake_case`
- Constants/enums: `kPascalCase` or `UPPER_SNAKE_CASE`
- Files: `snake_case.h` / `snake_case.cc` / `snake_case_test.cc`
- Transforms: `target_from_source` (e.g., `cam_from_world`)
- Coordinates: `x_in_y` (e.g., `point3D_in_world`)

### Index and Identifier Types (util/types.h)

- Generic indexes: int, size_t
- Special identifiers: camera_t, image_t, image_pair_t, frame_t, rig_t, point2D_t, point3D_t, sensor_t, data_t, pose_prior_t

### Formatting

```bash
scripts/format/c++.sh
scripts/format/python.sh
```

## External Dependencies

### Core

| Library | Role |
|---------|------|
| Eigen3 | Linear algebra, matrices, geometry |
| Ceres Solver | Nonlinear optimization |
| Boost | Graph algorithms, CLI options, etc. |
| glog | Structured logging |
| SQLite3 | Feature/match database |
| OpenImageIO | Image I/O and processing |
| CHOLMOD | Sparse Cholesky |
| Metis | Graph partitioning |
| PoseLib | Minimal pose solvers |
| FAISS | Fast ANN for descriptor matching |

### Optional

| Library | Role | Gate |
|---------|------|------|
| CUDA | GPU PatchMatch, SiftGPU, Ceres GPU BA | CUDA_ENABLED |
| ONNX Runtime | ALIKED, LightGlue neural features | ONNX_ENABLED |
| Qt5/6 | GUI | GUI_ENABLED |
| OpenGL/GLEW | 3D visualization, SiftGPU | OPENGL_ENABLED |
| CGAL | Delaunay meshing | CGAL_ENABLED |

### Bundled (src/thirdparty/)

| Library | Role |
|---------|------|
| VLFeat | CPU SIFT |
| SiftGPU | GPU SIFT |
| PoissonRecon | Surface reconstruction |
| LSD | Line detection |
