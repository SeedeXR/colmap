# COLMAP on Apple Silicon (macOS) — build & tuning guide

This guide covers building and running COLMAP efficiently on Apple Silicon
Macs (M-series), with an emphasis on **16 GB unified-memory** machines. All
numbers below are measured on an Apple **M4 / 16 GiB / 10 cores (4 P + 6 E)**.

## Building

Dependencies are available via Homebrew:

```bash
brew install cmake ninja libomp qt eigen ceres-solver glog gflags \
             boost faiss openimageio onnxruntime
```

Configure + build (CUDA is unavailable on macOS; the GUI is optional):

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ENABLED=OFF \
  -DTESTS_ENABLED=ON          # turn ON during development
# Optional, experimental Apple GPU acceleration (see "Metal" below):
#  -DMETAL_ENABLED=ON
cmake --build build -j
```

OpenMP is required; the Homebrew `libomp` is detected automatically.

## Diagnose your machine

```bash
colmap system_info            # RAM, P/E cores, recommended threads, build info
colmap system_info --format json
```

## Resource-aware operation (16 GB machines)

CPU SIFT feature extraction is the most memory-hungry stage. COLMAP on Apple
Silicon now defaults to the **performance-core count** for automatic threading
(not every logical core), which is both faster and lighter — measured: the old
all-cores default thrashed (157.9 s, ~8.4 GB peak, swapping) on 16 × 7 MP
images; the P-core default runs the same in **44.5 s (~3.5× faster)**.

Useful flags:

| Flag | Effect |
|---|---|
| `--FeatureExtraction.num_threads -1` | automatic (P-cores on Apple Silicon) |
| `--FeatureExtraction.max_memory_gb 6` | cap CPU-SIFT threads to fit a RAM budget |
| `--SiftExtraction.first_octave 0` | **big win for ≥2 MP images** (see below) |
| `--FeatureExtraction.max_image_size N` | down-scale large images |

### The single biggest lever: `--SiftExtraction.first_octave 0`

The default `first_octave = -1` upsamples each image 2× before building the
SIFT scale space (4× the pixels). For images larger than ~2 MP this is wasted.
Measured (south-building, 7 MP), full extract→match→map:

| `first_octave` | extract time | extract RAM | registered | reproj error |
|---|---|---|---|---|
| -1 (default) | 38.6 s | 8.50 GB | 15 | 0.49 px |
| **0** | **7.0 s** | **2.22 GB** | 15 | 0.52 px |

≈ **3.8× faster and 3.8× less RAM at equivalent reconstruction quality**.
COLMAP prints a hint recommending this for large images. (Keep the default for
small / low-resolution images, which genuinely benefit from the upsample.)

### Recommended low-footprint recipe

```bash
colmap feature_extractor --image_path IMAGES --database_path DB \
    --SiftExtraction.first_octave 0 --FeatureExtraction.max_memory_gb 6
```

Measured end-to-end (extract → Metal match → map) this stays at **~2.2 GB peak**
across 4–32 images, with all images registered (see
`scripts/macos/pipeline_regression.sh`).

## Feature matching on the GPU (Metal)

macOS has no CUDA and the legacy GLSL `SiftMatchGPU` is deprecated, so matching
historically ran on the CPU. With `METAL_ENABLED`, `--FeatureMatching.use_gpu 1`
runs brute-force descriptor matching on the Apple GPU (MPS) — the macOS
equivalent of CUDA's `SiftMatchGPU`. Measured: **~60× faster** brute-force
matching (730 s → 10.8 s on 24 images) with **identical** match results.

```bash
colmap exhaustive_matcher --database_path DB --FeatureMatching.use_gpu 1
```

Set `COLMAP_DISABLE_METAL=1` to force the CPU path (debugging/benchmarking).

## Choosing a mapper

```bash
colmap mapper_advisor --database_path DB          # recommendation + rationale
colmap mapper_advisor --database_path DB --format json
```

Or let `automatic_reconstructor` decide:

```bash
colmap automatic_reconstructor --workspace_path WS --image_path IMAGES --mapper auto
```

Rule of thumb: incremental for small/medium or sequential sets; global (GLOMAP)
for large, well-connected, unordered collections; hierarchical for very large
scenes.

## Vocabulary-tree matching

Vocab-tree matching only prunes pairs when `--VocabTreeMatching.num_images` is
**much smaller** than the dataset size. On a few hundred images the default
(100) retrieves nearly everything and is no faster than `exhaustive_matcher`.
Use a small `num_images` for thousands of images; otherwise prefer exhaustive.

## Machine-readable progress (for orchestrators / scripting)

Every command accepts `--progress_format {none,plain,jsonl}` (default `none` =
unchanged behavior). `jsonl` streams structured events on **stdout**
(`stage_started`/`progress`/`stage_completed`/`metric`/`warning`) while glog
logging stays on stderr:

```bash
colmap mapper ... --progress_format jsonl
```

## Dense reconstruction (MVS) status

Dense MVS (`patch_match_stereo`) is **CUDA-only** and does not run on macOS in
this build. A Metal port is in progress — the photometric core (NCC cost,
plane-induced homography), plane-sweep depth, and the PatchMatch optimizer are
validated on Metal, but the full pipeline integration is not complete. Track
status in `memory/porting.md` §9. For now, run dense MVS on a CUDA machine.

## Verifying your build

```bash
ctest --test-dir build                       # unit tests
scripts/macos/pipeline_regression.sh         # end-to-end + RAM-budget gate
```
