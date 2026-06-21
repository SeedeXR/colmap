// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <filesystem>
#include <string>

namespace colmap {
namespace mvs {

struct PatchMatchOptions;  // mvs/patch_match_options.h

// Metal (Apple GPU) compute primitives for dense PatchMatch multi-view stereo.
//
// Dense MVS (`patch_match_stereo`) is CUDA-only upstream and therefore does not
// run on macOS at all. This module is the foundation of a Metal backend: it
// ports COLMAP's PatchMatch GPU kernels (`mvs/patch_match_cuda.cu`) to Metal
// Shading Language one validated piece at a time. Each kernel is checked
// against an independent reference (CPU math here; the CUDA golden depth/normal
// maps in golden_task/colmap_golden_bundle/ once the full pipeline is wired) so
// the port is provably the SAME algorithm, not an approximation.
//
// The interface is intentionally free of Metal and Eigen types so it can be
// included from any translation unit (plain C++ headers, no Objective-C).

// Whether a usable Metal GPU is available at runtime. Always false in builds
// without COLMAP_METAL_ENABLED, or when no Metal device is present (e.g. the
// COLMAP_DISABLE_METAL=1 escape hatch).
bool IsPatchMatchMetalAvailable();

// Computes, on the Metal GPU, the plane-induced homographies that warp the
// reference image into ONE source image for a batch of per-pixel plane
// hypotheses. This is the geometric core of PatchMatch's photometric cost:
//
//     H = K_src * (R - T * n^T / d) * K_ref^-1
//
// ported verbatim from the device function `ComposeHomography` in
// `patch_match_cuda.cu` (same scalar expansion, same float arithmetic).
//
// Calibrations are packed exactly as `patch_match_cuda.cu` packs them:
//   ref_inv_K = {1/fx, -cx/fx, 1/fy, -cy/fy}   reference inverse intrinsics
//   src_K     = {fx, cx, fy, cy}               source intrinsics
//   src_R     = row-major 3x3 relative rotation (reference -> source)
//   src_T     = relative translation (reference -> source), length 3
// Per hypothesis i in [0, num): reference pixel (rows[i], cols[i]), depth
// depths[i], unit surface normal normals[3i .. 3i+2] in the reference frame.
//
// Output: out_H must hold num * 9 floats; entry i is the row-major 3x3
// homography for hypothesis i. Returns false (writing nothing) if Metal is
// unavailable or num <= 0; callers then fall back to the CPU/CUDA path.
bool ComposePlaneHomographiesMetal(const float ref_inv_K[4],
                                   const float src_K[4],
                                   const float src_R[9],
                                   const float src_T[3],
                                   const int* rows,
                                   const int* cols,
                                   const float* depths,
                                   const float* normals,
                                   int num,
                                   float* out_H);

// Computes, on the Metal GPU, the bilateral-weighted NCC photometric-consistency
// cost for a batch of (reference pixel, plane homography) pairs -- the inner
// loop of PatchMatch (`PhotoConsistencyCostComputer::Compute` in
// patch_match_cuda.cu). For each hypothesis it walks a (2*window_radius+1)^2
// window (stride window_step) around the reference pixel, warps each window
// pixel into the source image through the per-hypothesis homography H, samples
// the source bilinearly, weights every sample by the same bilateral weight the
// CUDA path uses --
//   w = exp(-(dr^2+dc^2)/(2*sigma_spatial^2) - (c_center-c_ref)^2/(2*sigma_color^2))
// -- and returns cost = 1 - NCC in [0, 2] (kMaxCost=2.0 when either patch has
// variance < 1e-5, matching CUDA exactly).
//
//   ref_image/src_image: row-major width*height grayscale floats (the CUDA path
//       uses intensities normalized to [0, 1]; any consistent scale works).
//   homographies: num * 9 floats, row-major H per hypothesis (e.g. the output
//       of ComposePlaneHomographiesMetal). Window pixel (col+dc, row+dr) maps to
//       source (x/z, y/z) with [x y z]^T = H * [col+dc, row+dr, 1]^T.
//   out_cost: caller-allocated num floats.
// Pixels sampled outside either image contribute 0 (border). Returns false
// (writing nothing) if Metal is unavailable or the inputs are invalid.
bool ComputeNCCPhotometricCostMetal(const float* ref_image,
                                    const float* src_image,
                                    int width,
                                    int height,
                                    int window_radius,
                                    int window_step,
                                    float sigma_spatial,
                                    float sigma_color,
                                    const int* rows,
                                    const int* cols,
                                    const float* homographies,
                                    int num,
                                    float* out_cost);

// Baseline plane-sweep depth estimator on Metal: for each reference pixel,
// evaluates the bilateral-NCC photometric cost against ONE source image over a
// set of candidate depths along a fixed plane normal, and returns the
// lowest-cost depth (and its cost) per pixel. It composes the two validated
// primitives above -- ComposePlaneHomographiesMetal (one homography per pixel
// per candidate depth) then ComputeNCCPhotometricCostMetal -- and keeps the
// per-pixel arg-min on the host.
//
// This is the simplest depth estimator and serves as the bring-up/validation
// baseline for the (stochastic) PatchMatch propagation+refinement optimizer
// that replaces it; it is EXHAUSTIVE (cost grows with num_depths), not the
// final algorithm, and uses a single fixed normal rather than per-pixel normals.
//
//   normal: assumed plane normal in the reference frame (e.g. {0, 0, -1}
//           fronto-parallel).
//   depth_candidates: num_depths candidate depths to sweep.
//   rows/cols: the num_pixels reference pixels to estimate.
//   out_depth/out_cost: caller-allocated num_pixels floats (best depth + cost).
// Returns false if Metal is unavailable or the inputs are invalid.
bool PlaneSweepDepthMetal(const float* ref_image,
                          const float* src_image,
                          int width,
                          int height,
                          const float ref_inv_K[4],
                          const float src_K[4],
                          const float src_R[9],
                          const float src_T[3],
                          const float normal[3],
                          const float* depth_candidates,
                          int num_depths,
                          int window_radius,
                          int window_step,
                          float sigma_spatial,
                          float sigma_color,
                          const int* rows,
                          const int* cols,
                          int num_pixels,
                          float* out_depth,
                          float* out_cost);

// Fused per-pixel hypothesis-cost evaluation: composes the plane homography from
// each pixel's (depth, normal) hypothesis and returns its bilateral-NCC cost in
// a single Metal kernel (no host round-trip). This is the inner-loop primitive
// of the PatchMatch optimizer below. Same camera packing as the functions above;
// rows/cols/depths/normals are num parallel arrays; out_cost holds num floats.
// Returns false if Metal is unavailable or inputs are invalid.
bool EvaluateHypothesisCostMetal(const float* ref_image,
                                 const float* src_image,
                                 int width,
                                 int height,
                                 const float ref_inv_K[4],
                                 const float src_K[4],
                                 const float src_R[9],
                                 const float src_T[3],
                                 int window_radius,
                                 int window_step,
                                 float sigma_spatial,
                                 float sigma_color,
                                 const int* rows,
                                 const int* cols,
                                 const float* depths,
                                 const float* normals,
                                 int num,
                                 float* out_cost);

// Multi-view fused hypothesis cost: like EvaluateHypothesisCostMetal but scores
// each pixel's (depth, normal) hypothesis against `num_src` source views and
// aggregates as the MEAN of the best (lowest) `best_k` per-view costs -- a
// robust multi-view score that tolerates a minority of occluded/mismatched
// views. (Deterministic robust aggregation, not COLMAP's probabilistic
// monte-carlo view selection, which is a later refinement.)
//   src_images: num_src planes of width*height row-major floats, concatenated.
//   src_K/src_R/src_T: per-source intrinsics/rotation/translation, num_src * {4,9,3}.
//   best_k: number of lowest per-view costs to average (clamped to [1, num_src]).
// Returns false if Metal is unavailable or inputs are invalid.
bool EvaluateHypothesisCostMultiViewMetal(const float* ref_image,
                                          const float* src_images,
                                          int num_src,
                                          int width,
                                          int height,
                                          const float ref_inv_K[4],
                                          const float* src_K,
                                          const float* src_R,
                                          const float* src_T,
                                          int best_k,
                                          int window_radius,
                                          int window_step,
                                          float sigma_spatial,
                                          float sigma_color,
                                          const int* rows,
                                          const int* cols,
                                          const float* depths,
                                          const float* normals,
                                          int num,
                                          float* out_cost);

// Geometric-consistency cost: forward-backward reprojection error of each
// pixel's depth hypothesis against the source views' depth maps, averaged over
// sources and truncated at max_cost. Verbatim port of ComputeGeomConsistencyCost
// (patch_match_cuda.cu) -- the inter-view consistency term of PatchMatch's
// geometric refinement pass.
//   ref_inv_K = {1/fx, -cx/fx, 1/fy, -cy/fy}; ref_K = {fx, cx, fy, cy}.
//   src_P:     num_src * 12 row-major 3x4 source projection matrices
//              P = K_src * [R | T] mapping a reference-frame point to a source pixel.
//   src_inv_P: num_src * 12 row-major 3x4 inverse projections
//              [R^T * K_src^-1 | -R^T T] mapping a source pixel*depth back to the
//              reference frame.
//   src_depth_maps: num_src planes of width*height row-major depths (0 = invalid).
// Returns false if Metal is unavailable or inputs are invalid.
bool ComputeGeomConsistencyCostMetal(int width,
                                     int height,
                                     const float ref_inv_K[4],
                                     const float ref_K[4],
                                     const float* src_P,
                                     const float* src_inv_P,
                                     const float* src_depth_maps,
                                     int num_src,
                                     float max_cost,
                                     const int* rows,
                                     const int* cols,
                                     const float* depths,
                                     int num,
                                     float* out_cost);

// Fused geometric-consistency count: like ComputeGeomConsistencyCostMetal but
// evaluates ALL source views in a single dispatch and writes, per pixel, the
// number of sources whose forward-backward reprojection error is <= max_cost.
// This is the count-based consistency filter COLMAP applies (a pixel survives if
// it is consistent in >= filter_min_num_consistent views); doing it in one
// dispatch over a single contiguous source-depth upload avoids the per-source
// dispatch loop. A degenerate reprojection does not count. Same argument layout
// as ComputeGeomConsistencyCostMetal except `out_count` (length `num`).
// Returns false if Metal is unavailable or inputs are invalid.
bool ComputeGeomConsistencyCountMetal(int width,
                                      int height,
                                      const float ref_inv_K[4],
                                      const float ref_K[4],
                                      const float* src_P,
                                      const float* src_inv_P,
                                      const float* src_depth_maps,
                                      int num_src,
                                      float max_cost,
                                      const int* rows,
                                      const int* cols,
                                      const float* depths,
                                      int num,
                                      int* out_count);

// PatchMatch optimizer for ONE source view: iteratively improves a per-pixel
// (depth, normal) field over the whole image by (a) spatial PROPAGATION of each
// of the 4 neighbors' hypotheses (depth transferred along the local plane, as in
// COLMAP's PropagateDepth) and (b) REFINEMENT via multiplicative depth
// perturbation and small normal tilts, at a perturbation scale that shrinks each
// iteration. Candidates are accepted greedily when they lower the bilateral-NCC
// cost, so the per-pixel cost is monotonically non-increasing (stable).
//
// This is a deterministic propagation+refinement scheme (no RNG) -- it captures
// PatchMatch's propagate-then-refine structure and converges to the photometric
// optimum; COLMAP's stochastic curand sampling + the red-black sweep order are a
// later refinement. View selection and geometric consistency are not included.
//
//   depth_min/depth_max: valid depth range (candidates outside are rejected).
//   io_depth: in/out, width*height (caller seeds the initial field).
//   io_normal: in/out, width*height*3 unit normals in the reference frame.
// Returns false if Metal is unavailable or inputs are invalid.
bool RunPatchMatchOptimizerMetal(const float* ref_image,
                                 const float* src_image,
                                 int width,
                                 int height,
                                 const float ref_inv_K[4],
                                 const float src_K[4],
                                 const float src_R[9],
                                 const float src_T[3],
                                 int window_radius,
                                 int window_step,
                                 float sigma_spatial,
                                 float sigma_color,
                                 float depth_min,
                                 float depth_max,
                                 int num_iterations,
                                 float* io_depth,
                                 float* io_normal);

// Multi-view PatchMatch optimizer: identical propagate-then-refine scheme as
// RunPatchMatchOptimizerMetal, but each candidate is scored against `num_src`
// source views via the best-K aggregation (EvaluateHypothesisCostMultiViewMetal).
// src_images/src_K/src_R/src_T are packed as in EvaluateHypothesisCostMultiViewMetal.
// Returns false if Metal is unavailable or inputs are invalid.
bool RunPatchMatchMultiViewOptimizerMetal(const float* ref_image,
                                          const float* src_images,
                                          int num_src,
                                          int width,
                                          int height,
                                          const float ref_inv_K[4],
                                          const float* src_K,
                                          const float* src_R,
                                          const float* src_T,
                                          int best_k,
                                          int window_radius,
                                          int window_step,
                                          float sigma_spatial,
                                          float sigma_color,
                                          float depth_min,
                                          float depth_max,
                                          int num_iterations,
                                          float* io_depth,
                                          float* io_normal);

// Integration bridge from COLMAP camera conventions to the Metal optimizer:
// given a reference grayscale image + its 3x3 K and world->cam (R, T), and
// num_src source images each with their own K/R/T (all COLMAP world->cam), it
// derives the reference inverse intrinsics and the reference->source relative
// poses (via mvs::ComputeRelativePose), initializes a fronto-parallel field at
// the middle of [depth_min, depth_max], and runs the multi-view PatchMatch
// optimizer. This is the bridge a PatchMatchController/Workspace driver calls
// per reference image.
//   ref_gray/src_gray: row-major width*height grayscale floats (src concatenated).
//   ref_K9/src_K9: row-major 3x3 intrinsics (fx,_,cx,_,fy,cy,_,_,1).
//   ref_R9/src_R9, ref_T3/src_T3: world->cam rotation (row-major 3x3) + translation.
//   filter_min_ncc: after optimization, invalidate (depth=0) pixels whose final
//     NCC is below this threshold -- COLMAP's low-confidence rejection. <= 0
//     disables filtering (every pixel kept).
//   out_depth: width*height; out_normal: width*height*3 (caller-allocated).
// Returns false if Metal is unavailable or inputs are invalid.
bool EstimateDepthMapMetalFromArrays(int width,
                                     int height,
                                     const float* ref_gray,
                                     const float ref_K9[9],
                                     const float ref_R9[9],
                                     const float ref_T3[3],
                                     int num_src,
                                     const float* src_gray,
                                     const float* src_K9,
                                     const float* src_R9,
                                     const float* src_T3,
                                     int best_k,
                                     int window_radius,
                                     int window_step,
                                     float sigma_spatial,
                                     float sigma_color,
                                     float depth_min,
                                     float depth_max,
                                     int num_iterations,
                                     float filter_min_ncc,
                                     float* out_depth,
                                     float* out_normal);

// Metal MVS controller (macOS analogue of the CUDA-only mvs::PatchMatchController):
// reads a COLMAP dense workspace, runs the Metal PatchMatch optimizer per
// reference image, applies geometric-consistency filtering, and writes the
// stereo/{depth,normal}_maps. Defined only when COLMAP_METAL_ENABLED (in
// patch_match_metal_controller.cc); call site must guard on it.
void RunPatchMatchStereoMetal(const PatchMatchOptions& options,
                              const std::filesystem::path& workspace_path,
                              const std::string& workspace_format);

}  // namespace mvs
}  // namespace colmap
