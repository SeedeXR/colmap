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

#include "colmap/mvs/patch_match_metal.h"

#include "colmap/mvs/image.h"  // mvs::ComputeRelativePose

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace colmap {
namespace mvs {
namespace {

// Max source views the multi-view kernel evaluates per pixel. MUST match the
// `kMaxViews` constant in the MSL source below; the host rejects num_src above
// this rather than letting the kernel silently truncate.
constexpr int kMaxMetalMvsViews = 32;

// Plane-induced homography kernel, runtime-compiled (no .metallib to ship).
// This is a verbatim MSL port of ComposeHomography from patch_match_cuda.cu:
// the same scalar expansion of H = K_src * (R - T * n^T / d) * K_ref^-1, using
// the same float arithmetic so results match the CUDA path within float error.
constexpr char kHomographySource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct HomographyParams {
  float ref_inv_K[4];  // {1/fx, -cx/fx, 1/fy, -cy/fy} of the reference camera.
  float K[4];          // {fx, cx, fy, cy} of the source camera.
  float R[9];          // row-major relative rotation (reference -> source).
  float T[3];          // relative translation (reference -> source).
  uint num;            // number of plane hypotheses.
};

struct NccParams {
  int width;
  int height;
  int window_radius;
  int window_step;
  float spatial_norm;  // 1 / (2 * sigma_spatial^2)
  float color_norm;    // 1 / (2 * sigma_color^2)
  uint num;
};

// Fused (depth, normal) -> homography -> cost evaluation for one source view.
struct HypothesisParams {
  float ref_inv_K[4];
  float K[4];
  float R[9];
  float T[3];
  int width;
  int height;
  int window_radius;
  int window_step;
  float spatial_norm;
  float color_norm;
  uint num;
};

// Multi-view variant: per-source K/R/T live in separate device buffers; here we
// carry only the reference calibration + scalars.
struct HypothesisMVParams {
  float ref_inv_K[4];
  int width;
  int height;
  int window_radius;
  int window_step;
  float spatial_norm;
  float color_norm;
  int num_src;
  int best_k;
  uint num;
};

// Plane-induced homography, verbatim port of ComposeHomography in
// patch_match_cuda.cu (H = K_src * (R + T * n^T / d) * K_ref^-1). Shared by all
// kernels so the formula lives in exactly one place.
inline void ComposeHomographyDev(thread const float* rik, thread const float* K,
                                 thread const float* R, thread const float* T,
                                 int row, int col, float depth,
                                 thread const float* normal, thread float* H) {
  const float dist = depth * (normal[0] * (rik[0] * col + rik[1]) +
                              normal[1] * (rik[2] * row + rik[3]) + normal[2]);
  // Plane through (or grazing) the camera center -> degenerate homography. Emit
  // a non-finite inv_dist so the consuming cost kernels' isfinite() guard
  // rejects this hypothesis instead of producing a huge finite garbage warp.
  const float inv_dist = (fabs(dist) > 1e-9f) ? (1.0f / dist) : INFINITY;
  const float n0 = inv_dist * normal[0];
  const float n1 = inv_dist * normal[1];
  const float n2 = inv_dist * normal[2];
  H[0] = rik[0] * (K[0] * (R[0] + n0 * T[0]) + K[1] * (R[6] + n0 * T[2]));
  H[1] = rik[2] * (K[0] * (R[1] + n1 * T[0]) + K[1] * (R[7] + n1 * T[2]));
  H[2] = K[0] * (R[2] + n2 * T[0]) + K[1] * (R[8] + n2 * T[2]) +
         rik[1] * (K[0] * (R[0] + n0 * T[0]) + K[1] * (R[6] + n0 * T[2])) +
         rik[3] * (K[0] * (R[1] + n1 * T[0]) + K[1] * (R[7] + n1 * T[2]));
  H[3] = rik[0] * (K[2] * (R[3] + n0 * T[1]) + K[3] * (R[6] + n0 * T[2]));
  H[4] = rik[2] * (K[2] * (R[4] + n1 * T[1]) + K[3] * (R[7] + n1 * T[2]));
  H[5] = K[2] * (R[5] + n2 * T[1]) + K[3] * (R[8] + n2 * T[2]) +
         rik[1] * (K[2] * (R[3] + n0 * T[1]) + K[3] * (R[6] + n0 * T[2])) +
         rik[3] * (K[2] * (R[4] + n1 * T[1]) + K[3] * (R[7] + n1 * T[2]));
  H[6] = rik[0] * (R[6] + n0 * T[2]);
  H[7] = rik[2] * (R[7] + n1 * T[2]);
  H[8] = R[8] + rik[1] * (R[6] + n0 * T[2]) + rik[3] * (R[7] + n1 * T[2]) +
         n2 * T[2];
}

// Nearest (point) fetch with border = 0, matching the reference-image filter
// (cudaFilterModePoint + cudaAddressModeBorder).
inline float SamplePoint(device const float* img, int W, int H, int c, int r) {
  if (c < 0 || c >= W || r < 0 || r >= H) return 0.0f;
  return img[(uint)r * W + c];
}

// Bilinear fetch with border = 0. COLMAP's CUDA path samples the source texture
// at (coord + 0.5) with hardware linear filtering and pixel centers at
// integer+0.5, which is exactly bilinear interpolation at `coord` on the integer
// pixel grid -- reproduced here so no hardware sampler is needed.
inline float SampleBilinear(device const float* img, int W, int H, float x,
                            float y) {
  const int c0 = (int)floor(x);
  const int r0 = (int)floor(y);
  const float ac = x - (float)c0;
  const float ar = y - (float)r0;
  const float v00 = SamplePoint(img, W, H, c0, r0);
  const float v01 = SamplePoint(img, W, H, c0 + 1, r0);
  const float v10 = SamplePoint(img, W, H, c0, r0 + 1);
  const float v11 = SamplePoint(img, W, H, c0 + 1, r0 + 1);
  return (1.0f - ar) * ((1.0f - ac) * v00 + ac * v01) +
         ar * ((1.0f - ac) * v10 + ac * v11);
}

// Bilateral-weighted NCC cost (1 - NCC) for the patch at (crow, ccol) warped by
// homography H. Verbatim port of PhotoConsistencyCostComputer::Compute.
inline float NccCostDev(device const float* ref_img, device const float* src_img,
                       int W, int Ht, int wr, int ws, float sn, float cn,
                       int crow, int ccol, thread const float* H) {
  const float center = SamplePoint(ref_img, W, Ht, ccol, crow);
  float ref_sum = 0.0f, ref_sq = 0.0f;
  float src_sum = 0.0f, src_sq = 0.0f, src_ref = 0.0f, w_sum = 0.0f;
  for (int dr = -wr; dr <= wr; dr += ws) {
    for (int dc = -wr; dc <= wr; dc += ws) {
      const int cc = ccol + dc;
      const int rr = crow + dr;
      const float ref_color = SamplePoint(ref_img, W, Ht, cc, rr);
      const float color_diff = center - ref_color;
      const float w =
          exp(-(float)(dr * dr + dc * dc) * sn - color_diff * color_diff * cn);
      const float xs = H[0] * cc + H[1] * rr + H[2];
      const float ys = H[3] * cc + H[4] * rr + H[5];
      const float zs = H[6] * cc + H[7] * rr + H[8];
      // Degenerate hypothesis: a window pixel projects to/behind the source
      // camera plane (zs <= 0) or the homography is non-finite (e.g. a plane
      // through the camera center made dist ~ 0). Reject the whole pixel as
      // worst-cost rather than divide by ~0 and sample garbage.
      if (!(zs > 1e-6f) || !isfinite(zs)) return 2.0f;  // kMaxCost
      const float inv_z = 1.0f / zs;
      const float src_color =
          SampleBilinear(src_img, W, Ht, inv_z * xs, inv_z * ys);
      ref_sum += w * ref_color;
      ref_sq += w * ref_color * ref_color;
      const float w_src = w * src_color;
      src_sum += w_src;
      src_sq += w_src * src_color;
      src_ref += w_src * ref_color;
      w_sum += w;
    }
  }
  const float inv_w = 1.0f / w_sum;
  ref_sum *= inv_w;
  ref_sq *= inv_w;
  src_sum *= inv_w;
  src_sq *= inv_w;
  src_ref *= inv_w;
  const float ref_var = ref_sq - ref_sum * ref_sum;
  const float src_var = src_sq - src_sum * src_sum;
  const float kMaxCost = 2.0f;
  const float kMinVar = 1e-5f;
  if (ref_var < kMinVar || src_var < kMinVar) return kMaxCost;
  const float covar = src_ref - ref_sum * src_sum;
  return max(0.0f, min(kMaxCost, 1.0f - covar / sqrt(ref_var * src_var)));
}

kernel void compose_homographies(constant HomographyParams& p [[buffer(0)]],
                                 device const int* rows       [[buffer(1)]],
                                 device const int* cols       [[buffer(2)]],
                                 device const float* depths   [[buffer(3)]],
                                 device const float* normals  [[buffer(4)]],
                                 device float* out_H          [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= p.num) return;
  const float normal[3] = {normals[3 * gid + 0], normals[3 * gid + 1],
                           normals[3 * gid + 2]};
  float rik[4], K[4], R[9], T[3];
  for (int i = 0; i < 4; ++i) { rik[i] = p.ref_inv_K[i]; K[i] = p.K[i]; }
  for (int i = 0; i < 9; ++i) R[i] = p.R[i];
  for (int i = 0; i < 3; ++i) T[i] = p.T[i];
  float H[9];
  ComposeHomographyDev(rik, K, R, T, rows[gid], cols[gid], depths[gid], normal,
                       H);
  device float* out = out_H + (size_t)gid * 9;
  for (int i = 0; i < 9; ++i) out[i] = H[i];
}

kernel void ncc_photometric_cost(constant NccParams& p     [[buffer(0)]],
                                device const float* ref_img [[buffer(1)]],
                                device const float* src_img [[buffer(2)]],
                                device const int* rows      [[buffer(3)]],
                                device const int* cols      [[buffer(4)]],
                                device const float* H_all   [[buffer(5)]],
                                device float* out_cost      [[buffer(6)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= p.num) return;
  float H[9];
  for (int i = 0; i < 9; ++i) H[i] = H_all[(size_t)gid * 9 + i];
  out_cost[gid] =
      NccCostDev(ref_img, src_img, p.width, p.height, p.window_radius,
                 p.window_step, p.spatial_norm, p.color_norm, rows[gid],
                 cols[gid], H);
}

// Fused: compose the homography from each pixel's (depth, normal) hypothesis and
// return its NCC cost -- the inner-loop primitive of the PatchMatch optimizer.
kernel void evaluate_hypothesis_cost(constant HypothesisParams& p [[buffer(0)]],
                                    device const float* ref_img    [[buffer(1)]],
                                    device const float* src_img    [[buffer(2)]],
                                    device const int* rows         [[buffer(3)]],
                                    device const int* cols         [[buffer(4)]],
                                    device const float* depths     [[buffer(5)]],
                                    device const float* normals    [[buffer(6)]],
                                    device float* out_cost         [[buffer(7)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.num) return;
  const float normal[3] = {normals[3 * gid + 0], normals[3 * gid + 1],
                           normals[3 * gid + 2]};
  float rik[4], K[4], R[9], T[3];
  for (int i = 0; i < 4; ++i) { rik[i] = p.ref_inv_K[i]; K[i] = p.K[i]; }
  for (int i = 0; i < 9; ++i) R[i] = p.R[i];
  for (int i = 0; i < 3; ++i) T[i] = p.T[i];
  float H[9];
  ComposeHomographyDev(rik, K, R, T, rows[gid], cols[gid], depths[gid], normal,
                       H);
  out_cost[gid] =
      NccCostDev(ref_img, src_img, p.width, p.height, p.window_radius,
                 p.window_step, p.spatial_norm, p.color_norm, rows[gid],
                 cols[gid], H);
}

// Multi-view fused cost: evaluates the hypothesis against `num_src` source views
// (each with its own K/R/T and image plane in the packed src_imgs buffer) and
// aggregates as the MEAN of the best (lowest) `best_k` per-view costs -- a robust
// multi-view score that tolerates a minority of occluded/mismatched views. (This
// is a deterministic robust aggregation, not COLMAP's probabilistic monte-carlo
// view selection, which is a later refinement.)
kernel void evaluate_hypothesis_cost_mv(constant HypothesisMVParams& p [[buffer(0)]],
                                       device const float* ref_img    [[buffer(1)]],
                                       device const float* src_imgs   [[buffer(2)]],
                                       device const float* src_K      [[buffer(3)]],
                                       device const float* src_R      [[buffer(4)]],
                                       device const float* src_T      [[buffer(5)]],
                                       device const int* rows         [[buffer(6)]],
                                       device const int* cols         [[buffer(7)]],
                                       device const float* depths     [[buffer(8)]],
                                       device const float* normals    [[buffer(9)]],
                                       device float* out_cost         [[buffer(10)]],
                                       uint gid [[thread_position_in_grid]]) {
  if (gid >= p.num) return;
  const int crow = rows[gid];
  const int ccol = cols[gid];
  const float normal[3] = {normals[3 * gid + 0], normals[3 * gid + 1],
                           normals[3 * gid + 2]};
  float rik[4];
  for (int i = 0; i < 4; ++i) rik[i] = p.ref_inv_K[i];

  const float kMaxCost = 2.0f;
  // Per-view costs (bounded small num_src); insertion into a sorted best list.
  const int kMaxViews = 32;
  float costs[kMaxViews];
  int nsrc = min(p.num_src, kMaxViews);
  const size_t plane = (size_t)p.width * p.height;
  for (int m = 0; m < nsrc; ++m) {
    float K[4], R[9], T[3];
    for (int i = 0; i < 4; ++i) K[i] = src_K[m * 4 + i];
    for (int i = 0; i < 9; ++i) R[i] = src_R[m * 9 + i];
    for (int i = 0; i < 3; ++i) T[i] = src_T[m * 3 + i];
    float H[9];
    ComposeHomographyDev(rik, K, R, T, crow, ccol, depths[gid], normal, H);
    costs[m] = NccCostDev(ref_img, src_imgs + m * plane, p.width, p.height,
                          p.window_radius, p.window_step, p.spatial_norm,
                          p.color_norm, crow, ccol, H);
  }
  // Selection of the best_k smallest costs, averaged.
  int k = min(p.best_k, nsrc);
  if (k < 1) k = 1;
  float sum = 0.0f;
  for (int sel = 0; sel < k; ++sel) {
    int min_idx = -1;
    float min_val = kMaxCost + 1.0f;
    for (int m = 0; m < nsrc; ++m) {
      if (costs[m] < min_val) { min_val = costs[m]; min_idx = m; }
    }
    sum += min_val;
    if (min_idx >= 0) costs[min_idx] = kMaxCost + 2.0f;  // remove from pool.
  }
  out_cost[gid] = sum / (float)k;
}

struct GeomParams {
  float ref_inv_K[4];  // {1/fx, -cx/fx, 1/fy, -cy/fy} reference inverse intrinsics.
  float ref_K[4];      // {fx, cx, fy, cy} reference (forward) intrinsics.
  int width;
  int height;
  int num_src;
  float max_cost;
  uint num;
};

// Nearest-pixel depth fetch (depth maps are not interpolated across edges);
// COLMAP samples at (coord + 0.5) -> nearest integer pixel. Border = 0 (invalid).
inline float SampleDepth(device const float* dmap, int W, int H, float x,
                         float y) {
  const int c = (int)floor(x + 0.5f);
  const int r = (int)floor(y + 0.5f);
  if (c < 0 || c >= W || r < 0 || r >= H) return 0.0f;
  return dmap[(uint)r * W + c];
}

// Forward-backward reprojection error of a reference-frame point (px,py,pz) at
// reference pixel (col,row) against ONE source view (projection P, inverse iP,
// depth map src_depth). Returns the pixel reprojection error, or a negative
// sentinel if the reprojection is degenerate (point at/behind the source
// camera, or no source depth at the reprojection). Shared by both geometric-
// consistency kernels below so the math stays a single verbatim port of
// ComputeGeomConsistencyCost (patch_match_cuda.cu).
inline float GeomReprojErrorDev(constant GeomParams& p,
                                device const float* P,
                                device const float* iP,
                                device const float* src_depth,
                                float px, float py, float pz,
                                int col, int row) {
  const float fz = P[8] * px + P[9] * py + P[10] * pz + P[11];
  if (!(fz > 1e-6f) || !isfinite(fz)) return -1.0f;  // at/behind source camera.
  const float inv_fz = 1.0f / fz;
  float sc = inv_fz * (P[0] * px + P[1] * py + P[2] * pz + P[3]);
  float sr = inv_fz * (P[4] * px + P[5] * py + P[6] * pz + P[7]);
  const float sd = SampleDepth(src_depth, p.width, p.height, sc, sr);
  if (sd == 0.0f) return -1.0f;  // no source depth at the reprojection.
  sc *= sd;
  sr *= sd;
  const float bx = iP[0] * sc + iP[1] * sr + iP[2] * sd + iP[3];
  const float by = iP[4] * sc + iP[5] * sr + iP[6] * sd + iP[7];
  const float bz = iP[8] * sc + iP[9] * sr + iP[10] * sd + iP[11];
  if (!(bz > 1e-6f) || !isfinite(bz)) return -1.0f;  // back-projection degenerate.
  const float inv_bz = 1.0f / bz;
  const float bcol = inv_bz * (p.ref_K[0] * bx + p.ref_K[1] * bz);
  const float brow = inv_bz * (p.ref_K[2] * by + p.ref_K[3] * bz);
  const float dcol = col - bcol;
  const float drow = row - brow;
  return sqrt(dcol * dcol + drow * drow);
}

// Geometric-consistency cost: forward-backward reprojection error of the depth
// hypothesis against the source views' depth maps. Verbatim port of
// ComputeGeomConsistencyCost (patch_match_cuda.cu), averaged over sources.
kernel void geom_consistency_cost(constant GeomParams& p       [[buffer(0)]],
                                 device const float* src_P      [[buffer(1)]],
                                 device const float* src_inv_P  [[buffer(2)]],
                                 device const float* src_depths [[buffer(3)]],
                                 device const int* rows         [[buffer(4)]],
                                 device const int* cols         [[buffer(5)]],
                                 device const float* depths     [[buffer(6)]],
                                 device float* out_cost         [[buffer(7)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= p.num) return;
  const int col = cols[gid];
  const int row = rows[gid];
  const float depth = depths[gid];
  // Reference-frame 3D point at the hypothesised depth (ComputePointAtDepth).
  const float px = depth * (p.ref_inv_K[0] * col + p.ref_inv_K[1]);
  const float py = depth * (p.ref_inv_K[2] * row + p.ref_inv_K[3]);
  const float pz = depth;

  const size_t plane = (size_t)p.width * p.height;
  float total = 0.0f;
  for (int m = 0; m < p.num_src; ++m) {
    const float err = GeomReprojErrorDev(p, src_P + m * 12, src_inv_P + m * 12,
                                         src_depths + m * plane, px, py, pz,
                                         col, row);
    // Degenerate reprojections and large errors are truncated at max_cost.
    total += (err < 0.0f) ? p.max_cost : min(p.max_cost, err);
  }
  out_cost[gid] = total / (float)max(p.num_src, 1);
}

// Fused geometric-consistency count: like geom_consistency_cost but evaluates
// ALL source views in one dispatch and outputs, per pixel, the number of
// sources whose forward-backward reprojection error is <= max_cost (the
// consistency threshold). This lets the controller apply COLMAP's count-based
// filter with a single dispatch and a single contiguous source-depth upload
// instead of one dispatch per source. A degenerate reprojection (point behind
// a camera, missing source depth) simply does not count toward consistency.
kernel void geom_consistency_count(constant GeomParams& p       [[buffer(0)]],
                                  device const float* src_P      [[buffer(1)]],
                                  device const float* src_inv_P  [[buffer(2)]],
                                  device const float* src_depths [[buffer(3)]],
                                  device const int* rows         [[buffer(4)]],
                                  device const int* cols         [[buffer(5)]],
                                  device const float* depths     [[buffer(6)]],
                                  device int* out_count          [[buffer(7)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid >= p.num) return;
  const int col = cols[gid];
  const int row = rows[gid];
  const float depth = depths[gid];
  const float px = depth * (p.ref_inv_K[0] * col + p.ref_inv_K[1]);
  const float py = depth * (p.ref_inv_K[2] * row + p.ref_inv_K[3]);
  const float pz = depth;

  const size_t plane = (size_t)p.width * p.height;
  int count = 0;
  for (int m = 0; m < p.num_src; ++m) {
    const float err = GeomReprojErrorDev(p, src_P + m * 12, src_inv_P + m * 12,
                                         src_depths + m * plane, px, py, pz,
                                         col, row);
    // A degenerate reprojection (err < 0) does not count toward consistency.
    if (err >= 0.0f && err <= p.max_cost) ++count;
  }
  out_count[gid] = count;
}
)METAL";

// Host mirror of the MSL HomographyParams. Plain scalar float arrays (no simd
// vector types) so the byte layout matches the MSL struct exactly.
struct HomographyParams {
  float ref_inv_K[4];
  float K[4];
  float R[9];
  float T[3];
  uint32_t num;
};

// Host mirror of the MSL NccParams (same scalar layout).
struct NccParams {
  int32_t width;
  int32_t height;
  int32_t window_radius;
  int32_t window_step;
  float spatial_norm;
  float color_norm;
  uint32_t num;
};

// Host mirror of the MSL HypothesisParams (same scalar layout).
struct HypothesisParams {
  float ref_inv_K[4];
  float K[4];
  float R[9];
  float T[3];
  int32_t width;
  int32_t height;
  int32_t window_radius;
  int32_t window_step;
  float spatial_norm;
  float color_norm;
  uint32_t num;
};

// Host mirror of the MSL HypothesisMVParams (same scalar layout).
struct HypothesisMVParams {
  float ref_inv_K[4];
  int32_t width;
  int32_t height;
  int32_t window_radius;
  int32_t window_step;
  float spatial_norm;
  float color_norm;
  int32_t num_src;
  int32_t best_k;
  uint32_t num;
};

// Host mirror of the MSL GeomParams (same scalar layout).
struct GeomParams {
  float ref_inv_K[4];
  float ref_K[4];
  int32_t width;
  int32_t height;
  int32_t num_src;
  float max_cost;
  uint32_t num;
};

// Persistent Metal context: device, queue, and the compiled homography pipeline
// are created once and reused for every dispatch.
struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> homography_pipeline = nil;
  id<MTLComputePipelineState> ncc_pipeline = nil;
  id<MTLComputePipelineState> hypothesis_pipeline = nil;
  id<MTLComputePipelineState> hypothesis_mv_pipeline = nil;
  id<MTLComputePipelineState> geom_pipeline = nil;
  id<MTLComputePipelineState> geom_count_pipeline = nil;
  bool valid = false;

  MetalContext() {
    // Escape hatch mirroring the matcher: COLMAP_DISABLE_METAL=1 forces the
    // non-Metal path without recompiling.
    const char* disable = std::getenv("COLMAP_DISABLE_METAL");
    if (disable != nullptr && disable[0] == '1') {
      return;
    }
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      if (device == nil) {
        return;
      }
      queue = [device newCommandQueue];
      if (queue == nil) {
        return;
      }
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithSource:[NSString stringWithUTF8String:kHomographySource]
                       options:nil
                         error:&error];
      if (library == nil) {
        return;
      }
      id<MTLFunction> fn = [library newFunctionWithName:@"compose_homographies"];
      if (fn == nil) {
        return;
      }
      homography_pipeline =
          [device newComputePipelineStateWithFunction:fn error:&error];
      if (homography_pipeline == nil) {
        return;
      }
      id<MTLFunction> ncc_fn =
          [library newFunctionWithName:@"ncc_photometric_cost"];
      if (ncc_fn == nil) {
        return;
      }
      ncc_pipeline =
          [device newComputePipelineStateWithFunction:ncc_fn error:&error];
      if (ncc_pipeline == nil) {
        return;
      }
      id<MTLFunction> hyp_fn =
          [library newFunctionWithName:@"evaluate_hypothesis_cost"];
      if (hyp_fn == nil) {
        return;
      }
      hypothesis_pipeline =
          [device newComputePipelineStateWithFunction:hyp_fn error:&error];
      if (hypothesis_pipeline == nil) {
        return;
      }
      id<MTLFunction> hyp_mv_fn =
          [library newFunctionWithName:@"evaluate_hypothesis_cost_mv"];
      if (hyp_mv_fn == nil) {
        return;
      }
      hypothesis_mv_pipeline =
          [device newComputePipelineStateWithFunction:hyp_mv_fn error:&error];
      if (hypothesis_mv_pipeline == nil) {
        return;
      }
      id<MTLFunction> geom_fn =
          [library newFunctionWithName:@"geom_consistency_cost"];
      if (geom_fn == nil) {
        return;
      }
      geom_pipeline =
          [device newComputePipelineStateWithFunction:geom_fn error:&error];
      if (geom_pipeline == nil) {
        return;
      }
      id<MTLFunction> geom_count_fn =
          [library newFunctionWithName:@"geom_consistency_count"];
      if (geom_count_fn == nil) {
        return;
      }
      geom_count_pipeline =
          [device newComputePipelineStateWithFunction:geom_count_fn error:&error];
      if (geom_count_pipeline == nil) {
        return;
      }
      valid = true;
    }
  }
};

MetalContext& GetContext() {
  static MetalContext context;
  return context;
}

}  // namespace

bool IsPatchMatchMetalAvailable() { return GetContext().valid; }

bool ComposePlaneHomographiesMetal(const float ref_inv_K[4],
                                   const float src_K[4],
                                   const float src_R[9],
                                   const float src_T[3],
                                   const int* rows,
                                   const int* cols,
                                   const float* depths,
                                   const float* normals,
                                   int num,
                                   float* out_H) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num <= 0) {
    return false;
  }

  @autoreleasepool {
    HomographyParams params;
    std::memcpy(params.ref_inv_K, ref_inv_K, sizeof(params.ref_inv_K));
    std::memcpy(params.K, src_K, sizeof(params.K));
    std::memcpy(params.R, src_R, sizeof(params.R));
    std::memcpy(params.T, src_T, sizeof(params.T));
    params.num = static_cast<uint32_t>(num);

    const size_t n = static_cast<size_t>(num);
    // Unified-memory shared buffers (zero copy on Apple Silicon).
    id<MTLBuffer> rows_buf =
        [ctx.device newBufferWithBytes:rows
                                length:n * sizeof(int)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> cols_buf =
        [ctx.device newBufferWithBytes:cols
                                length:n * sizeof(int)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> depths_buf =
        [ctx.device newBufferWithBytes:depths
                                length:n * sizeof(float)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> normals_buf =
        [ctx.device newBufferWithBytes:normals
                                length:n * 3 * sizeof(float)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf =
        [ctx.device newBufferWithLength:n * 9 * sizeof(float)
                                options:MTLResourceStorageModeShared];
    if (rows_buf == nil || cols_buf == nil || depths_buf == nil ||
        normals_buf == nil || out_buf == nil) {
      return false;
    }

    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.homography_pipeline];
    [enc setBytes:&params length:sizeof(params) atIndex:0];
    [enc setBuffer:rows_buf offset:0 atIndex:1];
    [enc setBuffer:cols_buf offset:0 atIndex:2];
    [enc setBuffer:depths_buf offset:0 atIndex:3];
    [enc setBuffer:normals_buf offset:0 atIndex:4];
    [enc setBuffer:out_buf offset:0 atIndex:5];

    NSUInteger tg = ctx.homography_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    std::memcpy(out_H, [out_buf contents], n * 9 * sizeof(float));
  }
  return true;
}

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
                                    float* out_cost) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num <= 0 || width <= 0 || height <= 0 ||
      window_radius < 1 || window_step <= 0 || sigma_spatial <= 0.0f ||
      sigma_color <= 0.0f) {
    return false;
  }

  @autoreleasepool {
    NccParams params;
    params.width = width;
    params.height = height;
    params.window_radius = window_radius;
    params.window_step = window_step;
    params.spatial_norm = 1.0f / (2.0f * sigma_spatial * sigma_spatial);
    params.color_norm = 1.0f / (2.0f * sigma_color * sigma_color);
    params.num = static_cast<uint32_t>(num);

    const size_t n = static_cast<size_t>(num);
    const size_t img_bytes =
        static_cast<size_t>(width) * height * sizeof(float);
    id<MTLBuffer> ref_buf =
        [ctx.device newBufferWithBytes:ref_image
                                length:img_bytes
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> src_buf =
        [ctx.device newBufferWithBytes:src_image
                                length:img_bytes
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> rows_buf =
        [ctx.device newBufferWithBytes:rows
                                length:n * sizeof(int)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> cols_buf =
        [ctx.device newBufferWithBytes:cols
                                length:n * sizeof(int)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> H_buf =
        [ctx.device newBufferWithBytes:homographies
                                length:n * 9 * sizeof(float)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> cost_buf =
        [ctx.device newBufferWithLength:n * sizeof(float)
                                options:MTLResourceStorageModeShared];
    if (ref_buf == nil || src_buf == nil || rows_buf == nil ||
        cols_buf == nil || H_buf == nil || cost_buf == nil) {
      return false;
    }

    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.ncc_pipeline];
    [enc setBytes:&params length:sizeof(params) atIndex:0];
    [enc setBuffer:ref_buf offset:0 atIndex:1];
    [enc setBuffer:src_buf offset:0 atIndex:2];
    [enc setBuffer:rows_buf offset:0 atIndex:3];
    [enc setBuffer:cols_buf offset:0 atIndex:4];
    [enc setBuffer:H_buf offset:0 atIndex:5];
    [enc setBuffer:cost_buf offset:0 atIndex:6];

    NSUInteger tg = ctx.ncc_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    std::memcpy(out_cost, [cost_buf contents], n * sizeof(float));
  }
  return true;
}

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
                          float* out_cost) {
  if (!IsPatchMatchMetalAvailable() || num_pixels <= 0 || num_depths <= 0) {
    return false;
  }
  const size_t np = static_cast<size_t>(num_pixels);

  // The plane normal is the same for every pixel/hypothesis in this baseline.
  std::vector<float> normals(np * 3);
  for (size_t i = 0; i < np; ++i) {
    normals[3 * i + 0] = normal[0];
    normals[3 * i + 1] = normal[1];
    normals[3 * i + 2] = normal[2];
  }

  std::vector<float> depths(np);
  std::vector<float> H(np * 9);
  std::vector<float> cost(np);
  std::vector<float> best(np, std::numeric_limits<float>::max());

  for (int d = 0; d < num_depths; ++d) {
    const float dv = depth_candidates[d];
    std::fill(depths.begin(), depths.end(), dv);
    if (!ComposePlaneHomographiesMetal(ref_inv_K, src_K, src_R, src_T, rows,
                                       cols, depths.data(), normals.data(),
                                       num_pixels, H.data())) {
      return false;
    }
    if (!ComputeNCCPhotometricCostMetal(
            ref_image, src_image, width, height, window_radius, window_step,
            sigma_spatial, sigma_color, rows, cols, H.data(), num_pixels,
            cost.data())) {
      return false;
    }
    for (size_t i = 0; i < np; ++i) {
      if (cost[i] < best[i]) {
        best[i] = cost[i];
        out_depth[i] = dv;
      }
    }
  }
  for (size_t i = 0; i < np; ++i) {
    out_cost[i] = best[i];
  }
  return true;
}

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
                                 float* out_cost) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num <= 0 || width <= 0 || height <= 0 ||
      window_radius < 1 || window_step <= 0 || sigma_spatial <= 0.0f ||
      sigma_color <= 0.0f) {
    return false;
  }
  @autoreleasepool {
    HypothesisParams params;
    std::memcpy(params.ref_inv_K, ref_inv_K, sizeof(params.ref_inv_K));
    std::memcpy(params.K, src_K, sizeof(params.K));
    std::memcpy(params.R, src_R, sizeof(params.R));
    std::memcpy(params.T, src_T, sizeof(params.T));
    params.width = width;
    params.height = height;
    params.window_radius = window_radius;
    params.window_step = window_step;
    params.spatial_norm = 1.0f / (2.0f * sigma_spatial * sigma_spatial);
    params.color_norm = 1.0f / (2.0f * sigma_color * sigma_color);
    params.num = static_cast<uint32_t>(num);

    const size_t n = static_cast<size_t>(num);
    const size_t img_bytes =
        static_cast<size_t>(width) * height * sizeof(float);
    id<MTLBuffer> ref_buf =
        [ctx.device newBufferWithBytes:ref_image
                                length:img_bytes
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> src_buf =
        [ctx.device newBufferWithBytes:src_image
                                length:img_bytes
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> rows_buf =
        [ctx.device newBufferWithBytes:rows
                                length:n * sizeof(int)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> cols_buf =
        [ctx.device newBufferWithBytes:cols
                                length:n * sizeof(int)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> depths_buf =
        [ctx.device newBufferWithBytes:depths
                                length:n * sizeof(float)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> normals_buf =
        [ctx.device newBufferWithBytes:normals
                                length:n * 3 * sizeof(float)
                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> cost_buf =
        [ctx.device newBufferWithLength:n * sizeof(float)
                                options:MTLResourceStorageModeShared];
    if (ref_buf == nil || src_buf == nil || rows_buf == nil ||
        cols_buf == nil || depths_buf == nil || normals_buf == nil ||
        cost_buf == nil) {
      return false;
    }

    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.hypothesis_pipeline];
    [enc setBytes:&params length:sizeof(params) atIndex:0];
    [enc setBuffer:ref_buf offset:0 atIndex:1];
    [enc setBuffer:src_buf offset:0 atIndex:2];
    [enc setBuffer:rows_buf offset:0 atIndex:3];
    [enc setBuffer:cols_buf offset:0 atIndex:4];
    [enc setBuffer:depths_buf offset:0 atIndex:5];
    [enc setBuffer:normals_buf offset:0 atIndex:6];
    [enc setBuffer:cost_buf offset:0 atIndex:7];

    NSUInteger tg = ctx.hypothesis_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    std::memcpy(out_cost, [cost_buf contents], n * sizeof(float));
  }
  return true;
}

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
                                          float* out_cost) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num <= 0 || num_src <= 0 || num_src > kMaxMetalMvsViews ||
      width <= 0 || height <= 0 || window_radius < 1 || window_step <= 0 ||
      sigma_spatial <= 0.0f || sigma_color <= 0.0f) {
    return false;
  }
  @autoreleasepool {
    HypothesisMVParams params;
    std::memcpy(params.ref_inv_K, ref_inv_K, sizeof(params.ref_inv_K));
    params.width = width;
    params.height = height;
    params.window_radius = window_radius;
    params.window_step = window_step;
    params.spatial_norm = 1.0f / (2.0f * sigma_spatial * sigma_spatial);
    params.color_norm = 1.0f / (2.0f * sigma_color * sigma_color);
    params.num_src = num_src;
    params.best_k = best_k;
    params.num = static_cast<uint32_t>(num);

    const size_t n = static_cast<size_t>(num);
    const size_t m = static_cast<size_t>(num_src);
    const size_t plane = static_cast<size_t>(width) * height;
    auto buf = [&](const void* ptr, size_t bytes) {
      return [ctx.device newBufferWithBytes:ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> ref_buf = buf(ref_image, plane * sizeof(float));
    id<MTLBuffer> src_buf = buf(src_images, m * plane * sizeof(float));
    id<MTLBuffer> srcK_buf = buf(src_K, m * 4 * sizeof(float));
    id<MTLBuffer> srcR_buf = buf(src_R, m * 9 * sizeof(float));
    id<MTLBuffer> srcT_buf = buf(src_T, m * 3 * sizeof(float));
    id<MTLBuffer> rows_buf = buf(rows, n * sizeof(int));
    id<MTLBuffer> cols_buf = buf(cols, n * sizeof(int));
    id<MTLBuffer> depths_buf = buf(depths, n * sizeof(float));
    id<MTLBuffer> normals_buf = buf(normals, n * 3 * sizeof(float));
    id<MTLBuffer> cost_buf =
        [ctx.device newBufferWithLength:n * sizeof(float)
                                options:MTLResourceStorageModeShared];
    if (ref_buf == nil || src_buf == nil || srcK_buf == nil || srcR_buf == nil ||
        srcT_buf == nil || rows_buf == nil || cols_buf == nil ||
        depths_buf == nil || normals_buf == nil || cost_buf == nil) {
      return false;
    }

    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.hypothesis_mv_pipeline];
    [enc setBytes:&params length:sizeof(params) atIndex:0];
    [enc setBuffer:ref_buf offset:0 atIndex:1];
    [enc setBuffer:src_buf offset:0 atIndex:2];
    [enc setBuffer:srcK_buf offset:0 atIndex:3];
    [enc setBuffer:srcR_buf offset:0 atIndex:4];
    [enc setBuffer:srcT_buf offset:0 atIndex:5];
    [enc setBuffer:rows_buf offset:0 atIndex:6];
    [enc setBuffer:cols_buf offset:0 atIndex:7];
    [enc setBuffer:depths_buf offset:0 atIndex:8];
    [enc setBuffer:normals_buf offset:0 atIndex:9];
    [enc setBuffer:cost_buf offset:0 atIndex:10];

    NSUInteger tg = ctx.hypothesis_mv_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    std::memcpy(out_cost, [cost_buf contents], n * sizeof(float));
  }
  return true;
}

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
                                      int* out_count) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num <= 0 || num_src <= 0 || num_src > kMaxMetalMvsViews ||
      width <= 0 || height <= 0) {
    return false;
  }
  @autoreleasepool {
    GeomParams params;
    std::memcpy(params.ref_inv_K, ref_inv_K, sizeof(params.ref_inv_K));
    std::memcpy(params.ref_K, ref_K, sizeof(params.ref_K));
    params.width = width;
    params.height = height;
    params.num_src = num_src;
    params.max_cost = max_cost;
    params.num = static_cast<uint32_t>(num);

    const size_t n = static_cast<size_t>(num);
    const size_t m = static_cast<size_t>(num_src);
    const size_t plane = static_cast<size_t>(width) * height;
    auto buf = [&](const void* ptr, size_t bytes) {
      return [ctx.device newBufferWithBytes:ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> P_buf = buf(src_P, m * 12 * sizeof(float));
    id<MTLBuffer> iP_buf = buf(src_inv_P, m * 12 * sizeof(float));
    id<MTLBuffer> depths_map_buf = buf(src_depth_maps, m * plane * sizeof(float));
    id<MTLBuffer> rows_buf = buf(rows, n * sizeof(int));
    id<MTLBuffer> cols_buf = buf(cols, n * sizeof(int));
    id<MTLBuffer> depths_buf = buf(depths, n * sizeof(float));
    id<MTLBuffer> count_buf =
        [ctx.device newBufferWithLength:n * sizeof(int)
                                options:MTLResourceStorageModeShared];
    if (P_buf == nil || iP_buf == nil || depths_map_buf == nil ||
        rows_buf == nil || cols_buf == nil || depths_buf == nil ||
        count_buf == nil) {
      return false;
    }

    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.geom_count_pipeline];
    [enc setBytes:&params length:sizeof(params) atIndex:0];
    [enc setBuffer:P_buf offset:0 atIndex:1];
    [enc setBuffer:iP_buf offset:0 atIndex:2];
    [enc setBuffer:depths_map_buf offset:0 atIndex:3];
    [enc setBuffer:rows_buf offset:0 atIndex:4];
    [enc setBuffer:cols_buf offset:0 atIndex:5];
    [enc setBuffer:depths_buf offset:0 atIndex:6];
    [enc setBuffer:count_buf offset:0 atIndex:7];

    NSUInteger tg = ctx.geom_count_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    std::memcpy(out_count, [count_buf contents], n * sizeof(int));
  }
  return true;
}

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
                                     float* out_cost) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num <= 0 || num_src <= 0 || width <= 0 || height <= 0) {
    return false;
  }
  @autoreleasepool {
    GeomParams params;
    std::memcpy(params.ref_inv_K, ref_inv_K, sizeof(params.ref_inv_K));
    std::memcpy(params.ref_K, ref_K, sizeof(params.ref_K));
    params.width = width;
    params.height = height;
    params.num_src = num_src;
    params.max_cost = max_cost;
    params.num = static_cast<uint32_t>(num);

    const size_t n = static_cast<size_t>(num);
    const size_t m = static_cast<size_t>(num_src);
    const size_t plane = static_cast<size_t>(width) * height;
    auto buf = [&](const void* ptr, size_t bytes) {
      return [ctx.device newBufferWithBytes:ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> P_buf = buf(src_P, m * 12 * sizeof(float));
    id<MTLBuffer> iP_buf = buf(src_inv_P, m * 12 * sizeof(float));
    id<MTLBuffer> depths_map_buf = buf(src_depth_maps, m * plane * sizeof(float));
    id<MTLBuffer> rows_buf = buf(rows, n * sizeof(int));
    id<MTLBuffer> cols_buf = buf(cols, n * sizeof(int));
    id<MTLBuffer> depths_buf = buf(depths, n * sizeof(float));
    id<MTLBuffer> cost_buf =
        [ctx.device newBufferWithLength:n * sizeof(float)
                                options:MTLResourceStorageModeShared];
    if (P_buf == nil || iP_buf == nil || depths_map_buf == nil ||
        rows_buf == nil || cols_buf == nil || depths_buf == nil ||
        cost_buf == nil) {
      return false;
    }

    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.geom_pipeline];
    [enc setBytes:&params length:sizeof(params) atIndex:0];
    [enc setBuffer:P_buf offset:0 atIndex:1];
    [enc setBuffer:iP_buf offset:0 atIndex:2];
    [enc setBuffer:depths_map_buf offset:0 atIndex:3];
    [enc setBuffer:rows_buf offset:0 atIndex:4];
    [enc setBuffer:cols_buf offset:0 atIndex:5];
    [enc setBuffer:depths_buf offset:0 atIndex:6];
    [enc setBuffer:cost_buf offset:0 atIndex:7];

    NSUInteger tg = ctx.geom_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    std::memcpy(out_cost, [cost_buf contents], n * sizeof(float));
  }
  return true;
}

namespace {

// Transfer depth on the local plane from the viewing ray at row1 to row2
// (COLMAP PropagateDepth). Uses the vertical (row) inverse-calibration terms.
float PropagateDepthRow(const float rik[4], float depth1, const float normal1[3],
                        float row1, float row2) {
  const float x1 = depth1 * (rik[2] * row1 + rik[3]);
  const float y1 = depth1;
  const float x2 = x1 + normal1[2];
  const float y2 = y1 - normal1[1];
  const float x4 = rik[2] * row2 + rik[3];
  const float denom = x2 - x1 + x4 * (y1 - y2);
  if (std::abs(denom) < 1e-5f) return depth1;
  return (y1 * x2 - x1 * y2) / denom;
}

// Column analogue of PropagateDepthRow (horizontal neighbors): mirrors the row
// formula with the column inverse-calibration terms and the normal x-component.
float PropagateDepthCol(const float rik[4], float depth1, const float normal1[3],
                        float col1, float col2) {
  const float x1 = depth1 * (rik[0] * col1 + rik[1]);
  const float y1 = depth1;
  const float x2 = x1 + normal1[2];
  const float y2 = y1 - normal1[0];
  const float x4 = rik[0] * col2 + rik[1];
  const float denom = x2 - x1 + x4 * (y1 - y2);
  if (std::abs(denom) < 1e-5f) return depth1;
  return (y1 * x2 - x1 * y2) / denom;
}

// Unbiased random unit normal oriented toward the camera (Marsaglia 1972),
// matching COLMAP's GenerateRandomNormal in patch_match_cuda.cu.
void RandomNormal(std::mt19937& rng, const float rik[4], int row, int col,
                  float n[3]) {
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  float v1 = 0.0f, v2 = 0.0f, s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * u(rng) - 1.0f;
    v2 = 2.0f * u(rng) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }
  const float sn = std::sqrt(1.0f - s);
  n[0] = 2.0f * v1 * sn;
  n[1] = 2.0f * v2 * sn;
  n[2] = 1.0f - 2.0f * s;
  // Flip to look away from the camera (toward negative view-ray dot).
  const float ray[3] = {rik[0] * col + rik[1], rik[2] * row + rik[3], 1.0f};
  if (n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2] > 0.0f) {
    n[0] = -n[0];
    n[1] = -n[1];
    n[2] = -n[2];
  }
}

// Randomly perturb a unit normal by a small random rotation (R = Rx*Ry*Rz with
// angles in +/- perturbation), keeping it pointing toward the camera. Verbatim
// port of COLMAP's PerturbNormal (patch_match_cuda.cu): wider `perturbation`
// early (anchored exploration around the current normal) shrinking to local
// refinement. General PatchMatch refinement, not dataset-specific.
void PerturbNormal(std::mt19937& rng, const float rik[4], int row, int col,
                   const float normal[3], float perturbation, float out[3]) {
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const float ray[3] = {rik[0] * col + rik[1], rik[2] * row + rik[3], 1.0f};
  float pert = perturbation;
  for (int trial = 0; trial < 4; ++trial) {
    const float a1 = (u(rng) - 0.5f) * pert;
    const float a2 = (u(rng) - 0.5f) * pert;
    const float a3 = (u(rng) - 0.5f) * pert;
    const float s1 = std::sin(a1), s2 = std::sin(a2), s3 = std::sin(a3);
    const float c1 = std::cos(a1), c2 = std::cos(a2), c3 = std::cos(a3);
    const float R0 = c2 * c3, R1 = -c2 * s3, R2 = s2;
    const float R3 = c1 * s3 + c3 * s1 * s2, R4 = c1 * c3 - s1 * s2 * s3,
                R5 = -c2 * s1;
    const float R6 = s1 * s3 - c1 * c3 * s2, R7 = c3 * s1 + c1 * s2 * s3,
                R8 = c1 * c2;
    const float p0 = R0 * normal[0] + R1 * normal[1] + R2 * normal[2];
    const float p1 = R3 * normal[0] + R4 * normal[1] + R5 * normal[2];
    const float p2 = R6 * normal[0] + R7 * normal[1] + R8 * normal[2];
    if (p0 * ray[0] + p1 * ray[1] + p2 * ray[2] >= 0.0f) {
      pert *= 0.5f;  // perturbed normal faces away from camera; retry smaller.
      continue;
    }
    const float inv = 1.0f / std::sqrt(p0 * p0 + p1 * p1 + p2 * p2 + 1e-20f);
    out[0] = p0 * inv;
    out[1] = p1 * inv;
    out[2] = p2 * inv;
    return;
  }
  out[0] = normal[0];
  out[1] = normal[1];
  out[2] = normal[2];
}

// Row-major coordinates for every pixel of a width x height grid.
void BuildFullGridCoords(int width, int height, std::vector<int>& rows,
                         std::vector<int>& cols) {
  const size_t N = static_cast<size_t>(width) * height;
  rows.resize(N);
  cols.resize(N);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      rows[static_cast<size_t>(r) * width + c] = r;
      cols[static_cast<size_t>(r) * width + c] = c;
    }
  }
}

}  // namespace

// Forward declaration; defined after the public optimizer wrappers below.
static bool OptimizeWithEval(
    int W, int Hh, const float ref_inv_K[4], float depth_min, float depth_max,
    int num_iterations, float* io_depth, float* io_normal,
    const std::function<bool(const std::vector<float>&,
                             const std::vector<float>&, std::vector<float>&)>&
        eval);

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
                                 float* io_normal) {
  if (!IsPatchMatchMetalAvailable() || width <= 0 || height <= 0 ||
      num_iterations <= 0 || depth_min <= 0.0f || depth_max <= depth_min) {
    return false;
  }
  std::vector<int> rows, cols;
  BuildFullGridCoords(width, height, rows, cols);
  auto eval = [&](const std::vector<float>& d, const std::vector<float>& n,
                  std::vector<float>& cost) -> bool {
    return EvaluateHypothesisCostMetal(
        ref_image, src_image, width, height, ref_inv_K, src_K, src_R, src_T,
        window_radius, window_step, sigma_spatial, sigma_color, rows.data(),
        cols.data(), d.data(), n.data(), static_cast<int>(rows.size()),
        cost.data());
  };
  return OptimizeWithEval(width, height, ref_inv_K, depth_min, depth_max,
                          num_iterations, io_depth, io_normal, eval);
}

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
                                          float* io_normal) {
  if (!IsPatchMatchMetalAvailable() || num_src <= 0 ||
      num_src > kMaxMetalMvsViews || width <= 0 ||
      height <= 0 || num_iterations <= 0 || depth_min <= 0.0f ||
      depth_max <= depth_min) {
    return false;
  }
  std::vector<int> rows, cols;
  BuildFullGridCoords(width, height, rows, cols);
  auto eval = [&](const std::vector<float>& d, const std::vector<float>& n,
                  std::vector<float>& cost) -> bool {
    return EvaluateHypothesisCostMultiViewMetal(
        ref_image, src_images, num_src, width, height, ref_inv_K, src_K, src_R,
        src_T, best_k, window_radius, window_step, sigma_spatial, sigma_color,
        rows.data(), cols.data(), d.data(), n.data(),
        static_cast<int>(rows.size()), cost.data());
  };
  return OptimizeWithEval(width, height, ref_inv_K, depth_min, depth_max,
                          num_iterations, io_depth, io_normal, eval);
}

// Shared propagate-then-refine loop. `eval(depth, normal, cost)` fills the
// per-pixel cost of a full-grid hypothesis field; the single- and multi-view
// optimizers above differ only in that callback, so the loop lives here once.
static bool OptimizeWithEval(
    int W, int Hh, const float ref_inv_K[4], float depth_min, float depth_max,
    int num_iterations, float* io_depth, float* io_normal,
    const std::function<bool(const std::vector<float>&,
                             const std::vector<float>&, std::vector<float>&)>&
        eval) {
  const size_t N = static_cast<size_t>(W) * Hh;
  // Fixed-seed PRNG: the stochastic restarts below mirror COLMAP's random
  // sampling (same algorithm) but are reproducible for tests; the exact curand
  // sequence is not reproduced (and need not be -- acceptance is greedy/monotone).
  std::mt19937 rng(0x5eed1234u);
  std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

  std::vector<float> best_depth(io_depth, io_depth + N);
  std::vector<float> best_normal(io_normal, io_normal + 3 * N);
  std::vector<float> best_cost(N);
  if (!eval(best_depth, best_normal, best_cost)) return false;

  std::vector<float> cand_depth(N), cand_normal(3 * N), cand_cost(N);
  // Snapshot of the maps at the start of an iteration (propagation reads these
  // so acceptance order within an iteration does not matter).
  std::vector<float> snap_depth(N), snap_normal(3 * N);

  auto accept = [&]() {
    for (size_t i = 0; i < N; ++i) {
      if (cand_cost[i] < best_cost[i]) {
        best_cost[i] = cand_cost[i];
        best_depth[i] = cand_depth[i];
        best_normal[3 * i + 0] = cand_normal[3 * i + 0];
        best_normal[3 * i + 1] = cand_normal[3 * i + 1];
        best_normal[3 * i + 2] = cand_normal[3 * i + 2];
      }
    }
  };
  auto copy_best_to_cand = [&]() {
    cand_depth = best_depth;
    cand_normal = best_normal;
  };

  for (int iter = 0; iter < num_iterations; ++iter) {
    snap_depth = best_depth;
    snap_normal = best_normal;
    const float pert = 0.20f * std::pow(0.6f, static_cast<float>(iter));

    // --- Spatial propagation from the 4 neighbors (read from the snapshot). ---
    const int dctab[4] = {-1, 1, 0, 0};
    const int drtab[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
      copy_best_to_cand();
      for (int r = 0; r < Hh; ++r) {
        for (int c = 0; c < W; ++c) {
          const int nc = c + dctab[k], nr = r + drtab[k];
          if (nc < 0 || nc >= W || nr < 0 || nr >= Hh) continue;
          const size_t i = static_cast<size_t>(r) * W + c;
          const size_t j = static_cast<size_t>(nr) * W + nc;
          const float nn[3] = {snap_normal[3 * j + 0], snap_normal[3 * j + 1],
                               snap_normal[3 * j + 2]};
          float prop;
          if (drtab[k] != 0) {
            prop = PropagateDepthRow(ref_inv_K, snap_depth[j], nn,
                                     static_cast<float>(nr),
                                     static_cast<float>(r));
          } else {
            prop = PropagateDepthCol(ref_inv_K, snap_depth[j], nn,
                                     static_cast<float>(nc),
                                     static_cast<float>(c));
          }
          if (prop < depth_min || prop > depth_max) continue;
          cand_depth[i] = prop;
          cand_normal[3 * i + 0] = nn[0];
          cand_normal[3 * i + 1] = nn[1];
          cand_normal[3 * i + 2] = nn[2];
        }
      }
      if (!eval(cand_depth, cand_normal, cand_cost)) return false;
      accept();
    }

    // --- Depth refinement: multiplicative perturbation down/up. ---
    for (int s = 0; s < 2; ++s) {
      const float factor = (s == 0) ? (1.0f - pert) : (1.0f + pert);
      copy_best_to_cand();
      for (size_t i = 0; i < N; ++i) {
        const float d = best_depth[i] * factor;
        if (d >= depth_min && d <= depth_max) cand_depth[i] = d;
      }
      if (!eval(cand_depth, cand_normal, cand_cost)) return false;
      accept();
    }

    // --- Normal refinement: COLMAP-style PerturbNormal (random 3-axis rotation
    //     of the CURRENT normal), keeping depth fixed. Anchored to the current
    //     estimate and wide early (so a fronto-parallel normal can swing toward a
    //     steeply slanted surface), shrinking to local refinement -- unlike the
    //     fully-random {keep-depth, rand-n} candidate above, this converges. Two
    //     independent draws per iteration. perturbation in radians, schedule
    //     from the iteration index (general, not dataset-tuned). ---
    const float normal_pert = 2.0f * std::pow(0.5f, static_cast<float>(iter));
    for (int s = 0; s < 2; ++s) {
      copy_best_to_cand();
      for (size_t i = 0; i < N; ++i) {
        const int r = static_cast<int>(i / W);
        const int c = static_cast<int>(i % W);
        float pn[3];
        PerturbNormal(rng, ref_inv_K, r, c, &best_normal[3 * i], normal_pert,
                      pn);
        cand_normal[3 * i + 0] = pn[0];
        cand_normal[3 * i + 1] = pn[1];
        cand_normal[3 * i + 2] = pn[2];
      }
      if (!eval(cand_depth, cand_normal, cand_cost)) return false;
      accept();
    }

    // --- Stochastic restart: a fully random plane (uniform depth + Marsaglia
    //     normal) per pixel, as in COLMAP's random sampling. Greedy acceptance
    //     means this can only escape local minima, never worsen the estimate. ---
    {
      copy_best_to_cand();
      for (size_t i = 0; i < N; ++i) {
        const int r = static_cast<int>(i / W);
        const int c = static_cast<int>(i % W);
        cand_depth[i] = depth_min + uni01(rng) * (depth_max - depth_min);
        float rn[3];
        RandomNormal(rng, ref_inv_K, r, c, rn);
        cand_normal[3 * i + 0] = rn[0];
        cand_normal[3 * i + 1] = rn[1];
        cand_normal[3 * i + 2] = rn[2];
      }
      if (!eval(cand_depth, cand_normal, cand_cost)) return false;
      accept();
    }

    // --- Decoupled normal resample: KEEP the (already well-converged) depth and
    //     draw a fresh full-hemisphere normal. This is COLMAP's {curr_d, rand_n}
    //     candidate (patch_match_cuda.cu): the joint random restart above pairs a
    //     good normal with a random (usually wrong) depth so it is rejected and
    //     the normal can never improve independently -- this candidate lets a
    //     correct slanted normal be accepted while the good depth is held fixed.
    //     General PatchMatch move (Bleyer 2011 / Schonberger 2016), not tuned to
    //     any dataset. ---
    {
      copy_best_to_cand();  // cand_depth = best_depth (kept).
      for (size_t i = 0; i < N; ++i) {
        const int r = static_cast<int>(i / W);
        const int c = static_cast<int>(i % W);
        float rn[3];
        RandomNormal(rng, ref_inv_K, r, c, rn);
        cand_normal[3 * i + 0] = rn[0];
        cand_normal[3 * i + 1] = rn[1];
        cand_normal[3 * i + 2] = rn[2];
      }
      if (!eval(cand_depth, cand_normal, cand_cost)) return false;
      accept();
    }
  }

  std::copy(best_depth.begin(), best_depth.end(), io_depth);
  std::copy(best_normal.begin(), best_normal.end(), io_normal);
  return true;
}

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
                                     float* out_normal) {
  if (!IsPatchMatchMetalAvailable() || num_src <= 0 ||
      num_src > kMaxMetalMvsViews || width <= 0 ||
      height <= 0 || depth_min <= 0.0f || depth_max <= depth_min) {
    return false;
  }
  // Reference inverse intrinsics from the 3x3 K (row-major: fx,_,cx,_,fy,cy).
  const float fxr = ref_K9[0], cxr = ref_K9[2], fyr = ref_K9[4], cyr = ref_K9[5];
  const float ref_inv_K[4] = {1.0f / fxr, -cxr / fxr, 1.0f / fyr, -cyr / fyr};

  // Per-source relative pose (reference -> source) via COLMAP's convention, and
  // the source intrinsics packed as {fx, cx, fy, cy}.
  std::vector<float> src_K(static_cast<size_t>(num_src) * 4);
  std::vector<float> src_R(static_cast<size_t>(num_src) * 9);
  std::vector<float> src_T(static_cast<size_t>(num_src) * 3);
  for (int m = 0; m < num_src; ++m) {
    float Rrel[9], Trel[3];
    ComputeRelativePose(ref_R9, ref_T3, src_R9 + m * 9, src_T3 + m * 3, Rrel,
                        Trel);
    for (int i = 0; i < 9; ++i) src_R[m * 9 + i] = Rrel[i];
    for (int i = 0; i < 3; ++i) src_T[m * 3 + i] = Trel[i];
    const float* K = src_K9 + m * 9;
    src_K[m * 4 + 0] = K[0];  // fx
    src_K[m * 4 + 1] = K[2];  // cx
    src_K[m * 4 + 2] = K[4];  // fy
    src_K[m * 4 + 3] = K[5];  // cy
  }

  // Initialize at the middle of the depth range, fronto-parallel.
  const size_t N = static_cast<size_t>(width) * height;
  const float d0 = 0.5f * (depth_min + depth_max);
  for (size_t i = 0; i < N; ++i) {
    out_depth[i] = d0;
    out_normal[3 * i + 0] = 0.0f;
    out_normal[3 * i + 1] = 0.0f;
    out_normal[3 * i + 2] = -1.0f;
  }

  if (!RunPatchMatchMultiViewOptimizerMetal(
          ref_gray, src_gray, num_src, width, height, ref_inv_K, src_K.data(),
          src_R.data(), src_T.data(), best_k, window_radius, window_step,
          sigma_spatial, sigma_color, depth_min, depth_max, num_iterations,
          out_depth, out_normal)) {
    return false;
  }

  // Photometric-confidence filtering (COLMAP's filter_min_ncc): re-score the
  // final field and invalidate (depth = 0) pixels whose NCC is below the
  // threshold, mirroring COLMAP's low-confidence rejection. cost = 1 - NCC, so
  // keep iff cost <= 1 - filter_min_ncc. <= 0 disables filtering.
  if (filter_min_ncc > 0.0f) {
    std::vector<int> rows, cols;
    BuildFullGridCoords(width, height, rows, cols);
    std::vector<float> cost(N);
    if (EvaluateHypothesisCostMultiViewMetal(
            ref_gray, src_gray, num_src, width, height, ref_inv_K, src_K.data(),
            src_R.data(), src_T.data(), best_k, window_radius, window_step,
            sigma_spatial, sigma_color, rows.data(), cols.data(), out_depth,
            out_normal, static_cast<int>(N), cost.data())) {
      const float max_cost = 1.0f - filter_min_ncc;
      for (size_t i = 0; i < N; ++i) {
        if (cost[i] > max_cost) {
          out_depth[i] = 0.0f;  // invalid.
          out_normal[3 * i + 0] = 0.0f;
          out_normal[3 * i + 1] = 0.0f;
          out_normal[3 * i + 2] = 0.0f;
        }
      }
    }
  }
  return true;
}

}  // namespace mvs
}  // namespace colmap
