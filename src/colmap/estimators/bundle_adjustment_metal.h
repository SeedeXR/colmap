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

// Caspar-style Metal bundle-adjustment proof-of-concept (branch: caspar-style).
//
// Goal of this PoC is NOT a full BA solver. It answers the prerequisite
// question for a Metal BA accelerator: can the Apple GPU compute COLMAP's
// reprojection factor -- the per-observation residual AND its analytic
// Jacobians -- with output equivalent to the CPU/Ceres path, and faster when
// batched over many observations? This is the universal factor in COLMAP BA
// (every dataset/camera uses it), so a positive result generalizes.
//
// The interface is free of Metal/Objective-C types so it is includable from any
// translation unit; the implementation lives in bundle_adjustment_metal.mm,
// compiled on an Apple COLMAP_METAL_ENABLED build (stub otherwise) -- and only
// when the build opts in via the CMake option BA_METAL_POC_ENABLED (default
// OFF). A normal build, including a normal Apple Metal build, does not compile
// this PoC or its test at all.
//
// STATUS / LIMITATIONS (read before using):
//   * EXPERIMENTAL, research-only. These functions are NOT wired into COLMAP's
//     BA pipeline (bundle_adjustment.cc never calls them); nothing in the
//     product dispatches here. The end-to-end solver that ties them together
//     (Schur/Power-BA LM) currently lives only in bundle_adjustment_metal_test
//     and is not exposed as a callable library entry point. Decision on the
//     caspar-style branch: equivalent output to Ceres is proven, no speed win
//     is demonstrated, so this stays out of production (Ceres remains the BA).
//   * Pinhole model only, INTRINSICS FIXED, NO robust loss (raw squared error).
//     Real COLMAP BA handles distorted models, refines intrinsics, and uses a
//     robust loss by default -- so equivalence to Ceres holds on clean data but
//     would diverge on real data with outliers / other camera models.
//   * fp32 throughout (matches the CUDA Caspar f32 path); adequate on the
//     validated problems but an implicit ceiling on ill-conditioned/large
//     scenes where Ceres' fp64 keeps precision this loses.
//   * Each call is stateless: it allocates fresh (shared) MTLBuffers and copies
//     inputs in / outputs out. An iterative caller (e.g. the LM solver in the
//     test) therefore reallocates per iteration even for inputs that do not
//     change. A production design would keep persistent shared buffers and
//     write in place; that is intentionally NOT done here -- this PoC measures
//     batched factor-evaluation throughput, not end-to-end solver wall-time.

#include <cstddef>

namespace colmap {

// Whether a usable Metal GPU is available at runtime. Always false in builds
// without COLMAP_METAL_ENABLED.
bool IsBundleAdjustmentMetalAvailable();

// Batched reprojection residual + analytic Jacobians for the Pinhole camera
// model, evaluated on the Metal GPU (one thread per observation). Mirrors
// ReprojErrorCostFunctor<PinholeCameraModel>:
//   p_cam   = R(q) * X_world + t
//   img     = (fx * p_cam.x / p_cam.z + cx, fy * p_cam.y / p_cam.z + cy)
//   residual = img - observed_pixel
//
// All buffers are caller-owned, fp32, row-major, indexed by observation:
//   poses  [7*N]: (qx, qy, qz, qw, tx, ty, tz)  -- cam_from_world (Eigen quat)
//   points [3*N]: (X, Y, Z) in world
//   cams   [4*N]: (fx, fy, cx, cy)
//   obs    [2*N]: observed pixel (x, y)
// Outputs (caller-allocated):
//   residuals [2*N]
//   jac_point [6*N] : 2x3 row-major, d residual / d point3D_in_world
//   jac_pose  [14*N]: 2x7 row-major, d residual / d cam_from_world (quat|trans)
//   jac_cam   [8*N] : 2x4 row-major, d residual / d camera_params
// The Jacobian block order matches the ReprojErrorCostFunctor parameter order
// (point3D_in_world, cam_from_world, camera_params). Returns false if Metal is
// unavailable or inputs are invalid.
bool ComputeReprojErrorMetalPinhole(int num_obs,
                                    const float* poses,
                                    const float* points,
                                    const float* cams,
                                    const float* obs,
                                    float* residuals,
                                    float* jac_point,
                                    float* jac_pose,
                                    float* jac_cam);

// Structure-only (point) bundle adjustment on the Metal GPU: with cameras
// fixed, refine each 3D point by Levenberg-Marquardt minimizing its
// reprojection error over the observations that see it (Pinhole model). One GPU
// thread per point -- the points are independent (no shared state, no manifold;
// points are Euclidean), so this is the embarrassingly-parallel structure half
// of BA and the per-point C-block that a Schur/Power-BA camera solver
// eliminates.
//
// Observations are stored CSR-style, grouped by point:
//   point3D    [3*num_points] : INOUT, optimized in place (X,Y,Z world)
//   obs_offset [num_points+1] : point p's obs span [obs_offset[p], offset[p+1])
//   obs_pose   [7*num_obs]    : (qx,qy,qz,qw,tx,ty,tz) cam_from_world per obs
//   obs_cam    [4*num_obs]    : (fx,fy,cx,cy) per obs
//   obs_pixel  [2*num_obs]    : observed pixel (x,y) per obs
// max_iters bounds the per-point LM iterations. Returns false if Metal is
// unavailable or inputs are invalid.
bool RefinePointsMetalPinhole(int num_points,
                              int num_obs,
                              float* point3D,
                              const int* obs_offset,
                              const float* obs_pose,
                              const float* obs_cam,
                              const float* obs_pixel,
                              int max_iters);

// Camera-solver factor (full BA): per-observation reprojection residual, the
// pose Jacobian on the SE(3) tangent, and the point Jacobian (Pinhole,
// intrinsics fixed). The pose tangent uses the right perturbation
// q <- q (x) exp(dtheta), t <- t + dt (6-DoF, dtheta first), so
//   d p_cam / d dtheta = -R(q) [X]_x ,  d p_cam / d dt = I.
// These are the blocks needed to assemble the reduced camera system (Schur
// complement) for a Metal camera/Power-BA solver. Layout per observation:
//   poses [7*N], points [3*N], cams [4*N], obs [2*N]  (as above)
//   residuals        [2*N]
//   jac_cam_tangent  [12*N] : 2x6 row-major, d residual / d (dtheta, dt)
//   jac_point        [6*N]  : 2x3 row-major, d residual / d point
// Returns false if Metal is unavailable or inputs are invalid.
bool ComputePoseTangentJacMetalPinhole(int num_obs,
                                       const float* poses,
                                       const float* points,
                                       const float* cams,
                                       const float* obs,
                                       float* residuals,
                                       float* jac_cam_tangent,
                                       float* jac_point);

}  // namespace colmap
