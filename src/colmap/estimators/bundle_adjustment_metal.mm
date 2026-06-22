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

#include "colmap/estimators/bundle_adjustment_metal.h"

#include "colmap/util/logging.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>
#include <cstring>

namespace colmap {
namespace {

// One thread per observation. Computes the Pinhole reprojection residual and the
// analytic Jacobian blocks w.r.t. point3D, cam_from_world (quaternion|trans) and
// camera params -- a verbatim port of the math in
// estimators/cost_functions/reprojection_error.h. The rotation R(q)*p uses the
// same quaternion formula Ceres autodiff differentiates, so the analytic
// Jacobian matches the autodiff Jacobian (up to fp32 rounding).
constexpr char kReprojErrorSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Smallest camera-frame depth w treated as "in front of the camera". Mirrors
// PinholeCameraModel::ImgFromCam, which rejects w < numeric_limits<T>::epsilon().
// These kernels are fp32, so the faithful analogue is the float epsilon
// (FLT_EPSILON ~1.19e-7); the CPU/Ceres reference runs in fp64 (epsilon
// ~2.2e-16). Points whose depth lies in the narrow band between the two are the
// ONLY place the GPU and CPU can disagree on validity -- an inherent fp32 limit,
// NOT the much wider 1e-6 cutoff this used to apply.
constant float kMinValidDepth = 1.1920929e-7f;  // FLT_EPSILON.

// Shared forward projection + projection-Jacobian core for the Pinhole model.
// ALL reprojection kernels below go through this, so the math AND the
// behind-camera guard live in exactly one place. valid=false means the point is
// at/behind the camera (w < kMinValidDepth), mirroring
// PinholeCameraModel::ImgFromCam returning false. For an invalid observation the
// LINEARIZED contribution is zeroed (residual + Jacobian = 0): a behind-camera
// term has no meaningful first-order expansion, and zeroing avoids inf/nan. The
// NONLINEAR cost of an invalid observation is handled separately by PointCost (a
// large penalty), so an LM trial that pushes a point behind a camera is still
// rejected. The two policies are consistent and intentional: feasible terms
// drive the step through their Jacobian, while an infeasible term contributes no
// gradient pull but blocks acceptance through the cost.
struct ReprojCore {
  bool valid;
  float u, v, w, inv_w;
  float r0, r1;            // residual.
  float M[9];              // R(q), row-major.
  float jpx0, jpx2, jpy1, jpy2;  // nonzero entries of d(img)/d(p_cam).
};

static ReprojCore ComputeReprojCore(float qx, float qy, float qz, float qw,
                                    float tx, float ty, float tz,
                                    float f1, float f2, float c1, float c2,
                                    float X, float Y, float Z,
                                    float ox, float oy) {
  ReprojCore c;
  const float vxp0 = qy * Z - qz * Y;
  const float vxp1 = qz * X - qx * Z;
  const float vxp2 = qx * Y - qy * X;
  c.u = X + 2.0f * (qw * vxp0 + (qy * vxp2 - qz * vxp1)) + tx;
  c.v = Y + 2.0f * (qw * vxp1 + (qz * vxp0 - qx * vxp2)) + ty;
  c.w = Z + 2.0f * (qw * vxp2 + (qx * vxp1 - qy * vxp0)) + tz;
  c.valid = c.w >= kMinValidDepth;
  if (!c.valid) {
    c.inv_w = 0.0f;
    c.r0 = 0.0f;
    c.r1 = 0.0f;
    for (int i = 0; i < 9; ++i) c.M[i] = 0.0f;
    c.jpx0 = c.jpx2 = c.jpy1 = c.jpy2 = 0.0f;
    return c;
  }
  c.inv_w = 1.0f / c.w;
  const float inv_w2 = c.inv_w * c.inv_w;
  c.r0 = f1 * c.u * c.inv_w + c1 - ox;
  c.r1 = f2 * c.v * c.inv_w + c2 - oy;
  c.jpx0 = f1 * c.inv_w;
  c.jpx2 = -f1 * c.u * inv_w2;
  c.jpy1 = f2 * c.inv_w;
  c.jpy2 = -f2 * c.v * inv_w2;
  c.M[0] = 1.0f - 2.0f * (qy * qy + qz * qz);
  c.M[1] = 2.0f * (qx * qy - qw * qz);
  c.M[2] = 2.0f * (qx * qz + qw * qy);
  c.M[3] = 2.0f * (qx * qy + qw * qz);
  c.M[4] = 1.0f - 2.0f * (qx * qx + qz * qz);
  c.M[5] = 2.0f * (qy * qz - qw * qx);
  c.M[6] = 2.0f * (qx * qz - qw * qy);
  c.M[7] = 2.0f * (qy * qz + qw * qx);
  c.M[8] = 1.0f - 2.0f * (qx * qx + qy * qy);
  return c;
}

// J_point (2x3, row-major) = J_proj * R(q), written to a device buffer. Shared
// by reproj_error_pinhole and pose_tangent_jac_pinhole so the assembly lives in
// one place. (ReprojPointJac has a thread-memory variant; MSL address spaces
// differ, so the two cannot share a single helper.)
static void WriteJPoint(ReprojCore c, device float* jp) {
  jp[0] = c.jpx0 * c.M[0] + c.jpx2 * c.M[6];
  jp[1] = c.jpx0 * c.M[1] + c.jpx2 * c.M[7];
  jp[2] = c.jpx0 * c.M[2] + c.jpx2 * c.M[8];
  jp[3] = c.jpy1 * c.M[3] + c.jpy2 * c.M[6];
  jp[4] = c.jpy1 * c.M[4] + c.jpy2 * c.M[7];
  jp[5] = c.jpy1 * c.M[5] + c.jpy2 * c.M[8];
}

kernel void reproj_error_pinhole(constant uint& num_obs       [[buffer(0)]],
                                 device const float* poses     [[buffer(1)]],
                                 device const float* points    [[buffer(2)]],
                                 device const float* cams       [[buffer(3)]],
                                 device const float* obs        [[buffer(4)]],
                                 device float* residuals        [[buffer(5)]],
                                 device float* jac_point        [[buffer(6)]],
                                 device float* jac_pose         [[buffer(7)]],
                                 device float* jac_cam          [[buffer(8)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= num_obs) return;
  const float qx = poses[gid * 7 + 0], qy = poses[gid * 7 + 1];
  const float qz = poses[gid * 7 + 2], qw = poses[gid * 7 + 3];
  const float X = points[gid * 3 + 0], Y = points[gid * 3 + 1];
  const float Z = points[gid * 3 + 2];
  const ReprojCore c = ComputeReprojCore(
      qx, qy, qz, qw, poses[gid * 7 + 4], poses[gid * 7 + 5],
      poses[gid * 7 + 6], cams[gid * 4 + 0], cams[gid * 4 + 1],
      cams[gid * 4 + 2], cams[gid * 4 + 3], X, Y, Z, obs[gid * 2 + 0],
      obs[gid * 2 + 1]);
  residuals[gid * 2 + 0] = c.r0;
  residuals[gid * 2 + 1] = c.r1;
  if (!c.valid) {  // point at/behind camera: zero residual + Jacobian.
    for (int k = 0; k < 6; ++k) jac_point[gid * 6 + k] = 0.0f;
    for (int k = 0; k < 14; ++k) jac_pose[gid * 14 + k] = 0.0f;
    for (int k = 0; k < 8; ++k) jac_cam[gid * 8 + k] = 0.0f;
    return;
  }

  WriteJPoint(c, jac_point + gid * 6);  // J_point (2x3) = J_proj * M.

  // d(R(q)*p) / dq (3x4) w.r.t. (qx,qy,qz,qw), original point (X,Y,Z). Matches
  // QuaternionRotatePointWithJac in reprojection_error.h.
  const float Jq00 = 2.0f * (qy * Y + qz * Z);
  const float Jq01 = 2.0f * (-2.0f * qy * X + qx * Y + qw * Z);
  const float Jq02 = 2.0f * (-2.0f * qz * X - qw * Y + qx * Z);
  const float Jq03 = 2.0f * (-qz * Y + qy * Z);
  const float Jq10 = 2.0f * (qy * X - 2.0f * qx * Y - qw * Z);
  const float Jq11 = 2.0f * (qx * X + qz * Z);
  const float Jq12 = 2.0f * (qw * X - 2.0f * qz * Y + qy * Z);
  const float Jq13 = 2.0f * (qz * X - qx * Z);
  const float Jq20 = 2.0f * (qz * X + qw * Y - 2.0f * qx * Z);
  const float Jq21 = 2.0f * (-qw * X + qz * Y - 2.0f * qy * Z);
  const float Jq22 = 2.0f * (qx * X + qy * Y);
  const float Jq23 = 2.0f * (-qy * X + qx * Y);

  // J_pose (2x7) = [J_quat = J_proj*Jq | J_trans = J_proj].
  jac_pose[gid * 14 + 0] = c.jpx0 * Jq00 + c.jpx2 * Jq20;
  jac_pose[gid * 14 + 1] = c.jpx0 * Jq01 + c.jpx2 * Jq21;
  jac_pose[gid * 14 + 2] = c.jpx0 * Jq02 + c.jpx2 * Jq22;
  jac_pose[gid * 14 + 3] = c.jpx0 * Jq03 + c.jpx2 * Jq23;
  jac_pose[gid * 14 + 4] = c.jpx0;
  jac_pose[gid * 14 + 5] = 0.0f;
  jac_pose[gid * 14 + 6] = c.jpx2;
  jac_pose[gid * 14 + 7] = c.jpy1 * Jq10 + c.jpy2 * Jq20;
  jac_pose[gid * 14 + 8] = c.jpy1 * Jq11 + c.jpy2 * Jq21;
  jac_pose[gid * 14 + 9] = c.jpy1 * Jq12 + c.jpy2 * Jq22;
  jac_pose[gid * 14 + 10] = c.jpy1 * Jq13 + c.jpy2 * Jq23;
  jac_pose[gid * 14 + 11] = 0.0f;
  jac_pose[gid * 14 + 12] = c.jpy1;
  jac_pose[gid * 14 + 13] = c.jpy2;

  // J_cam (2x4): d residual / d (f1, f2, c1, c2).
  jac_cam[gid * 8 + 0] = c.u * c.inv_w;
  jac_cam[gid * 8 + 1] = 0.0f;
  jac_cam[gid * 8 + 2] = 1.0f;
  jac_cam[gid * 8 + 3] = 0.0f;
  jac_cam[gid * 8 + 4] = 0.0f;
  jac_cam[gid * 8 + 5] = c.v * c.inv_w;
  jac_cam[gid * 8 + 6] = 0.0f;
  jac_cam[gid * 8 + 7] = 1.0f;
}

// Pinhole reprojection residual (r[2]) and point Jacobian (J row-major 2x3) for
// ONE observation. Returns false if the point is at/behind the source camera.
static bool ReprojPointJac(float qx, float qy, float qz, float qw,
                           float tx, float ty, float tz,
                           float f1, float f2, float c1, float c2,
                           float X, float Y, float Z, float ox, float oy,
                           thread float* r, thread float* J) {
  const ReprojCore c = ComputeReprojCore(qx, qy, qz, qw, tx, ty, tz, f1, f2, c1,
                                         c2, X, Y, Z, ox, oy);
  if (!c.valid) return false;
  r[0] = c.r0;
  r[1] = c.r1;
  // J_point (2x3) = J_proj * M.
  J[0] = c.jpx0 * c.M[0] + c.jpx2 * c.M[6];
  J[1] = c.jpx0 * c.M[1] + c.jpx2 * c.M[7];
  J[2] = c.jpx0 * c.M[2] + c.jpx2 * c.M[8];
  J[3] = c.jpy1 * c.M[3] + c.jpy2 * c.M[6];
  J[4] = c.jpy1 * c.M[4] + c.jpy2 * c.M[7];
  J[5] = c.jpy1 * c.M[5] + c.jpy2 * c.M[8];
  return true;
}

// Sum of squared reprojection residuals of point (X,Y,Z) over its observations.
// An observation that falls at/behind the camera (cheirality violation) is
// charged a large fixed penalty rather than skipped, so an LM trial that pushes
// the point behind its cameras strictly INCREASES the cost and is rejected --
// without this, such a trial would score 0 and be wrongly accepted, committing
// a point behind the cameras.
static float PointCost(uint begin, uint end,
                       device const float* obs_pose,
                       device const float* obs_cam,
                       device const float* obs_pixel,
                       float X, float Y, float Z) {
  constexpr float kCheiralityPenalty = 1e12f;
  float cost = 0.0f;
  float r[2], J[6];
  for (uint o = begin; o < end; ++o) {
    if (ReprojPointJac(obs_pose[o * 7 + 0], obs_pose[o * 7 + 1],
                       obs_pose[o * 7 + 2], obs_pose[o * 7 + 3],
                       obs_pose[o * 7 + 4], obs_pose[o * 7 + 5],
                       obs_pose[o * 7 + 6], obs_cam[o * 4 + 0],
                       obs_cam[o * 4 + 1], obs_cam[o * 4 + 2],
                       obs_cam[o * 4 + 3], X, Y, Z, obs_pixel[o * 2 + 0],
                       obs_pixel[o * 2 + 1], r, J)) {
      cost += r[0] * r[0] + r[1] * r[1];
    } else {
      cost += kCheiralityPenalty;
    }
  }
  return cost;
}

// Structure-only BA: one thread per point runs Levenberg-Marquardt on its 3D
// position (Euclidean, no manifold) minimizing reprojection error over its
// observations. Independent per point -- the GPU-ideal C-block of BA.
kernel void refine_points_pinhole(constant uint& num_points  [[buffer(0)]],
                                  constant uint& max_iters    [[buffer(1)]],
                                  device float* point3D       [[buffer(2)]],
                                  device const int* obs_offset [[buffer(3)]],
                                  device const float* obs_pose [[buffer(4)]],
                                  device const float* obs_cam  [[buffer(5)]],
                                  device const float* obs_pixel [[buffer(6)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid >= num_points) return;
  const uint begin = (uint)obs_offset[gid];
  const uint end = (uint)obs_offset[gid + 1];
  if (end <= begin) return;  // unobserved point: leave unchanged.

  float X = point3D[gid * 3 + 0];
  float Y = point3D[gid * 3 + 1];
  float Z = point3D[gid * 3 + 2];
  float cost = PointCost(begin, end, obs_pose, obs_cam, obs_pixel, X, Y, Z);
  float lambda = 1e-3f;

  for (uint it = 0; it < max_iters; ++it) {
    // Assemble H = sum J^T J (symmetric 3x3) and g = sum J^T r.
    float h00 = 0, h01 = 0, h02 = 0, h11 = 0, h12 = 0, h22 = 0;
    float g0 = 0, g1 = 0, g2 = 0;
    float r[2], J[6];
    for (uint o = begin; o < end; ++o) {
      if (!ReprojPointJac(obs_pose[o * 7 + 0], obs_pose[o * 7 + 1],
                          obs_pose[o * 7 + 2], obs_pose[o * 7 + 3],
                          obs_pose[o * 7 + 4], obs_pose[o * 7 + 5],
                          obs_pose[o * 7 + 6], obs_cam[o * 4 + 0],
                          obs_cam[o * 4 + 1], obs_cam[o * 4 + 2],
                          obs_cam[o * 4 + 3], X, Y, Z, obs_pixel[o * 2 + 0],
                          obs_pixel[o * 2 + 1], r, J)) {
        continue;
      }
      for (int row = 0; row < 2; ++row) {
        const float j0 = J[row * 3 + 0], j1 = J[row * 3 + 1], j2 = J[row * 3 + 2];
        h00 += j0 * j0; h01 += j0 * j1; h02 += j0 * j2;
        h11 += j1 * j1; h12 += j1 * j2; h22 += j2 * j2;
        g0 += j0 * r[row]; g1 += j1 * r[row]; g2 += j2 * r[row];
      }
    }
    if (sqrt(g0 * g0 + g1 * g1 + g2 * g2) < 1e-7f) break;  // converged.

    bool accepted = false;
    for (int tries = 0; tries < 6 && !accepted; ++tries) {
      // A = H + lambda * max(diag(H), floor) (Levenberg-Marquardt with Jacobi
      // scaling). The floor keeps damping effective even when a Hessian diagonal
      // is ~0 (a structurally unconstrained direction); pure relative scaling
      // h_ii + lambda*h_ii would stay singular there for every lambda.
      constexpr float kDiagFloor = 1e-9f;
      const float a00 = h00 + lambda * max(h00, kDiagFloor);
      const float a11 = h11 + lambda * max(h11, kDiagFloor);
      const float a22 = h22 + lambda * max(h22, kDiagFloor);
      const float a01 = h01, a02 = h02, a12 = h12;
      // Symmetric 3x3 inverse via cofactors; solve A dx = -g.
      const float co00 = a11 * a22 - a12 * a12;
      const float co01 = a02 * a12 - a01 * a22;
      const float co02 = a01 * a12 - a02 * a11;
      const float co11 = a00 * a22 - a02 * a02;
      const float co12 = a01 * a02 - a00 * a12;
      const float co22 = a00 * a11 - a01 * a01;
      const float det = a00 * co00 + a01 * co01 + a02 * co02;
      if (fabs(det) < 1e-20f) {
        lambda = min(lambda * 10.0f, 1e12f);
        continue;
      }
      const float inv_det = 1.0f / det;
      const float b0 = -g0, b1 = -g1, b2 = -g2;
      const float dx = (co00 * b0 + co01 * b1 + co02 * b2) * inv_det;
      const float dy = (co01 * b0 + co11 * b1 + co12 * b2) * inv_det;
      const float dz = (co02 * b0 + co12 * b1 + co22 * b2) * inv_det;
      const float Xt = X + dx, Yt = Y + dy, Zt = Z + dz;
      const float cost_t =
          PointCost(begin, end, obs_pose, obs_cam, obs_pixel, Xt, Yt, Zt);
      if (cost_t < cost) {
        X = Xt; Y = Yt; Z = Zt; cost = cost_t;
        lambda = max(lambda * 0.3f, 1e-12f);
        accepted = true;
      } else {
        lambda = min(lambda * 10.0f, 1e12f);
      }
    }
    if (!accepted) break;  // damping could not improve -> at a minimum.
  }

  point3D[gid * 3 + 0] = X;
  point3D[gid * 3 + 1] = Y;
  point3D[gid * 3 + 2] = Z;
}

// Camera-solver factor: residual + pose Jacobian on the SE(3) tangent (right
// perturbation: d p_cam/d dtheta = -R[X]x, d p_cam/d dt = I) + point Jacobian.
// Intrinsics fixed. Blocks for assembling the reduced camera system in full BA.
kernel void pose_tangent_jac_pinhole(constant uint& num_obs   [[buffer(0)]],
                                     device const float* poses [[buffer(1)]],
                                     device const float* points [[buffer(2)]],
                                     device const float* cams   [[buffer(3)]],
                                     device const float* obs    [[buffer(4)]],
                                     device float* residuals    [[buffer(5)]],
                                     device float* jac_cam_tangent [[buffer(6)]],
                                     device float* jac_point    [[buffer(7)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= num_obs) return;
  const float qx = poses[gid * 7 + 0], qy = poses[gid * 7 + 1];
  const float qz = poses[gid * 7 + 2], qw = poses[gid * 7 + 3];
  const float X = points[gid * 3 + 0], Y = points[gid * 3 + 1];
  const float Z = points[gid * 3 + 2];
  const ReprojCore c = ComputeReprojCore(
      qx, qy, qz, qw, poses[gid * 7 + 4], poses[gid * 7 + 5],
      poses[gid * 7 + 6], cams[gid * 4 + 0], cams[gid * 4 + 1],
      cams[gid * 4 + 2], cams[gid * 4 + 3], X, Y, Z, obs[gid * 2 + 0],
      obs[gid * 2 + 1]);
  residuals[gid * 2 + 0] = c.r0;
  residuals[gid * 2 + 1] = c.r1;
  if (!c.valid) {  // point at/behind camera: zero residual + Jacobian.
    for (int k = 0; k < 6; ++k) jac_point[gid * 6 + k] = 0.0f;
    for (int k = 0; k < 12; ++k) jac_cam_tangent[gid * 12 + k] = 0.0f;
    return;
  }

  WriteJPoint(c, jac_point + gid * 6);  // J_point (2x3) = J_proj * M.

  // M*[X]x (3x3), then d p_cam/d dtheta = -(M*[X]x).
  const float MX00 = c.M[1] * Z - c.M[2] * Y;
  const float MX10 = c.M[4] * Z - c.M[5] * Y;
  const float MX20 = c.M[7] * Z - c.M[8] * Y;
  const float MX01 = -c.M[0] * Z + c.M[2] * X;
  const float MX11 = -c.M[3] * Z + c.M[5] * X;
  const float MX21 = -c.M[6] * Z + c.M[8] * X;
  const float MX02 = c.M[0] * Y - c.M[1] * X;
  const float MX12 = c.M[3] * Y - c.M[4] * X;
  const float MX22 = c.M[6] * Y - c.M[7] * X;

  // J_rot (2x3) = J_proj * (-(M*[X]x)).
  const float jr00 = -(c.jpx0 * MX00 + c.jpx2 * MX20);
  const float jr01 = -(c.jpx0 * MX01 + c.jpx2 * MX21);
  const float jr02 = -(c.jpx0 * MX02 + c.jpx2 * MX22);
  const float jr10 = -(c.jpy1 * MX10 + c.jpy2 * MX20);
  const float jr11 = -(c.jpy1 * MX11 + c.jpy2 * MX21);
  const float jr12 = -(c.jpy1 * MX12 + c.jpy2 * MX22);

  // jac_cam_tangent (2x6) = [J_rot (2x3) | J_trans (2x3) = J_proj].
  jac_cam_tangent[gid * 12 + 0] = jr00;
  jac_cam_tangent[gid * 12 + 1] = jr01;
  jac_cam_tangent[gid * 12 + 2] = jr02;
  jac_cam_tangent[gid * 12 + 3] = c.jpx0;
  jac_cam_tangent[gid * 12 + 4] = 0.0f;
  jac_cam_tangent[gid * 12 + 5] = c.jpx2;
  jac_cam_tangent[gid * 12 + 6] = jr10;
  jac_cam_tangent[gid * 12 + 7] = jr11;
  jac_cam_tangent[gid * 12 + 8] = jr12;
  jac_cam_tangent[gid * 12 + 9] = 0.0f;
  jac_cam_tangent[gid * 12 + 10] = c.jpy1;
  jac_cam_tangent[gid * 12 + 11] = c.jpy2;
}
)METAL";

// Device, queue, and compiled pipeline are created once and reused.
struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLComputePipelineState> points_pipeline = nil;
  id<MTLComputePipelineState> tangent_pipeline = nil;
  bool valid = false;

  MetalContext() {
    const char* disable = std::getenv("COLMAP_DISABLE_METAL");
    if (disable != nullptr && disable[0] == '1') return;
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      if (device == nil) return;
      queue = [device newCommandQueue];
      if (queue == nil) return;
      // Log (rather than silently swallow) build failures, so a kernel that
      // fails to compile on some toolchain is distinguishable from a machine
      // that simply has no Metal GPU. A device that is merely absent returns
      // early above without a warning; reaching here means the GPU exists but
      // the PoC pipeline could not be built.
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithSource:[NSString stringWithUTF8String:kReprojErrorSource]
                       options:nil
                         error:&error];
      if (library == nil) {
        LOG(WARNING) << "Metal BA PoC: kernel library failed to compile: "
                     << (error != nil ? error.localizedDescription.UTF8String
                                       : "unknown error");
        return;
      }
      id<MTLFunction> fn = [library newFunctionWithName:@"reproj_error_pinhole"];
      if (fn == nil) return;
      pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
      if (pipeline == nil) {
        LOG(WARNING) << "Metal BA PoC: reproj_error pipeline failed: "
                     << (error != nil ? error.localizedDescription.UTF8String
                                       : "unknown error");
        return;
      }
      id<MTLFunction> pts_fn =
          [library newFunctionWithName:@"refine_points_pinhole"];
      if (pts_fn == nil) return;
      points_pipeline =
          [device newComputePipelineStateWithFunction:pts_fn error:&error];
      if (points_pipeline == nil) {
        LOG(WARNING) << "Metal BA PoC: refine_points pipeline failed: "
                     << (error != nil ? error.localizedDescription.UTF8String
                                       : "unknown error");
        return;
      }
      id<MTLFunction> tan_fn =
          [library newFunctionWithName:@"pose_tangent_jac_pinhole"];
      if (tan_fn == nil) return;
      tangent_pipeline =
          [device newComputePipelineStateWithFunction:tan_fn error:&error];
      if (tangent_pipeline == nil) {
        LOG(WARNING) << "Metal BA PoC: pose_tangent pipeline failed: "
                     << (error != nil ? error.localizedDescription.UTF8String
                                       : "unknown error");
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

// Shared unified-memory buffer creation + 1D dispatch, so the host wrappers
// below don't each re-implement the same scaffolding.
id<MTLBuffer> MakeInBuf(id<MTLDevice> device, const void* ptr, size_t bytes) {
  return [device newBufferWithBytes:ptr
                             length:bytes
                            options:MTLResourceStorageModeShared];
}
id<MTLBuffer> MakeOutBuf(id<MTLDevice> device, size_t bytes) {
  return [device newBufferWithLength:bytes
                             options:MTLResourceStorageModeShared];
}
// Clamp the threadgroup size, dispatch `threads` threads, commit and block.
// Returns false if the command buffer did not complete.
bool Dispatch1DAndWait(id<MTLCommandBuffer> cmd,
                       id<MTLComputeCommandEncoder> enc,
                       id<MTLComputePipelineState> pipeline,
                       NSUInteger threads) {
  NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
  if (tg > threads) tg = threads;
  if (tg == 0) tg = 1;
  [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  return cmd.status == MTLCommandBufferStatusCompleted;
}

}  // namespace

bool IsBundleAdjustmentMetalAvailable() { return GetContext().valid; }

bool ComputeReprojErrorMetalPinhole(int num_obs,
                                    const float* poses,
                                    const float* points,
                                    const float* cams,
                                    const float* obs,
                                    float* residuals,
                                    float* jac_point,
                                    float* jac_pose,
                                    float* jac_cam) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num_obs <= 0 || poses == nullptr || points == nullptr ||
      cams == nullptr || obs == nullptr || residuals == nullptr ||
      jac_point == nullptr || jac_pose == nullptr || jac_cam == nullptr) {
    return false;
  }

  @autoreleasepool {
    const NSUInteger n = static_cast<NSUInteger>(num_obs);
    id<MTLDevice> dev = ctx.device;
    id<MTLBuffer> poses_buf = MakeInBuf(dev, poses, n * 7 * sizeof(float));
    id<MTLBuffer> points_buf = MakeInBuf(dev, points, n * 3 * sizeof(float));
    id<MTLBuffer> cams_buf = MakeInBuf(dev, cams, n * 4 * sizeof(float));
    id<MTLBuffer> obs_buf = MakeInBuf(dev, obs, n * 2 * sizeof(float));
    id<MTLBuffer> res_buf = MakeOutBuf(dev, n * 2 * sizeof(float));
    id<MTLBuffer> jp_buf = MakeOutBuf(dev, n * 6 * sizeof(float));
    id<MTLBuffer> jpose_buf = MakeOutBuf(dev, n * 14 * sizeof(float));
    id<MTLBuffer> jcam_buf = MakeOutBuf(dev, n * 8 * sizeof(float));
    if (poses_buf == nil || points_buf == nil || cams_buf == nil ||
        obs_buf == nil || res_buf == nil || jp_buf == nil || jpose_buf == nil ||
        jcam_buf == nil) {
      return false;
    }

    uint32_t num_obs_u32 = static_cast<uint32_t>(num_obs);
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.pipeline];
    [enc setBytes:&num_obs_u32 length:sizeof(num_obs_u32) atIndex:0];
    [enc setBuffer:poses_buf offset:0 atIndex:1];
    [enc setBuffer:points_buf offset:0 atIndex:2];
    [enc setBuffer:cams_buf offset:0 atIndex:3];
    [enc setBuffer:obs_buf offset:0 atIndex:4];
    [enc setBuffer:res_buf offset:0 atIndex:5];
    [enc setBuffer:jp_buf offset:0 atIndex:6];
    [enc setBuffer:jpose_buf offset:0 atIndex:7];
    [enc setBuffer:jcam_buf offset:0 atIndex:8];
    if (!Dispatch1DAndWait(cmd, enc, ctx.pipeline, n)) return false;

    std::memcpy(residuals, [res_buf contents], n * 2 * sizeof(float));
    std::memcpy(jac_point, [jp_buf contents], n * 6 * sizeof(float));
    std::memcpy(jac_pose, [jpose_buf contents], n * 14 * sizeof(float));
    std::memcpy(jac_cam, [jcam_buf contents], n * 8 * sizeof(float));
  }
  return true;
}

bool RefinePointsMetalPinhole(int num_points,
                              int num_obs,
                              float* point3D,
                              const int* obs_offset,
                              const float* obs_pose,
                              const float* obs_cam,
                              const float* obs_pixel,
                              int max_iters) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num_points <= 0 || num_obs <= 0 || max_iters <= 0 ||
      point3D == nullptr || obs_offset == nullptr || obs_pose == nullptr ||
      obs_cam == nullptr || obs_pixel == nullptr) {
    return false;
  }
  // Validate the CSR offsets on the host. The kernel casts each obs_offset
  // entry to uint and indexes obs_pose/obs_cam/obs_pixel with it; a negative,
  // out-of-range, or non-monotonic offset would read out of bounds on the GPU
  // (and a descending pair underflows the unsigned cast to ~4e9). Requiring
  // 0 <= obs_offset[i] <= obs_offset[i+1] <= num_obs makes every index in
  // [begin, end) provably < num_obs.
  if (obs_offset[0] < 0 || obs_offset[num_points] > num_obs) {
    return false;
  }
  for (int i = 0; i < num_points; ++i) {
    if (obs_offset[i] > obs_offset[i + 1]) {
      return false;
    }
  }

  @autoreleasepool {
    const NSUInteger p = static_cast<NSUInteger>(num_points);
    const NSUInteger m = static_cast<NSUInteger>(num_obs);
    id<MTLDevice> dev = ctx.device;
    // point3D is read-write (optimized in place).
    id<MTLBuffer> pts_buf = MakeInBuf(dev, point3D, p * 3 * sizeof(float));
    id<MTLBuffer> off_buf = MakeInBuf(dev, obs_offset, (p + 1) * sizeof(int));
    id<MTLBuffer> pose_buf = MakeInBuf(dev, obs_pose, m * 7 * sizeof(float));
    id<MTLBuffer> cam_buf = MakeInBuf(dev, obs_cam, m * 4 * sizeof(float));
    id<MTLBuffer> pix_buf = MakeInBuf(dev, obs_pixel, m * 2 * sizeof(float));
    if (pts_buf == nil || off_buf == nil || pose_buf == nil || cam_buf == nil ||
        pix_buf == nil) {
      return false;
    }

    uint32_t num_points_u32 = static_cast<uint32_t>(num_points);
    uint32_t max_iters_u32 = static_cast<uint32_t>(max_iters);
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.points_pipeline];
    [enc setBytes:&num_points_u32 length:sizeof(num_points_u32) atIndex:0];
    [enc setBytes:&max_iters_u32 length:sizeof(max_iters_u32) atIndex:1];
    [enc setBuffer:pts_buf offset:0 atIndex:2];
    [enc setBuffer:off_buf offset:0 atIndex:3];
    [enc setBuffer:pose_buf offset:0 atIndex:4];
    [enc setBuffer:cam_buf offset:0 atIndex:5];
    [enc setBuffer:pix_buf offset:0 atIndex:6];
    if (!Dispatch1DAndWait(cmd, enc, ctx.points_pipeline, p)) return false;

    std::memcpy(point3D, [pts_buf contents], p * 3 * sizeof(float));
  }
  return true;
}

bool ComputePoseTangentJacMetalPinhole(int num_obs,
                                       const float* poses,
                                       const float* points,
                                       const float* cams,
                                       const float* obs,
                                       float* residuals,
                                       float* jac_cam_tangent,
                                       float* jac_point) {
  MetalContext& ctx = GetContext();
  if (!ctx.valid || num_obs <= 0 || poses == nullptr || points == nullptr ||
      cams == nullptr || obs == nullptr || residuals == nullptr ||
      jac_cam_tangent == nullptr || jac_point == nullptr) {
    return false;
  }
  @autoreleasepool {
    const NSUInteger n = static_cast<NSUInteger>(num_obs);
    id<MTLDevice> dev = ctx.device;
    id<MTLBuffer> poses_buf = MakeInBuf(dev, poses, n * 7 * sizeof(float));
    id<MTLBuffer> points_buf = MakeInBuf(dev, points, n * 3 * sizeof(float));
    id<MTLBuffer> cams_buf = MakeInBuf(dev, cams, n * 4 * sizeof(float));
    id<MTLBuffer> obs_buf = MakeInBuf(dev, obs, n * 2 * sizeof(float));
    id<MTLBuffer> res_buf = MakeOutBuf(dev, n * 2 * sizeof(float));
    id<MTLBuffer> jct_buf = MakeOutBuf(dev, n * 12 * sizeof(float));
    id<MTLBuffer> jp_buf = MakeOutBuf(dev, n * 6 * sizeof(float));
    if (poses_buf == nil || points_buf == nil || cams_buf == nil ||
        obs_buf == nil || res_buf == nil || jct_buf == nil || jp_buf == nil) {
      return false;
    }
    uint32_t num_obs_u32 = static_cast<uint32_t>(num_obs);
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx.tangent_pipeline];
    [enc setBytes:&num_obs_u32 length:sizeof(num_obs_u32) atIndex:0];
    [enc setBuffer:poses_buf offset:0 atIndex:1];
    [enc setBuffer:points_buf offset:0 atIndex:2];
    [enc setBuffer:cams_buf offset:0 atIndex:3];
    [enc setBuffer:obs_buf offset:0 atIndex:4];
    [enc setBuffer:res_buf offset:0 atIndex:5];
    [enc setBuffer:jct_buf offset:0 atIndex:6];
    [enc setBuffer:jp_buf offset:0 atIndex:7];
    if (!Dispatch1DAndWait(cmd, enc, ctx.tangent_pipeline, n)) return false;
    std::memcpy(residuals, [res_buf contents], n * 2 * sizeof(float));
    std::memcpy(jac_cam_tangent, [jct_buf contents], n * 12 * sizeof(float));
    std::memcpy(jac_point, [jp_buf contents], n * 6 * sizeof(float));
  }
  return true;
}

}  // namespace colmap
