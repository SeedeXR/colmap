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

  const float qx = poses[gid * 7 + 0];
  const float qy = poses[gid * 7 + 1];
  const float qz = poses[gid * 7 + 2];
  const float qw = poses[gid * 7 + 3];
  const float tx = poses[gid * 7 + 4];
  const float ty = poses[gid * 7 + 5];
  const float tz = poses[gid * 7 + 6];
  const float X = points[gid * 3 + 0];
  const float Y = points[gid * 3 + 1];
  const float Z = points[gid * 3 + 2];
  const float f1 = cams[gid * 4 + 0];
  const float f2 = cams[gid * 4 + 1];
  const float c1 = cams[gid * 4 + 2];
  const float c2 = cams[gid * 4 + 3];
  const float ox = obs[gid * 2 + 0];
  const float oy = obs[gid * 2 + 1];

  // p_cam = R(q) * p + t, with R(q)*p = p + 2w(v x p) + 2 v x (v x p).
  const float vxp0 = qy * Z - qz * Y;
  const float vxp1 = qz * X - qx * Z;
  const float vxp2 = qx * Y - qy * X;
  const float vvxp0 = qy * vxp2 - qz * vxp1;
  const float vvxp1 = qz * vxp0 - qx * vxp2;
  const float vvxp2 = qx * vxp1 - qy * vxp0;
  const float u = X + 2.0f * (qw * vxp0 + vvxp0) + tx;
  const float v = Y + 2.0f * (qw * vxp1 + vvxp1) + ty;
  const float w = Z + 2.0f * (qw * vxp2 + vvxp2) + tz;

  const float inv_w = 1.0f / w;
  const float inv_w2 = inv_w * inv_w;

  // Residual = pinhole projection - observation.
  residuals[gid * 2 + 0] = f1 * u * inv_w + c1 - ox;
  residuals[gid * 2 + 1] = f2 * v * inv_w + c2 - oy;

  // d residual / d p_cam (2x3 projection Jacobian).
  const float jpx0 = f1 * inv_w;            // d x / d u
  const float jpx2 = -f1 * u * inv_w2;      // d x / d w
  const float jpy1 = f2 * inv_w;            // d y / d v
  const float jpy2 = -f2 * v * inv_w2;      // d y / d w

  // d p_cam / d point = R(q) = I + 2w[v]x + 2[v]x^2.
  const float M00 = 1.0f - 2.0f * (qy * qy + qz * qz);
  const float M01 = 2.0f * (qx * qy - qw * qz);
  const float M02 = 2.0f * (qx * qz + qw * qy);
  const float M10 = 2.0f * (qx * qy + qw * qz);
  const float M11 = 1.0f - 2.0f * (qx * qx + qz * qz);
  const float M12 = 2.0f * (qy * qz - qw * qx);
  const float M20 = 2.0f * (qx * qz - qw * qy);
  const float M21 = 2.0f * (qy * qz + qw * qx);
  const float M22 = 1.0f - 2.0f * (qx * qx + qy * qy);

  // J_point (2x3) = J_proj * M.
  jac_point[gid * 6 + 0] = jpx0 * M00 + jpx2 * M20;
  jac_point[gid * 6 + 1] = jpx0 * M01 + jpx2 * M21;
  jac_point[gid * 6 + 2] = jpx0 * M02 + jpx2 * M22;
  jac_point[gid * 6 + 3] = jpy1 * M10 + jpy2 * M20;
  jac_point[gid * 6 + 4] = jpy1 * M11 + jpy2 * M21;
  jac_point[gid * 6 + 5] = jpy1 * M12 + jpy2 * M22;

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

  // J_pose (2x7) = [J_quat (2x4) | J_trans (2x3)].
  // J_quat = J_proj * Jq ; J_trans = J_proj (d p_cam / d t = I).
  jac_pose[gid * 14 + 0] = jpx0 * Jq00 + jpx2 * Jq20;
  jac_pose[gid * 14 + 1] = jpx0 * Jq01 + jpx2 * Jq21;
  jac_pose[gid * 14 + 2] = jpx0 * Jq02 + jpx2 * Jq22;
  jac_pose[gid * 14 + 3] = jpx0 * Jq03 + jpx2 * Jq23;
  jac_pose[gid * 14 + 4] = jpx0;            // d x / d tx
  jac_pose[gid * 14 + 5] = 0.0f;            // d x / d ty
  jac_pose[gid * 14 + 6] = jpx2;            // d x / d tz
  jac_pose[gid * 14 + 7] = jpy1 * Jq10 + jpy2 * Jq20;
  jac_pose[gid * 14 + 8] = jpy1 * Jq11 + jpy2 * Jq21;
  jac_pose[gid * 14 + 9] = jpy1 * Jq12 + jpy2 * Jq22;
  jac_pose[gid * 14 + 10] = jpy1 * Jq13 + jpy2 * Jq23;
  jac_pose[gid * 14 + 11] = 0.0f;           // d y / d tx
  jac_pose[gid * 14 + 12] = jpy1;           // d y / d ty
  jac_pose[gid * 14 + 13] = jpy2;           // d y / d tz

  // J_cam (2x4): d residual / d (f1, f2, c1, c2).
  jac_cam[gid * 8 + 0] = u * inv_w;
  jac_cam[gid * 8 + 1] = 0.0f;
  jac_cam[gid * 8 + 2] = 1.0f;
  jac_cam[gid * 8 + 3] = 0.0f;
  jac_cam[gid * 8 + 4] = 0.0f;
  jac_cam[gid * 8 + 5] = v * inv_w;
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
  const float vxp0 = qy * Z - qz * Y;
  const float vxp1 = qz * X - qx * Z;
  const float vxp2 = qx * Y - qy * X;
  const float vvxp0 = qy * vxp2 - qz * vxp1;
  const float vvxp1 = qz * vxp0 - qx * vxp2;
  const float vvxp2 = qx * vxp1 - qy * vxp0;
  const float u = X + 2.0f * (qw * vxp0 + vvxp0) + tx;
  const float v = Y + 2.0f * (qw * vxp1 + vvxp1) + ty;
  const float w = Z + 2.0f * (qw * vxp2 + vvxp2) + tz;
  if (w <= 1e-6f) return false;
  const float inv_w = 1.0f / w;
  const float inv_w2 = inv_w * inv_w;
  r[0] = f1 * u * inv_w + c1 - ox;
  r[1] = f2 * v * inv_w + c2 - oy;

  const float jpx0 = f1 * inv_w, jpx2 = -f1 * u * inv_w2;
  const float jpy1 = f2 * inv_w, jpy2 = -f2 * v * inv_w2;
  const float M00 = 1.0f - 2.0f * (qy * qy + qz * qz);
  const float M01 = 2.0f * (qx * qy - qw * qz);
  const float M02 = 2.0f * (qx * qz + qw * qy);
  const float M10 = 2.0f * (qx * qy + qw * qz);
  const float M11 = 1.0f - 2.0f * (qx * qx + qz * qz);
  const float M12 = 2.0f * (qy * qz - qw * qx);
  const float M20 = 2.0f * (qx * qz - qw * qy);
  const float M21 = 2.0f * (qy * qz + qw * qx);
  const float M22 = 1.0f - 2.0f * (qx * qx + qy * qy);
  J[0] = jpx0 * M00 + jpx2 * M20;
  J[1] = jpx0 * M01 + jpx2 * M21;
  J[2] = jpx0 * M02 + jpx2 * M22;
  J[3] = jpy1 * M10 + jpy2 * M20;
  J[4] = jpy1 * M11 + jpy2 * M21;
  J[5] = jpy1 * M12 + jpy2 * M22;
  return true;
}

// Sum of squared reprojection residuals of point (X,Y,Z) over its observations.
static float PointCost(uint begin, uint end,
                       device const float* obs_pose,
                       device const float* obs_cam,
                       device const float* obs_pixel,
                       float X, float Y, float Z) {
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
      // A = H + lambda * diag(H)  (Levenberg-Marquardt, Jacobi scaling).
      const float a00 = h00 + lambda * h00;
      const float a11 = h11 + lambda * h11;
      const float a22 = h22 + lambda * h22;
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
  const float tx = poses[gid * 7 + 4], ty = poses[gid * 7 + 5];
  const float tz = poses[gid * 7 + 6];
  const float X = points[gid * 3 + 0], Y = points[gid * 3 + 1];
  const float Z = points[gid * 3 + 2];
  const float f1 = cams[gid * 4 + 0], f2 = cams[gid * 4 + 1];
  const float c1 = cams[gid * 4 + 2], c2 = cams[gid * 4 + 3];

  const float vxp0 = qy * Z - qz * Y;
  const float vxp1 = qz * X - qx * Z;
  const float vxp2 = qx * Y - qy * X;
  const float u = X + 2.0f * (qw * vxp0 + (qy * vxp2 - qz * vxp1)) + tx;
  const float v = Y + 2.0f * (qw * vxp1 + (qz * vxp0 - qx * vxp2)) + ty;
  const float w = Z + 2.0f * (qw * vxp2 + (qx * vxp1 - qy * vxp0)) + tz;
  const float inv_w = 1.0f / w;
  const float inv_w2 = inv_w * inv_w;
  residuals[gid * 2 + 0] = f1 * u * inv_w + c1 - obs[gid * 2 + 0];
  residuals[gid * 2 + 1] = f2 * v * inv_w + c2 - obs[gid * 2 + 1];

  const float jpx0 = f1 * inv_w, jpx2 = -f1 * u * inv_w2;
  const float jpy1 = f2 * inv_w, jpy2 = -f2 * v * inv_w2;
  const float M00 = 1.0f - 2.0f * (qy * qy + qz * qz);
  const float M01 = 2.0f * (qx * qy - qw * qz);
  const float M02 = 2.0f * (qx * qz + qw * qy);
  const float M10 = 2.0f * (qx * qy + qw * qz);
  const float M11 = 1.0f - 2.0f * (qx * qx + qz * qz);
  const float M12 = 2.0f * (qy * qz - qw * qx);
  const float M20 = 2.0f * (qx * qz - qw * qy);
  const float M21 = 2.0f * (qy * qz + qw * qx);
  const float M22 = 1.0f - 2.0f * (qx * qx + qy * qy);

  // J_point = J_proj * R.
  jac_point[gid * 6 + 0] = jpx0 * M00 + jpx2 * M20;
  jac_point[gid * 6 + 1] = jpx0 * M01 + jpx2 * M21;
  jac_point[gid * 6 + 2] = jpx0 * M02 + jpx2 * M22;
  jac_point[gid * 6 + 3] = jpy1 * M10 + jpy2 * M20;
  jac_point[gid * 6 + 4] = jpy1 * M11 + jpy2 * M21;
  jac_point[gid * 6 + 5] = jpy1 * M12 + jpy2 * M22;

  // M*[X]x (3x3), then d p_cam/d dtheta = -(M*[X]x).
  const float MX00 = M01 * Z - M02 * Y;
  const float MX10 = M11 * Z - M12 * Y;
  const float MX20 = M21 * Z - M22 * Y;
  const float MX01 = -M00 * Z + M02 * X;
  const float MX11 = -M10 * Z + M12 * X;
  const float MX21 = -M20 * Z + M22 * X;
  const float MX02 = M00 * Y - M01 * X;
  const float MX12 = M10 * Y - M11 * X;
  const float MX22 = M20 * Y - M21 * X;

  // J_rot (2x3) = J_proj * (-(M*[X]x)).
  const float jr00 = -(jpx0 * MX00 + jpx2 * MX20);
  const float jr01 = -(jpx0 * MX01 + jpx2 * MX21);
  const float jr02 = -(jpx0 * MX02 + jpx2 * MX22);
  const float jr10 = -(jpy1 * MX10 + jpy2 * MX20);
  const float jr11 = -(jpy1 * MX11 + jpy2 * MX21);
  const float jr12 = -(jpy1 * MX12 + jpy2 * MX22);

  // jac_cam_tangent (2x6) = [J_rot (2x3) | J_trans (2x3) = J_proj].
  jac_cam_tangent[gid * 12 + 0] = jr00;
  jac_cam_tangent[gid * 12 + 1] = jr01;
  jac_cam_tangent[gid * 12 + 2] = jr02;
  jac_cam_tangent[gid * 12 + 3] = jpx0;
  jac_cam_tangent[gid * 12 + 4] = 0.0f;
  jac_cam_tangent[gid * 12 + 5] = jpx2;
  jac_cam_tangent[gid * 12 + 6] = jr10;
  jac_cam_tangent[gid * 12 + 7] = jr11;
  jac_cam_tangent[gid * 12 + 8] = jr12;
  jac_cam_tangent[gid * 12 + 9] = 0.0f;
  jac_cam_tangent[gid * 12 + 10] = jpy1;
  jac_cam_tangent[gid * 12 + 11] = jpy2;
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
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithSource:[NSString stringWithUTF8String:kReprojErrorSource]
                       options:nil
                         error:&error];
      if (library == nil) return;
      id<MTLFunction> fn = [library newFunctionWithName:@"reproj_error_pinhole"];
      if (fn == nil) return;
      pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
      if (pipeline == nil) return;
      id<MTLFunction> pts_fn =
          [library newFunctionWithName:@"refine_points_pinhole"];
      if (pts_fn == nil) return;
      points_pipeline =
          [device newComputePipelineStateWithFunction:pts_fn error:&error];
      if (points_pipeline == nil) return;
      id<MTLFunction> tan_fn =
          [library newFunctionWithName:@"pose_tangent_jac_pinhole"];
      if (tan_fn == nil) return;
      tangent_pipeline =
          [device newComputePipelineStateWithFunction:tan_fn error:&error];
      if (tangent_pipeline == nil) return;
      valid = true;
    }
  }
};

MetalContext& GetContext() {
  static MetalContext context;
  return context;
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
    auto in_buf = [&](const void* ptr, size_t bytes) {
      return [ctx.device newBufferWithBytes:ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared];
    };
    auto out_buf = [&](size_t bytes) {
      return [ctx.device newBufferWithLength:bytes
                                     options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> poses_buf = in_buf(poses, n * 7 * sizeof(float));
    id<MTLBuffer> points_buf = in_buf(points, n * 3 * sizeof(float));
    id<MTLBuffer> cams_buf = in_buf(cams, n * 4 * sizeof(float));
    id<MTLBuffer> obs_buf = in_buf(obs, n * 2 * sizeof(float));
    id<MTLBuffer> res_buf = out_buf(n * 2 * sizeof(float));
    id<MTLBuffer> jp_buf = out_buf(n * 6 * sizeof(float));
    id<MTLBuffer> jpose_buf = out_buf(n * 14 * sizeof(float));
    id<MTLBuffer> jcam_buf = out_buf(n * 8 * sizeof(float));
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

    NSUInteger tg = ctx.pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;

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

  @autoreleasepool {
    const NSUInteger p = static_cast<NSUInteger>(num_points);
    const NSUInteger m = static_cast<NSUInteger>(num_obs);
    auto in_buf = [&](const void* ptr, size_t bytes) {
      return [ctx.device newBufferWithBytes:ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared];
    };
    // point3D is read-write (optimized in place).
    id<MTLBuffer> pts_buf = in_buf(point3D, p * 3 * sizeof(float));
    id<MTLBuffer> off_buf = in_buf(obs_offset, (p + 1) * sizeof(int));
    id<MTLBuffer> pose_buf = in_buf(obs_pose, m * 7 * sizeof(float));
    id<MTLBuffer> cam_buf = in_buf(obs_cam, m * 4 * sizeof(float));
    id<MTLBuffer> pix_buf = in_buf(obs_pixel, m * 2 * sizeof(float));
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

    NSUInteger tg = ctx.points_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > p) tg = p;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(p, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;

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
    auto in_buf = [&](const void* ptr, size_t bytes) {
      return [ctx.device newBufferWithBytes:ptr
                                     length:bytes
                                    options:MTLResourceStorageModeShared];
    };
    auto out_buf = [&](size_t bytes) {
      return [ctx.device newBufferWithLength:bytes
                                     options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> poses_buf = in_buf(poses, n * 7 * sizeof(float));
    id<MTLBuffer> points_buf = in_buf(points, n * 3 * sizeof(float));
    id<MTLBuffer> cams_buf = in_buf(cams, n * 4 * sizeof(float));
    id<MTLBuffer> obs_buf = in_buf(obs, n * 2 * sizeof(float));
    id<MTLBuffer> res_buf = out_buf(n * 2 * sizeof(float));
    id<MTLBuffer> jct_buf = out_buf(n * 12 * sizeof(float));
    id<MTLBuffer> jp_buf = out_buf(n * 6 * sizeof(float));
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
    NSUInteger tg = ctx.tangent_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n;
    if (tg == 0) tg = 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;
    std::memcpy(residuals, [res_buf contents], n * 2 * sizeof(float));
    std::memcpy(jac_cam_tangent, [jct_buf contents], n * 12 * sizeof(float));
    std::memcpy(jac_point, [jp_buf contents], n * 6 * sizeof(float));
  }
  return true;
}

}  // namespace colmap
