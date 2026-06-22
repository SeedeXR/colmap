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

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/math/random.h"
#include "colmap/sensor/models.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Builds `n` random reprojection observations (pose, point in front of the
// camera, pinhole intrinsics, observed pixel) into the flat fp32 layout the
// Metal kernel consumes, and keeps a double copy for the Ceres reference.
struct ReprojData {
  std::vector<float> poses, points, cams, obs;       // fp32 for the kernel.
  std::vector<std::array<double, 7>> poses_d;
  std::vector<std::array<double, 3>> points_d;
  std::vector<std::array<double, 4>> cams_d;
  std::vector<Eigen::Vector2d> obs_d;
};

ReprojData MakeRandomData(int n) {
  SetPRNGSeed(42);
  ReprojData d;
  d.poses.resize(n * 7);
  d.points.resize(n * 3);
  d.cams.resize(n * 4);
  d.obs.resize(n * 2);
  d.poses_d.resize(n);
  d.points_d.resize(n);
  d.cams_d.resize(n);
  d.obs_d.resize(n);
  for (int i = 0; i < n; ++i) {
    const Eigen::Quaterniond q(
        Eigen::AngleAxisd(RandomUniformReal<double>(0, 2 * EIGEN_PI),
                          Eigen::Vector3d(RandomUniformReal<double>(-1, 1),
                                          RandomUniformReal<double>(-1, 1),
                                          RandomUniformReal<double>(-1, 1))
                              .normalized()));
    const Eigen::Vector3d t(RandomUniformReal<double>(-0.5, 0.5),
                            RandomUniformReal<double>(-0.5, 0.5),
                            RandomUniformReal<double>(-0.5, 0.5));
    const Rigid3d cam_from_world(q, t);

    // Point comfortably in front of the camera (z_cam in [2, 8]).
    Eigen::Vector3d point_in_cam(RandomUniformReal<double>(-2, 2),
                                 RandomUniformReal<double>(-2, 2),
                                 RandomUniformReal<double>(2, 8));
    const Eigen::Vector3d point3D = Inverse(cam_from_world) * point_in_cam;

    const std::array<double, 4> cam = {RandomUniformReal<double>(800, 1200),
                                       RandomUniformReal<double>(800, 1200),
                                       RandomUniformReal<double>(300, 700),
                                       RandomUniformReal<double>(300, 700)};
    // Observed pixel = true projection + small noise.
    double px, py;
    PinholeCameraModel::ImgFromCam(cam.data(), point_in_cam.x(),
                                   point_in_cam.y(), point_in_cam.z(), &px, &py);
    const Eigen::Vector2d ob(px + RandomUniformReal<double>(-1, 1),
                             py + RandomUniformReal<double>(-1, 1));

    d.poses_d[i] = {cam_from_world.params[0], cam_from_world.params[1],
                    cam_from_world.params[2], cam_from_world.params[3],
                    cam_from_world.params[4], cam_from_world.params[5],
                    cam_from_world.params[6]};
    d.points_d[i] = {point3D.x(), point3D.y(), point3D.z()};
    d.cams_d[i] = cam;
    d.obs_d[i] = ob;
    for (int k = 0; k < 7; ++k) d.poses[i * 7 + k] = (float)d.poses_d[i][k];
    for (int k = 0; k < 3; ++k) d.points[i * 3 + k] = (float)d.points_d[i][k];
    for (int k = 0; k < 4; ++k) d.cams[i * 4 + k] = (float)cam[k];
    d.obs[i * 2 + 0] = (float)ob.x();
    d.obs[i * 2 + 1] = (float)ob.y();
  }
  return d;
}

// The Metal reprojection residual + analytic Jacobians must match COLMAP's
// Ceres ReprojErrorCostFunctor<PinholeCameraModel> (the factor BA minimizes),
// up to fp32 rounding. This is the "equivalent output" half of the PoC.
TEST(BundleAdjustmentMetal, ReprojErrorMatchesCeresPinhole) {
  if (!IsBundleAdjustmentMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  constexpr int kN = 512;
  ReprojData d = MakeRandomData(kN);

  std::vector<float> res(kN * 2), jp(kN * 6), jpose(kN * 14), jcam(kN * 8);
  ASSERT_TRUE(ComputeReprojErrorMetalPinhole(kN, d.poses.data(), d.points.data(),
                                             d.cams.data(), d.obs.data(),
                                             res.data(), jp.data(), jpose.data(),
                                             jcam.data()));

  double max_res_err = 0, max_jac_rel_err = 0;
  for (int i = 0; i < kN; ++i) {
    std::unique_ptr<ceres::CostFunction> cost(
        ReprojErrorCostFunctor<PinholeCameraModel>::Create(d.obs_d[i]));
    const double* params[3] = {d.points_d[i].data(), d.poses_d[i].data(),
                               d.cams_d[i].data()};
    double ref_res[2];
    double ref_jp[6], ref_jpose[14], ref_jcam[8];
    double* jacs[3] = {ref_jp, ref_jpose, ref_jcam};
    ASSERT_TRUE(cost->Evaluate(params, ref_res, jacs));

    for (int k = 0; k < 2; ++k)
      max_res_err = std::max(max_res_err, std::abs(res[i * 2 + k] - ref_res[k]));
    auto rel = [&](float gpu, double ref) {
      max_jac_rel_err =
          std::max(max_jac_rel_err, std::abs(gpu - ref) / (std::abs(ref) + 1.0));
    };
    for (int k = 0; k < 6; ++k) rel(jp[i * 6 + k], ref_jp[k]);
    for (int k = 0; k < 14; ++k) rel(jpose[i * 14 + k], ref_jpose[k]);
    for (int k = 0; k < 8; ++k) rel(jcam[i * 8 + k], ref_jcam[k]);
  }

  std::printf(
      "[ba-metal] N=%d  max residual abs err=%.3e px  max jac rel err=%.3e\n",
      kN, max_res_err, max_jac_rel_err);
  // fp32 over the reprojection chain: residuals match to well under 0.01 px and
  // Jacobians to ~1e-4 relative -- numerically equivalent to the Ceres factor.
  EXPECT_LT(max_res_err, 1e-2);
  EXPECT_LT(max_jac_rel_err, 1e-3);
}

// A point at/behind the camera (cheirality violation) must yield a zeroed
// residual + zeroed Jacobians (finite), matching PinholeCameraModel::ImgFromCam
// returning false -- NOT inf/nan. Exercises the guard the synthetic in-front
// fixtures never reach.
TEST(BundleAdjustmentMetal, HandlesBehindCameraObservation) {
  if (!IsBundleAdjustmentMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  // Identity pose, point at z=-1 (behind the camera), arbitrary intrinsics/obs.
  const float poses[7] = {0, 0, 0, 1, 0, 0, 0};
  const float points[3] = {0, 0, -1};
  const float cams[4] = {1000, 1000, 640, 480};
  const float obs[2] = {100, 100};

  float res[2], jp[6], jpose[14], jcam[8];
  ASSERT_TRUE(ComputeReprojErrorMetalPinhole(1, poses, points, cams, obs, res,
                                             jp, jpose, jcam));
  for (float x : res) { EXPECT_TRUE(std::isfinite(x)); EXPECT_EQ(x, 0.0f); }
  for (float x : jp) { EXPECT_TRUE(std::isfinite(x)); EXPECT_EQ(x, 0.0f); }
  for (float x : jpose) { EXPECT_TRUE(std::isfinite(x)); EXPECT_EQ(x, 0.0f); }
  for (float x : jcam) { EXPECT_TRUE(std::isfinite(x)); EXPECT_EQ(x, 0.0f); }

  float res2[2], jct[12], jp2[6];
  ASSERT_TRUE(ComputePoseTangentJacMetalPinhole(1, poses, points, cams, obs,
                                                res2, jct, jp2));
  for (float x : res2) { EXPECT_TRUE(std::isfinite(x)); EXPECT_EQ(x, 0.0f); }
  for (float x : jct) { EXPECT_TRUE(std::isfinite(x)); EXPECT_EQ(x, 0.0f); }
  for (float x : jp2) { EXPECT_TRUE(std::isfinite(x)); EXPECT_EQ(x, 0.0f); }
}

// Speed signal: GPU batch evaluation vs a single-threaded CPU Ceres loop over
// the same observations. Not a production matcher -- an order-of-magnitude
// indicator of whether Metal is worth a full BA accelerator. Env-gated so it
// never gates CI on timing.
TEST(BundleAdjustmentMetal, ReprojErrorThroughputSignal) {
  if (!IsBundleAdjustmentMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  if (std::getenv("COLMAP_BA_METAL_BENCH") == nullptr) {
    GTEST_SKIP() << "set COLMAP_BA_METAL_BENCH=1 to run the throughput signal.";
  }
  constexpr int kN = 500000;
  ReprojData d = MakeRandomData(kN);
  std::vector<float> res(kN * 2), jp(kN * 6), jpose(kN * 14), jcam(kN * 8);

  // Warm up (compile/allocate) then time the GPU batch.
  ComputeReprojErrorMetalPinhole(1, d.poses.data(), d.points.data(),
                                 d.cams.data(), d.obs.data(), res.data(),
                                 jp.data(), jpose.data(), jcam.data());
  auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(ComputeReprojErrorMetalPinhole(
      kN, d.poses.data(), d.points.data(), d.cams.data(), d.obs.data(),
      res.data(), jp.data(), jpose.data(), jcam.data()));
  auto t1 = std::chrono::steady_clock::now();
  const double gpu_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  // CPU reference: single-threaded Ceres Evaluate loop. The cost functions are
  // created up front (as a real BA does once at setup) so the timed region is
  // pure residual+Jacobian evaluation, the apples-to-apples comparison to the
  // GPU kernel. NB: real BA evaluates multi-threaded (~4 P-cores here), so the
  // per-core eval advantage of the GPU is ~1/4 of the printed single-thread
  // ratio; and evaluation is only part of BA (the Schur/PCG solve is separate).
  std::vector<std::unique_ptr<ceres::CostFunction>> costs(kN);
  for (int i = 0; i < kN; ++i) {
    costs[i].reset(ReprojErrorCostFunctor<PinholeCameraModel>::Create(d.obs_d[i]));
  }
  std::vector<double> cres(kN * 2), cjp(kN * 6), cjpose(kN * 14), cjcam(kN * 8);
  auto c0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kN; ++i) {
    const double* params[3] = {d.points_d[i].data(), d.poses_d[i].data(),
                               d.cams_d[i].data()};
    double* jacs[3] = {&cjp[i * 6], &cjpose[i * 14], &cjcam[i * 8]};
    costs[i]->Evaluate(params, &cres[i * 2], jacs);
  }
  auto c1 = std::chrono::steady_clock::now();
  const double cpu_ms =
      std::chrono::duration<double, std::milli>(c1 - c0).count();

  std::printf(
      "[ba-metal] %d obs residual+Jacobian: GPU %.1f ms, CPU(1-thread) %.1f ms, "
      "speedup %.1fx (GPU includes buffer alloc + copy back)\n",
      kN, gpu_ms, cpu_ms, cpu_ms / gpu_ms);
  SUCCEED();
}

// The pose-tangent (2x6) + point (2x3) Jacobians from the camera-solver kernel
// must match central finite differences of the reprojection residual under the
// same right-perturbation manifold (q<-q*exp(dtheta), t<-t+dt). Validates the
// SE(3) tangent derivation independent of any solver/Ceres convention.
TEST(BundleAdjustmentMetal, PoseTangentJacobianMatchesFiniteDiff) {
  if (!IsBundleAdjustmentMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  constexpr int kN = 512;
  ReprojData d = MakeRandomData(kN);
  std::vector<float> res(kN * 2), jct(kN * 12), jp(kN * 6);
  ASSERT_TRUE(ComputePoseTangentJacMetalPinhole(
      kN, d.poses.data(), d.points.data(), d.cams.data(), d.obs.data(),
      res.data(), jct.data(), jp.data()));

  auto residual = [](const std::array<double, 7>& pose,
                     const Eigen::Vector3d& X, const std::array<double, 4>& cam,
                     const Eigen::Vector2d& ob) {
    const Eigen::Quaterniond q(pose[3], pose[0], pose[1], pose[2]);  // w,x,y,z
    const Eigen::Vector3d t(pose[4], pose[5], pose[6]);
    const Eigen::Vector3d pc = q * X + t;
    return Eigen::Vector2d(cam[0] * pc.x() / pc.z() + cam[2] - ob.x(),
                           cam[1] * pc.y() / pc.z() + cam[3] - ob.y());
  };
  const double eps = 1e-5;
  double max_rel = 0;
  for (int i = 0; i < kN; ++i) {
    const auto& pose = d.poses_d[i];
    const Eigen::Vector3d X(d.points_d[i][0], d.points_d[i][1], d.points_d[i][2]);
    const auto& cam = d.cams_d[i];
    const Eigen::Vector2d ob = d.obs_d[i];
    const Eigen::Quaterniond q(pose[3], pose[0], pose[1], pose[2]);
    const Eigen::Vector3d t(pose[4], pose[5], pose[6]);

    auto check = [&](const Eigen::Vector2d& num, double gpu0, double gpu1) {
      max_rel = std::max(max_rel, std::abs(gpu0 - num.x()) / (std::abs(num.x()) + 1.0));
      max_rel = std::max(max_rel, std::abs(gpu1 - num.y()) / (std::abs(num.y()) + 1.0));
    };
    // Rotation tangent columns 0..2.
    for (int a = 0; a < 3; ++a) {
      Eigen::Vector3d axis = Eigen::Vector3d::Zero();
      axis[a] = 1.0;
      auto qd = [&](double s) {
        return q * Eigen::Quaterniond(Eigen::AngleAxisd(s, axis));
      };
      auto pose_of = [&](const Eigen::Quaterniond& qq) {
        return std::array<double, 7>{qq.x(), qq.y(), qq.z(), qq.w(),
                                     t.x(), t.y(), t.z()};
      };
      const Eigen::Vector2d num =
          (residual(pose_of(qd(eps)), X, cam, ob) -
           residual(pose_of(qd(-eps)), X, cam, ob)) / (2 * eps);
      check(num, jct[i * 12 + a], jct[i * 12 + 6 + a]);
    }
    // Translation tangent columns 3..5.
    for (int a = 0; a < 3; ++a) {
      auto pose_dt = [&](double s) {
        std::array<double, 7> pp = pose;
        pp[4 + a] += s;
        return pp;
      };
      const Eigen::Vector2d num =
          (residual(pose_dt(eps), X, cam, ob) -
           residual(pose_dt(-eps), X, cam, ob)) / (2 * eps);
      check(num, jct[i * 12 + 3 + a], jct[i * 12 + 6 + 3 + a]);
    }
    // Point columns 0..2.
    for (int a = 0; a < 3; ++a) {
      Eigen::Vector3d dp = Eigen::Vector3d::Zero();
      dp[a] = eps;
      const Eigen::Vector2d num =
          (residual(pose, X + dp, cam, ob) - residual(pose, X - dp, cam, ob)) /
          (2 * eps);
      check(num, jp[i * 6 + a], jp[i * 6 + 3 + a]);
    }
  }
  std::printf("[ba-metal] pose-tangent+point Jacobian vs finite-diff: max rel "
              "err=%.3e\n", max_rel);
  EXPECT_LT(max_rel, 5e-3);
}

// A synthetic point-BA problem: cameras orbiting the origin looking inward, a
// cloud of points near the origin (so every point is in front of every camera),
// observations grouped by point (CSR). Keeps double copies for the Ceres
// reference and the ground-truth points for the noise-free check.
struct PointBAData {
  int num_points = 0;
  int num_obs = 0;
  std::vector<float> point3D;            // 3*P, perturbed initial guess (INOUT).
  std::vector<double> point3D_true;      // 3*P ground truth.
  std::vector<int> obs_offset;           // P+1.
  std::vector<float> obs_pose, obs_cam, obs_pixel;  // CSR fp32 for the kernel.
  std::vector<std::array<double, 7>> obs_pose_d;
  std::vector<std::array<double, 4>> obs_cam_d;
  std::vector<Eigen::Vector2d> obs_pixel_d;
};

Rigid3d LookAt(const Eigen::Vector3d& eye) {
  const Eigen::Vector3d f = (-eye).normalized();  // +z optical axis -> scene.
  Eigen::Vector3d up(0, 1, 0);
  if (std::abs(f.dot(up)) > 0.9) up = Eigen::Vector3d(1, 0, 0);
  const Eigen::Vector3d x = f.cross(up).normalized();   // +x right.
  const Eigen::Vector3d y = f.cross(x);                 // +y down.
  Eigen::Matrix3d R_cw;
  R_cw.row(0) = x;
  R_cw.row(1) = y;
  R_cw.row(2) = f;
  return Rigid3d(Eigen::Quaterniond(R_cw), -(R_cw * eye));
}

PointBAData MakePointBA(int num_points, int num_cameras, double pixel_noise) {
  SetPRNGSeed(7);
  PointBAData d;
  d.num_points = num_points;

  std::vector<Rigid3d> cams_from_world(num_cameras);
  std::vector<std::array<double, 4>> intrinsics(num_cameras);
  for (int c = 0; c < num_cameras; ++c) {
    const Eigen::Vector3d eye(RandomUniformReal<double>(-1, 1),
                              RandomUniformReal<double>(-1, 1),
                              RandomUniformReal<double>(-1, 1));
    cams_from_world[c] = LookAt(eye.normalized() * 6.0);  // radius 6 sphere.
    intrinsics[c] = {RandomUniformReal<double>(900, 1100),
                     RandomUniformReal<double>(900, 1100), 640.0, 480.0};
  }

  d.obs_offset.push_back(0);
  for (int p = 0; p < num_points; ++p) {
    const Eigen::Vector3d Xtrue(RandomUniformReal<double>(-1, 1),
                                RandomUniformReal<double>(-1, 1),
                                RandomUniformReal<double>(-1, 1));
    d.point3D_true.insert(d.point3D_true.end(),
                          {Xtrue.x(), Xtrue.y(), Xtrue.z()});
    // Perturbed initial guess.
    const Eigen::Vector3d Xinit =
        Xtrue + Eigen::Vector3d(RandomUniformReal<double>(-0.1, 0.1),
                                RandomUniformReal<double>(-0.1, 0.1),
                                RandomUniformReal<double>(-0.1, 0.1));
    d.point3D.insert(d.point3D.end(),
                     {(float)Xinit.x(), (float)Xinit.y(), (float)Xinit.z()});
    for (int c = 0; c < num_cameras; ++c) {
      const Eigen::Vector3d pc = cams_from_world[c] * Xtrue;
      if (pc.z() <= 0.1) continue;
      double px, py;
      PinholeCameraModel::ImgFromCam(intrinsics[c].data(), pc.x(), pc.y(),
                                     pc.z(), &px, &py);
      const Eigen::Vector2d pix(px + RandomUniformReal<double>(-1, 1) * pixel_noise,
                                py + RandomUniformReal<double>(-1, 1) * pixel_noise);
      const auto& q = cams_from_world[c];
      d.obs_pose_d.push_back({q.params[0], q.params[1], q.params[2], q.params[3],
                              q.params[4], q.params[5], q.params[6]});
      d.obs_cam_d.push_back(intrinsics[c]);
      d.obs_pixel_d.push_back(pix);
      for (int k = 0; k < 7; ++k) d.obs_pose.push_back((float)q.params[k]);
      for (int k = 0; k < 4; ++k) d.obs_cam.push_back((float)intrinsics[c][k]);
      d.obs_pixel.push_back((float)pix.x());
      d.obs_pixel.push_back((float)pix.y());
    }
    d.obs_offset.push_back(static_cast<int>(d.obs_pixel_d.size()));
  }
  d.num_obs = static_cast<int>(d.obs_pixel_d.size());
  return d;
}

double MaxPointError(const std::vector<float>& pts,
                     const std::vector<double>& ref) {
  double e = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    e = std::max(e, std::abs((double)pts[i] - ref[i]));
  return e;
}

// Noise-free: Metal point-BA must drive each perturbed point back to its true
// position (the observations are exactly consistent). Proves the LM loop +
// 3x3 solve + convergence work end-to-end on the GPU.
TEST(BundleAdjustmentMetal, RefinePointsRecoversTruthNoiseFree) {
  if (!IsBundleAdjustmentMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  PointBAData d = MakePointBA(/*num_points=*/2000, /*num_cameras=*/8,
                              /*pixel_noise=*/0.0);
  const double init_err = MaxPointError(d.point3D, d.point3D_true);
  ASSERT_TRUE(RefinePointsMetalPinhole(d.num_points, d.num_obs,
                                       d.point3D.data(), d.obs_offset.data(),
                                       d.obs_pose.data(), d.obs_cam.data(),
                                       d.obs_pixel.data(), /*max_iters=*/30));
  const double final_err = MaxPointError(d.point3D, d.point3D_true);
  std::printf("[ba-metal] point-BA noise-free: init max err %.3e -> %.3e\n",
              init_err, final_err);
  EXPECT_LT(final_err, 1e-3);
}

// Noisy: Metal point-BA must converge to the same points as Ceres point-only BA
// (cameras held constant) -- the "equivalent output" check for the solver.
TEST(BundleAdjustmentMetal, RefinePointsMatchesCeresNoisy) {
  if (!IsBundleAdjustmentMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  PointBAData d = MakePointBA(/*num_points=*/400, /*num_cameras=*/8,
                              /*pixel_noise=*/0.5);

  // Metal refine (in place on a copy).
  std::vector<float> metal_pts = d.point3D;
  ASSERT_TRUE(RefinePointsMetalPinhole(d.num_points, d.num_obs,
                                       metal_pts.data(), d.obs_offset.data(),
                                       d.obs_pose.data(), d.obs_cam.data(),
                                       d.obs_pixel.data(), /*max_iters=*/50));

  // Ceres point-only BA: vary points, hold poses + intrinsics constant.
  std::vector<std::array<double, 3>> ceres_pts(d.num_points);
  for (int p = 0; p < d.num_points; ++p) {
    ceres_pts[p] = {d.point3D[p * 3 + 0], d.point3D[p * 3 + 1],
                    d.point3D[p * 3 + 2]};
  }
  ceres::Problem problem;
  for (int p = 0; p < d.num_points; ++p) {
    for (int o = d.obs_offset[p]; o < d.obs_offset[p + 1]; ++o) {
      problem.AddResidualBlock(
          ReprojErrorCostFunctor<PinholeCameraModel>::Create(d.obs_pixel_d[o]),
          nullptr, ceres_pts[p].data(), d.obs_pose_d[o].data(),
          d.obs_cam_d[o].data());
      problem.SetParameterBlockConstant(d.obs_pose_d[o].data());
      problem.SetParameterBlockConstant(d.obs_cam_d[o].data());
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 50;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  double max_diff = 0;
  for (int p = 0; p < d.num_points; ++p) {
    for (int k = 0; k < 3; ++k) {
      max_diff = std::max(
          max_diff, std::abs((double)metal_pts[p * 3 + k] - ceres_pts[p][k]));
    }
  }
  std::printf(
      "[ba-metal] point-BA noisy: max |Metal - Ceres| point coord diff = %.3e\n",
      max_diff);
  // Loose tolerance: fp32 Metal LM vs fp64 Ceres, and a near-degenerate (low-
  // parallax) point among the 400 can legitimately differ more than a tight
  // bound under fp32 rounding across GPUs.
  EXPECT_LT(max_diff, 2e-2);
}

// Speed signal for the per-point solver (env-gated).
TEST(BundleAdjustmentMetal, RefinePointsThroughputSignal) {
  if (!IsBundleAdjustmentMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  if (std::getenv("COLMAP_BA_METAL_BENCH") == nullptr) {
    GTEST_SKIP() << "set COLMAP_BA_METAL_BENCH=1 to run the throughput signal.";
  }
  PointBAData d = MakePointBA(/*num_points=*/100000, /*num_cameras=*/8,
                              /*pixel_noise=*/0.5);
  std::vector<float> pts = d.point3D;
  auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(RefinePointsMetalPinhole(d.num_points, d.num_obs, pts.data(),
                                       d.obs_offset.data(), d.obs_pose.data(),
                                       d.obs_cam.data(), d.obs_pixel.data(),
                                       /*max_iters=*/30));
  auto t1 = std::chrono::steady_clock::now();
  std::printf("[ba-metal] point-BA: %d points / %d obs refined in %.1f ms\n",
              d.num_points, d.num_obs,
              std::chrono::duration<double, std::milli>(t1 - t0).count());
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Hybrid full bundle adjustment: GPU computes per-observation residual + pose-
// tangent (2x6) + point (2x3) Jacobians each iteration; the CPU assembles the
// normal equations, eliminates points via the Schur complement (reduced camera
// system, dense solve), back-substitutes, retracts on the SE(3) manifold, and
// runs Levenberg-Marquardt. Camera 0 is held fixed (gauge). Intrinsics fixed.
// This is the camera-side solver the Power-BA path later moves fully onto GPU;
// here it proves end-to-end convergence to Ceres.
struct Camera {
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
  std::array<double, 4> intr;
};
struct Obs {
  int cam, point;
  Eigen::Vector2d pixel;
};

double FullBACost(const std::vector<Camera>& cams,
                  const std::vector<Eigen::Vector3d>& pts,
                  const std::vector<Obs>& obs) {
  constexpr double kCheiralityPenalty = 1e12;
  double cost = 0;
  for (const Obs& o : obs) {
    const Eigen::Vector3d pc = cams[o.cam].q * pts[o.point] + cams[o.cam].t;
    double px, py;
    // Use COLMAP's own projection (single source of truth); it returns false
    // for points at/behind the camera, which we charge a large penalty so a
    // cheirality-violating trial is rejected by LM.
    if (PinholeCameraModel::ImgFromCam(cams[o.cam].intr.data(), pc.x(), pc.y(),
                                       pc.z(), &px, &py)) {
      const double dx = px - o.pixel.x(), dy = py - o.pixel.y();
      cost += dx * dx + dy * dy;
    } else {
      cost += kCheiralityPenalty;
    }
  }
  return cost;
}

// solve_mode: 0 = dense Schur LDLT, 1 = Power-BA power-series inverse Schur
// (block mat-vecs B^-1 E C^-1 E^T, no factorization -- the GPU-friendly method).
double MetalFullBA(std::vector<Camera>& cams,
                   std::vector<Eigen::Vector3d>& pts,
                   const std::vector<Obs>& obs, int max_iters,
                   int solve_mode = 0) {
  const int num_cams = static_cast<int>(cams.size());
  const int num_obs = static_cast<int>(obs.size());
  const int F = num_cams - 1;  // free cameras (camera 0 fixed); index c -> c-1.

  double lambda = 1e-3;
  double cost = FullBACost(cams, pts, obs);

  // Constant across iterations: per-observation intrinsics + observed pixels,
  // and the point->observations grouping. Build once. Only poses/points (the
  // optimized state) are refilled each iteration.
  std::vector<float> camsp(num_obs * 4), obsp(num_obs * 2);
  std::vector<std::vector<int>> point_obs(pts.size());
  for (int o = 0; o < num_obs; ++o) {
    const Camera& cm = cams[obs[o].cam];
    for (int k = 0; k < 4; ++k) camsp[o * 4 + k] = (float)cm.intr[k];
    obsp[o * 2 + 0] = (float)obs[o].pixel.x();
    obsp[o * 2 + 1] = (float)obs[o].pixel.y();
    point_obs[obs[o].point].push_back(o);
  }
  std::vector<float> poses(num_obs * 7), points(num_obs * 3);

  for (int it = 0; it < max_iters; ++it) {
    // GPU: residuals + Jacobians at the current state (refill poses + points).
    for (int o = 0; o < num_obs; ++o) {
      const Camera& cm = cams[obs[o].cam];
      poses[o * 7 + 0] = (float)cm.q.x(); poses[o * 7 + 1] = (float)cm.q.y();
      poses[o * 7 + 2] = (float)cm.q.z(); poses[o * 7 + 3] = (float)cm.q.w();
      poses[o * 7 + 4] = (float)cm.t.x(); poses[o * 7 + 5] = (float)cm.t.y();
      poses[o * 7 + 6] = (float)cm.t.z();
      const Eigen::Vector3d& X = pts[obs[o].point];
      points[o * 3 + 0] = (float)X.x(); points[o * 3 + 1] = (float)X.y();
      points[o * 3 + 2] = (float)X.z();
    }
    std::vector<float> res(num_obs * 2), jct(num_obs * 12), jp(num_obs * 6);
    if (!ComputePoseTangentJacMetalPinhole(num_obs, poses.data(), points.data(),
                                           camsp.data(), obsp.data(), res.data(),
                                           jct.data(), jp.data())) {
      return cost;
    }

    // Assemble B (per free cam 6x6), g_cam (6); C (per point 3x3), g_point (3);
    // and per-observation coupling E = Jc^T Jp (6x3).
    std::vector<Eigen::Matrix<double, 6, 6>> B(
        F, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> gcam(
        F, Eigen::Matrix<double, 6, 1>::Zero());
    std::vector<Eigen::Matrix3d> C(pts.size(), Eigen::Matrix3d::Zero());
    std::vector<Eigen::Vector3d> gpt(pts.size(), Eigen::Vector3d::Zero());
    std::vector<Eigen::Matrix<double, 6, 3>> E(num_obs);
    for (int o = 0; o < num_obs; ++o) {
      Eigen::Matrix<double, 2, 6> Jc;
      Eigen::Matrix<double, 2, 3> Jp;
      for (int r = 0; r < 2; ++r) {
        for (int k = 0; k < 6; ++k) Jc(r, k) = jct[o * 12 + r * 6 + k];
        for (int k = 0; k < 3; ++k) Jp(r, k) = jp[o * 6 + r * 3 + k];
      }
      const Eigen::Vector2d r(res[o * 2 + 0], res[o * 2 + 1]);
      E[o] = Jc.transpose() * Jp;
      const int p = obs[o].point;
      C[p] += Jp.transpose() * Jp;
      gpt[p] += Jp.transpose() * r;
      if (obs[o].cam != 0) {
        const int c = obs[o].cam - 1;
        B[c] += Jc.transpose() * Jc;
        gcam[c] += Jc.transpose() * r;
      }
    }

    bool accepted = false;
    for (int tries = 0; tries < 8 && !accepted; ++tries) {
      // Damped point blocks C^{-1} and the camera rhs b = -g_cam + E C^{-1} g_pt.
      // LM damping with a small diagonal floor so a structurally unconstrained
      // direction (zero Hessian diagonal) can still be regularized -- pure
      // relative scaling (d += lambda*d) would stay singular there.
      constexpr double kDiagFloor = 1e-9;
      std::vector<Eigen::Matrix3d> Cinv(pts.size());
      for (size_t p = 0; p < pts.size(); ++p) {
        Eigen::Matrix3d Cd = C[p];
        for (int k = 0; k < 3; ++k)
          Cd(k, k) += lambda * std::max(Cd(k, k), kDiagFloor);
        Cinv[p] = Cd.inverse();
      }
      std::vector<Eigen::Matrix<double, 6, 6>> Bd(F);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * F);
      for (int c = 0; c < F; ++c) {
        Bd[c] = B[c];
        for (int k = 0; k < 6; ++k)
          Bd[c](k, k) += lambda * std::max(Bd[c](k, k), kDiagFloor);
        b.segment<6>(6 * c) = -gcam[c];
      }
      for (size_t p = 0; p < pts.size(); ++p) {
        for (int oi : point_obs[p]) {
          if (obs[oi].cam == 0) continue;
          b.segment<6>(6 * (obs[oi].cam - 1)) += E[oi] * Cinv[p] * gpt[p];
        }
      }

      Eigen::VectorXd dcam;
      if (solve_mode == 0) {
        // Dense Schur: S = blkdiag(Bd) - sum_p E C^{-1} E^T, solved by LDLT.
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(6 * F, 6 * F);
        for (int c = 0; c < F; ++c) S.block<6, 6>(6 * c, 6 * c) = Bd[c];
        for (size_t p = 0; p < pts.size(); ++p) {
          for (int oi : point_obs[p]) {
            if (obs[oi].cam == 0) continue;
            const int ci = obs[oi].cam - 1;
            for (int oj : point_obs[p]) {
              if (obs[oj].cam == 0) continue;
              S.block<6, 6>(6 * ci, 6 * (obs[oj].cam - 1)) -=
                  E[oi] * Cinv[p] * E[oj].transpose();
            }
          }
        }
        dcam = S.ldlt().solve(b);
      } else {
        // Power-BA: S^{-1} b = (sum_k M^k) B^{-1} b, M = B^{-1} E C^{-1} E^T.
        // Each M*v is block mat-vecs (E^T, C^{-1}, E, B^{-1}) -- no factorization,
        // exactly the structure a Metal/GPU solver parallelizes.
        std::vector<Eigen::Matrix<double, 6, 6>> Binv(F);
        for (int c = 0; c < F; ++c) Binv[c] = Bd[c].inverse();
        auto apply_M = [&](const Eigen::VectorXd& v) {
          std::vector<Eigen::Vector3d> a(pts.size(), Eigen::Vector3d::Zero());
          for (int o = 0; o < num_obs; ++o) {
            if (obs[o].cam == 0) continue;
            a[obs[o].point] +=
                E[o].transpose() * v.segment<6>(6 * (obs[o].cam - 1));
          }
          for (size_t p = 0; p < pts.size(); ++p) a[p] = Cinv[p] * a[p];
          Eigen::VectorXd e = Eigen::VectorXd::Zero(6 * F);
          for (int o = 0; o < num_obs; ++o) {
            if (obs[o].cam == 0) continue;
            e.segment<6>(6 * (obs[o].cam - 1)) += E[o] * a[obs[o].point];
          }
          Eigen::VectorXd Mv = Eigen::VectorXd::Zero(6 * F);
          for (int c = 0; c < F; ++c) Mv.segment<6>(6 * c) = Binv[c] * e.segment<6>(6 * c);
          return Mv;
        };
        Eigen::VectorXd u(6 * F);
        for (int c = 0; c < F; ++c) u.segment<6>(6 * c) = Binv[c] * b.segment<6>(6 * c);
        dcam = u;
        Eigen::VectorXd term = u;
        for (int k = 0; k < 50; ++k) {
          term = apply_M(term);
          dcam += term;
          if (term.norm() < 1e-10 * (dcam.norm() + 1e-30)) break;
        }
      }

      // Back-substitute point updates and build trial state.
      std::vector<Camera> cams_t = cams;
      std::vector<Eigen::Vector3d> pts_t = pts;
      for (int c = 0; c < F; ++c) {
        const Eigen::Vector3d dth = dcam.segment<3>(6 * c);
        const Eigen::Vector3d dt = dcam.segment<3>(6 * c + 3);
        const double ang = dth.norm();
        const Eigen::Quaterniond dq =
            ang < 1e-12 ? Eigen::Quaterniond::Identity()
                        : Eigen::Quaterniond(Eigen::AngleAxisd(ang, dth / ang));
        cams_t[c + 1].q = (cams[c + 1].q * dq).normalized();
        cams_t[c + 1].t = cams[c + 1].t + dt;
      }
      for (size_t p = 0; p < pts.size(); ++p) {
        Eigen::Vector3d acc = -gpt[p];
        for (int oi : point_obs[p]) {
          if (obs[oi].cam == 0) continue;
          acc -= E[oi].transpose() * dcam.segment<6>(6 * (obs[oi].cam - 1));
        }
        pts_t[p] = pts[p] + Cinv[p] * acc;
      }

      const double cost_t = FullBACost(cams_t, pts_t, obs);
      if (cost_t < cost) {
        cams = cams_t; pts = pts_t; cost = cost_t;
        lambda = std::max(lambda * 0.3, 1e-12);
        accepted = true;
      } else {
        lambda = std::min(lambda * 10.0, 1e12);
      }
    }
    if (!accepted) break;
  }
  return cost;
}

TEST(BundleAdjustmentMetal, FullBAConvergesLikeCeres) {
  if (!IsBundleAdjustmentMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  SetPRNGSeed(123);
  const int num_cams = 6, num_points = 120;

  // Ground-truth cameras (orbit) + points (cloud near origin).
  std::vector<Camera> cams_true(num_cams);
  for (int c = 0; c < num_cams; ++c) {
    const Eigen::Vector3d eye(RandomUniformReal<double>(-1, 1),
                              RandomUniformReal<double>(-1, 1),
                              RandomUniformReal<double>(-1, 1));
    const Rigid3d cfw = LookAt(eye.normalized() * 6.0);
    cams_true[c].q = Eigen::Quaterniond(cfw.rotation());
    cams_true[c].t = cfw.translation();
    cams_true[c].intr = {1000.0, 1000.0, 640.0, 480.0};
  }
  std::vector<Eigen::Vector3d> pts_true(num_points);
  for (int p = 0; p < num_points; ++p) {
    pts_true[p] = Eigen::Vector3d(RandomUniformReal<double>(-1, 1),
                                  RandomUniformReal<double>(-1, 1),
                                  RandomUniformReal<double>(-1, 1));
  }
  std::vector<Obs> obs;
  for (int p = 0; p < num_points; ++p) {
    for (int c = 0; c < num_cams; ++c) {
      const Eigen::Vector3d pc = cams_true[c].q * pts_true[p] + cams_true[c].t;
      if (pc.z() <= 0.1) continue;
      const auto& ic = cams_true[c].intr;
      obs.push_back({c, p,
                     Eigen::Vector2d(
                         ic[0] * pc.x() / pc.z() + ic[2] +
                             RandomUniformReal<double>(-1, 1) * 0.5,
                         ic[1] * pc.y() / pc.z() + ic[3] +
                             RandomUniformReal<double>(-1, 1) * 0.5)});
    }
  }

  // Perturbed initial state (camera 0 left at truth = the fixed gauge).
  std::vector<Camera> cams0 = cams_true;
  std::vector<Eigen::Vector3d> pts0 = pts_true;
  for (int c = 1; c < num_cams; ++c) {
    cams0[c].q = (cams0[c].q * Eigen::Quaterniond(Eigen::AngleAxisd(
                      0.02, Eigen::Vector3d::UnitX())))
                     .normalized();
    cams0[c].t += Eigen::Vector3d(0.05, -0.05, 0.05);
  }
  for (auto& X : pts0)
    X += Eigen::Vector3d(RandomUniformReal<double>(-0.05, 0.05),
                         RandomUniformReal<double>(-0.05, 0.05),
                         RandomUniformReal<double>(-0.05, 0.05));

  const double init_cost = FullBACost(cams0, pts0, obs);

  // Metal hybrid BA: dense Schur LDLT.
  std::vector<Camera> cams_m = cams0;
  std::vector<Eigen::Vector3d> pts_m = pts0;
  const double metal_cost =
      MetalFullBA(cams_m, pts_m, obs, /*max_iters=*/50, /*solve_mode=*/0);

  // Metal hybrid BA: Power-BA power-series inverse Schur (GPU-friendly solve).
  std::vector<Camera> cams_pw = cams0;
  std::vector<Eigen::Vector3d> pts_pw = pts0;
  const double power_cost =
      MetalFullBA(cams_pw, pts_pw, obs, /*max_iters=*/50, /*solve_mode=*/1);

  // Ceres full BA: same problem, camera 0 fixed, intrinsics fixed.
  std::vector<std::array<double, 7>> cere_pose(num_cams);
  std::vector<std::array<double, 4>> cere_intr(num_cams);
  std::vector<std::array<double, 3>> cere_pts(num_points);
  for (int c = 0; c < num_cams; ++c) {
    cere_pose[c] = {cams0[c].q.x(), cams0[c].q.y(), cams0[c].q.z(),
                    cams0[c].q.w(), cams0[c].t.x(), cams0[c].t.y(),
                    cams0[c].t.z()};
    cere_intr[c] = cams0[c].intr;
  }
  for (int p = 0; p < num_points; ++p)
    cere_pts[p] = {pts0[p].x(), pts0[p].y(), pts0[p].z()};
  ceres::Problem problem;
  for (const Obs& o : obs) {
    problem.AddResidualBlock(
        ReprojErrorCostFunctor<PinholeCameraModel>::Create(o.pixel), nullptr,
        cere_pts[o.point].data(), cere_pose[o.cam].data(),
        cere_intr[o.cam].data());
  }
  // cam_from_world is a 7-param [quat|trans] block -> product manifold (tangent
  // 6), matching the SE(3) tangent the Metal solver uses. Intrinsics fixed,
  // camera 0 fixed (gauge).
  for (int c = 0; c < num_cams; ++c) {
    SetManifold(&problem, cere_pose[c].data(),
                CreateProductManifold(ceres::EigenQuaternionManifold(),
                                      ceres::EuclideanManifold<3>()));
    problem.SetParameterBlockConstant(cere_intr[c].data());
  }
  problem.SetParameterBlockConstant(cere_pose[0].data());
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.max_num_iterations = 50;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  const double ceres_cost = 2.0 * summary.final_cost;  // Ceres cost = 0.5*sum r^2.

  std::printf(
      "[ba-metal] full BA: init cost %.3f -> Metal(dense) %.4f, "
      "Metal(Power-BA) %.4f, Ceres %.4f (dense rel %.2e, power rel %.2e)\n",
      init_cost, metal_cost, power_cost, ceres_cost,
      std::abs(metal_cost - ceres_cost) / ceres_cost,
      std::abs(power_cost - ceres_cost) / ceres_cost);
  // Both Metal solvers must optimize substantially and reach Ceres' minimum.
  EXPECT_LT(metal_cost, init_cost * 0.5);
  EXPECT_LT(std::abs(metal_cost - ceres_cost) / ceres_cost, 0.02);
  EXPECT_LT(std::abs(power_cost - ceres_cost) / ceres_cost, 0.02);
}

}  // namespace
}  // namespace colmap
