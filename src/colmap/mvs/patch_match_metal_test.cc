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

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/workspace.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

// Independent reference for the plane-induced homography, built from matrices
// (not the expanded scalar form the kernel uses) so it genuinely cross-checks
// the port. Uses COLMAP's exact convention from patch_match_cuda.cu:
//   H = K_src * (R + T * n^T / d) * K_ref^-1
// with the plane distance d = depth * n . (K_ref^-1 * [col, row, 1]^T). (The
// sign is +; COLMAP folds the plane orientation into n and d so the T*n^T term
// is added, matching the `R[i] + inv_dist_N*T[j]` terms in the kernel.)
Eigen::Matrix3f ReferenceHomography(const float ref_inv_K[4],
                                    const float src_K[4],
                                    const Eigen::Matrix3f& R,
                                    const Eigen::Vector3f& T,
                                    int row,
                                    int col,
                                    float depth,
                                    const Eigen::Vector3f& normal) {
  Eigen::Matrix3f K_ref_inv;
  K_ref_inv << ref_inv_K[0], 0.0f, ref_inv_K[1], 0.0f, ref_inv_K[2],
      ref_inv_K[3], 0.0f, 0.0f, 1.0f;
  Eigen::Matrix3f K_src;
  K_src << src_K[0], 0.0f, src_K[1], 0.0f, src_K[2], src_K[3], 0.0f, 0.0f, 1.0f;

  const Eigen::Vector3f ray =
      K_ref_inv * Eigen::Vector3f(static_cast<float>(col),
                                  static_cast<float>(row), 1.0f);
  const float dist = depth * normal.dot(ray);
  return K_src * (R + (T * normal.transpose()) / dist) * K_ref_inv;
}

// Combined relative+absolute tolerance: the kernel runs in fp32 while the
// reference accumulates the matrix product, so a few ULP of disagreement per
// element is expected and acceptable.
void ExpectClose(const Eigen::Matrix3f& ref, const float* h) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const float expected = ref(r, c);
      const float actual = h[r * 3 + c];
      EXPECT_NEAR(actual, expected, 1e-4f * std::abs(expected) + 1e-5f)
          << "mismatch at H[" << r << "," << c << "]";
    }
  }
}

TEST(PatchMatchMetal, ComposeHomographyMatchesEigenReference) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available; skipping GPU homography test.";
  }

  // Reference camera fx=fy=1000, cx=320, cy=240.
  const float ref_inv_K[4] = {1.0f / 1000.0f, -320.0f / 1000.0f,
                              1.0f / 1000.0f, -240.0f / 1000.0f};
  // Source camera with slightly different intrinsics.
  const float src_K[4] = {1010.0f, 322.0f, 1005.0f, 238.0f};

  // A small relative rotation and a baseline translation (reference -> source).
  const Eigen::Matrix3f R =
      (Eigen::AngleAxisf(0.05f, Eigen::Vector3f::UnitY()) *
       Eigen::AngleAxisf(0.02f, Eigen::Vector3f::UnitX()))
          .toRotationMatrix();
  const Eigen::Vector3f T(0.12f, -0.03f, 0.04f);

  // A batch of hypotheses: varied pixels, depths, and (unit) normals, including
  // a fronto-parallel normal and slanted ones.
  struct Hyp {
    int row, col;
    float depth;
    Eigen::Vector3f normal;
  };
  std::vector<Hyp> hyps = {
      {240, 320, 3.0f, Eigen::Vector3f(0.0f, 0.0f, -1.0f)},
      {100, 150, 5.0f, Eigen::Vector3f(0.1f, -0.2f, -1.0f).normalized()},
      {400, 600, 1.8f, Eigen::Vector3f(-0.3f, 0.15f, -1.0f).normalized()},
      {50, 500, 8.0f, Eigen::Vector3f(0.0f, 0.4f, -1.0f).normalized()},
  };

  const int num = static_cast<int>(hyps.size());
  std::vector<int> rows(num), cols(num);
  std::vector<float> depths(num), normals(num * 3);
  float src_R[9], src_T[3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) src_R[i * 3 + j] = R(i, j);
    src_T[i] = T(i);
  }
  for (int i = 0; i < num; ++i) {
    rows[i] = hyps[i].row;
    cols[i] = hyps[i].col;
    depths[i] = hyps[i].depth;
    normals[3 * i + 0] = hyps[i].normal.x();
    normals[3 * i + 1] = hyps[i].normal.y();
    normals[3 * i + 2] = hyps[i].normal.z();
  }

  std::vector<float> out_H(num * 9, 0.0f);
  ASSERT_TRUE(ComposePlaneHomographiesMetal(ref_inv_K, src_K, src_R, src_T,
                                            rows.data(), cols.data(),
                                            depths.data(), normals.data(), num,
                                            out_H.data()));

  for (int i = 0; i < num; ++i) {
    const Eigen::Matrix3f ref = ReferenceHomography(
        ref_inv_K, src_K, R, T, hyps[i].row, hyps[i].col, hyps[i].depth,
        hyps[i].normal);
    ExpectClose(ref, out_H.data() + i * 9);
  }
}

TEST(PatchMatchMetal, FrontoParallelIdentityWhenSameCamera) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available; skipping GPU homography test.";
  }
  // Identical reference/source camera, identity rotation, zero translation:
  // the homography must be the identity (up to fp error), independent of depth.
  const float ref_inv_K[4] = {1.0f / 800.0f, -400.0f / 800.0f, 1.0f / 800.0f,
                              -300.0f / 800.0f};
  const float src_K[4] = {800.0f, 400.0f, 800.0f, 300.0f};
  float src_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float src_T[3] = {0, 0, 0};
  int rows[1] = {300}, cols[1] = {400};
  float depths[1] = {4.2f};
  float normals[3] = {0.0f, 0.0f, -1.0f};
  float h[9] = {0};
  ASSERT_TRUE(ComposePlaneHomographiesMetal(ref_inv_K, src_K, src_R, src_T,
                                            rows, cols, depths, normals, 1, h));
  const float kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(h[i], kIdentity[i], 1e-4f) << "at index " << i;
  }
}

TEST(PatchMatchMetal, ReturnsFalseForEmptyBatch) {
  float ref_inv_K[4] = {0.001f, -0.32f, 0.001f, -0.24f};
  float src_K[4] = {1000, 320, 1000, 240};
  float src_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float src_T[3] = {0, 0, 0};
  float out = 0.0f;
  EXPECT_FALSE(ComposePlaneHomographiesMetal(ref_inv_K, src_K, src_R, src_T,
                                            nullptr, nullptr, nullptr, nullptr,
                                            0, &out));
}

// ---------------------------------------------------------------------------
// Bilateral-weighted NCC photometric cost.
// ---------------------------------------------------------------------------

namespace {

// Deterministic textured image in roughly [0.15, 0.85].
std::vector<float> MakeTexture(int width, int height) {
  std::vector<float> img(static_cast<size_t>(width) * height);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      img[static_cast<size_t>(r) * width + c] =
          0.5f + 0.25f * std::sin(0.30f * c) * std::cos(0.21f * r) +
          0.10f * std::sin(0.07f * (c + r));
    }
  }
  return img;
}

// Faithful CPU reference for the bilateral-weighted NCC cost (same math as the
// MSL kernel and patch_match_cuda.cu's PhotoConsistencyCostComputer).
float NCCCostCpuReference(const std::vector<float>& ref,
                          const std::vector<float>& src, int W, int H, int wr,
                          int ws, float sigma_spatial, float sigma_color,
                          int crow, int ccol, const float* Hm) {
  auto pt = [&](const std::vector<float>& img, int c, int r) -> float {
    if (c < 0 || c >= W || r < 0 || r >= H) return 0.0f;
    return img[static_cast<size_t>(r) * W + c];
  };
  auto bil = [&](const std::vector<float>& img, float x, float y) -> float {
    const int c0 = static_cast<int>(std::floor(x));
    const int r0 = static_cast<int>(std::floor(y));
    const float ac = x - c0;
    const float ar = y - r0;
    return (1 - ar) * ((1 - ac) * pt(img, c0, r0) + ac * pt(img, c0 + 1, r0)) +
           ar * ((1 - ac) * pt(img, c0, r0 + 1) + ac * pt(img, c0 + 1, r0 + 1));
  };
  const float spatial_norm = 1.0f / (2.0f * sigma_spatial * sigma_spatial);
  const float color_norm = 1.0f / (2.0f * sigma_color * sigma_color);
  const float center = pt(ref, ccol, crow);
  float rs = 0, rsq = 0, ss = 0, ssq = 0, sr = 0, wsum = 0;
  for (int dr = -wr; dr <= wr; dr += ws) {
    for (int dc = -wr; dc <= wr; dc += ws) {
      const int cc = ccol + dc, rr = crow + dr;
      const float rc = pt(ref, cc, rr);
      const float cd = center - rc;
      const float w = std::exp(-static_cast<float>(dr * dr + dc * dc) *
                                   spatial_norm -
                               cd * cd * color_norm);
      const float xs = Hm[0] * cc + Hm[1] * rr + Hm[2];
      const float ys = Hm[3] * cc + Hm[4] * rr + Hm[5];
      const float zs = Hm[6] * cc + Hm[7] * rr + Hm[8];
      const float inv = 1.0f / zs;
      const float sc = bil(src, inv * xs, inv * ys);
      rs += w * rc;
      rsq += w * rc * rc;
      const float wsrc = w * sc;
      ss += wsrc;
      ssq += wsrc * sc;
      sr += wsrc * rc;
      wsum += w;
    }
  }
  const float invw = 1.0f / wsum;
  rs *= invw;
  rsq *= invw;
  ss *= invw;
  ssq *= invw;
  sr *= invw;
  const float rv = rsq - rs * rs;
  const float sv = ssq - ss * ss;
  if (rv < 1e-5f || sv < 1e-5f) return 2.0f;
  const float covar = sr - rs * ss;
  return std::max(0.0f, std::min(2.0f, 1.0f - covar / std::sqrt(rv * sv)));
}

const float kIdentityH[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

}  // namespace

TEST(PatchMatchMetalNCC, IdentityHomographySamePatchIsZeroCost) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 128, H = 96, wr = 5, ws = 1;
  const std::vector<float> img = MakeTexture(W, H);
  // Reference == source, identity warp -> patches are identical -> NCC = 1 ->
  // cost = 0. Use interior pixels so the window stays in bounds.
  const std::vector<int> rows = {30, 48, 70};
  const std::vector<int> cols = {40, 64, 90};
  const int num = static_cast<int>(rows.size());
  std::vector<float> H_all;
  for (int i = 0; i < num; ++i)
    H_all.insert(H_all.end(), kIdentityH, kIdentityH + 9);
  std::vector<float> cost(num, -1.0f);
  ASSERT_TRUE(ComputeNCCPhotometricCostMetal(
      img.data(), img.data(), W, H, wr, ws, 3.0f, 0.2f, rows.data(),
      cols.data(), H_all.data(), num, cost.data()));
  for (int i = 0; i < num; ++i) {
    EXPECT_NEAR(cost[i], 0.0f, 1e-4f) << "pixel " << i;
  }
}

TEST(PatchMatchMetalNCC, ConstantPatchReturnsMaxCost) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 64, H = 64, wr = 4, ws = 1;
  const std::vector<float> flat(static_cast<size_t>(W) * H, 0.5f);
  const int rows[1] = {32}, cols[1] = {32};
  std::vector<float> H_all(kIdentityH, kIdentityH + 9);
  float cost = -1.0f;
  ASSERT_TRUE(ComputeNCCPhotometricCostMetal(flat.data(), flat.data(), W, H, wr,
                                             ws, 3.0f, 0.2f, rows, cols,
                                             H_all.data(), 1, &cost));
  // Zero variance -> kMaxCost.
  EXPECT_FLOAT_EQ(cost, 2.0f);
}

TEST(PatchMatchMetalNCC, MatchesCpuReferenceForNonTrivialHomography) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 160, H = 120, wr = 5, ws = 1;
  const float sigma_spatial = 3.0f, sigma_color = 0.2f;
  const std::vector<float> ref = MakeTexture(W, H);
  const std::vector<float> src = MakeTexture(W, H);  // same content.

  // A mild projective warp (scale + skew + slight perspective).
  const float Hm[9] = {1.02f, 0.03f,  1.5f, -0.01f, 0.99f,
                       -2.0f, 1e-4f, 5e-5f, 1.0f};
  const std::vector<int> rows = {35, 50, 60, 80};
  const std::vector<int> cols = {45, 70, 100, 120};
  const int num = static_cast<int>(rows.size());
  std::vector<float> H_all;
  for (int i = 0; i < num; ++i) H_all.insert(H_all.end(), Hm, Hm + 9);

  std::vector<float> cost(num, -1.0f);
  ASSERT_TRUE(ComputeNCCPhotometricCostMetal(
      ref.data(), src.data(), W, H, wr, ws, sigma_spatial, sigma_color,
      rows.data(), cols.data(), H_all.data(), num, cost.data()));

  for (int i = 0; i < num; ++i) {
    const float expected =
        NCCCostCpuReference(ref, src, W, H, wr, ws, sigma_spatial, sigma_color,
                            rows[i], cols[i], Hm);
    EXPECT_NEAR(cost[i], expected, 1e-4f) << "pixel " << i;
  }
}

// ---------------------------------------------------------------------------
// Plane-sweep depth estimator: recover synthetic ground-truth depth.
// ---------------------------------------------------------------------------

namespace {

// Lower-frequency texture so the photometric minimum is unique across the
// disparity search range (avoids aliasing into a wrong local minimum).
std::vector<float> MakeStereoTexture(int width, int height) {
  std::vector<float> img(static_cast<size_t>(width) * height);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      img[static_cast<size_t>(r) * width + c] =
          0.5f + 0.25f * std::sin(0.10f * c + 0.06f * r) +
          0.12f * std::sin(0.31f * c) * std::cos(0.17f * r);
    }
  }
  return img;
}

float SampleBilinearHost(const std::vector<float>& img, int W, int H, float x,
                         float y) {
  auto pt = [&](int c, int r) -> float {
    if (c < 0 || c >= W || r < 0 || r >= H) return 0.0f;
    return img[static_cast<size_t>(r) * W + c];
  };
  const int c0 = static_cast<int>(std::floor(x));
  const int r0 = static_cast<int>(std::floor(y));
  const float ac = x - c0, ar = y - r0;
  return (1 - ar) * ((1 - ac) * pt(c0, r0) + ac * pt(c0 + 1, r0)) +
         ar * ((1 - ac) * pt(c0, r0 + 1) + ac * pt(c0 + 1, r0 + 1));
}

}  // namespace

// Rectified stereo (R = I, T = (b, 0, 0)) viewing a fronto-parallel plane at
// known depth d_gt. The induced homography reduces to a horizontal shift of
// disparity = fx * b / d_gt, so the source image is the reference shifted right
// by that disparity. The plane sweep must recover d_gt as the lowest-cost depth.
TEST(PatchMatchMetalSweep, RecoversSyntheticGroundTruthDepth) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 220, H = 90;
  const float fx = 400.0f, fy = 400.0f, cx = 110.0f, cy = 45.0f;
  const float baseline = 0.15f;
  const float d_gt = 5.0f;
  const float disp_gt = fx * baseline / d_gt;  // = 12.0 px

  const std::vector<float> ref = MakeStereoTexture(W, H);
  // src(c, r) = ref(c - disp_gt, r): ref pixel (col) appears at (col + disp_gt).
  std::vector<float> src(static_cast<size_t>(W) * H);
  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      src[static_cast<size_t>(r) * W + c] =
          SampleBilinearHost(ref, W, H, c - disp_gt, r);
    }
  }

  const float ref_inv_K[4] = {1.0f / fx, -cx / fx, 1.0f / fy, -cy / fy};
  const float src_K[4] = {fx, cx, fy, cy};
  const float src_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float src_T[3] = {baseline, 0, 0};
  const float normal[3] = {0.0f, 0.0f, -1.0f};  // fronto-parallel.

  // Sweep depth over a range bracketing d_gt.
  std::vector<float> depth_candidates;
  const int kNumDepths = 36;
  const float d_lo = 3.5f, d_hi = 7.0f;
  for (int i = 0; i < kNumDepths; ++i) {
    depth_candidates.push_back(d_lo + (d_hi - d_lo) * i / (kNumDepths - 1));
  }

  // Interior, textured pixels (window + max disparity stay in bounds).
  std::vector<int> rows, cols;
  for (int r = 30; r <= 60; r += 6) {
    for (int c = 50; c <= 150; c += 10) {
      rows.push_back(r);
      cols.push_back(c);
    }
  }
  const int num = static_cast<int>(rows.size());
  std::vector<float> out_depth(num, 0.0f), out_cost(num, 0.0f);

  ASSERT_TRUE(PlaneSweepDepthMetal(
      ref.data(), src.data(), W, H, ref_inv_K, src_K, src_R, src_T, normal,
      depth_candidates.data(), kNumDepths, /*window_radius=*/5,
      /*window_step=*/1, /*sigma_spatial=*/3.0f, /*sigma_color=*/0.2f,
      rows.data(), cols.data(), num, out_depth.data(), out_cost.data()));

  // Most pixels should recover d_gt within ~half the search resolution; the
  // median error must be small. (A few low-texture pixels may miss.)
  std::vector<float> errors;
  int within_half = 0;
  for (int i = 0; i < num; ++i) {
    const float err = std::abs(out_depth[i] - d_gt);
    errors.push_back(err);
    if (err <= 0.5f) ++within_half;
  }
  std::sort(errors.begin(), errors.end());
  const float median_err = errors[errors.size() / 2];
  EXPECT_LT(median_err, 0.3f) << "median depth error too large";
  EXPECT_GT(within_half, num * 0.8) << "too few pixels recovered near GT depth";
}

// ---------------------------------------------------------------------------
// Fused hypothesis-cost evaluation + PatchMatch optimizer.
// ---------------------------------------------------------------------------

// The fused (depth,normal)->H->cost kernel must agree with composing the
// homography and evaluating the NCC cost in two separate steps.
TEST(PatchMatchMetalHypothesis, FusedCostMatchesSeparateComposeAndNcc) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 160, Ht = 120, wr = 5, ws = 1;
  const float ss = 3.0f, sc = 0.2f;
  const std::vector<float> ref = MakeTexture(W, Ht);
  const std::vector<float> src = MakeTexture(W, Ht);
  const float fx = 500, fy = 500, cx = 80, cy = 60;
  const float ref_inv_K[4] = {1.0f / fx, -cx / fx, 1.0f / fy, -cy / fy};
  const float src_K[4] = {fx, cx, fy, cy};
  const float src_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float src_T[3] = {0.1f, 0.0f, 0.0f};

  const std::vector<int> rows = {40, 55, 70, 60};
  const std::vector<int> cols = {50, 70, 90, 110};
  const int num = static_cast<int>(rows.size());
  std::vector<float> depths = {4.0f, 5.0f, 3.5f, 6.0f};
  std::vector<float> normals;
  for (int i = 0; i < num; ++i) {
    normals.push_back(0.05f * i);
    normals.push_back(-0.05f);
    normals.push_back(-1.0f);
  }
  for (int i = 0; i < num; ++i) {  // Normalize.
    float* n = normals.data() + 3 * i;
    const float inv = 1.0f / std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    n[0] *= inv;
    n[1] *= inv;
    n[2] *= inv;
  }

  std::vector<float> Hmat(num * 9), cost_separate(num), cost_fused(num);
  ASSERT_TRUE(ComposePlaneHomographiesMetal(ref_inv_K, src_K, src_R, src_T,
                                            rows.data(), cols.data(),
                                            depths.data(), normals.data(), num,
                                            Hmat.data()));
  ASSERT_TRUE(ComputeNCCPhotometricCostMetal(
      ref.data(), src.data(), W, Ht, wr, ws, ss, sc, rows.data(), cols.data(),
      Hmat.data(), num, cost_separate.data()));
  ASSERT_TRUE(EvaluateHypothesisCostMetal(
      ref.data(), src.data(), W, Ht, ref_inv_K, src_K, src_R, src_T, wr, ws, ss,
      sc, rows.data(), cols.data(), depths.data(), normals.data(), num,
      cost_fused.data()));
  for (int i = 0; i < num; ++i) {
    EXPECT_NEAR(cost_fused[i], cost_separate[i], 1e-5f) << "pixel " << i;
  }
}

// Multi-view aggregation: two rectified source views (left + right baselines)
// both consistent with a fronto-parallel plane at d_gt. The best-1 aggregated
// cost must be minimized at d_gt, and an added third OCCLUDED view (garbage)
// must be ignored by the best-2-of-3 selection.
TEST(PatchMatchMetalMultiView, BestKAggregationIsRobust) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 180, Ht = 80;
  const float fx = 400, fy = 400, cx = 90, cy = 40;
  const float d_gt = 5.0f;
  const float ref_inv_K[4] = {1.0f / fx, -cx / fx, 1.0f / fy, -cy / fy};

  const std::vector<float> ref = MakeStereoTexture(W, Ht);
  // Source 0: baseline +0.15 (disp +12), source 1: baseline -0.12 (disp -9.6).
  auto shifted = [&](float disp) {
    std::vector<float> s(static_cast<size_t>(W) * Ht);
    for (int r = 0; r < Ht; ++r)
      for (int c = 0; c < W; ++c)
        s[static_cast<size_t>(r) * W + c] = SampleBilinearHost(ref, W, Ht, c - disp, r);
    return s;
  };
  const float b0 = 0.15f, b1 = -0.12f;
  const std::vector<float> s0 = shifted(fx * b0 / d_gt);
  const std::vector<float> s1 = shifted(fx * b1 / d_gt);
  // Occluded/garbage third view: constant texture (never matches).
  const std::vector<float> s2(static_cast<size_t>(W) * Ht, 0.5f);

  // Pack 3 source planes + per-source K/R/T (rectified: R=I, T=(b,0,0)).
  std::vector<float> src_imgs;
  src_imgs.insert(src_imgs.end(), s0.begin(), s0.end());
  src_imgs.insert(src_imgs.end(), s1.begin(), s1.end());
  src_imgs.insert(src_imgs.end(), s2.begin(), s2.end());
  const float src_K[12] = {fx, cx, fy, cy, fx, cx, fy, cy, fx, cx, fy, cy};
  const float src_R[27] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                           0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float src_T[9] = {b0, 0, 0, b1, 0, 0, 0.2f, 0, 0};

  // One central pixel; sweep depth and confirm the best-2-of-3 cost dips at d_gt.
  const int rows[1] = {40}, cols[1] = {90};
  const float normal[3] = {0, 0, -1};
  float best_cost = 1e9f, best_depth = -1.0f;
  for (int i = 0; i <= 40; ++i) {
    const float d = 3.0f + (7.0f - 3.0f) * i / 40.0f;
    const float depths[1] = {d};
    float cost = -1.0f;
    ASSERT_TRUE(EvaluateHypothesisCostMultiViewMetal(
        ref.data(), src_imgs.data(), /*num_src=*/3, W, Ht, ref_inv_K, src_K,
        src_R, src_T, /*best_k=*/2, /*window_radius=*/5, /*window_step=*/1,
        3.0f, 0.2f, rows, cols, depths, normal, 1, &cost));
    if (cost < best_cost) {
      best_cost = cost;
      best_depth = d;
    }
  }
  // The two good views agree at d_gt; best-2-of-3 ignores the garbage view, so
  // the minimum is at the true depth.
  EXPECT_NEAR(best_depth, d_gt, 0.25f);
}

// The optimizer must recover a fronto-parallel ground-truth plane's depth from a
// poor (too-near) initialization, via propagation + depth refinement.
TEST(PatchMatchMetalOptimizer, RecoversFrontoParallelDepthFromPoorInit) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 160, H = 70;
  const float fx = 400, fy = 400, cx = 80, cy = 35;
  const float baseline = 0.15f, d_gt = 5.0f;
  const float disp_gt = fx * baseline / d_gt;  // 12 px

  const std::vector<float> ref = MakeStereoTexture(W, H);
  std::vector<float> src(static_cast<size_t>(W) * H);
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c)
      src[static_cast<size_t>(r) * W + c] =
          SampleBilinearHost(ref, W, H, c - disp_gt, r);

  const float ref_inv_K[4] = {1.0f / fx, -cx / fx, 1.0f / fy, -cy / fy};
  const float src_K[4] = {fx, cx, fy, cy};
  const float src_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float src_T[3] = {baseline, 0, 0};

  // Poor init: every pixel at the near plane, fronto-parallel.
  const size_t N = static_cast<size_t>(W) * H;
  std::vector<float> depth(N, 3.0f), normal(3 * N);
  for (size_t i = 0; i < N; ++i) {
    normal[3 * i + 0] = 0.0f;
    normal[3 * i + 1] = 0.0f;
    normal[3 * i + 2] = -1.0f;
  }

  ASSERT_TRUE(RunPatchMatchOptimizerMetal(
      ref.data(), src.data(), W, H, ref_inv_K, src_K, src_R, src_T,
      /*window_radius=*/5, /*window_step=*/1, /*sigma_spatial=*/3.0f,
      /*sigma_color=*/0.2f, /*depth_min=*/2.5f, /*depth_max=*/8.0f,
      /*num_iterations=*/24, depth.data(), normal.data()));

  // Check recovered depth on interior, textured pixels.
  std::vector<float> errors;
  for (int r = 25; r <= 45; r += 4) {
    for (int c = 40; c <= 120; c += 8) {
      errors.push_back(std::abs(depth[static_cast<size_t>(r) * W + c] - d_gt));
    }
  }
  std::sort(errors.begin(), errors.end());
  const float median_err = errors[errors.size() / 2];
  EXPECT_LT(median_err, 0.4f) << "optimizer did not converge to GT depth";
}

// The multi-view optimizer must recover the GT depth from a poor init using two
// source views (left + right baselines) via propagation + refinement.
TEST(PatchMatchMetalOptimizer, MultiViewRecoversDepthFromPoorInit) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 160, Ht = 70;
  const float fx = 400, fy = 400, cx = 80, cy = 35;
  const float d_gt = 5.0f;
  const float ref_inv_K[4] = {1.0f / fx, -cx / fx, 1.0f / fy, -cy / fy};
  const std::vector<float> ref = MakeStereoTexture(W, Ht);
  auto shifted = [&](float disp) {
    std::vector<float> s(static_cast<size_t>(W) * Ht);
    for (int r = 0; r < Ht; ++r)
      for (int c = 0; c < W; ++c)
        s[static_cast<size_t>(r) * W + c] = SampleBilinearHost(ref, W, Ht, c - disp, r);
    return s;
  };
  const float b0 = 0.15f, b1 = -0.12f;
  const std::vector<float> s0 = shifted(fx * b0 / d_gt);
  const std::vector<float> s1 = shifted(fx * b1 / d_gt);
  std::vector<float> src_imgs;
  src_imgs.insert(src_imgs.end(), s0.begin(), s0.end());
  src_imgs.insert(src_imgs.end(), s1.begin(), s1.end());
  const float src_K[8] = {fx, cx, fy, cy, fx, cx, fy, cy};
  const float src_R[18] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float src_T[6] = {b0, 0, 0, b1, 0, 0};

  const size_t N = static_cast<size_t>(W) * Ht;
  std::vector<float> depth(N, 3.0f), normal(3 * N);
  for (size_t i = 0; i < N; ++i) {
    normal[3 * i + 2] = -1.0f;
  }
  ASSERT_TRUE(RunPatchMatchMultiViewOptimizerMetal(
      ref.data(), src_imgs.data(), /*num_src=*/2, W, Ht, ref_inv_K, src_K,
      src_R, src_T, /*best_k=*/2, /*window_radius=*/5, /*window_step=*/1, 3.0f,
      0.2f, /*depth_min=*/2.5f, /*depth_max=*/8.0f, /*num_iterations=*/24,
      depth.data(), normal.data()));
  std::vector<float> errors;
  for (int r = 25; r <= 45; r += 4)
    for (int c = 40; c <= 120; c += 8)
      errors.push_back(std::abs(depth[static_cast<size_t>(r) * W + c] - d_gt));
  std::sort(errors.begin(), errors.end());
  EXPECT_LT(errors[errors.size() / 2], 0.4f) << "multi-view optimizer failed";
}

// The optimizer's stochastic restarts use a fixed seed, so identical inputs must
// give bit-identical outputs (reproducibility -- important for regression tests).
TEST(PatchMatchMetalOptimizer, IsReproducible) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 120, Ht = 60;
  const float fx = 400, fy = 400, cx = 60, cy = 30;
  const float d_gt = 5.0f, disp = fx * 0.15f / d_gt;
  const std::vector<float> ref = MakeStereoTexture(W, Ht);
  std::vector<float> src(static_cast<size_t>(W) * Ht);
  for (int r = 0; r < Ht; ++r)
    for (int c = 0; c < W; ++c)
      src[static_cast<size_t>(r) * W + c] = SampleBilinearHost(ref, W, Ht, c - disp, r);
  const float ref_inv_K[4] = {1.0f / fx, -cx / fx, 1.0f / fy, -cy / fy};
  const float src_K[4] = {fx, cx, fy, cy};
  const float src_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float src_T[3] = {0.15f, 0, 0};
  const size_t N = static_cast<size_t>(W) * Ht;

  auto run = [&](std::vector<float>& d, std::vector<float>& n) {
    d.assign(N, 3.0f);
    n.assign(3 * N, 0.0f);
    for (size_t i = 0; i < N; ++i) n[3 * i + 2] = -1.0f;
    ASSERT_TRUE(RunPatchMatchOptimizerMetal(
        ref.data(), src.data(), W, Ht, ref_inv_K, src_K, src_R, src_T, 5, 1,
        3.0f, 0.2f, 2.5f, 8.0f, 12, d.data(), n.data()));
  };
  std::vector<float> d1, n1, d2, n2;
  run(d1, n1);
  run(d2, n2);
  EXPECT_EQ(d1, d2);
  EXPECT_EQ(n1, n2);
}

// Geometric consistency: a fronto-parallel plane at d_gt seen by a reference and
// a source camera. The source depth map is the plane's true source-frame depth.
// The geom cost (forward-backward reprojection error) must be ~0 at d_gt and
// grow for wrong depths.
TEST(PatchMatchMetalGeom, ConsistentDepthHasLowReprojectionError) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 180, Ht = 80;
  const float fx = 400, fy = 400, cx = 90, cy = 40;
  const float d_gt = 5.0f;
  const float ref_inv_K[4] = {1.0f / fx, -cx / fx, 1.0f / fy, -cy / fy};
  const float ref_K[4] = {fx, cx, fy, cy};

  // Source camera: small rotation about y + a baseline (reference -> source).
  const Eigen::Matrix3f R =
      Eigen::AngleAxisf(0.03f, Eigen::Vector3f::UnitY()).toRotationMatrix();
  const Eigen::Vector3f T(0.2f, 0.0f, 0.02f);
  Eigen::Matrix3f Ks;
  Ks << fx, 0, cx, 0, fy, cy, 0, 0, 1;

  // P = Ks [R | T]; inv_P = [R^T Ks^-1 | -R^T T].
  Eigen::Matrix<float, 3, 4> RT;
  RT.block<3, 3>(0, 0) = R;
  RT.col(3) = T;
  const Eigen::Matrix<float, 3, 4> P = Ks * RT;
  Eigen::Matrix<float, 3, 4> invP;
  invP.block<3, 3>(0, 0) = R.transpose() * Ks.inverse();
  invP.col(3) = -R.transpose() * T;

  float src_P[12], src_inv_P[12];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c) {
      src_P[r * 4 + c] = P(r, c);
      src_inv_P[r * 4 + c] = invP(r, c);
    }

  // Source depth map: source-frame depth of the plane (z_ref = d_gt) at each
  // source pixel. Ray dir = Ks^-1 [u, v, 1]; t solves (R^T(t*dir - T))_z = d_gt.
  const Eigen::Vector3f r2 = R.col(2);  // third column = R^T's third row.
  std::vector<float> src_depth(static_cast<size_t>(W) * Ht, 0.0f);
  for (int v = 0; v < Ht; ++v) {
    for (int u = 0; u < W; ++u) {
      const Eigen::Vector3f dir((u - cx) / fx, (v - cy) / fy, 1.0f);
      const float t = (d_gt + r2.dot(T)) / r2.dot(dir);
      if (t > 0) src_depth[static_cast<size_t>(v) * W + u] = t;
    }
  }

  // Sweep the central reference pixel's depth; geom cost must dip at d_gt.
  const int rows[1] = {40}, cols[1] = {90};
  float cost_at_gt = -1.0f, best_cost = 1e9f, best_depth = -1.0f;
  for (int i = 0; i <= 40; ++i) {
    const float d = 3.0f + (7.0f - 3.0f) * i / 40.0f;
    const float depths[1] = {d};
    float cost = -1.0f;
    ASSERT_TRUE(ComputeGeomConsistencyCostMetal(
        W, Ht, ref_inv_K, ref_K, src_P, src_inv_P, src_depth.data(),
        /*num_src=*/1, /*max_cost=*/5.0f, rows, cols, depths, 1, &cost));
    if (std::abs(d - d_gt) < 1e-3f) cost_at_gt = cost;
    if (cost < best_cost) {
      best_cost = cost;
      best_depth = d;
    }
  }
  EXPECT_LT(cost_at_gt, 0.7f) << "reprojection error at GT depth too large";
  EXPECT_NEAR(best_depth, d_gt, 0.4f) << "geom cost not minimized at GT depth";
}

// The fused multi-source count kernel (ComputeGeomConsistencyCountMetal) must
// agree, per pixel, with summing the single-source cost kernel's consistency
// verdicts -- the exact equivalence the controller relies on when it replaces
// the per-source dispatch loop with one fused dispatch. Uses two sources, one of
// which is invalid (zero depth map), so the per-source verdicts differ.
TEST(PatchMatchMetalGeom, FusedCountMatchesPerSourceVerdicts) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 180, Ht = 80;
  const float fx = 400, fy = 400, cx = 90, cy = 40;
  const float d_gt = 5.0f;
  const float ref_inv_K[4] = {1.0f / fx, -cx / fx, 1.0f / fy, -cy / fy};
  const float ref_K[4] = {fx, cx, fy, cy};

  const Eigen::Matrix3f Rot =
      Eigen::AngleAxisf(0.03f, Eigen::Vector3f::UnitY()).toRotationMatrix();
  const Eigen::Vector3f T(0.2f, 0.0f, 0.02f);
  Eigen::Matrix3f Ks;
  Ks << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  Eigen::Matrix<float, 3, 4> RT;
  RT.block<3, 3>(0, 0) = Rot;
  RT.col(3) = T;
  const Eigen::Matrix<float, 3, 4> P = Ks * RT;
  Eigen::Matrix<float, 3, 4> invP;
  invP.block<3, 3>(0, 0) = Rot.transpose() * Ks.inverse();
  invP.col(3) = -Rot.transpose() * T;

  // Two sources: source 0 has the true plane depth map, source 1 is all-invalid
  // (zero), so source 0 is consistent near d_gt and source 1 never is.
  std::vector<float> src_P(24), src_inv_P(24);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c) {
      src_P[r * 4 + c] = P(r, c);
      src_inv_P[r * 4 + c] = invP(r, c);
      src_P[12 + r * 4 + c] = P(r, c);
      src_inv_P[12 + r * 4 + c] = invP(r, c);
    }

  const Eigen::Vector3f r2 = Rot.col(2);
  const size_t plane = static_cast<size_t>(W) * Ht;
  std::vector<float> src_depths(2 * plane, 0.0f);  // source 1 stays zero.
  for (int v = 0; v < Ht; ++v) {
    for (int u = 0; u < W; ++u) {
      const Eigen::Vector3f dir((u - cx) / fx, (v - cy) / fy, 1.0f);
      const float t = (d_gt + r2.dot(T)) / r2.dot(dir);
      if (t > 0) src_depths[static_cast<size_t>(v) * W + u] = t;
    }
  }

  // Reference pixels swept across depths so verdicts span 0..2 consistent.
  std::vector<int> rows, cols;
  std::vector<float> depths;
  for (int i = 0; i <= 40; ++i) {
    rows.push_back(40);
    cols.push_back(90);
    depths.push_back(3.0f + (7.0f - 3.0f) * i / 40.0f);
  }
  const int num = static_cast<int>(depths.size());
  const float cap = 10.0f, keep_max = 1.0f;

  // Per-source verdicts via the single-source cost kernel.
  std::vector<int> expected(num, 0);
  for (int s = 0; s < 2; ++s) {
    std::vector<float> cost(num, -1.0f);
    ASSERT_TRUE(ComputeGeomConsistencyCostMetal(
        W, Ht, ref_inv_K, ref_K, src_P.data() + s * 12,
        src_inv_P.data() + s * 12, src_depths.data() + s * plane,
        /*num_src=*/1, cap, rows.data(), cols.data(), depths.data(), num,
        cost.data()));
    for (int p = 0; p < num; ++p)
      if (cost[p] <= keep_max) ++expected[p];
  }

  std::vector<int> count(num, -1);
  ASSERT_TRUE(ComputeGeomConsistencyCountMetal(
      W, Ht, ref_inv_K, ref_K, src_P.data(), src_inv_P.data(),
      src_depths.data(), /*num_src=*/2, keep_max, rows.data(), cols.data(),
      depths.data(), num, count.data()));

  int max_seen = 0;
  for (int p = 0; p < num; ++p) {
    EXPECT_EQ(count[p], expected[p]) << "pixel " << p;
    max_seen = std::max(max_seen, count[p]);
  }
  EXPECT_GE(max_seen, 1) << "expected at least some consistent depths near GT";
}

// Integration bridge: cameras given in COLMAP world->cam convention. Both share
// the same rotation Q (so the reference->source relative rotation must cancel to
// identity) and differ by a horizontal baseline, giving pure disparity. The
// bridge must derive the relative pose via ComputeRelativePose and recover depth.
TEST(PatchMatchMetalBridge, RecoversDepthFromWorldToCamPoses) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const int W = 160, Ht = 70;
  const float fx = 400, fy = 400, cx = 80, cy = 35;
  const float d_gt = 5.0f, tx = 0.15f;
  const float disp = fx * tx / d_gt;  // 12 px

  const std::vector<float> ref = MakeStereoTexture(W, Ht);
  std::vector<float> src(static_cast<size_t>(W) * Ht);
  for (int r = 0; r < Ht; ++r)
    for (int c = 0; c < W; ++c)
      src[static_cast<size_t>(r) * W + c] = SampleBilinearHost(ref, W, Ht, c - disp, r);

  const float K9[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
  // Shared (non-identity) world->cam rotation for both cameras.
  const Eigen::Matrix3f Q =
      Eigen::AngleAxisf(0.2f, Eigen::Vector3f(0.3f, 1.0f, 0.1f).normalized())
          .toRotationMatrix();
  float Q9[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) Q9[r * 3 + c] = Q(r, c);
  // Equal rotations -> relative R = Q Q^T = I; T_rel = T_src - T_ref = (tx,0,0).
  const float ref_T[3] = {0.10f, 0.05f, 0.0f};
  const float src_T[3] = {0.10f + tx, 0.05f, 0.0f};

  const size_t N = static_cast<size_t>(W) * Ht;
  std::vector<float> depth(N), normal(3 * N);
  ASSERT_TRUE(EstimateDepthMapMetalFromArrays(
      W, Ht, ref.data(), K9, Q9, ref_T, /*num_src=*/1, src.data(), K9, Q9,
      src_T, /*best_k=*/1, /*window_radius=*/5, /*window_step=*/1, 3.0f, 0.2f,
      /*depth_min=*/2.5f, /*depth_max=*/8.0f, /*num_iterations=*/24,
      /*filter_min_ncc=*/0.0f, depth.data(), normal.data()));

  std::vector<float> errors;
  for (int r = 25; r <= 45; r += 4)
    for (int c = 40; c <= 120; c += 8)
      errors.push_back(std::abs(depth[static_cast<size_t>(r) * W + c] - d_gt));
  std::sort(errors.begin(), errors.end());
  EXPECT_LT(errors[errors.size() / 2], 0.4f)
      << "bridge failed to recover depth from world->cam poses";
}

// Real-data end-to-end run on a COLMAP dense workspace, gated on the
// COLMAP_DENSE_WORKSPACE env var (skips otherwise, so CI is unaffected). It
// estimates the depth map for one reference image with the Metal pipeline,
// writes it (COLMAP_OUT_DEPTH) for an external golden diff, and prints timing.
TEST(PatchMatchMetalRealData, SouthBuildingDepthMap) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const char* ws = std::getenv("COLMAP_DENSE_WORKSPACE");
  if (ws == nullptr) {
    GTEST_SKIP() << "set COLMAP_DENSE_WORKSPACE to run the real-data test.";
  }

  Workspace::Options opts;
  opts.workspace_path = ws;
  opts.workspace_format = "COLMAP";
  opts.input_type = "photometric";
  opts.image_as_rgb = false;  // load grayscale.
  opts.cache_size = 4.0;
  CachedWorkspace workspace(opts);  // loads bitmaps on demand.
  const Model& model = workspace.GetModel();

  const char* ref_env = std::getenv("COLMAP_REF_IMAGE");
  const std::string ref_name = ref_env ? ref_env : "P1180141.JPG";
  const int ref_idx = model.GetImageIdx(ref_name);
  ASSERT_GE(ref_idx, 0) << "reference image not in model: " << ref_name;

  const int num_src_want = 4;
  const auto overlap = model.GetMaxOverlappingImages(num_src_want, 1.0);
  std::vector<int> srcs = overlap.at(ref_idx);
  if (static_cast<int>(srcs.size()) > num_src_want) srcs.resize(num_src_want);
  ASSERT_FALSE(srcs.empty());

  const auto ranges = model.ComputeDepthRanges();
  const float dmin = ranges.at(ref_idx).first;
  const float dmax = ranges.at(ref_idx).second;
  ASSERT_GT(dmax, dmin);

  auto to_gray = [&](int idx, int& W, int& H) {
    const Bitmap& bmp = workspace.GetBitmap(idx);
    W = bmp.Width();
    H = bmp.Height();
    const std::vector<uint8_t>& d = bmp.RowMajorData();
    std::vector<float> g(static_cast<size_t>(W) * H);
    for (size_t i = 0; i < g.size(); ++i) g[i] = d[i] / 255.0f;
    return g;
  };

  int W = 0, H = 0;
  const std::vector<float> ref_gray = to_gray(ref_idx, W, H);
  const Image& ref_img = model.images[ref_idx];

  std::vector<float> src_gray, src_K9, src_R9, src_T3;
  int num_src = 0;
  for (int s : srcs) {
    int sw = 0, sh = 0;
    std::vector<float> g = to_gray(s, sw, sh);
    if (sw != W || sh != H) continue;  // kernels assume shared dimensions.
    src_gray.insert(src_gray.end(), g.begin(), g.end());
    const Image& si = model.images[s];
    src_K9.insert(src_K9.end(), si.GetK(), si.GetK() + 9);
    src_R9.insert(src_R9.end(), si.GetR(), si.GetR() + 9);
    src_T3.insert(src_T3.end(), si.GetT(), si.GetT() + 3);
    ++num_src;
  }
  ASSERT_GT(num_src, 0);

  const char* iters_env = std::getenv("COLMAP_ITERS");
  const int iters = iters_env ? std::atoi(iters_env) : 5;
  const char* filt_env = std::getenv("COLMAP_FILTER_MIN_NCC");
  const float filter_min_ncc = filt_env ? std::atof(filt_env) : 0.0f;
  std::vector<float> depth(static_cast<size_t>(W) * H);
  std::vector<float> normal(static_cast<size_t>(W) * H * 3);

  Timer timer;
  timer.Start();
  ASSERT_TRUE(EstimateDepthMapMetalFromArrays(
      W, H, ref_gray.data(), ref_img.GetK(), ref_img.GetR(), ref_img.GetT(),
      num_src, src_gray.data(), src_K9.data(), src_R9.data(), src_T3.data(),
      /*best_k=*/(num_src + 1) / 2, /*window_radius=*/5, /*window_step=*/1,
      /*sigma_spatial=*/3.0f, /*sigma_color=*/0.2f, dmin, dmax, iters,
      filter_min_ncc, depth.data(), normal.data()));
  const double secs = timer.ElapsedSeconds();

  size_t valid = 0;
  for (float d : depth)
    if (d > 0.0f) ++valid;
  std::printf(
      "[real-data] %s  %dx%d  num_src=%d  iters=%d  depth_range=[%.2f,%.2f]  "
      "wall=%.1fs  in-range=%.1f%%\n",
      ref_name.c_str(), W, H, num_src, iters, dmin, dmax, secs,
      100.0 * valid / depth.size());

  const char* out = std::getenv("COLMAP_OUT_DEPTH");
  if (out != nullptr) {
    DepthMap dm(W, H, dmin, dmax);
    for (int r = 0; r < H; ++r)
      for (int c = 0; c < W; ++c)
        dm.Set(r, c, depth[static_cast<size_t>(r) * W + c]);
    dm.Write(out);
    std::printf("[real-data] wrote depth map -> %s\n", out);
  }
  EXPECT_GT(valid, 0u);
}

namespace {

// Read a COLMAP dense binary map (W&H&C& header + column-major float32).
std::vector<float> ReadColmapArray(const std::string& path, int& W, int& H) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return {};
  std::string hdr;
  int amp = 0;
  while (amp < 3) {
    int ch = std::fgetc(f);
    if (ch == EOF) {
      std::fclose(f);
      return {};
    }
    hdr.push_back(static_cast<char>(ch));
    if (ch == '&') ++amp;
  }
  int c = 0;
  std::sscanf(hdr.c_str(), "%d&%d&%d&", &W, &H, &c);
  std::vector<float> data(static_cast<size_t>(W) * H * c);
  const size_t n = std::fread(data.data(), sizeof(float), data.size(), f);
  std::fclose(f);
  if (n != data.size()) return {};
  // COLMAP Mat is row-major (data[slice*W*H + row*W + col]); channel 0 is the
  // depth, already in row-major (row*W + col) order -- return it directly.
  data.resize(static_cast<size_t>(W) * H);
  return data;
}

// Build P = K*[R|T] and inv_P = [R^T K^-1 | -R^T T] (row-major 3x4) for the geom
// kernel, from source intrinsics K9 and the reference->source relative pose.
void BuildPInvP(const float K9[9], const float Rrel[9], const float Trel[3],
                float P[12], float invP[12]) {
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> K(K9);
  const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R(Rrel);
  const Eigen::Map<const Eigen::Vector3f> T(Trel);
  Eigen::Matrix<float, 3, 4> RT;
  RT.block<3, 3>(0, 0) = R;
  RT.col(3) = T;
  const Eigen::Matrix<float, 3, 4> Pm = K * RT;
  Eigen::Matrix<float, 3, 4> iPm;
  iPm.block<3, 3>(0, 0) = R.transpose() * K.inverse();
  iPm.col(3) = -R.transpose() * T;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c) {
      P[r * 4 + c] = Pm(r, c);
      invP[r * 4 + c] = iPm(r, c);
    }
}

}  // namespace

// All-views geometric-consistency filtering: compute photometric depth for a
// golden reference AND its source views, then reject pixels of the reference
// whose forward-backward reprojection error against the sources' depth maps
// exceeds a threshold. This is what removes the GEOMETRIC outliers (good NCC at
// the wrong depth) that photometric filtering cannot -- so the depth error on
// the surviving pixels (vs the CUDA golden) should drop. Env-gated.
TEST(PatchMatchMetalRealData, GeomConsistencyShrinksOutliers) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const char* ws = std::getenv("COLMAP_DENSE_WORKSPACE");
  const char* gold_dir = std::getenv("COLMAP_GOLDEN_DEPTH_DIR");
  if (ws == nullptr || gold_dir == nullptr) {
    GTEST_SKIP() << "set COLMAP_DENSE_WORKSPACE and COLMAP_GOLDEN_DEPTH_DIR.";
  }

  Workspace::Options opts;
  opts.workspace_path = ws;
  opts.workspace_format = "COLMAP";
  opts.input_type = "photometric";
  opts.image_as_rgb = false;
  opts.cache_size = 6.0;
  CachedWorkspace workspace(opts);
  const Model& model = workspace.GetModel();

  const int num_src = 4;
  const auto overlap = model.GetMaxOverlappingImages(num_src, 1.0);
  const auto ranges = model.ComputeDepthRanges();
  const char* iters_env = std::getenv("COLMAP_ITERS");
  const int iters = iters_env ? std::atoi(iters_env) : 8;

  int W = 0, Hh = 0;
  auto gray = [&](int idx) {
    const Bitmap& b = workspace.GetBitmap(idx);
    W = b.Width();
    Hh = b.Height();
    const auto& d = b.RowMajorData();
    std::vector<float> g(static_cast<size_t>(W) * Hh);
    for (size_t i = 0; i < g.size(); ++i) g[i] = d[i] / 255.0f;
    return g;
  };

  // Photometric depth for image `idx` using its top sources (cached).
  std::map<int, std::vector<float>> depth_cache;
  auto photo_depth = [&](int idx) -> const std::vector<float>& {
    auto it = depth_cache.find(idx);
    if (it != depth_cache.end()) return it->second;
    int w0 = 0, h0 = 0;
    std::vector<float> rg = gray(idx);
    w0 = W;
    h0 = Hh;
    std::vector<float> sg, sK, sR, sT;
    int ns = 0;
    for (int s : overlap[idx]) {
      if (ns >= num_src) break;
      std::vector<float> g = gray(s);
      if (W != w0 || Hh != h0) continue;
      sg.insert(sg.end(), g.begin(), g.end());
      sK.insert(sK.end(), model.images[s].GetK(), model.images[s].GetK() + 9);
      sR.insert(sR.end(), model.images[s].GetR(), model.images[s].GetR() + 9);
      sT.insert(sT.end(), model.images[s].GetT(), model.images[s].GetT() + 3);
      ++ns;
    }
    W = w0;
    Hh = h0;
    std::vector<float> depth(static_cast<size_t>(W) * Hh);
    std::vector<float> normal(static_cast<size_t>(W) * Hh * 3);
    const Image& ri = model.images[idx];
    EstimateDepthMapMetalFromArrays(
        W, Hh, rg.data(), ri.GetK(), ri.GetR(), ri.GetT(), ns, sg.data(),
        sK.data(), sR.data(), sT.data(), (ns + 1) / 2, 5, 1, 3.0f, 0.2f,
        ranges[idx].first, ranges[idx].second, iters, /*filter_min_ncc=*/0.0f,
        depth.data(), normal.data());
    depth_cache[idx] = std::move(depth);
    return depth_cache[idx];
  };

  const std::string ref_name =
      std::getenv("COLMAP_REF_IMAGE") ? std::getenv("COLMAP_REF_IMAGE")
                                      : "P1180141.JPG";
  const int R = model.GetImageIdx(ref_name);
  ASSERT_GE(R, 0);

  std::vector<float> depthR = photo_depth(R);  // copy (cache may grow).
  const int Wr = W, Hr = Hh;

  // Build per-source P/inv_P + depth maps for the geometric cross-check.
  std::vector<float> P_all, invP_all, srcdepth_all;
  int ns = 0;
  for (int s : overlap[R]) {
    if (ns >= num_src) break;
    const std::vector<float>& ds = photo_depth(s);
    if (static_cast<int>(ds.size()) != Wr * Hr) continue;
    float Rrel[9], Trel[3], P[12], invP[12];
    ComputeRelativePose(model.images[R].GetR(), model.images[R].GetT(),
                        model.images[s].GetR(), model.images[s].GetT(), Rrel,
                        Trel);
    BuildPInvP(model.images[s].GetK(), Rrel, Trel, P, invP);
    P_all.insert(P_all.end(), P, P + 12);
    invP_all.insert(invP_all.end(), invP, invP + 12);
    srcdepth_all.insert(srcdepth_all.end(), ds.begin(), ds.end());
    ++ns;
  }
  ASSERT_GT(ns, 0);

  const float* K9 = model.images[R].GetK();
  const float ref_inv_K[4] = {1.0f / K9[0], -K9[2] / K9[0], 1.0f / K9[4],
                              -K9[5] / K9[4]};
  const float ref_K[4] = {K9[0], K9[2], K9[4], K9[5]};
  std::vector<int> rows, cols;
  for (int r = 0; r < Hr; ++r)
    for (int c = 0; c < Wr; ++c) {
      rows.push_back(r);
      cols.push_back(c);
    }
  std::vector<float> geom(static_cast<size_t>(Wr) * Hr);
  ASSERT_TRUE(ComputeGeomConsistencyCostMetal(
      Wr, Hr, ref_inv_K, ref_K, P_all.data(), invP_all.data(),
      srcdepth_all.data(), ns, /*max_cost=*/3.0f, rows.data(), cols.data(),
      depthR.data(), static_cast<int>(depthR.size()), geom.data()));

  // Golden depth map for the reference.
  int gw = 0, gh = 0;
  const std::vector<float> golden =
      ReadColmapArray(std::string(gold_dir) + "/" + ref_name + ".geometric.bin",
                      gw, gh);
  ASSERT_EQ(gw, Wr);
  ASSERT_EQ(gh, Hr);

  // Median relative error vs golden, on golden-valid pixels, before vs after a
  // geometric-consistency reject (reprojection error > 1 px).
  auto p_metrics = [&](bool apply_geom) {
    std::vector<float> errs;
    for (size_t i = 0; i < depthR.size(); ++i) {
      if (golden[i] <= 0.0f || depthR[i] <= 0.0f) continue;
      if (apply_geom && geom[i] > 1.0f) continue;  // geometric reject.
      errs.push_back(std::abs(depthR[i] - golden[i]) / golden[i]);
    }
    std::sort(errs.begin(), errs.end());
    float med = errs.empty() ? 0 : errs[errs.size() / 2];
    float p95 = errs.empty() ? 0 : errs[std::min(errs.size() - 1,
                                                  errs.size() * 95 / 100)];
    return std::make_tuple(errs.size(), med, p95);
  };
  auto [n_before, med_before, p95_before] = p_metrics(false);
  auto [n_after, med_after, p95_after] = p_metrics(true);
  std::printf(
      "[geom-filter] %s  before: n=%zu med=%.4f p95=%.4f | after: n=%zu "
      "med=%.4f p95=%.4f (kept %.1f%%)\n",
      ref_name.c_str(), n_before, med_before, p95_before, n_after, med_after,
      p95_after, 100.0 * n_after / std::max<size_t>(1, n_before));
  EXPECT_LT(p95_after, p95_before) << "geom filtering should shrink the tail";
}

// Fast single-image NORMAL-quality probe (env-gated). Computes the photometric
// depth+normal for one reference and reports the median/p90 angular error of the
// normals vs the CUDA golden -- the tight loop for tuning normal refinement.
// COLMAP_DENSE_WORKSPACE + COLMAP_GOLDEN_NORMAL_DIR; COLMAP_ITERS optional.
TEST(PatchMatchMetalRealData, NormalQualityVsGolden) {
  if (!IsPatchMatchMetalAvailable()) {
    GTEST_SKIP() << "Metal not available.";
  }
  const char* ws = std::getenv("COLMAP_DENSE_WORKSPACE");
  const char* gold_dir = std::getenv("COLMAP_GOLDEN_NORMAL_DIR");
  if (ws == nullptr || gold_dir == nullptr) {
    GTEST_SKIP() << "set COLMAP_DENSE_WORKSPACE and COLMAP_GOLDEN_NORMAL_DIR.";
  }
  Workspace::Options opts;
  opts.workspace_path = ws;
  opts.workspace_format = "COLMAP";
  opts.input_type = "photometric";
  opts.image_as_rgb = false;
  opts.cache_size = 6.0;
  CachedWorkspace workspace(opts);
  const Model& model = workspace.GetModel();
  const std::string ref_name =
      std::getenv("COLMAP_REF_IMAGE") ? std::getenv("COLMAP_REF_IMAGE")
                                      : "P1180141.JPG";
  const int R = model.GetImageIdx(ref_name);
  ASSERT_GE(R, 0);
  const int num_src = 6;
  const auto overlap = model.GetMaxOverlappingImages(num_src, 1.0);
  const auto ranges = model.ComputeDepthRanges();
  const char* iters_env = std::getenv("COLMAP_ITERS");
  const int iters = iters_env ? std::atoi(iters_env) : 8;

  int W = 0, H = 0;
  auto gray = [&](int idx, int& w, int& h) {
    const Bitmap& b = workspace.GetBitmap(idx);
    w = b.Width();
    h = b.Height();
    const auto& d = b.RowMajorData();
    std::vector<float> g(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < g.size(); ++i) g[i] = d[i] / 255.0f;
    return g;
  };
  const std::vector<float> ref_gray = gray(R, W, H);
  std::vector<float> sg, sK, sR, sT;
  int ns = 0;
  for (int s : overlap[R]) {
    int sw, sh;
    std::vector<float> g = gray(s, sw, sh);
    if (sw != W || sh != H) continue;
    sg.insert(sg.end(), g.begin(), g.end());
    sK.insert(sK.end(), model.images[s].GetK(), model.images[s].GetK() + 9);
    sR.insert(sR.end(), model.images[s].GetR(), model.images[s].GetR() + 9);
    sT.insert(sT.end(), model.images[s].GetT(), model.images[s].GetT() + 3);
    ++ns;
  }
  const size_t n = static_cast<size_t>(W) * H;
  std::vector<float> depth(n), normal(n * 3);
  const Image& ri = model.images[R];
  ASSERT_TRUE(EstimateDepthMapMetalFromArrays(
      W, H, ref_gray.data(), ri.GetK(), ri.GetR(), ri.GetT(), ns, sg.data(),
      sK.data(), sR.data(), sT.data(), (ns + 1) / 2, 5, 1, 3.0f, 0.2f,
      ranges[R].first, ranges[R].second, iters, /*filter_min_ncc=*/0.0f,
      depth.data(), normal.data()));

  // Golden normal map (3 channels, Mat layout data[k*n + r*W + c]).
  int gw = 0, gh = 0, gc = 0;
  std::vector<float> gN;
  {
    std::FILE* f = std::fopen(
        (std::string(gold_dir) + "/" + ref_name + ".geometric.bin").c_str(),
        "rb");
    ASSERT_NE(f, nullptr);
    std::string hdr;
    int amp = 0;
    while (amp < 3) {
      int ch = std::fgetc(f);
      hdr.push_back(static_cast<char>(ch));
      if (ch == '&') ++amp;
    }
    std::sscanf(hdr.c_str(), "%d&%d&%d&", &gw, &gh, &gc);
    gN.resize(static_cast<size_t>(gw) * gh * gc);
    std::fread(gN.data(), sizeof(float), gN.size(), f);
    std::fclose(f);
  }
  ASSERT_EQ(gw, W);
  ASSERT_EQ(gh, H);
  ASSERT_EQ(gc, 3);

  std::vector<float> errs;
  for (size_t p = 0; p < n; ++p) {
    if (depth[p] <= 0.0f) continue;
    const float gx = gN[p], gy = gN[n + p], gz = gN[2 * n + p];
    const float gnorm = std::sqrt(gx * gx + gy * gy + gz * gz);
    if (gnorm < 0.5f) continue;  // golden invalid.
    float mx = normal[3 * p + 0], my = normal[3 * p + 1], mz = normal[3 * p + 2];
    const float mnorm = std::sqrt(mx * mx + my * my + mz * mz);
    if (mnorm < 1e-6f) continue;
    const float dot = (mx * gx + my * gy + mz * gz) / (mnorm * gnorm);
    errs.push_back(std::acos(std::max(-1.0f, std::min(1.0f, dot))) * 57.2958f);
  }
  std::sort(errs.begin(), errs.end());
  const float med = errs.empty() ? -1 : errs[errs.size() / 2];
  const float p90 = errs.empty() ? -1 : errs[std::min(errs.size() - 1, errs.size() * 9 / 10)];
  std::printf("[normal-quality] %s iters=%d n=%zu  median=%.1f deg  p90=%.1f deg\n",
              ref_name.c_str(), iters, errs.size(), med, p90);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
