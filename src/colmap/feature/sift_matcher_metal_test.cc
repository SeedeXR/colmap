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

#include "colmap/feature/sift_matcher_metal.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Reference: exact integer dot products between row-major uint8 descriptors.
std::vector<float> CpuDotProducts(const std::vector<uint8_t>& d1,
                                  int n1,
                                  const std::vector<uint8_t>& d2,
                                  int n2,
                                  int dim) {
  std::vector<float> out(static_cast<size_t>(n1) * n2);
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n2; ++j) {
      int64_t dot = 0;
      for (int k = 0; k < dim; ++k) {
        dot += static_cast<int64_t>(d1[i * dim + k]) * d2[j * dim + k];
      }
      out[static_cast<size_t>(i) * n2 + j] = static_cast<float>(dot);
    }
  }
  return out;
}

TEST(SiftMetalMatcher, AvailabilityIsConsistent) {
  // Calling the availability query must be safe regardless of build config.
  const bool available = IsSiftMetalMatcherAvailable();
  // When unavailable, the compute function must refuse (return false).
  if (!available) {
    std::vector<uint8_t> d(128, 1);
    std::vector<float> out(1);
    EXPECT_FALSE(
        ComputeSiftDotProductsMetal(d.data(), 1, d.data(), 1, 128, out.data()));
  }
}

TEST(SiftMetalMatcher, BitExactVsCpu) {
  if (!IsSiftMetalMatcherAvailable()) {
    GTEST_SKIP() << "Metal matcher unavailable in this build/runtime.";
  }
  constexpr int kDim = 128;
  const int n1 = 257, n2 = 311;  // non-round sizes to exercise edge handling.
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> u(0, 255);
  std::vector<uint8_t> d1(n1 * kDim), d2(n2 * kDim);
  for (auto& v : d1) v = static_cast<uint8_t>(u(rng));
  for (auto& v : d2) v = static_cast<uint8_t>(u(rng));

  std::vector<float> gpu(static_cast<size_t>(n1) * n2, -1.0f);
  ASSERT_TRUE(ComputeSiftDotProductsMetal(
      d1.data(), n1, d2.data(), n2, kDim, gpu.data()));

  const std::vector<float> cpu = CpuDotProducts(d1, n1, d2, n2, kDim);
  ASSERT_EQ(gpu.size(), cpu.size());
  // fp32 represents these integer dot products exactly (< 2^24), so the GPU
  // result must equal the CPU reference bit-for-bit.
  double max_abs_diff = 0.0;
  for (size_t i = 0; i < gpu.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff,
                            static_cast<double>(std::abs(gpu[i] - cpu[i])));
  }
  EXPECT_EQ(max_abs_diff, 0.0);
}

// CPU reference replicating FindBestMatchesOneWayBruteForce (rows direction).
std::vector<int> CpuBestMatches(const std::vector<float>& dots,
                                int n1,
                                int n2,
                                float max_distance,
                                float max_ratio) {
  constexpr double kSqNorm = 512.0 * 512.0;
  const float inv = static_cast<float>(1.0 / kSqNorm);
  std::vector<int> best(n1, -1);
  for (int i1 = 0; i1 < n1; ++i1) {
    int bi = -1;
    float bv = 0.f, sv = 0.f;
    for (int j = 0; j < n2; ++j) {
      const float v = dots[static_cast<size_t>(i1) * n2 + j];
      if (v > bv) {
        sv = bv;
        bv = v;
        bi = j;
      } else if (v > sv) {
        sv = v;
      }
    }
    if (bi < 0) continue;
    const float bd = std::acos(std::min(inv * bv, 1.0f));
    if (bd > max_distance) continue;
    const float sd = std::acos(std::min(inv * sv, 1.0f));
    if (bd >= max_ratio * sd) continue;
    best[i1] = bi;
  }
  return best;
}

// Builds a uint8 descriptor set L2-normalized so each row's squared norm is
// ~512^2 (kSqSiftDescriptorNorm), matching real SIFT descriptors -- required
// for the acos-based angular distance/ratio test to behave realistically.
std::vector<uint8_t> MakeNormalizedDescriptors(int n, int dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> uf(0.f, 1.f);
  std::vector<uint8_t> out(static_cast<size_t>(n) * dim);
  for (int i = 0; i < n; ++i) {
    std::vector<float> row(dim);
    double sq = 0.0;
    for (int k = 0; k < dim; ++k) {
      row[k] = uf(rng);
      sq += static_cast<double>(row[k]) * row[k];
    }
    const double scale = (sq > 0.0) ? 512.0 / std::sqrt(sq) : 0.0;
    for (int k = 0; k < dim; ++k) {
      int v = static_cast<int>(std::lround(row[k] * scale));
      out[static_cast<size_t>(i) * dim + k] =
          static_cast<uint8_t>(std::min(std::max(v, 0), 255));
    }
  }
  return out;
}

TEST(SiftMetalMatcher, GpuMemoryInstrumentationAndPerCallCost) {
  if (!IsSiftMetalMatcherAvailable()) {
    GTEST_SKIP() << "Metal matcher unavailable in this build/runtime.";
  }
  constexpr int kDim = 128;
  const int n1 = 4000, n2 = 4000;  // realistic SIFT counts per image.
  std::mt19937 rng(3);
  std::vector<uint8_t> d1 = MakeNormalizedDescriptors(n1, kDim, rng);
  std::vector<uint8_t> d2 = MakeNormalizedDescriptors(n2, kDim, rng);
  std::vector<int> best(n1);

  // Warm up (first call pays MPS kernel compilation).
  ASSERT_TRUE(MatchSiftDescriptorsMetal(d1.data(), n1, d2.data(), n2, kDim,
                                        0.7f, 0.9f, false, best.data(),
                                        nullptr));
  const uint64_t gpu_bytes = SiftMetalMatcherCurrentAllocatedBytes();
  EXPECT_GT(gpu_bytes, 0u);  // instrumentation reports real GPU memory.

  // Measure steady-state per-call cost over repeated matches. If buffer
  // allocation dominated, this would be large; in practice the MPS GEMM +
  // reduction dominate and per-pair allocation on unified memory is cheap.
  const int kReps = 30;
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kReps; ++i) {
    ASSERT_TRUE(MatchSiftDescriptorsMetal(d1.data(), n1, d2.data(), n2, kDim,
                                          0.7f, 0.9f, false, best.data(),
                                          nullptr));
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double per_call_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / kReps;
  std::printf(
      "[ INFO ] Metal matcher %dx%d: %.2f ms/call, GPU allocated %.1f MB\n",
      n1, n2, per_call_ms, gpu_bytes / (1024.0 * 1024.0));
  // Sanity: a 4000x4000x128 match should be well under 1 s/call.
  EXPECT_LT(per_call_ms, 1000.0);
}

TEST(SiftMetalMatcher, FullMatchEquivalentToCpu) {
  if (!IsSiftMetalMatcherAvailable()) {
    GTEST_SKIP() << "Metal matcher unavailable in this build/runtime.";
  }
  constexpr int kDim = 128;
  const int n1 = 400, n2 = 380;
  std::mt19937 rng(7);
  std::vector<uint8_t> d1 = MakeNormalizedDescriptors(n1, kDim, rng);
  std::vector<uint8_t> d2 = MakeNormalizedDescriptors(n2, kDim, rng);
  // Plant exact-duplicate pairs so real matches survive the ratio test,
  // exercising the kept-match path (not just rejections).
  const int planted = 50;
  for (int i = 0; i < planted && i < n1 && i < n2; ++i) {
    for (int k = 0; k < kDim; ++k) d2[i * kDim + k] = d1[i * kDim + k];
  }

  const float max_distance = 0.7f, max_ratio = 0.9f;
  std::vector<int> gpu_1to2(n1, -2);
  ASSERT_TRUE(MatchSiftDescriptorsMetal(d1.data(), n1, d2.data(), n2, kDim,
                                        max_distance, max_ratio,
                                        /*compute_2to1=*/false,
                                        gpu_1to2.data(), nullptr));

  const std::vector<float> dots = CpuDotProducts(d1, n1, d2, n2, kDim);
  const std::vector<int> cpu_1to2 =
      CpuBestMatches(dots, n1, n2, max_distance, max_ratio);

  // Primary validation: the GPU reduction must agree with the CPU reduction.
  // Allow a tiny number of ratio-boundary disagreements from acos ULP
  // differences between MSL and std::acos.
  int disagreements = 0;
  int gpu_kept = 0;
  for (int i = 0; i < n1; ++i) {
    if (gpu_1to2[i] != cpu_1to2[i]) ++disagreements;
    if (gpu_1to2[i] >= 0) ++gpu_kept;
  }
  EXPECT_LE(disagreements, std::max(1, n1 / 100));  // <= ~1% boundary diffs.
  EXPECT_GT(gpu_kept, 0) << "no matches survived; test would be vacuous";
}

}  // namespace
}  // namespace colmap
