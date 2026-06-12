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

#include <cstdint>

namespace colmap {

// Metal (Apple GPU) accelerated SIFT descriptor matching primitive.
//
// On macOS, COLMAP has no GPU feature matcher (CUDA is absent and the SiftGPU
// GLSL path is deprecated), so brute-force descriptor matching runs on the CPU.
// This module provides the Metal equivalent of the dot-product matrix that the
// CUDA SiftMatchGPU computes: C[i,j] = <descriptor1_i, descriptor2_j>.
//
// Numerical note: SIFT descriptors are uint8 (128-D); the maximum possible dot
// product is 128 * 255 * 255 = 8,323,200 < 2^24, so the fp32 GEMM used here is
// BIT-EXACT to the integer dot product. The resulting matches are therefore
// identical to the CPU brute-force matcher (verified by tests), not merely
// "close". The interface is intentionally free of Metal and Eigen types so it
// can be included from any translation unit; callers wrap it with Eigen.

// Whether a usable Metal GPU matcher is available at runtime. Always false in
// builds without COLMAP_METAL_ENABLED, or when no Metal device is present.
bool IsSiftMetalMatcherAvailable();

// Current total memory allocated by the Metal device for this process, in bytes
// (MTLDevice.currentAllocatedSize). This is the honest GPU/unified-memory
// working-set figure for the matcher's buffers, which /usr/bin/time maxRSS does
// NOT include (Metal buffers are wired/GPU-accounted separately). Returns 0
// when Metal is unavailable.
uint64_t SiftMetalMatcherCurrentAllocatedBytes();

// Computes the row-major dot-product matrix between two sets of uint8
// descriptors on the Metal GPU.
//   descriptors1: num_descriptors1 * dim row-major bytes.
//   descriptors2: num_descriptors2 * dim row-major bytes.
//   dot_products: caller-allocated num_descriptors1 * num_descriptors2 floats,
//                 row-major (entry [i*num_descriptors2 + j]).
// Returns true on success; false if Metal is unavailable or inputs are invalid
// (in which case the caller must fall back to the CPU implementation).
bool ComputeSiftDotProductsMetal(const uint8_t* descriptors1,
                                 int num_descriptors1,
                                 const uint8_t* descriptors2,
                                 int num_descriptors2,
                                 int dim,
                                 float* dot_products);

// Full GPU brute-force match: computes the dot-product matrix (MPS GEMM) AND
// the best/second-best ratio-test reduction entirely on the GPU, so the large
// N1*N2 matrix is never copied to the CPU -- only the per-descriptor best
// indices are returned. This is the high-throughput path (the GEMM result
// stays in device memory between the GEMM and the reduction).
//
//   best_idx_1to2: caller-allocated num_descriptors1 ints; on success, entry i
//     is the index of the best match in set 2 for descriptor i of set 1, or -1
//     if it fails the distance/ratio test.
//   best_idx_2to1: caller-allocated num_descriptors2 ints, filled only when
//     compute_2to1 is true (for the caller's cross-check); may be null
//     otherwise.
//
// Distance/ratio semantics match the CPU brute-force matcher
// (FindBestMatchesOneWayBruteForce): angular distance acos(min(dot/kSqNorm,1)),
// thresholds max_distance and max_ratio. The distance test is applied on the
// raw dot product (no acos needed); only the ratio test uses acos. Because the
// GPU acos may differ from std::acos by ~1 ULP, the resulting matches are
// numerically equivalent to (not necessarily bit-identical with) the CPU path
// at the ratio boundary; callers validate by match-count / reconstruction
// tolerance.
//
// Returns false if Metal is unavailable or inputs are invalid (fall back to
// CPU).
bool MatchSiftDescriptorsMetal(const uint8_t* descriptors1,
                               int num_descriptors1,
                               const uint8_t* descriptors2,
                               int num_descriptors2,
                               int dim,
                               float max_distance,
                               float max_ratio,
                               bool compute_2to1,
                               int* best_idx_1to2,
                               int* best_idx_2to1);

}  // namespace colmap
