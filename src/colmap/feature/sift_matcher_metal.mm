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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdlib>
#include <cstring>

namespace colmap {
namespace {

// Reduction kernels (runtime-compiled, so no .metallib needs to ship). They
// read the GEMM dot-product matrix in place on the GPU and emit only the
// per-descriptor best index, mirroring FindBestMatchesOneWayBruteForce:
// angular distance acos(min(dot/kSqNorm, 1)), distance + ratio thresholds.
// kSqNorm = 512*512 (kSqSiftDescriptorNorm).
constexpr char kReductionSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;
constant float kInvSqNorm = 1.0f / (512.0f * 512.0f);

struct MatchParams { uint n1; uint n2; float max_distance; float max_ratio; };

// One thread per row i1: best match in set 2 for descriptor i1 of set 1.
kernel void best_match_rows(device const float* dots [[buffer(0)]],
                            device int* best_idx     [[buffer(1)]],
                            constant MatchParams& p  [[buffer(2)]],
                            uint i1 [[thread_position_in_grid]]) {
  if (i1 >= p.n1) return;
  const device float* row = dots + (size_t)i1 * p.n2;
  int best = -1; float bestv = 0.0f; float secondv = 0.0f;
  for (uint j = 0; j < p.n2; ++j) {
    float v = row[j];
    if (v > bestv) { secondv = bestv; bestv = v; best = int(j); }
    else if (v > secondv) { secondv = v; }
  }
  best_idx[i1] = -1;
  if (best < 0) return;
  float bd = acos(min(kInvSqNorm * bestv, 1.0f));
  if (bd > p.max_distance) return;
  float sd = acos(min(kInvSqNorm * secondv, 1.0f));
  if (bd >= p.max_ratio * sd) return;
  best_idx[i1] = best;
}

// One thread per column i2: best match in set 1 for descriptor i2 of set 2
// (the 2->1 direction; equivalent to reducing the transpose). Strided reads.
kernel void best_match_cols(device const float* dots [[buffer(0)]],
                            device int* best_idx     [[buffer(1)]],
                            constant MatchParams& p  [[buffer(2)]],
                            uint i2 [[thread_position_in_grid]]) {
  if (i2 >= p.n2) return;
  int best = -1; float bestv = 0.0f; float secondv = 0.0f;
  for (uint i1 = 0; i1 < p.n1; ++i1) {
    float v = dots[(size_t)i1 * p.n2 + i2];
    if (v > bestv) { secondv = bestv; bestv = v; best = int(i1); }
    else if (v > secondv) { secondv = v; }
  }
  best_idx[i2] = -1;
  if (best < 0) return;
  float bd = acos(min(kInvSqNorm * bestv, 1.0f));
  if (bd > p.max_distance) return;
  float sd = acos(min(kInvSqNorm * secondv, 1.0f));
  if (bd >= p.max_ratio * sd) return;
  best_idx[i2] = best;
}
)METAL";

struct MatchParams {
  uint32_t n1;
  uint32_t n2;
  float max_distance;
  float max_ratio;
};

// Persistent Metal context. The device, command queue and reduction pipelines
// are created once and reused for every matched pair; MPS caches its compiled
// GEMM kernels globally, so only the first dispatch pays the compilation cost.
struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> rows_pipeline = nil;
  id<MTLComputePipelineState> cols_pipeline = nil;
  bool valid = false;

  MetalContext() {
    // Operational escape hatch: COLMAP_DISABLE_METAL=1 forces the CPU path
    // (useful for benchmarking and for sidestepping a driver issue without
    // recompiling).
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
          newLibraryWithSource:[NSString stringWithUTF8String:kReductionSource]
                       options:nil
                         error:&error];
      if (library == nil) {
        return;
      }
      id<MTLFunction> rows_fn = [library newFunctionWithName:@"best_match_rows"];
      id<MTLFunction> cols_fn = [library newFunctionWithName:@"best_match_cols"];
      if (rows_fn == nil || cols_fn == nil) {
        return;
      }
      rows_pipeline = [device newComputePipelineStateWithFunction:rows_fn
                                                            error:&error];
      cols_pipeline = [device newComputePipelineStateWithFunction:cols_fn
                                                            error:&error];
      valid = (rows_pipeline != nil && cols_pipeline != nil);
    }
  }
};

MetalContext& GetMetalContext() {
  static MetalContext context;
  return context;
}

}  // namespace

bool IsSiftMetalMatcherAvailable() { return GetMetalContext().valid; }

uint64_t SiftMetalMatcherCurrentAllocatedBytes() {
  MetalContext& context = GetMetalContext();
  if (!context.valid) {
    return 0;
  }
  return static_cast<uint64_t>([context.device currentAllocatedSize]);
}

bool ComputeSiftDotProductsMetal(const uint8_t* descriptors1,
                                 const int num_descriptors1,
                                 const uint8_t* descriptors2,
                                 const int num_descriptors2,
                                 const int dim,
                                 float* dot_products) {
  MetalContext& context = GetMetalContext();
  if (!context.valid || descriptors1 == nullptr || descriptors2 == nullptr ||
      dot_products == nullptr || num_descriptors1 <= 0 ||
      num_descriptors2 <= 0 || dim <= 0) {
    return false;
  }

  @autoreleasepool {
    id<MTLDevice> device = context.device;

    const NSUInteger n1 = static_cast<NSUInteger>(num_descriptors1);
    const NSUInteger n2 = static_cast<NSUInteger>(num_descriptors2);
    const NSUInteger k = static_cast<NSUInteger>(dim);

    const NSUInteger a_bytes = n1 * k * sizeof(float);
    const NSUInteger b_bytes = n2 * k * sizeof(float);
    const NSUInteger c_bytes = n1 * n2 * sizeof(float);

    // Unified-memory shared buffers: CPU fills them and the GPU reads the same
    // pages directly, so there is no separate host->device upload.
    id<MTLBuffer> a_buf =
        [device newBufferWithLength:a_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_buf =
        [device newBufferWithLength:b_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c_buf =
        [device newBufferWithLength:c_bytes options:MTLResourceStorageModeShared];
    if (a_buf == nil || b_buf == nil || c_buf == nil) {
      return false;
    }

    // Widen uint8 -> fp32 (exact; values fit well within fp32's integer range).
    float* a_ptr = static_cast<float*>(a_buf.contents);
    for (NSUInteger i = 0; i < n1 * k; ++i) {
      a_ptr[i] = static_cast<float>(descriptors1[i]);
    }
    float* b_ptr = static_cast<float*>(b_buf.contents);
    for (NSUInteger i = 0; i < n2 * k; ++i) {
      b_ptr[i] = static_cast<float>(descriptors2[i]);
    }

    MPSMatrixDescriptor* a_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n1
                                              columns:k
                                             rowBytes:k * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* b_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n2
                                              columns:k
                                             rowBytes:k * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* c_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n1
                                              columns:n2
                                             rowBytes:n2 * sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix* a_mat = [[MPSMatrix alloc] initWithBuffer:a_buf descriptor:a_desc];
    MPSMatrix* b_mat = [[MPSMatrix alloc] initWithBuffer:b_buf descriptor:b_desc];
    MPSMatrix* c_mat = [[MPSMatrix alloc] initWithBuffer:c_buf descriptor:c_desc];

    // C[n1 x n2] = A[n1 x k] * B[n2 x k]^T  (transposeRight=YES).
    MPSMatrixMultiplication* gemm =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          transposeLeft:NO
                                         transposeRight:YES
                                             resultRows:n1
                                          resultColumns:n2
                                        interiorColumns:k
                                                  alpha:1.0
                                                   beta:0.0];

    id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
    [gemm encodeToCommandBuffer:command_buffer
                     leftMatrix:a_mat
                    rightMatrix:b_mat
                   resultMatrix:c_mat];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }

    const float* c_ptr = static_cast<const float*>(c_buf.contents);
    std::memcpy(dot_products, c_ptr, c_bytes);
  }

  return true;
}

bool MatchSiftDescriptorsMetal(const uint8_t* descriptors1,
                               const int num_descriptors1,
                               const uint8_t* descriptors2,
                               const int num_descriptors2,
                               const int dim,
                               const float max_distance,
                               const float max_ratio,
                               const bool compute_2to1,
                               int* best_idx_1to2,
                               int* best_idx_2to1) {
  MetalContext& context = GetMetalContext();
  if (!context.valid || descriptors1 == nullptr || descriptors2 == nullptr ||
      best_idx_1to2 == nullptr || num_descriptors1 <= 0 ||
      num_descriptors2 <= 0 || dim <= 0 ||
      (compute_2to1 && best_idx_2to1 == nullptr)) {
    return false;
  }

  @autoreleasepool {
    id<MTLDevice> device = context.device;
    const NSUInteger n1 = static_cast<NSUInteger>(num_descriptors1);
    const NSUInteger n2 = static_cast<NSUInteger>(num_descriptors2);
    const NSUInteger k = static_cast<NSUInteger>(dim);

    id<MTLBuffer> a_buf = [device newBufferWithLength:n1 * k * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_buf = [device newBufferWithLength:n2 * k * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> c_buf = [device newBufferWithLength:n1 * n2 * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> idx1_buf = [device newBufferWithLength:n1 * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> idx2_buf =
        compute_2to1 ? [device newBufferWithLength:n2 * sizeof(int32_t)
                                           options:MTLResourceStorageModeShared]
                     : nil;
    if (a_buf == nil || b_buf == nil || c_buf == nil || idx1_buf == nil ||
        (compute_2to1 && idx2_buf == nil)) {
      return false;
    }

    float* a_ptr = static_cast<float*>(a_buf.contents);
    for (NSUInteger i = 0; i < n1 * k; ++i)
      a_ptr[i] = static_cast<float>(descriptors1[i]);
    float* b_ptr = static_cast<float*>(b_buf.contents);
    for (NSUInteger i = 0; i < n2 * k; ++i)
      b_ptr[i] = static_cast<float>(descriptors2[i]);

    MPSMatrixDescriptor* a_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n1 columns:k
                                             rowBytes:k * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* b_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n2 columns:k
                                             rowBytes:k * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* c_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:n1 columns:n2
                                             rowBytes:n2 * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrix* a_mat = [[MPSMatrix alloc] initWithBuffer:a_buf descriptor:a_desc];
    MPSMatrix* b_mat = [[MPSMatrix alloc] initWithBuffer:b_buf descriptor:b_desc];
    MPSMatrix* c_mat = [[MPSMatrix alloc] initWithBuffer:c_buf descriptor:c_desc];
    MPSMatrixMultiplication* gemm =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          transposeLeft:NO
                                         transposeRight:YES
                                             resultRows:n1
                                          resultColumns:n2
                                        interiorColumns:k
                                                  alpha:1.0
                                                   beta:0.0];

    MatchParams params{static_cast<uint32_t>(n1),
                       static_cast<uint32_t>(n2),
                       max_distance,
                       max_ratio};

    id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
    // 1) GEMM: c = a * b^T (dot-product matrix stays in device memory).
    [gemm encodeToCommandBuffer:command_buffer
                     leftMatrix:a_mat
                    rightMatrix:b_mat
                   resultMatrix:c_mat];
    // 2) Reduction(s): read c in place, emit only the best indices.
    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
    [enc setComputePipelineState:context.rows_pipeline];
    [enc setBuffer:c_buf offset:0 atIndex:0];
    [enc setBuffer:idx1_buf offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    NSUInteger tg_rows = context.rows_pipeline.maxTotalThreadsPerThreadgroup;
    if (tg_rows > n1) tg_rows = n1;
    [enc dispatchThreads:MTLSizeMake(n1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_rows, 1, 1)];
    if (compute_2to1) {
      [enc setComputePipelineState:context.cols_pipeline];
      [enc setBuffer:c_buf offset:0 atIndex:0];
      [enc setBuffer:idx2_buf offset:0 atIndex:1];
      [enc setBytes:&params length:sizeof(params) atIndex:2];
      NSUInteger tg_cols = context.cols_pipeline.maxTotalThreadsPerThreadgroup;
      if (tg_cols > n2) tg_cols = n2;
      [enc dispatchThreads:MTLSizeMake(n2, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg_cols, 1, 1)];
    }
    [enc endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      return false;
    }

    std::memcpy(best_idx_1to2, idx1_buf.contents, n1 * sizeof(int32_t));
    if (compute_2to1) {
      std::memcpy(best_idx_2to1, idx2_buf.contents, n2 * sizeof(int32_t));
    }
  }

  return true;
}

}  // namespace colmap
