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

#include "colmap/util/sysinfo.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// The host machine is unknown at test time, so these assert invariants that
// must hold on any real platform rather than exact values.
TEST(SystemInfo, CoreInvariants) {
  const SystemInfo& info = GetSystemInfo();
  // Any machine that can run a test has at least one logical core.
  EXPECT_GE(info.num_logical_cores, 1);
  // Performance cores are always populated (>= 1) and never exceed logical.
  EXPECT_GE(info.num_performance_cores, 1);
  EXPECT_LE(info.num_performance_cores, info.num_logical_cores);
  // Efficiency cores are non-negative and never exceed logical.
  EXPECT_GE(info.num_efficiency_cores, 0);
  EXPECT_LE(info.num_efficiency_cores, info.num_logical_cores);
  // Physical cores, when known, never exceed logical cores.
  if (info.num_physical_cores > 0) {
    EXPECT_LE(info.num_physical_cores, info.num_logical_cores);
  }
  EXPECT_FALSE(info.os_name.empty());
}

TEST(SystemInfo, MemoryInvariants) {
  const SystemInfo& info = GetSystemInfo();
  // When total memory is known, available must not exceed it.
  if (info.total_physical_memory_bytes > 0 &&
      info.available_memory_bytes > 0) {
    EXPECT_LE(info.available_memory_bytes, info.total_physical_memory_bytes);
  }
}

TEST(SystemInfo, Caching) {
  // GetSystemInfo() returns a stable cached reference.
  const SystemInfo& a = GetSystemInfo();
  const SystemInfo& b = GetSystemInfo();
  EXPECT_EQ(&a, &b);
}

TEST(GetAvailableMemoryBytes, NonNegative) {
  // Always a valid call; 0 means "unknown" which is acceptable.
  const uint64_t available = GetAvailableMemoryBytes();
  const SystemInfo& info = GetSystemInfo();
  if (available > 0 && info.total_physical_memory_bytes > 0) {
    EXPECT_LE(available, info.total_physical_memory_bytes);
  }
}

TEST(GetNumPerformanceCores, AtLeastOne) {
  EXPECT_GE(GetNumPerformanceCores(), 1);
}

TEST(ByteConversions, RoundTrip) {
  EXPECT_DOUBLE_EQ(BytesToGiB(0), 0.0);
  EXPECT_DOUBLE_EQ(BytesToGiB(1024ull * 1024ull * 1024ull), 1.0);
  EXPECT_DOUBLE_EQ(BytesToGiB(GiBToBytes(4.0)), 4.0);
  EXPECT_EQ(GiBToBytes(0.0), 0u);
  EXPECT_EQ(GiBToBytes(-1.0), 0u);
}

TEST(FormatSystemInfo, ContainsKeyFields) {
  const std::string text = FormatSystemInfo(GetSystemInfo());
  EXPECT_THAT(text, testing::HasSubstr("Logical cores"));
  EXPECT_THAT(text, testing::HasSubstr("Performance cores"));
  EXPECT_THAT(text, testing::HasSubstr("memory"));
}

TEST(RecommendNumThreadsForMemory, Nominal) {
  // 6 GB budget, 1 GB base, 1.8 GB/thread -> floor(5/1.8)=2 threads (measured
  // sweet spot for CPU SIFT on 7 MP images, first_octave=-1).
  const uint64_t gb = 1024ull * 1024ull * 1024ull;
  EXPECT_EQ(RecommendNumThreadsForMemory(6 * gb, 1 * gb, 1800ull * 1024 * 1024,
                                         /*max_threads=*/10),
            2);
  // 8 GB budget -> floor(7/1.8)=3 threads.
  EXPECT_EQ(RecommendNumThreadsForMemory(8 * gb, 1 * gb, 1800ull * 1024 * 1024,
                                         10),
            3);
  // Generous budget is capped by max_threads.
  EXPECT_EQ(RecommendNumThreadsForMemory(64 * gb, 1 * gb, gb, 8), 8);
  // Always at least one worker, even when over budget.
  EXPECT_EQ(RecommendNumThreadsForMemory(1 * gb, 1 * gb, 4 * gb, 8), 1);
  // Zero budget or zero per-thread estimate disables the cap.
  EXPECT_EQ(RecommendNumThreadsForMemory(0, 1 * gb, gb, 8), 8);
  EXPECT_EQ(RecommendNumThreadsForMemory(8 * gb, 1 * gb, 0, 8), 8);
  // max_threads < 1 is clamped to 1.
  EXPECT_EQ(RecommendNumThreadsForMemory(8 * gb, 0, gb, 0), 1);
}

TEST(FormatSystemInfoJson, IsParseableShape) {
  const std::string json = FormatSystemInfoJson(GetSystemInfo());
  EXPECT_FALSE(json.empty());
  EXPECT_EQ(json.front(), '{');
  EXPECT_EQ(json.back(), '}');
  EXPECT_THAT(json, testing::HasSubstr("\"num_logical_cores\":"));
  EXPECT_THAT(json, testing::HasSubstr("\"total_physical_memory_bytes\":"));
  EXPECT_THAT(json, testing::HasSubstr("\"is_apple_silicon\":"));
}

}  // namespace
}  // namespace colmap
