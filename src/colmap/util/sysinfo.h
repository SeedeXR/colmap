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
#include <string>

namespace colmap {

// Portable description of the host machine's compute resources. Used to derive
// resource-aware defaults (cache sizes, thread counts) so COLMAP coexists with
// other applications instead of exhausting the machine. All fields are
// best-effort: a value of 0 means "could not be determined on this platform",
// and callers must treat 0 as "unknown" (i.e. fall back to existing behavior),
// never as a real measurement.
struct SystemInfo {
  // Total installed physical RAM in bytes. 0 if undetectable.
  uint64_t total_physical_memory_bytes = 0;
  // Memory available for allocation right now, in bytes. 0 if undetectable.
  // On macOS this is (free + inactive + speculative + purgeable) pages; on
  // Linux it is MemAvailable; on Windows it is ullAvailPhys.
  uint64_t available_memory_bytes = 0;
  // Number of logical (hyperthread/SMT) processors. >= 1 on any real machine.
  int num_logical_cores = 0;
  // Number of physical cores. 0 if undetectable.
  int num_physical_cores = 0;
  // Apple Silicon performance ("P") cores. On platforms without an asymmetric
  // core topology this equals num_logical_cores.
  int num_performance_cores = 0;
  // Apple Silicon efficiency ("E") cores. 0 on symmetric platforms.
  int num_efficiency_cores = 0;
  // Best-effort CPU brand string (e.g. "Apple M1 Pro"). Empty if unknown.
  std::string cpu_brand;
  // Coarse OS name: "macOS", "Linux", "Windows", or "Unknown".
  std::string os_name;
  // True on arm64 Apple hardware (the unified-memory target of this port).
  bool is_apple_silicon = false;
};

// Returns cached system info, detected once on first call. Cheap thereafter.
// Note: total_physical_memory and the core topology are stable, but
// available_memory_bytes is a snapshot from first call; use
// GetAvailableMemoryBytes() for a fresh reading.
const SystemInfo& GetSystemInfo();

// Fresh snapshot of currently-available memory in bytes (0 if undetectable).
uint64_t GetAvailableMemoryBytes();

// Number of performance cores to size default thread pools against on
// asymmetric hardware (Apple Silicon hw.perflevel0.logicalcpu). Falls back to
// the logical core count on symmetric platforms. Always >= 1.
int GetNumPerformanceCores();

// Largest worker count in [1, max_threads] whose combined per-worker memory
// fits a budget:
//   clamp(floor((budget_bytes - base_overhead_bytes) / per_thread_bytes),
//         1, max_threads).
// A budget_bytes or per_thread_bytes of 0 disables the cap and returns
// max_threads unchanged (so callers that cannot estimate memory keep existing
// behavior). Never returns less than 1. This is platform-independent: it helps
// any machine (Linux/Windows included) avoid oversubscribing RAM with large
// images, while on symmetric platforms the default behavior is preserved when
// no budget is supplied.
int RecommendNumThreadsForMemory(uint64_t budget_bytes,
                                 uint64_t base_overhead_bytes,
                                 uint64_t per_thread_bytes,
                                 int max_threads);

// Convenience: bytes -> gibibytes (1024^3).
double BytesToGiB(uint64_t bytes);

// Convenience: gibibytes -> bytes.
uint64_t GiBToBytes(double gib);

// Renders SystemInfo as a human-readable multi-line block (for `system_info`).
std::string FormatSystemInfo(const SystemInfo& info);

// Renders SystemInfo as a single-line JSON object (for `system_info --format
// json` and bug reports). Field names are stable.
std::string FormatSystemInfoJson(const SystemInfo& info);

}  // namespace colmap
