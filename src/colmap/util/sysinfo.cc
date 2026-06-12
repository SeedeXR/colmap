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

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
// clang-format off
#include <windows.h>
#include <sysinfoapi.h>
// clang-format on
#else  // Linux and other POSIX.
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <utility>
#endif

namespace colmap {
namespace {

#if defined(__APPLE__)

// Reads a scalar sysctl by name into `out`. Returns false if unavailable
// (e.g. hw.perflevel0.* on Intel Macs, where the key does not exist).
template <typename T>
bool SysctlScalar(const char* name, T* out) {
  T value{};
  size_t len = sizeof(value);
  if (sysctlbyname(name, &value, &len, nullptr, 0) != 0 || len != sizeof(value)) {
    return false;
  }
  *out = value;
  return true;
}

std::string SysctlString(const char* name) {
  size_t len = 0;
  if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) {
    return std::string();
  }
  std::string buffer(len, '\0');
  if (sysctlbyname(name, buffer.data(), &len, nullptr, 0) != 0) {
    return std::string();
  }
  // Drop the trailing NUL terminator that sysctl includes in `len`.
  if (!buffer.empty() && buffer.back() == '\0') {
    buffer.pop_back();
  }
  return buffer;
}

uint64_t AppleAvailableMemoryBytes() {
  mach_port_t host = mach_host_self();
  vm_size_t page_size = 0;
  if (host_page_size(host, &page_size) != KERN_SUCCESS) {
    return 0;
  }
  vm_statistics64_data_t vm_stats{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(host,
                        HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm_stats),
                        &count) != KERN_SUCCESS) {
    return 0;
  }
  // Memory the OS can hand out without swapping: truly free pages plus pages
  // it can reclaim cheaply (inactive, speculative, purgeable file caches).
  const uint64_t reclaimable =
      static_cast<uint64_t>(vm_stats.free_count) +
      static_cast<uint64_t>(vm_stats.inactive_count) +
      static_cast<uint64_t>(vm_stats.speculative_count) +
      static_cast<uint64_t>(vm_stats.purgeable_count);
  return reclaimable * static_cast<uint64_t>(page_size);
}

void DetectApple(SystemInfo* info) {
  info->os_name = "macOS";

  uint64_t mem = 0;
  if (SysctlScalar("hw.memsize", &mem)) {
    info->total_physical_memory_bytes = mem;
  }
  info->available_memory_bytes = AppleAvailableMemoryBytes();

  int logical = 0;
  if (SysctlScalar("hw.logicalcpu", &logical) && logical > 0) {
    info->num_logical_cores = logical;
  }
  int physical = 0;
  if (SysctlScalar("hw.physicalcpu", &physical) && physical > 0) {
    info->num_physical_cores = physical;
  }

  // Asymmetric core topology (Apple Silicon). perflevel0 = performance cores,
  // perflevel1 = efficiency cores. Absent on Intel Macs -> graceful fallback.
  int p_cores = 0;
  int e_cores = 0;
  const bool has_p = SysctlScalar("hw.perflevel0.logicalcpu", &p_cores);
  const bool has_e = SysctlScalar("hw.perflevel1.logicalcpu", &e_cores);
  if (has_p && p_cores > 0) {
    info->num_performance_cores = p_cores;
    info->num_efficiency_cores = (has_e && e_cores > 0) ? e_cores : 0;
  } else {
    // Symmetric (Intel): treat all logical cores as performance cores.
    info->num_performance_cores = info->num_logical_cores;
    info->num_efficiency_cores = 0;
  }

  info->cpu_brand = SysctlString("machdep.cpu.brand_string");

#if defined(__aarch64__) || defined(__arm64__)
  info->is_apple_silicon = true;
#else
  info->is_apple_silicon = false;
#endif
}

#elif defined(_WIN32)

void DetectWindows(SystemInfo* info) {
  info->os_name = "Windows";

  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    info->total_physical_memory_bytes = status.ullTotalPhys;
    info->available_memory_bytes = status.ullAvailPhys;
  }

  SYSTEM_INFO sys_info{};
  GetSystemInfo(&sys_info);
  info->num_logical_cores = static_cast<int>(sys_info.dwNumberOfProcessors);

  // Physical core count via processor relationships (best effort).
  DWORD buffer_len = 0;
  GetLogicalProcessorInformation(nullptr, &buffer_len);
  if (buffer_len > 0) {
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
        buffer_len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (GetLogicalProcessorInformation(buffer.data(), &buffer_len)) {
      int physical = 0;
      for (const auto& entry : buffer) {
        if (entry.Relationship == RelationProcessorCore) {
          ++physical;
        }
      }
      info->num_physical_cores = physical;
    }
  }

  // Windows core topology is symmetric for our purposes.
  info->num_performance_cores = info->num_logical_cores;
  info->num_efficiency_cores = 0;
  info->is_apple_silicon = false;
}

#else  // Linux / generic POSIX.

uint64_t ReadMemInfoKb(const char* key) {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open()) {
    return 0;
  }
  std::string line;
  const size_t key_len = std::strlen(key);
  while (std::getline(meminfo, line)) {
    // Match the full key followed by ':' to avoid "MemFree" matching
    // "MemFreeFoo"-style prefixes.
    if (line.compare(0, key_len, key) == 0 && line.size() > key_len &&
        line[key_len] == ':') {
      // Format: "MemTotal:       16331640 kB".
      std::istringstream parse(line.substr(key_len + 1));
      uint64_t value_kb = 0;
      if (parse >> value_kb) {
        return value_kb * 1024ull;
      }
      return 0;
    }
  }
  return 0;
}

void DetectLinux(SystemInfo* info) {
  info->os_name = "Linux";

  info->total_physical_memory_bytes = ReadMemInfoKb("MemTotal");
  uint64_t available = ReadMemInfoKb("MemAvailable");
  if (available == 0) {
    // Older kernels lack MemAvailable; approximate with MemFree + Cached.
    available = ReadMemInfoKb("MemFree") + ReadMemInfoKb("Cached");
  }
  info->available_memory_bytes = available;

  const long online = sysconf(_SC_NPROCESSORS_ONLN);
  if (online > 0) {
    info->num_logical_cores = static_cast<int>(online);
  }

  // Physical core count: count unique (physical id, core id) pairs in
  // /proc/cpuinfo. Falls back to logical count if parsing fails.
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    std::set<std::pair<int, int>> cores;
    std::string line;
    int physical_id = -1;
    int core_id = -1;
    auto flush = [&]() {
      if (physical_id >= 0 && core_id >= 0) {
        cores.emplace(physical_id, core_id);
      }
      physical_id = -1;
      core_id = -1;
    };
    while (std::getline(cpuinfo, line)) {
      if (line.empty()) {
        flush();
      } else if (line.rfind("physical id", 0) == 0) {
        const size_t pos = line.find(':');
        if (pos != std::string::npos) physical_id = std::atoi(line.c_str() + pos + 1);
      } else if (line.rfind("core id", 0) == 0) {
        const size_t pos = line.find(':');
        if (pos != std::string::npos) core_id = std::atoi(line.c_str() + pos + 1);
      }
    }
    flush();
    if (!cores.empty()) {
      info->num_physical_cores = static_cast<int>(cores.size());
    }
  }

  info->num_performance_cores = info->num_logical_cores;
  info->num_efficiency_cores = 0;
  info->is_apple_silicon = false;
}

#endif

SystemInfo DetectSystemInfo() {
  SystemInfo info;
#if defined(__APPLE__)
  DetectApple(&info);
#elif defined(_WIN32)
  DetectWindows(&info);
#else
  DetectLinux(&info);
#endif

  // Universal fallbacks so callers always get sane, nonzero core counts.
  if (info.num_logical_cores <= 0) {
    const unsigned hw = std::thread::hardware_concurrency();
    info.num_logical_cores = hw > 0 ? static_cast<int>(hw) : 1;
  }
  if (info.num_performance_cores <= 0) {
    info.num_performance_cores = info.num_logical_cores;
  }
  if (info.os_name.empty()) {
    info.os_name = "Unknown";
  }
  return info;
}

std::string JsonEscape(const std::string& str) {
  std::string out;
  out.reserve(str.size() + 2);
  for (const char c : str) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      default: out += c; break;
    }
  }
  return out;
}

}  // namespace

const SystemInfo& GetSystemInfo() {
  static const SystemInfo info = DetectSystemInfo();
  return info;
}

uint64_t GetAvailableMemoryBytes() {
#if defined(__APPLE__)
  return AppleAvailableMemoryBytes();
#elif defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    return status.ullAvailPhys;
  }
  return 0;
#else
  uint64_t available = ReadMemInfoKb("MemAvailable");
  if (available == 0) {
    available = ReadMemInfoKb("MemFree") + ReadMemInfoKb("Cached");
  }
  return available;
#endif
}

int GetNumPerformanceCores() {
  const int cores = GetSystemInfo().num_performance_cores;
  return cores > 0 ? cores : 1;
}

int RecommendNumThreadsForMemory(uint64_t budget_bytes,
                                 uint64_t base_overhead_bytes,
                                 uint64_t per_thread_bytes,
                                 int max_threads) {
  if (max_threads < 1) {
    max_threads = 1;
  }
  // No budget or no per-thread estimate -> cap disabled.
  if (budget_bytes == 0 || per_thread_bytes == 0) {
    return max_threads;
  }
  // Reserve the fixed base overhead; whatever remains is split among workers.
  uint64_t usable =
      budget_bytes > base_overhead_bytes ? budget_bytes - base_overhead_bytes : 0;
  int fit = static_cast<int>(usable / per_thread_bytes);
  if (fit < 1) {
    fit = 1;  // Always allow at least one worker, even over budget.
  }
  return std::min(fit, max_threads);
}

double BytesToGiB(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

uint64_t GiBToBytes(double gib) {
  if (gib <= 0.0) {
    return 0;
  }
  return static_cast<uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

std::string FormatSystemInfo(const SystemInfo& info) {
  std::ostringstream ss;
  ss << "System information\n";
  ss << "  OS:                " << info.os_name;
  if (info.is_apple_silicon) ss << " (Apple Silicon)";
  ss << "\n";
  if (!info.cpu_brand.empty()) {
    ss << "  CPU:               " << info.cpu_brand << "\n";
  }
  ss << "  Logical cores:     " << info.num_logical_cores << "\n";
  if (info.num_physical_cores > 0) {
    ss << "  Physical cores:    " << info.num_physical_cores << "\n";
  }
  ss << "  Performance cores: " << info.num_performance_cores << "\n";
  ss << "  Efficiency cores:  " << info.num_efficiency_cores << "\n";
  if (info.total_physical_memory_bytes > 0) {
    ss << "  Total memory:      " << BytesToGiB(info.total_physical_memory_bytes)
       << " GiB\n";
  } else {
    ss << "  Total memory:      unknown\n";
  }
  if (info.available_memory_bytes > 0) {
    ss << "  Available memory:  " << BytesToGiB(info.available_memory_bytes)
       << " GiB\n";
  } else {
    ss << "  Available memory:  unknown\n";
  }
  return ss.str();
}

std::string FormatSystemInfoJson(const SystemInfo& info) {
  std::ostringstream ss;
  ss << "{";
  ss << "\"os\":\"" << JsonEscape(info.os_name) << "\",";
  ss << "\"cpu_brand\":\"" << JsonEscape(info.cpu_brand) << "\",";
  ss << "\"is_apple_silicon\":" << (info.is_apple_silicon ? "true" : "false")
     << ",";
  ss << "\"num_logical_cores\":" << info.num_logical_cores << ",";
  ss << "\"num_physical_cores\":" << info.num_physical_cores << ",";
  ss << "\"num_performance_cores\":" << info.num_performance_cores << ",";
  ss << "\"num_efficiency_cores\":" << info.num_efficiency_cores << ",";
  ss << "\"total_physical_memory_bytes\":" << info.total_physical_memory_bytes
     << ",";
  ss << "\"available_memory_bytes\":" << info.available_memory_bytes;
  ss << "}";
  return ss.str();
}

}  // namespace colmap
