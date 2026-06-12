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

#include "colmap/util/progress.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>

namespace colmap {
namespace {

std::string ToLower(std::string str) {
  for (char& c : str) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return str;
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

// Formats a numeric value for JSON: integers without a fractional part, other
// values with up to 6 significant digits and no trailing-zero noise.
std::string JsonNumber(double value) {
  if (std::isnan(value) || std::isinf(value)) {
    return "null";
  }
  if (value == std::floor(value) && std::fabs(value) < 9.0e15) {
    return std::to_string(static_cast<int64_t>(value));
  }
  std::ostringstream ss;
  ss.precision(6);
  ss << value;
  return ss.str();
}

std::string JsonArray(const std::vector<std::string>& items) {
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) ss << ",";
    ss << "\"" << JsonEscape(items[i]) << "\"";
  }
  ss << "]";
  return ss.str();
}

}  // namespace

bool ParseProgressFormat(const std::string& str, ProgressFormat* format) {
  const std::string lower = ToLower(str);
  if (lower == "none" || lower.empty()) {
    *format = ProgressFormat::kNone;
  } else if (lower == "plain") {
    *format = ProgressFormat::kPlain;
  } else if (lower == "jsonl") {
    *format = ProgressFormat::kJsonl;
  } else {
    return false;
  }
  return true;
}

std::string ProgressFormatToString(ProgressFormat format) {
  switch (format) {
    case ProgressFormat::kNone: return "none";
    case ProgressFormat::kPlain: return "plain";
    case ProgressFormat::kJsonl: return "jsonl";
  }
  return "none";
}

ProgressReporter& ProgressReporter::Default() {
  static ProgressReporter reporter;
  return reporter;
}

void ProgressReporter::SetFormat(ProgressFormat format) {
  std::lock_guard<std::mutex> lock(mutex_);
  format_ = format;
}

ProgressFormat ProgressReporter::GetFormat() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return format_;
}

void ProgressReporter::SetProgressEvery(int every) {
  std::lock_guard<std::mutex> lock(mutex_);
  progress_every_ = every;
}

int ProgressReporter::GetProgressEvery() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return progress_every_;
}

void ProgressReporter::EmitLine(const std::string& line) {
  // Caller holds mutex_. Write to stdout and flush so the line streams even
  // when stdout is a pipe (orchestrator) rather than a TTY.
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

void ProgressReporter::StageStarted(const std::string& stage, int64_t total) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (format_ == ProgressFormat::kNone) return;
  std::ostringstream ss;
  if (format_ == ProgressFormat::kJsonl) {
    ss << "{";
    if (!schema_emitted_) {
      ss << "\"schema\":1,";
      schema_emitted_ = true;
    }
    ss << "\"type\":\"stage_started\",\"stage\":\"" << JsonEscape(stage) << "\"";
    if (total >= 0) ss << ",\"total\":" << total;
    ss << "}";
  } else {
    ss << "stage_started stage=" << stage;
    if (total >= 0) ss << " total=" << total;
  }
  EmitLine(ss.str());
}

void ProgressReporter::Progress(const std::string& stage,
                                int64_t item,
                                int64_t total,
                                double ms_per_item) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (format_ == ProgressFormat::kNone) return;
  std::ostringstream ss;
  if (format_ == ProgressFormat::kJsonl) {
    ss << "{\"type\":\"progress\",\"stage\":\"" << JsonEscape(stage) << "\""
       << ",\"item\":" << item;
    if (total >= 0) ss << ",\"total\":" << total;
    if (ms_per_item >= 0.0) ss << ",\"ms_per_item\":" << JsonNumber(ms_per_item);
    ss << "}";
  } else {
    ss << "stage=" << stage << " item=" << item;
    if (total >= 0) ss << "/" << total;
    if (ms_per_item >= 0.0) ss << " ms/item=" << JsonNumber(ms_per_item);
  }
  EmitLine(ss.str());
}

void ProgressReporter::Metric(const std::string& stage,
                              const std::vector<ProgressMetric>& metrics) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (format_ == ProgressFormat::kNone) return;
  std::ostringstream ss;
  if (format_ == ProgressFormat::kJsonl) {
    ss << "{\"type\":\"metric\",\"stage\":\"" << JsonEscape(stage) << "\"";
    for (const auto& [key, value] : metrics) {
      ss << ",\"" << JsonEscape(key) << "\":" << JsonNumber(value);
    }
    ss << "}";
  } else {
    ss << "metric stage=" << stage;
    for (const auto& [key, value] : metrics) {
      ss << " " << key << "=" << JsonNumber(value);
    }
  }
  EmitLine(ss.str());
}

void ProgressReporter::Warning(const std::string& stage,
                               const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (format_ == ProgressFormat::kNone) return;
  std::ostringstream ss;
  if (format_ == ProgressFormat::kJsonl) {
    ss << "{\"type\":\"warning\",\"stage\":\"" << JsonEscape(stage)
       << "\",\"msg\":\"" << JsonEscape(message) << "\"}";
  } else {
    ss << "warning stage=" << stage << " msg=" << message;
  }
  EmitLine(ss.str());
}

void ProgressReporter::Heartbeat(const std::string& stage, double elapsed_s) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (format_ == ProgressFormat::kNone) return;
  std::ostringstream ss;
  if (format_ == ProgressFormat::kJsonl) {
    ss << "{\"type\":\"heartbeat\",\"stage\":\"" << JsonEscape(stage)
       << "\",\"elapsed_s\":" << JsonNumber(elapsed_s) << "}";
  } else {
    ss << "heartbeat stage=" << stage << " elapsed_s=" << JsonNumber(elapsed_s);
  }
  EmitLine(ss.str());
}

void ProgressReporter::StageCompleted(const std::string& stage,
                                      double elapsed_s,
                                      const std::vector<std::string>& outputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (format_ == ProgressFormat::kNone) return;
  std::ostringstream ss;
  if (format_ == ProgressFormat::kJsonl) {
    ss << "{\"type\":\"stage_completed\",\"stage\":\"" << JsonEscape(stage)
       << "\",\"elapsed_s\":" << JsonNumber(elapsed_s);
    if (!outputs.empty()) ss << ",\"outputs\":" << JsonArray(outputs);
    ss << "}";
  } else {
    ss << "stage_completed stage=" << stage
       << " elapsed_s=" << JsonNumber(elapsed_s);
    for (const auto& output : outputs) ss << " output=" << output;
  }
  EmitLine(ss.str());
}

void ProgressReporter::Done(const std::string& command,
                            int exit_code,
                            const std::vector<std::string>& outputs,
                            bool interrupted) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (format_ == ProgressFormat::kNone) return;
  std::ostringstream ss;
  if (format_ == ProgressFormat::kJsonl) {
    ss << "{\"type\":\"done\",\"command\":\"" << JsonEscape(command)
       << "\",\"exit_code\":" << exit_code;
    if (interrupted) ss << ",\"interrupted\":true";
    if (!outputs.empty()) ss << ",\"outputs\":" << JsonArray(outputs);
    ss << "}";
  } else {
    // The msplat-style human summary line.
    ss << "Done: " << command << " exit_code=" << exit_code;
    if (interrupted) ss << " (interrupted)";
    for (const auto& output : outputs) ss << " wrote " << output;
  }
  EmitLine(ss.str());
}

}  // namespace colmap
