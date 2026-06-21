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

#include "colmap/util/timer.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace colmap {

// Machine-readable progress channel. Emits structured lifecycle events to
// stdout so pipeline orchestrators (and other tooling) can track COLMAP stages
// the same way they track downstream trainers. glog/stderr is left untouched;
// this is an additive, opt-in channel (default format is kNone = silent).
//
// Full specification: memory/process_contract.md. The jsonl schema is
// versioned ("schema":1) and field names are a frozen contract once shipped.
//
// Thread-safety: all emit methods are mutex-guarded and flush per line, so they
// are safe to call from controller worker threads.
enum class ProgressFormat {
  kNone,   // emit nothing (default; byte-identical to legacy behavior)
  kPlain,  // human/greppable: "stage=feature_extraction item=100/1500 ..."
  kJsonl,  // one JSON object per line (recommended for orchestrators)
};

// Parses "none" | "plain" | "jsonl" (case-insensitive). Returns false on an
// unrecognized string and leaves `format` unchanged.
bool ParseProgressFormat(const std::string& str, ProgressFormat* format);

std::string ProgressFormatToString(ProgressFormat format);

// A single numeric metric (key -> value). Integers are emitted without a
// fractional part in jsonl (e.g. registered_images:42, not 42.0).
using ProgressMetric = std::pair<std::string, double>;

class ProgressReporter {
 public:
  // Process-global sink configured once from CLI flags and shared by all
  // controllers in the command.
  static ProgressReporter& Default();

  void SetFormat(ProgressFormat format);
  ProgressFormat GetFormat() const;

  // Default cadence (in items/iterations) for Progress throttling helpers.
  void SetProgressEvery(int every);
  int GetProgressEvery() const;

  // A pipeline phase begins. `total` is the number of items to process; pass
  // a negative value when unknown (orchestrators then show an indeterminate
  // bar). Announcing `total` lets wrappers size a determinate progress bar.
  void StageStarted(const std::string& stage, int64_t total = -1);

  // One item of `stage` finished. `ms_per_item` is optional (negative = omit).
  void Progress(const std::string& stage,
                int64_t item,
                int64_t total,
                double ms_per_item = -1.0);

  // Numeric metrics for a stage (e.g. registered_images, mean_reproj_error_px).
  void Metric(const std::string& stage,
              const std::vector<ProgressMetric>& metrics);

  // Recoverable issue (e.g. a resource cap being applied). Also surfaced to the
  // human via the caller's normal logging.
  void Warning(const std::string& stage, const std::string& message);

  // Liveness signal during long silent phases so orchestrators don't kill us.
  void Heartbeat(const std::string& stage, double elapsed_s);

  // A pipeline phase finished. `outputs` are absolute paths it produced.
  void StageCompleted(const std::string& stage,
                      double elapsed_s,
                      const std::vector<std::string>& outputs = {});

  // The whole command finished. `exit_code` follows util/exit_codes.h.
  void Done(const std::string& command,
            int exit_code,
            const std::vector<std::string>& outputs = {},
            bool interrupted = false);

 private:
  ProgressReporter() = default;

  // Writes a fully-formatted line to stdout and flushes. Caller holds mutex_.
  void EmitLine(const std::string& line);

  mutable std::mutex mutex_;
  ProgressFormat format_ = ProgressFormat::kNone;
  int progress_every_ = 0;  // 0 = use per-stage default
  bool schema_emitted_ = false;
};

// Throttled liveness emitter for a single long, otherwise progress-silent phase
// (bundle adjustment, vocab-tree indexing, dense stereo, ...). Construct one at
// the start of the phase and call Tick() frequently (e.g. each iteration/image);
// it emits ProgressReporter::Default().Heartbeat(stage, elapsed) at most once per
// `interval_s`. A no-op when no --progress_format is set. Centralizes the
// timer + throttle so every phase reports liveness the same way.
class HeartbeatThrottle {
 public:
  explicit HeartbeatThrottle(std::string stage, double interval_s = 5.0);
  void Tick();

 private:
  const std::string stage_;
  const double interval_s_;
  double last_emit_s_ = 0.0;
  Timer timer_;
};

}  // namespace colmap
