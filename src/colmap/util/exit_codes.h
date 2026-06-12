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

namespace colmap {

// Distinct process exit codes so orchestrators (and shell scripts) can react to
// *why* a command failed, not just whether. Additive over the historical
// 0/1 contract: kSuccess and kGenericFailure are unchanged, every other value
// is new and opt-in (a command only returns them where it has the context to
// distinguish the cause). See memory/process_contract.md §7.
enum ExitCode {
  EXIT_CODE_SUCCESS = 0,
  // Generic failure, unchanged for backward compatibility (== EXIT_FAILURE).
  EXIT_CODE_FAILURE = 1,
  // Invalid command-line arguments / option validation failure.
  EXIT_CODE_ARGUMENT_ERROR = 2,
  // Could not load required input (database, images, or model missing/corrupt).
  EXIT_CODE_INPUT_LOAD_ERROR = 3,
  // GPU / Metal / CUDA initialization failed.
  EXIT_CODE_GPU_INIT_ERROR = 4,
  // Could not write an output file (permissions, disk full, bad path).
  EXIT_CODE_OUTPUT_WRITE_ERROR = 5,
  // Interrupted by SIGINT after a clean partial save (128 + 2).
  EXIT_CODE_INTERRUPTED_SIGINT = 130,
  // Interrupted by SIGTERM after a clean partial save (128 + 15).
  EXIT_CODE_INTERRUPTED_SIGTERM = 143,
};

}  // namespace colmap
