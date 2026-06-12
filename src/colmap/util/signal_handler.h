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

// Cooperative interrupt handling for long-running commands. Installs portable
// SIGINT/SIGTERM handlers (POSIX sigaction, Windows console control handler)
// that do nothing but set an async-signal-safe flag. The running command polls
// IsInterruptRequested() (typically by wiring it into a controller's
// "check if stopped" predicate) and shuts down cleanly, saving partial results,
// then exits with the matching code (130 for SIGINT, 143 for SIGTERM).
//
// Rationale for the poll-don't-call design: it is not safe to call arbitrary
// C++ (mutexes, allocation, std::function) from inside a signal handler, so the
// handler only flips an atomic; the cooperative stop happens on a normal thread.
// See memory/process_contract.md §8.

// Installs the handlers. Idempotent; safe to call once at program start.
void InstallInterruptHandlers();

// True once SIGINT/SIGTERM has been received. Async-signal-safe to read.
bool IsInterruptRequested();

// The signal number that triggered the interrupt (e.g. SIGINT=2, SIGTERM=15),
// or 0 if none received yet.
int GetInterruptSignal();

// Process exit code corresponding to the received signal (128 + signum), or
// 0 if no interrupt has been requested.
int GetInterruptExitCode();

// Clears the interrupt state. Intended for unit tests; not for normal use.
void ResetInterruptForTesting();

// Simulates receiving `signum` without raising a real signal. Test-only.
void SimulateInterruptForTesting(int signum);

}  // namespace colmap
