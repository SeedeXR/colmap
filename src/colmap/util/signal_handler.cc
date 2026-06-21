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

#include "colmap/util/signal_handler.h"

#include <csignal>

#if defined(_WIN32)
// clang-format off
#include <windows.h>
// clang-format on
#endif

namespace colmap {
namespace {

// Set from the (async-signal) handler; read from normal threads. sig_atomic_t
// is the only type the C/C++ standard guarantees safe to touch in a handler.
volatile std::sig_atomic_t g_interrupt_signal = 0;

#if defined(_WIN32)

BOOL WINAPI ConsoleCtrlHandler(DWORD ctrl_type) {
  switch (ctrl_type) {
    case CTRL_C_EVENT:
      g_interrupt_signal = SIGINT;
      return TRUE;
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
      g_interrupt_signal = SIGTERM;
      return TRUE;
    default:
      return FALSE;
  }
}

#else  // POSIX.

extern "C" void PosixSignalHandler(int signum) {
  // Only async-signal-safe work: record the signal. The handler is installed
  // with SA_RESETHAND, so it fires exactly once and the disposition reverts to
  // the default; a second identical signal then terminates the process via the
  // default disposition if the cooperative shutdown wedges.
  g_interrupt_signal = signum;
}

#endif

}  // namespace

void InstallInterruptHandlers() {
#if defined(_WIN32)
  SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);
#else
  struct sigaction action {};
  action.sa_handler = PosixSignalHandler;
  sigemptyset(&action.sa_mask);
  // No SA_RESTART (let blocking calls return EINTR). SA_RESETHAND restores the
  // default disposition after the first signal, so a second SIGINT/SIGTERM
  // hard-terminates the process if the cooperative shutdown wedges.
  action.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &action, nullptr);
  sigaction(SIGTERM, &action, nullptr);
#endif
}

bool IsInterruptRequested() { return g_interrupt_signal != 0; }

int GetInterruptSignal() { return static_cast<int>(g_interrupt_signal); }

int GetInterruptExitCode() {
  const int signum = static_cast<int>(g_interrupt_signal);
  return signum == 0 ? 0 : 128 + signum;
}

void ResetInterruptForTesting() { g_interrupt_signal = 0; }

void SimulateInterruptForTesting(int signum) {
  g_interrupt_signal = signum;
}

}  // namespace colmap
