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

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(SignalHandler, InitiallyNotRequested) {
  ResetInterruptForTesting();
  EXPECT_FALSE(IsInterruptRequested());
  EXPECT_EQ(GetInterruptSignal(), 0);
  EXPECT_EQ(GetInterruptExitCode(), 0);
}

TEST(SignalHandler, SigintExitCode) {
  ResetInterruptForTesting();
  SimulateInterruptForTesting(SIGINT);
  EXPECT_TRUE(IsInterruptRequested());
  EXPECT_EQ(GetInterruptSignal(), SIGINT);
  EXPECT_EQ(GetInterruptExitCode(), 128 + SIGINT);  // 130
  ResetInterruptForTesting();
}

TEST(SignalHandler, SigtermExitCode) {
  ResetInterruptForTesting();
  SimulateInterruptForTesting(SIGTERM);
  EXPECT_TRUE(IsInterruptRequested());
  EXPECT_EQ(GetInterruptSignal(), SIGTERM);
  EXPECT_EQ(GetInterruptExitCode(), 128 + SIGTERM);  // 143
  ResetInterruptForTesting();
}

TEST(SignalHandler, InstallIsIdempotent) {
  // Installing handlers must not throw or alter pending state.
  ResetInterruptForTesting();
  InstallInterruptHandlers();
  InstallInterruptHandlers();
  EXPECT_FALSE(IsInterruptRequested());
}

TEST(SignalHandler, RealSignalIsCaught) {
  // End-to-end: install the real handler and raise SIGINT in-process. The
  // handler must flip the flag without terminating the test.
  ResetInterruptForTesting();
  InstallInterruptHandlers();
  std::raise(SIGINT);
  EXPECT_TRUE(IsInterruptRequested());
  EXPECT_EQ(GetInterruptSignal(), SIGINT);
  ResetInterruptForTesting();
}

}  // namespace
}  // namespace colmap
