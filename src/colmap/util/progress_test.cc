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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

using testing::HasSubstr;
using testing::Not;

// Helper: run `body` with the Default() reporter set to `format` and return
// everything it wrote to stdout.
template <typename Body>
std::string CaptureEmit(ProgressFormat format, Body&& body) {
  ProgressReporter& reporter = ProgressReporter::Default();
  reporter.SetFormat(format);
  testing::internal::CaptureStdout();
  body(reporter);
  return testing::internal::GetCapturedStdout();
}

TEST(ParseProgressFormat, RoundTrip) {
  ProgressFormat format = ProgressFormat::kJsonl;
  EXPECT_TRUE(ParseProgressFormat("none", &format));
  EXPECT_EQ(format, ProgressFormat::kNone);
  EXPECT_TRUE(ParseProgressFormat("PLAIN", &format));
  EXPECT_EQ(format, ProgressFormat::kPlain);
  EXPECT_TRUE(ParseProgressFormat("jsonl", &format));
  EXPECT_EQ(format, ProgressFormat::kJsonl);
  EXPECT_FALSE(ParseProgressFormat("bogus", &format));
  // Unchanged on parse failure.
  EXPECT_EQ(format, ProgressFormat::kJsonl);
  EXPECT_EQ(ProgressFormatToString(ProgressFormat::kPlain), "plain");
}

TEST(ProgressReporter, NoneFormatEmitsNothing) {
  const std::string out = CaptureEmit(ProgressFormat::kNone, [](auto& r) {
    r.StageStarted("feature_extraction", 1500);
    r.Progress("feature_extraction", 100, 1500, 12.3);
    r.Done("cmd", 0);
  });
  EXPECT_TRUE(out.empty());
}

TEST(ProgressReporter, JsonlSchemaAndShape) {
  const std::string out = CaptureEmit(ProgressFormat::kJsonl, [](auto& r) {
    r.StageStarted("feature_extraction", 1500);
    r.Progress("feature_extraction", 100, 1500, 12.3);
    r.Metric("mapper", {{"registered_images", 42}, {"points3D", 118234}});
    r.Warning("mvs_patch_match", "cache_size capped to 4.0 GB");
    r.Heartbeat("bundle_adjustment", 45.0);
    r.StageCompleted("mapper", 312.4, {"/abs/path/sparse/0"});
    r.Done("automatic_reconstructor", 0, {"/abs/path/sparse/0"});
  });
  // Schema appears exactly once, on the first event.
  EXPECT_THAT(out, HasSubstr("\"schema\":1"));
  EXPECT_EQ(out.find("\"schema\":1"), out.rfind("\"schema\":1"));
  EXPECT_THAT(out, HasSubstr("\"type\":\"stage_started\",\"stage\":"
                             "\"feature_extraction\",\"total\":1500"));
  EXPECT_THAT(out, HasSubstr("\"item\":100"));
  EXPECT_THAT(out, HasSubstr("\"ms_per_item\":12.3"));
  // Integer metrics serialized without a fractional part.
  EXPECT_THAT(out, HasSubstr("\"registered_images\":42"));
  EXPECT_THAT(out, HasSubstr("\"points3D\":118234"));
  EXPECT_THAT(out, Not(HasSubstr("42.0")));
  EXPECT_THAT(out, HasSubstr("\"type\":\"warning\""));
  EXPECT_THAT(out, HasSubstr("\"type\":\"heartbeat\""));
  EXPECT_THAT(out, HasSubstr("\"type\":\"stage_completed\""));
  EXPECT_THAT(out, HasSubstr("\"outputs\":[\"/abs/path/sparse/0\"]"));
  EXPECT_THAT(out, HasSubstr("\"type\":\"done\",\"command\":"
                             "\"automatic_reconstructor\",\"exit_code\":0"));
}

TEST(ProgressReporter, JsonlInterruptedDone) {
  const std::string out = CaptureEmit(ProgressFormat::kJsonl, [](auto& r) {
    r.Done("mapper", 143, {"/abs/sparse/0"}, /*interrupted=*/true);
  });
  EXPECT_THAT(out, HasSubstr("\"exit_code\":143"));
  EXPECT_THAT(out, HasSubstr("\"interrupted\":true"));
}

TEST(ProgressReporter, PlainFormat) {
  const std::string out = CaptureEmit(ProgressFormat::kPlain, [](auto& r) {
    r.Progress("feature_extraction", 100, 1500, 12.3);
    r.Done("matcher", 0, {"/abs/db.db"});
  });
  EXPECT_THAT(out, HasSubstr("stage=feature_extraction item=100/1500"));
  EXPECT_THAT(out, HasSubstr("ms/item=12.3"));
  EXPECT_THAT(out, HasSubstr("Done: matcher"));
  EXPECT_THAT(out, HasSubstr("wrote /abs/db.db"));
  // Plain format must not emit JSON braces.
  EXPECT_THAT(out, Not(HasSubstr("{\"")));
}

TEST(ProgressReporter, JsonEscaping) {
  const std::string out = CaptureEmit(ProgressFormat::kJsonl, [](auto& r) {
    r.Warning("stage", "path \"with\" quotes\nand newline");
  });
  EXPECT_THAT(out, HasSubstr("\\\"with\\\""));
  EXPECT_THAT(out, HasSubstr("\\n"));
}

}  // namespace
}  // namespace colmap
