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

#include "colmap/exe/sfm.h"

#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/signal_handler.h"
#include "colmap/util/testing.h"

#include <csignal>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Builds a synthetic dataset (descriptors + verified matches) in a fresh
// database and returns its path along with the total number of frames.
std::filesystem::path SynthesizeMappingDataset(const std::filesystem::path& dir,
                                               int* total_frames) {
  const auto database_path = dir / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 2;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 8;  // 16 frames total -> room to interrupt early.
  options.num_points3D = 100;
  options.camera_has_prior_focal_length = false;
  SynthesizeDataset(options, &gt_reconstruction, database.get());
  *total_frames = options.num_rigs * options.num_frames_per_rig;
  return database_path;
}

// A SIGINT delivered mid-mapping (the cooperative stop wired in
// RunIncrementalMapperImpl) must still flush the in-progress reconstruction to
// disk so the partial work is recoverable.
TEST(RunIncrementalMapperImpl, WritesPartialReconstructionOnInterrupt) {
  ResetInterruptForTesting();

  const auto test_dir = CreateTestDir();
  int total_frames = 0;
  const auto database_path = SynthesizeMappingDataset(test_dir, &total_frames);
  const auto output_path = test_dir / "sparse";
  CreateDirIfNotExists(output_path);

  // Simulate Ctrl-C after a few images have been registered.
  constexpr int kInterruptAfter = 4;
  int num_registered = 0;
  auto next_image_callback = [&]() {
    if (++num_registered == kInterruptAfter) {
      SimulateInterruptForTesting(SIGINT);
    }
  };

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  const bool result =
      RunIncrementalMapperImpl(database_path,
                               /*image_path=*/test_dir,
                               output_path,
                               std::make_shared<IncrementalPipelineOptions>(),
                               reconstruction_manager,
                               /*initial_image_pair_callback=*/{},
                               next_image_callback);

  EXPECT_TRUE(result);
  EXPECT_TRUE(IsInterruptRequested());

  // The interrupted sub-model is written to <output_path>/0 even though its
  // per-model completion callback never fired.
  const auto model_path = output_path / "0";
  ASSERT_TRUE(ExistsFile(model_path / "images.bin"));
  Reconstruction written;
  written.Read(model_path);
  // Partial: some images registered, but fewer than the full dataset.
  EXPECT_GT(written.NumRegImages(), 0);
  EXPECT_LT(written.NumRegImages(), total_frames);

  ResetInterruptForTesting();
}

// Without an interrupt, the normal per-model write path is unaffected (the
// post-Run "write unwritten sub-models" loop must be a no-op, not a double
// write or a crash).
TEST(RunIncrementalMapperImpl, WritesFullReconstructionWithoutInterrupt) {
  ResetInterruptForTesting();

  const auto test_dir = CreateTestDir();
  int total_frames = 0;
  const auto database_path = SynthesizeMappingDataset(test_dir, &total_frames);
  const auto output_path = test_dir / "sparse";
  CreateDirIfNotExists(output_path);

  auto reconstruction_manager = std::make_shared<ReconstructionManager>();
  const bool result =
      RunIncrementalMapperImpl(database_path,
                               /*image_path=*/test_dir,
                               output_path,
                               std::make_shared<IncrementalPipelineOptions>(),
                               reconstruction_manager);

  EXPECT_TRUE(result);
  EXPECT_FALSE(IsInterruptRequested());

  const auto model_path = output_path / "0";
  ASSERT_TRUE(ExistsFile(model_path / "images.bin"));
  Reconstruction written;
  written.Read(model_path);
  EXPECT_EQ(written.NumRegImages(), total_frames);
}

}  // namespace
}  // namespace colmap
