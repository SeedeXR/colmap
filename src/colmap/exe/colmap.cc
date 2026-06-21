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

#include "colmap/exe/database.h"
#include "colmap/exe/feature.h"
#include "colmap/exe/gui.h"
#include "colmap/exe/image.h"
#include "colmap/exe/model.h"
#if defined(COLMAP_MVS_ENABLED)
#include "colmap/exe/mvs.h"
#endif
#include "colmap/exe/sfm.h"
#include "colmap/exe/vocab_tree.h"
#include "colmap/util/oiio_utils.h"
#include "colmap/util/signal_handler.h"
#include "colmap/util/string.h"
#include "colmap/util/sysinfo.h"
#include "colmap/util/version.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <set>

namespace {

using command_func_t = std::function<int(int, char**)>;

void ShowVersion() {
  std::cout << colmap::GetVersionInfo() << " (" << colmap::GetBuildInfo()
            << ")\n";
}

// `colmap system_info [--format json]`: report detected machine resources and
// build configuration, for tuning and bug reports. See process_contract.md §9.
int RunSystemInfo(int argc, char** argv) {
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--format" && i + 1 < argc) {
      json = std::string(argv[i + 1]) == "json";
    } else if (std::string(argv[i]) == "-h" ||
               std::string(argv[i]) == "--help") {
      std::cout << "Usage: colmap system_info [--format {text,json}]\n";
      return EXIT_SUCCESS;
    }
  }

  const colmap::SystemInfo& info = colmap::GetSystemInfo();
  int omp_max_threads = 1;
#ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
#endif
  const int recommended_perf = colmap::GetNumPerformanceCores();

  if (json) {
    std::string sys = colmap::FormatSystemInfoJson(info);
    // Splice build/runtime fields in before the closing brace. Locating the
    // last '}' (rather than blindly dropping the final char) is robust to any
    // trailing whitespace/newline FormatSystemInfoJson might emit.
    const std::string extra = colmap::StringPrintf(
        ",\"openmp_max_threads\":%d,\"recommended_extraction_threads\":%d,"
        "\"version\":\"%s\"",
        omp_max_threads,
        recommended_perf,
        colmap::JsonEscape(colmap::GetVersionInfo()).c_str());
    const size_t brace = sys.find_last_of('}');
    if (brace != std::string::npos) {
      sys.insert(brace, extra);
    }
    std::cout << sys << "\n";
  } else {
    std::cout << colmap::FormatSystemInfo(info);
    std::cout << "  OpenMP threads:    " << omp_max_threads << "\n";
    std::cout << "  Rec. extract thr:  " << recommended_perf
              << " (performance cores)\n";
    std::cout << "  Version:           " << colmap::GetVersionInfo() << "\n";
    std::cout << "  Build:             " << colmap::GetBuildInfo() << "\n";
  }
  return EXIT_SUCCESS;
}

void ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>>& commands) {
  ShowVersion();

  std::cout << "Usage:\n";
  std::cout << "  colmap [command] [options]\n";

  std::cout << "Documentation:\n";
  std::cout << "  https://colmap.github.io/\n";

  std::cout << "Example usage:\n";
  std::cout << "  colmap help [ -h, --help ]\n";
  std::cout << "  colmap gui\n";
  std::cout << "  colmap gui -h [ --help ]\n";
  std::cout << "  colmap automatic_reconstructor -h [ --help ]\n";
  std::cout << "  colmap automatic_reconstructor --image_path IMAGES "
               "--workspace_path WORKSPACE\n";
  std::cout << "  colmap feature_extractor --image_path IMAGES --database_path "
               "DATABASE\n";
  std::cout << "  colmap exhaustive_matcher --database_path DATABASE\n";
  std::cout << "  colmap mapper --image_path IMAGES --database_path DATABASE "
               "--output_path MODEL\n";
  std::cout << "  ...\n";

  // Command groups for a readable listing. This only organizes the display;
  // command dispatch still uses the flat `commands` list. Any command not
  // assigned to a group is printed under "Other" so new commands are never
  // hidden from --help.
  struct CommandGroup {
    std::string title;
    std::vector<std::string> names;
  };
  static const std::vector<CommandGroup> kGroups = {
      {"General",
       {"gui",
        "automatic_reconstructor",
        "project_generator",
        "mapper_advisor",
        "system_info"}},
      {"Database",
       {"database_creator", "database_cleaner", "database_merger"}},
      {"Feature extraction & matching",
       {"feature_extractor",
        "feature_importer",
        "exhaustive_matcher",
        "sequential_matcher",
        "spatial_matcher",
        "transitive_matcher",
        "vocab_tree_matcher",
        "vocab_tree_builder",
        "vocab_tree_retriever",
        "matches_importer",
        "geometric_verifier",
        "guided_geometric_verifier"}},
      {"Sparse reconstruction (SfM)",
       {"mapper",
        "global_mapper",
        "hierarchical_mapper",
        "pose_prior_mapper",
        "image_registrator",
        "point_triangulator",
        "bundle_adjuster",
        "rig_configurator",
        "view_graph_calibrator",
        "rotation_averager",
        "point_filtering",
        "color_extractor"}},
      {"Dense reconstruction & meshing",
       {"image_undistorter",
        "image_undistorter_standalone",
        "image_rectifier",
        "patch_match_stereo",
        "stereo_fusion",
        "poisson_mesher",
        "delaunay_mesher",
        "advancing_front_mesher",
        "mesh_simplifier",
        "mesh_texturer"}},
      {"Model utilities",
       {"model_analyzer",
        "model_aligner",
        "model_comparer",
        "model_converter",
        "model_cropper",
        "model_clusterer",
        "model_merger",
        "model_splitter",
        "model_transformer",
        "model_orientation_aligner",
        "image_deleter",
        "image_filterer"}},
  };

  std::set<std::string> available;
  for (const auto& command : commands) {
    available.insert(command.first);
  }

  std::cout << "Available commands:\n";
  std::set<std::string> printed;
  for (const auto& group : kGroups) {
    std::vector<std::string> names;
    for (const std::string& name : group.names) {
      if (available.count(name) > 0) {
        names.push_back(name);
        printed.insert(name);
      }
    }
    if (names.empty() && group.title != "General") {
      continue;
    }
    std::cout << "\n  " << group.title << ":\n";
    // The built-in help/version pseudo-commands live under General.
    if (group.title == "General") {
      std::cout << "    help\n";
      std::cout << "    version\n";
    }
    for (const std::string& name : names) {
      std::cout << "    " << name << '\n';
    }
  }

  std::vector<std::string> others;
  for (const auto& command : commands) {
    if (printed.count(command.first) == 0) {
      others.push_back(command.first);
    }
  }
  if (!others.empty()) {
    std::cout << "\n  Other:\n";
    for (const std::string& name : others) {
      std::cout << "    " << name << '\n';
    }
  }
  std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  colmap::EnsureOpenImageIOInitialized();
  // Install cooperative SIGINT/SIGTERM handlers so long-running commands can
  // shut down cleanly and save partial results (see util/signal_handler.h and
  // memory/process_contract.md §8).
  colmap::InstallInterruptHandlers();

  std::vector<std::pair<std::string, command_func_t>> commands;
  commands.emplace_back("gui", &colmap::RunGraphicalUserInterface);
  commands.emplace_back("automatic_reconstructor",
                        &colmap::RunAutomaticReconstructor);
  commands.emplace_back("bundle_adjuster", &colmap::RunBundleAdjuster);
  commands.emplace_back("color_extractor", &colmap::RunColorExtractor);
  commands.emplace_back("database_cleaner", &colmap::RunDatabaseCleaner);
  commands.emplace_back("database_creator", &colmap::RunDatabaseCreator);
  commands.emplace_back("database_merger", &colmap::RunDatabaseMerger);
#if defined(COLMAP_MVS_ENABLED)
  commands.emplace_back("advancing_front_mesher",
                        &colmap::RunAdvancingFrontMesher);
  commands.emplace_back("delaunay_mesher", &colmap::RunDelaunayMesher);
#endif
  commands.emplace_back("exhaustive_matcher", &colmap::RunExhaustiveMatcher);
  commands.emplace_back("feature_extractor", &colmap::RunFeatureExtractor);
  commands.emplace_back("feature_importer", &colmap::RunFeatureImporter);
  commands.emplace_back("geometric_verifier", &colmap::RunGeometricVerifier);
  commands.emplace_back("global_mapper", &colmap::RunGlobalMapper);
  commands.emplace_back("guided_geometric_verifier",
                        &colmap::RunGuidedGeometricVerifier);
  commands.emplace_back("hierarchical_mapper", &colmap::RunHierarchicalMapper);
  commands.emplace_back("image_deleter", &colmap::RunImageDeleter);
  commands.emplace_back("image_filterer", &colmap::RunImageFilterer);
  commands.emplace_back("image_rectifier", &colmap::RunImageRectifier);
  commands.emplace_back("image_registrator", &colmap::RunImageRegistrator);
  commands.emplace_back("image_undistorter", &colmap::RunImageUndistorter);
  commands.emplace_back("image_undistorter_standalone",
                        &colmap::RunImageUndistorterStandalone);
  commands.emplace_back("mapper", &colmap::RunMapper);
  commands.emplace_back("matches_importer", &colmap::RunMatchesImporter);
#if defined(COLMAP_MVS_ENABLED)
  commands.emplace_back("mesh_simplifier", &colmap::RunMeshSimplifier);
  commands.emplace_back("mesh_texturer", &colmap::RunMeshTexturer);
#endif
  commands.emplace_back("model_aligner", &colmap::RunModelAligner);
  commands.emplace_back("model_analyzer", &colmap::RunModelAnalyzer);
  commands.emplace_back("model_clusterer", &colmap::RunModelClusterer);
  commands.emplace_back("model_comparer", &colmap::RunModelComparer);
  commands.emplace_back("model_converter", &colmap::RunModelConverter);
  commands.emplace_back("model_cropper", &colmap::RunModelCropper);
  commands.emplace_back("model_merger", &colmap::RunModelMerger);
  commands.emplace_back("model_orientation_aligner",
                        &colmap::RunModelOrientationAligner);
  commands.emplace_back("model_splitter", &colmap::RunModelSplitter);
  commands.emplace_back("model_transformer", &colmap::RunModelTransformer);
#if defined(COLMAP_MVS_ENABLED)
  commands.emplace_back("patch_match_stereo", &colmap::RunPatchMatchStereo);
#endif
  commands.emplace_back("point_filtering", &colmap::RunPointFiltering);
  commands.emplace_back("point_triangulator", &colmap::RunPointTriangulator);
  commands.emplace_back("mapper_advisor", &colmap::RunMapperAdvisor);
  commands.emplace_back("pose_prior_mapper", &colmap::RunPosePriorMapper);
#if defined(COLMAP_MVS_ENABLED)
  commands.emplace_back("poisson_mesher", &colmap::RunPoissonMesher);
#endif
  commands.emplace_back("project_generator", &colmap::RunProjectGenerator);
  commands.emplace_back("rig_configurator", &colmap::RunRigConfigurator);
  commands.emplace_back("rotation_averager", &colmap::RunRotationAverager);
  commands.emplace_back("sequential_matcher", &colmap::RunSequentialMatcher);
  commands.emplace_back("spatial_matcher", &colmap::RunSpatialMatcher);
  commands.emplace_back("system_info", &RunSystemInfo);
#if defined(COLMAP_MVS_ENABLED)
  commands.emplace_back("stereo_fusion", &colmap::RunStereoFuser);
#endif
  commands.emplace_back("transitive_matcher", &colmap::RunTransitiveMatcher);
  commands.emplace_back("view_graph_calibrator",
                        &colmap::RunViewGraphCalibrator);
  commands.emplace_back("vocab_tree_builder", &colmap::RunVocabTreeBuilder);
  commands.emplace_back("vocab_tree_matcher", &colmap::RunVocabTreeMatcher);
  commands.emplace_back("vocab_tree_retriever", &colmap::RunVocabTreeRetriever);

  if (argc == 1) {
    ShowHelp(commands);
    return EXIT_SUCCESS;
  }

  const std::string command = argv[1];
  if (command == "help" || command == "--help" || command == "-h") {
    ShowHelp(commands);
    return EXIT_SUCCESS;
  } else if (command == "version" || command == "--version" ||
             command == "-v") {
    ShowVersion();
    return EXIT_SUCCESS;
  } else {
    command_func_t matched_command_func = nullptr;
    for (const auto& command_func : commands) {
      if (command == command_func.first) {
        matched_command_func = command_func.second;
        break;
      }
    }
    if (matched_command_func == nullptr) {
      LOG(ERROR) << colmap::StringPrintf(
          "Command `%s` not recognized. To list the "
          "available commands, run `colmap help`.",
          command.c_str());
      return EXIT_FAILURE;
    } else {
      int command_argc = argc - 1;
      char** command_argv = &argv[1];
      command_argv[0] = argv[0];
      const int command_exit_code =
          matched_command_func(command_argc, command_argv);
      // Uniform process contract: if the command was interrupted by a signal
      // (and reported success on its partial work), surface the signal's exit
      // code (130 for SIGINT, 143 for SIGTERM) so orchestrators don't read an
      // interrupted run as a clean success. A command that already reported its
      // own non-zero status is left untouched. See util/signal_handler.h.
      if (command_exit_code == EXIT_SUCCESS &&
          colmap::IsInterruptRequested()) {
        return colmap::GetInterruptExitCode();
      }
      return command_exit_code;
    }
  }

  ShowHelp(commands);
  return EXIT_SUCCESS;
}
