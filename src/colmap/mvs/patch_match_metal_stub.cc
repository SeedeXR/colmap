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

// Stub used when COLMAP_METAL_ENABLED is off (non-Apple builds, or Metal
// disabled). Reports the Metal MVS backend as unavailable so callers use the
// CPU/CUDA path. The real implementation lives in patch_match_metal.mm.

#include "colmap/mvs/patch_match_metal.h"

namespace colmap {
namespace mvs {

bool IsPatchMatchMetalAvailable() { return false; }

bool ComposePlaneHomographiesMetal(const float /*ref_inv_K*/[4],
                                   const float /*src_K*/[4],
                                   const float /*src_R*/[9],
                                   const float /*src_T*/[3],
                                   const int* /*rows*/,
                                   const int* /*cols*/,
                                   const float* /*depths*/,
                                   const float* /*normals*/,
                                   int /*num*/,
                                   float* /*out_H*/) {
  return false;
}

bool ComputeNCCPhotometricCostMetal(const float* /*ref_image*/,
                                    const float* /*src_image*/,
                                    int /*width*/,
                                    int /*height*/,
                                    int /*window_radius*/,
                                    int /*window_step*/,
                                    float /*sigma_spatial*/,
                                    float /*sigma_color*/,
                                    const int* /*rows*/,
                                    const int* /*cols*/,
                                    const float* /*homographies*/,
                                    int /*num*/,
                                    float* /*out_cost*/) {
  return false;
}

bool PlaneSweepDepthMetal(const float* /*ref_image*/,
                          const float* /*src_image*/,
                          int /*width*/,
                          int /*height*/,
                          const float /*ref_inv_K*/[4],
                          const float /*src_K*/[4],
                          const float /*src_R*/[9],
                          const float /*src_T*/[3],
                          const float /*normal*/[3],
                          const float* /*depth_candidates*/,
                          int /*num_depths*/,
                          int /*window_radius*/,
                          int /*window_step*/,
                          float /*sigma_spatial*/,
                          float /*sigma_color*/,
                          const int* /*rows*/,
                          const int* /*cols*/,
                          int /*num_pixels*/,
                          float* /*out_depth*/,
                          float* /*out_cost*/) {
  return false;
}

bool EvaluateHypothesisCostMetal(const float* /*ref_image*/,
                                 const float* /*src_image*/,
                                 int /*width*/,
                                 int /*height*/,
                                 const float /*ref_inv_K*/[4],
                                 const float /*src_K*/[4],
                                 const float /*src_R*/[9],
                                 const float /*src_T*/[3],
                                 int /*window_radius*/,
                                 int /*window_step*/,
                                 float /*sigma_spatial*/,
                                 float /*sigma_color*/,
                                 const int* /*rows*/,
                                 const int* /*cols*/,
                                 const float* /*depths*/,
                                 const float* /*normals*/,
                                 int /*num*/,
                                 float* /*out_cost*/) {
  return false;
}

bool EvaluateHypothesisCostMultiViewMetal(const float* /*ref_image*/,
                                          const float* /*src_images*/,
                                          int /*num_src*/,
                                          int /*width*/,
                                          int /*height*/,
                                          const float /*ref_inv_K*/[4],
                                          const float* /*src_K*/,
                                          const float* /*src_R*/,
                                          const float* /*src_T*/,
                                          int /*best_k*/,
                                          int /*window_radius*/,
                                          int /*window_step*/,
                                          float /*sigma_spatial*/,
                                          float /*sigma_color*/,
                                          const int* /*rows*/,
                                          const int* /*cols*/,
                                          const float* /*depths*/,
                                          const float* /*normals*/,
                                          int /*num*/,
                                          float* /*out_cost*/) {
  return false;
}

bool ComputeGeomConsistencyCostMetal(int /*width*/,
                                     int /*height*/,
                                     const float /*ref_inv_K*/[4],
                                     const float /*ref_K*/[4],
                                     const float* /*src_P*/,
                                     const float* /*src_inv_P*/,
                                     const float* /*src_depth_maps*/,
                                     int /*num_src*/,
                                     float /*max_cost*/,
                                     const int* /*rows*/,
                                     const int* /*cols*/,
                                     const float* /*depths*/,
                                     int /*num*/,
                                     float* /*out_cost*/) {
  return false;
}

bool RunPatchMatchOptimizerMetal(const float* /*ref_image*/,
                                 const float* /*src_image*/,
                                 int /*width*/,
                                 int /*height*/,
                                 const float /*ref_inv_K*/[4],
                                 const float /*src_K*/[4],
                                 const float /*src_R*/[9],
                                 const float /*src_T*/[3],
                                 int /*window_radius*/,
                                 int /*window_step*/,
                                 float /*sigma_spatial*/,
                                 float /*sigma_color*/,
                                 float /*depth_min*/,
                                 float /*depth_max*/,
                                 int /*num_iterations*/,
                                 float* /*io_depth*/,
                                 float* /*io_normal*/) {
  return false;
}

bool RunPatchMatchMultiViewOptimizerMetal(const float* /*ref_image*/,
                                          const float* /*src_images*/,
                                          int /*num_src*/,
                                          int /*width*/,
                                          int /*height*/,
                                          const float /*ref_inv_K*/[4],
                                          const float* /*src_K*/,
                                          const float* /*src_R*/,
                                          const float* /*src_T*/,
                                          int /*best_k*/,
                                          int /*window_radius*/,
                                          int /*window_step*/,
                                          float /*sigma_spatial*/,
                                          float /*sigma_color*/,
                                          float /*depth_min*/,
                                          float /*depth_max*/,
                                          int /*num_iterations*/,
                                          float* /*io_depth*/,
                                          float* /*io_normal*/) {
  return false;
}

bool EstimateDepthMapMetalFromArrays(int /*width*/,
                                     int /*height*/,
                                     const float* /*ref_gray*/,
                                     const float /*ref_K9*/[9],
                                     const float /*ref_R9*/[9],
                                     const float /*ref_T3*/[3],
                                     int /*num_src*/,
                                     const float* /*src_gray*/,
                                     const float* /*src_K9*/,
                                     const float* /*src_R9*/,
                                     const float* /*src_T3*/,
                                     int /*best_k*/,
                                     int /*window_radius*/,
                                     int /*window_step*/,
                                     float /*sigma_spatial*/,
                                     float /*sigma_color*/,
                                     float /*depth_min*/,
                                     float /*depth_max*/,
                                     int /*num_iterations*/,
                                     float /*filter_min_ncc*/,
                                     float* /*out_depth*/,
                                     float* /*out_normal*/) {
  return false;
}

}  // namespace mvs
}  // namespace colmap
