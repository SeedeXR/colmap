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

// Non-Metal stub: every entry point reports the Metal BA backend as
// unavailable (returns false). Built in place of bundle_adjustment_metal.mm
// only when the BA_METAL_POC_ENABLED build option is ON but Metal is
// unavailable (non-Apple, or METAL_ENABLED OFF); it exists so the PoC's parity
// test still links and skips. With BA_METAL_POC_ENABLED OFF (the default) the
// entire PoC -- including this stub -- is omitted from the build. See
// BA_METAL_POC_ENABLED in the root CMakeLists.

#include "colmap/estimators/bundle_adjustment_metal.h"

namespace colmap {

bool IsBundleAdjustmentMetalAvailable() { return false; }

bool ComputeReprojErrorMetalPinhole(int /*num_obs*/,
                                    const float* /*poses*/,
                                    const float* /*points*/,
                                    const float* /*cams*/,
                                    const float* /*obs*/,
                                    float* /*residuals*/,
                                    float* /*jac_point*/,
                                    float* /*jac_pose*/,
                                    float* /*jac_cam*/) {
  return false;
}

bool RefinePointsMetalPinhole(int /*num_points*/,
                              int /*num_obs*/,
                              float* /*point3D*/,
                              const int* /*obs_offset*/,
                              const float* /*obs_pose*/,
                              const float* /*obs_cam*/,
                              const float* /*obs_pixel*/,
                              int /*max_iters*/) {
  return false;
}

bool ComputePoseTangentJacMetalPinhole(int /*num_obs*/,
                                       const float* /*poses*/,
                                       const float* /*points*/,
                                       const float* /*cams*/,
                                       const float* /*obs*/,
                                       float* /*residuals*/,
                                       float* /*jac_cam_tangent*/,
                                       float* /*jac_point*/) {
  return false;
}

}  // namespace colmap
