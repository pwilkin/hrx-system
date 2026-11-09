// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/gpu_target.h"
#include "benchmark/benchmark.h"
#include "iree/hal/driver/amd/gpu/dispatch.h"

namespace {

// Immutable image/native facts are warm, as in repeated libhrx launch-size
// queries. This measures residency arithmetic, not executable lookup, public
// queue validation, submission, or device execution.
static void BM_QueryConcurrency(benchmark::State& state) {
  const iree_hal_amd_gpu_gfxip_version_t targets[] = {
      {9, 4, 2},
      {11, 0, 0},
      {12, 0, 0},
  };
  const auto version = targets[state.range(0)];
  const auto capabilities = amdf_gpu_target_query_capabilities(
      version.major, version.minor, version.stepping);
  const bool wave32 = capabilities.shader.wave_size_mask & 32;
  amdf_gpu_endpoint_info_t target = {
      .gfx_ip = {version.major, version.minor, version.stepping},
      .compute =
          {
              .wavefront_size = wave32 ? 32u : 64u,
              .compute_unit_count = 96,
              .maximum_wave_count_per_compute_unit = 32,
              .local_data_share_byte_length = 65536,
          },
  };
  iree_hal_amd_gpu_image_kernel_t kernel = {
      .compute_pgm_rsrc1 = 19u | (7u << 6),
      .code_properties = static_cast<uint16_t>(wave32 ? 1u << 10 : 0),
  };
  iree_hal_amd_gpu_metadata_kernel_t metadata = {
      .group_segment_fixed_size = 1024,
      .max_flat_workgroup_size = 1024,
  };
  iree_hal_queue_dispatch_concurrency_params_t params = {
      .workgroup_size = {256, 1, 1},
      .dynamic_workgroup_local_memory = static_cast<uint32_t>(state.range(1)),
  };
  benchmark::DoNotOptimize(target);
  benchmark::DoNotOptimize(kernel);
  benchmark::DoNotOptimize(metadata);
  benchmark::DoNotOptimize(params);
  for (auto _ : state) {
    iree_hal_queue_dispatch_concurrency_t concurrency;
    IREE_CHECK_OK(iree_hal_amd_gpu_dispatch_query_concurrency(
        &target, target.compute.compute_unit_count, &kernel, &metadata, 16,
        params, &concurrency));
    benchmark::DoNotOptimize(concurrency);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_QueryConcurrency)
    ->ArgNames({"target_case", "dynamic_lds"})
    ->Args({0, 0})
    ->Args({0, 32768})
    ->Args({1, 0})
    ->Args({1, 32768})
    ->Args({2, 0})
    ->Args({2, 32768});

}  // namespace
