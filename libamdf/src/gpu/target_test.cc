// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>

#include "amdf/gpu_target.h"
#include "gtest/gtest.h"

namespace {

TEST(GpuTargetTest, ShaderAndPublicationFactsCoverEveryEncodingFamily) {
  struct Case {
    // Native target identity.
    std::array<uint32_t, 3> version;
    // Accepted wave sizes in the native bit encoding.
    uint32_t waves;
    // SIMD count per physical compute unit.
    uint32_t simds;
    // Vector registers per SIMD for wave32 and wave64.
    std::array<uint32_t, 2> registers;
    // Per-wave vector allocation granules.
    std::array<uint32_t, 2> allocation;
    // Workgroup-local memory allocation granule in bytes.
    uint32_t lds;
    // Device-library grid barrier implementation.
    amdf_gpu_grid_barrier_t barrier;
    // Native resource-three register, or zero when not described.
    uint32_t resource3;
  };
  const Case cases[] = {
      {{9, 0, 0}, 64, 4, {0, 256}, {0, 4}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 0, 2}, 64, 4, {0, 256}, {0, 4}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 0, 4}, 64, 4, {0, 256}, {0, 4}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 0, 6}, 64, 4, {0, 256}, {0, 4}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 0, 8}, 64, 4, {0, 256}, {0, 4}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 0, 9}, 64, 4, {0, 256}, {0, 4}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 0, 10}, 64, 4, {0, 512}, {0, 8}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 0, 12}, 64, 4, {0, 256}, {0, 4}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 4, 0}, 64, 4, {0, 512}, {0, 8}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 4, 1}, 64, 4, {0, 512}, {0, 8}, 512, AMDF_GPU_GRID_BARRIER_GWS, 0},
      {{9, 4, 2},
       64,
       4,
       {0, 512},
       {0, 8},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0},
      {{9, 5, 0},
       64,
       4,
       {0, 512},
       {0, 8},
       1280,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0},
      {{10, 1, 0},
       96,
       2,
       {1024, 512},
       {8, 4},
       512,
       AMDF_GPU_GRID_BARRIER_GWS,
       0},
      {{10, 1, 3},
       96,
       2,
       {1024, 512},
       {8, 4},
       512,
       AMDF_GPU_GRID_BARRIER_GWS,
       0},
      {{10, 3, 0},
       96,
       2,
       {1024, 512},
       {16, 8},
       512,
       AMDF_GPU_GRID_BARRIER_GWS,
       0},
      {{10, 3, 6},
       96,
       2,
       {1024, 512},
       {16, 8},
       512,
       AMDF_GPU_GRID_BARRIER_GWS,
       0},
      {{11, 0, 0},
       96,
       2,
       {1536, 768},
       {24, 12},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{11, 0, 1},
       96,
       2,
       {1536, 768},
       {24, 12},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{11, 0, 2},
       96,
       2,
       {1024, 512},
       {16, 8},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{11, 0, 3},
       96,
       2,
       {1024, 512},
       {16, 8},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{11, 5, 0},
       96,
       2,
       {1024, 512},
       {16, 8},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{11, 5, 1},
       96,
       2,
       {1536, 768},
       {24, 12},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{11, 5, 3},
       96,
       2,
       {1024, 512},
       {16, 8},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{11, 7, 0},
       96,
       2,
       {1024, 512},
       {16, 8},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{11, 7, 2},
       96,
       2,
       {1024, 512},
       {16, 8},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{12, 0, 0},
       96,
       2,
       {1536, 768},
       {24, 12},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{12, 0, 1},
       96,
       2,
       {1536, 768},
       {24, 12},
       512,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e28},
      {{12, 5, 0},
       32,
       4,
       {1536, 768},
       {24, 12},
       2048,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e23},
      {{12, 5, 1},
       32,
       4,
       {1536, 768},
       {24, 12},
       2048,
       AMDF_GPU_GRID_BARRIER_MEMORY,
       0x2e23},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::PrintToString(test.version));
    const auto target = amdf_gpu_target_query_capabilities(
        test.version[0], test.version[1], test.version[2]);
    EXPECT_EQ(target.shader.wave_size_mask, test.waves);
    EXPECT_EQ(target.shader.simds_per_compute_unit, test.simds);
    EXPECT_EQ(target.shader.vector_register_count[0], test.registers[0]);
    EXPECT_EQ(target.shader.vector_register_count[1], test.registers[1]);
    EXPECT_EQ(target.shader.vector_allocation_granule[0], test.allocation[0]);
    EXPECT_EQ(target.shader.vector_allocation_granule[1], test.allocation[1]);
    EXPECT_EQ(target.shader.local_memory_granule, test.lds);
    EXPECT_EQ(target.grid_barrier, test.barrier);
    EXPECT_EQ(target.shader.global_wave_sync,
              test.version[0] == 9 || test.version[0] == 10);
    EXPECT_EQ(target.pm4.resource3_register, test.resource3);
  }
}

TEST(GpuTargetTest, CacheAndDescriptorFactsAreIndependentCapabilities) {
  const auto cdna = amdf_gpu_target_query_capabilities(9, 4, 2);
  EXPECT_EQ(cdna.pm4.instruction_cache_uses_coherency_control, 1u);
  EXPECT_EQ(cdna.shader.scalar_register_count, 800u);
  EXPECT_EQ(cdna.shader.unmodeled_resource3_mask, 1u << 16);
  EXPECT_EQ(cdna.instruction_prefetch_mask, 0u);
  const auto rdna = amdf_gpu_target_query_capabilities(11, 0, 0);
  EXPECT_EQ(rdna.pm4.shader_cache_control, 0x103e0u);
  EXPECT_EQ(rdna.pm4.release_cache_control, 0x18040u);
  EXPECT_EQ(rdna.pm4.acquire_cache_control, 0x1c3e0u);
  EXPECT_EQ(rdna.pm4.maximum_scratch_wave_byte_length, 0x7fff00u);
  EXPECT_EQ(rdna.pm4.command_processor_bypasses_l2, 0u);
  EXPECT_EQ(rdna.instruction_prefetch_mask, 0x3fu);
  const auto scalar_cache = amdf_gpu_target_query_capabilities(12, 0, 1);
  EXPECT_EQ(scalar_cache.pm4.shader_cache_control, 0x101c0u);
  EXPECT_EQ(scalar_cache.pm4.acquire_cache_control, 0x1c1c0u);
  EXPECT_EQ(scalar_cache.pm4.command_processor_bypasses_l2, 1u);
  EXPECT_EQ(scalar_cache.shader.dynamic_vector_resource2_mask, 1u << 6);
  EXPECT_EQ(scalar_cache.pm4.user_data_count_mask, 31u);
  const auto vector_cache = amdf_gpu_target_query_capabilities(12, 5, 0);
  EXPECT_EQ(vector_cache.pm4.shader_cache_control, 0x101c0u);
  EXPECT_EQ(vector_cache.pm4.release_cache_control, 0x18060u);
  EXPECT_EQ(vector_cache.pm4.acquire_cache_control, 0x1c1e0u);
  EXPECT_EQ(vector_cache.pm4.maximum_scratch_wave_byte_length, 0x3ffff00u);
  EXPECT_EQ(vector_cache.pm4.group_segment_unit_log2, 11u);
  EXPECT_EQ(vector_cache.pm4.user_data_count_mask, 63u);
  EXPECT_EQ(vector_cache.shader.workgroup_processor_mode, 0u);
  EXPECT_EQ(vector_cache.shader.dynamic_vector_resource3_mask, 1u << 17);
  EXPECT_EQ(vector_cache.requires_revision_metadata, 1u);
  EXPECT_EQ(
      amdf_gpu_target_query_capabilities(12, 5, 1).requires_revision_metadata,
      0u);
}

TEST(GpuTargetTest, ProgramSelectionUsesDeclaredInstructionsAndPreference) {
  struct Variant {
    // Native instruction requirements of this immutable compiler artifact.
    amdf_gpu_instruction_set_t instructions;
    // Distinct payload verifies that selection uses the caller's stride.
    uint64_t payload;
  };
  const Variant variants[] = {
      {AMDF_GPU_INSTRUCTION_SET_GFX90A, 1},
      {AMDF_GPU_INSTRUCTION_SET_GFX9, 2},
      {AMDF_GPU_INSTRUCTION_SET_GFX94, 3},
      {AMDF_GPU_INSTRUCTION_SET_GFX115, 4},
      {AMDF_GPU_INSTRUCTION_SET_GFX117, 5},
      {AMDF_GPU_INSTRUCTION_SET_GFX11, 6},
      {AMDF_GPU_INSTRUCTION_SET_GFX12, 7},
      {AMDF_GPU_INSTRUCTION_SET_GFX125, 8},
  };
  const uint32_t cases[][4] = {
      {9, 0, 8, 1},           {9, 0, 10, 0},          {9, 4, 2, 2},
      {9, 5, 0, 2},           {11, 0, 0, 5},          {11, 5, 1, 3},
      {11, 7, 0, 4},          {12, 0, 1, 6},          {12, 5, 0, 7},
      {10, 3, 0, UINT32_MAX}, {13, 0, 0, UINT32_MAX},
  };
  for (const auto& test : cases) {
    const auto target =
        amdf_gpu_target_query_capabilities(test[0], test[1], test[2]);
    EXPECT_EQ(amdf_gpu_target_select_instructions(target.instruction_sets,
                                                  std::size(variants), variants,
                                                  sizeof(Variant)),
              test[3]);
  }
  EXPECT_EQ(amdf_gpu_target_select_instructions(UINT32_MAX, 0, nullptr, 0),
            UINT32_MAX);
}

TEST(GpuTargetTest, UnknownArchitecturesHaveNoEncodingCapabilities) {
  const amdf_gpu_target_capabilities_t empty = {};
  for (uint32_t major : {0u, 8u, 13u, UINT32_MAX}) {
    const auto target = amdf_gpu_target_query_capabilities(major, 0, 0);
    EXPECT_EQ(std::memcmp(&target, &empty, sizeof(target)), 0);
  }
  EXPECT_TRUE(amdf_gpu_target_equal(9, 4, 2, 9, 4, 2));
  EXPECT_FALSE(amdf_gpu_target_equal(9, 4, 2, 9, 4, 1));
}

}  // namespace
