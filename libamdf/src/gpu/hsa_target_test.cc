// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/gpu_hsa.h"
#include "gtest/gtest.h"

namespace {

TEST(GpuHsaTargetTest, HostPublicationRequiresTheNativeHdpProtocol) {
  EXPECT_FALSE(amdf_gpu_hsa_query_target(9, 0, 7).host_data_path_flush);
  EXPECT_FALSE(amdf_gpu_hsa_query_target(9, 0, 8).host_data_path_flush);
  EXPECT_FALSE(amdf_gpu_hsa_query_target(9, 0, 10).host_data_path_flush);
  EXPECT_TRUE(amdf_gpu_hsa_query_target(9, 4, 0).host_data_path_flush);
  EXPECT_TRUE(amdf_gpu_hsa_query_target(9, 4, 2).host_data_path_flush);
  EXPECT_FALSE(amdf_gpu_hsa_query_target(9, 4, 3).host_data_path_flush);
  EXPECT_TRUE(amdf_gpu_hsa_query_target(9, 5, 0).host_data_path_flush);
  EXPECT_FALSE(amdf_gpu_hsa_query_target(10, 0, 0).host_data_path_flush);
  EXPECT_FALSE(amdf_gpu_hsa_query_target(10, 1, 0).host_data_path_flush);
  EXPECT_FALSE(amdf_gpu_hsa_query_target(10, 3, 0).host_data_path_flush);
  EXPECT_FALSE(amdf_gpu_hsa_query_target(11, 0, 0).host_data_path_flush);
  EXPECT_TRUE(amdf_gpu_hsa_query_target(12, 0, 0).host_data_path_flush);
  EXPECT_TRUE(amdf_gpu_hsa_query_target(12, 0, 1).host_data_path_flush);
  EXPECT_FALSE(amdf_gpu_hsa_query_target(12, 1, 0).host_data_path_flush);
  EXPECT_TRUE(amdf_gpu_hsa_query_target(12, 5, 0).host_data_path_flush);
}

TEST(GpuHsaTargetTest, AgentUnitsRemainDistinctFromPhysicalCuFacts) {
  const auto target = amdf_gpu_hsa_query_target(12, 5, 0);
  EXPECT_EQ(target.shader_resource_model, 1u);
  EXPECT_EQ(target.compute_units_per_domain[0], 2u);
  EXPECT_EQ(target.local_memory_byte_length[0], 320u * 1024);
  EXPECT_EQ(target.barrier_count[0], 16u);
  EXPECT_EQ(target.resource3_register, 0x2e23u);
  EXPECT_EQ(target.local_memory_granule, 2048u);
  EXPECT_EQ(target.user_data_count_mask, 63u);
  EXPECT_EQ(target.shader.workgroup_processor_mode, 0u);
  const auto native = amdf_gpu_target_query_capabilities(12, 5, 0);
  EXPECT_EQ(native.shader.simds_per_compute_unit, 4u);
  EXPECT_EQ(amdf_gpu_hsa_query_target(9, 5, 0).local_memory_granule, 1280u);
  EXPECT_EQ(amdf_gpu_hsa_query_target(13, 0, 0).shader_resource_model, 0u);
  EXPECT_EQ(amdf_gpu_hsa_query_target(12, 1, 0).compute_units_per_resource, 0u);
}

TEST(GpuHsaTargetTest, CooperativeProgressAndProfilingAreIndependentFacts) {
  const auto target = amdf_gpu_hsa_query_target(9, 0, 10);
  EXPECT_EQ(target.cooperative_compute_unit_granule, 8u);
  EXPECT_EQ(target.cooperative_reserved_compute_units, 8u);
  EXPECT_EQ(target.counter_encoding, AMDF_GPU_HSA_COUNTER_ENCODING_GFX90A);
  EXPECT_EQ(target.clock_copy, AMDF_GPU_HSA_CLOCK_COPY_MEMORY_STREAM);
  EXPECT_EQ(target.packet_features, 0x581bu);
  EXPECT_EQ(amdf_gpu_hsa_query_target(11, 0, 0).packet_features, 0x3bffdu);
  EXPECT_EQ(amdf_gpu_hsa_query_target(12, 0, 1).packet_features, 0x7bffdu);
  EXPECT_EQ(amdf_gpu_hsa_query_target(12, 0, 1).counter_encoding,
            AMDF_GPU_HSA_COUNTER_ENCODING_NONE);
}

}  // namespace
