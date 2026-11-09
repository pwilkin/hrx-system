// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/scratch.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ScratchTest, NativeGeometryAndExactPacketLayouts) {
  const struct {
    // Target identity sent to native capability discovery.
    amdf_gpu_endpoint_info_t target;
    // Largest represented private allocation per wave, in bytes.
    uint32_t maximum_wave_byte_length;
    // Expected register offset in the PM4 command stream.
    uint32_t size_register;
  } cases[] = {
      {{.gfx_ip = {11, 0, 0}}, 0x7fff00, 0x218},
      {{.gfx_ip = {11, 5, 0}}, 0x7fff00, 0x218},
      {{.gfx_ip = {11, 7, 0}}, 0x7fff00, 0x218},
      {{.gfx_ip = {12, 0, 0}}, 0x3ffff00, 0x218},
      {{.gfx_ip = {12, 5, 0}}, 0x3ffff00, 0x216},
  };
  for (const auto& test : cases) {
    amdf_gpu_endpoint_info_t target = test.target;
    target.topology.xcc_count = 1;
    target.topology.shader_engine_count_per_xcc = 3;
    target.compute.compute_unit_count = 17;
    target.compute.maximum_scratch_wave_count_per_compute_unit = 32;
    iree_hal_amd_gpu_pm4_scratch_t scratch = {};
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_scratch_profile_initialize(
        &target, 0, &scratch.profile));
    EXPECT_EQ(scratch.profile.maximum_wave_byte_length, 0u);
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_scratch_profile_initialize(
        &target, AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH,
        &scratch.profile));
    EXPECT_EQ(scratch.profile.waves_per_engine, 192u);
    EXPECT_EQ(scratch.profile.engine_count, 3u);
    EXPECT_EQ(scratch.profile.maximum_wave_byte_length,
              test.maximum_wave_byte_length);
    iree_hal_amd_gpu_pm4_scratch_storage_t storage = {
        .address = UINT64_C(0x123456789a00),
        .wave_byte_length = 2048,
    };
    scratch.current = &storage;
    scratch.binding_word_count = 9;
    std::array<uint32_t, 11> words;
    words.fill(0x759bf13d);
    EXPECT_EQ(iree_hal_amd_gpu_pm4_scratch_emit(words.data() + 1, &scratch, 9),
              words.data() + 10);
    const std::array<uint32_t, 9> expected = {
        0xc0004600, 0x80000407, 0xc0027602,         0x210, 0x3456789a,
        0x12,       0xc0017602, test.size_register, 0x80c0};
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(words[i + 1], expected[i]) << i;
    }
    EXPECT_EQ(words.front(), 0x759bf13du);
    EXPECT_EQ(words.back(), 0x759bf13du);
    EXPECT_EQ(iree_hal_amd_gpu_pm4_scratch_word_count(&scratch, 0, false), 0u);
    EXPECT_EQ(iree_hal_amd_gpu_pm4_scratch_word_count(&scratch, 2048, false),
              9u);
    EXPECT_EQ(iree_hal_amd_gpu_pm4_scratch_publish(&scratch, 7, 2048), nullptr);
    EXPECT_EQ(storage.last_use, 7u);
    EXPECT_EQ(iree_hal_amd_gpu_pm4_scratch_word_count(&scratch, 2048, false),
              0u);
    EXPECT_EQ(iree_hal_amd_gpu_pm4_scratch_word_count(&scratch, 2048, true),
              7u);
  }
}

TEST(ScratchTest, UnknownRegisterEncodingCannotPublishScratch) {
  amdf_gpu_endpoint_info_t target = {.gfx_ip = {13, 0, 0}};
  target.topology.xcc_count = 1;
  target.topology.shader_engine_count_per_xcc = 1;
  target.compute.compute_unit_count = 1;
  target.compute.maximum_scratch_wave_count_per_compute_unit = 32;
  iree_hal_amd_gpu_pm4_scratch_profile_t profile = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amd_gpu_pm4_scratch_profile_initialize(
          &target, AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH, &profile));
  EXPECT_EQ(profile.size_register, 0u);
  EXPECT_EQ(profile.maximum_wave_byte_length, 0u);
}

TEST(ScratchTest, AcceptedFifoCompletionOwnsReplacedGenerations) {
  iree_hal_amd_gpu_pm4_scratch_storage_t previous = {.last_use = 3};
  iree_hal_amd_gpu_pm4_scratch_storage_t current = {};
  iree_hal_amd_gpu_pm4_scratch_t scratch = {.current = &previous};
  iree_hal_amd_gpu_context_t context = {};
  // Native backing is deliberately absent: installation of a still-live
  // generation must not issue a release call through the dependency.
  iree_hal_amd_gpu_pm4_scratch_install(&context, &current, 2, &scratch);
  EXPECT_EQ(scratch.current, &current);
  EXPECT_EQ(scratch.retired, &previous);
  EXPECT_EQ(scratch.binding_word_count, 9u);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_scratch_publish(&scratch, 4, 2048), &previous);
  EXPECT_EQ(scratch.retired, nullptr);
  EXPECT_EQ(current.last_use, 4u);
}

}  // namespace
