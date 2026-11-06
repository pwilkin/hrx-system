// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/pipeline/resources.h"
#include "loom/pass/types.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/transforms/pipeline/memory_channel.h"
#include "loom/util/fact_table.h"

namespace {

class Aie2pChannelsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    pass_.arena = &arena_;
    pass_.diagnostic_emitter = diagnostics_.emitter();
    context_.pass = &pass_;
    context_.family = loom_xdna_npu2_array_family();
    tile_.coordinate = {0, 2};
    tile_.facts = loom_xdna_array_tile_facts(context_.family, tile_.coordinate);
    tile_.next_lock = 3;
    context_.pool_tiles = &tile_pointer_;
    source_.storage.base_byte_offset = loom_value_facts_make(32, 32, 32);
    source_.storage.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
    source_.capacity = 3;
    source_.geometry = &geometry_;
    occurrence_.source = &source_;
    occurrence_.allocation = &placement_;
    realization_.invocations.channels = &occurrence_;
    realization_.invocations.channel_count = 1;
    const auto view = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                          loom_dim_pack_static(16), 0);
    ASSERT_TRUE(loom_memory_channel_protocol_initialize(&source_, view, 0, 64,
                                                        4, 4, &protocol_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Select(bool expected_valid) {
    bool valid = false;
    IREE_ASSERT_OK(
        loom_aie2p_native_select_channels(&context_, &realization_, &valid));
    ASSERT_EQ(valid, expected_valid);
    EXPECT_EQ(diagnostics_.emissions.size(), expected_valid ? 0u : 1u);
  }

  // Backing arena for physical channel selection.
  iree_arena_block_pool_t pool_ = {};
  // Per-selection allocation lifetime.
  iree_arena_allocator_t arena_ = {};
  // Captures target admission failures without parsing source IR.
  loom::testing::DiagnosticEmissionCapture diagnostics_;
  // Allocation and diagnostic services consumed by admission.
  loom_pass_t pass_ = {};
  // Native inventory and selected channels.
  loom_aie2p_native_context_t context_ = {};
  // One admitted local memory owner with three previously assigned locks.
  loom_aie2p_native_tile_t tile_ = {};
  // Local backing-pool correspondence.
  loom_aie2p_native_tile_t* tile_pointer_ = &tile_;
  // Three 64-byte records separated by 32-byte padding.
  loom_pipeline_channel_geometry_t geometry_ = {
      .slot_bit_stride = 96 * 8,
      .record_element_count = 16,
      .record_bit_span = 64 * 8,
      .element_bit_count = 32,
      .record_dense = true,
  };
  // Retained source geometry and storage facts.
  loom_pipeline_resource_channel_t source_ = {};
  // Source allocation whose lifetime is independent of channel selection.
  loom_pipeline_resource_allocation_t allocation_ = {
      .pool_index = 0,
      .byte_length = 320,
      .byte_alignment = 32,
  };
  // Common construction placement supplied before target channel selection.
  loom_pipeline_storage_allocation_t placement_ = {&allocation_, 512};
  // One imported or locally constructed channel occurrence.
  loom_pipeline_invocation_channel_t occurrence_ = {};
  // Complete selected input for this channel admission.
  loom_pipeline_realization_t realization_ = {};
  // Shared slot geometry and generation-counter policy.
  loom_memory_channel_protocol_t protocol_ = {};
  // Physical entry arguments for shared payload and notifications.
  loom_memory_channel_binding_t binding_ = {&protocol_, 0, 1};
  // Optional parent selection, indexed like the channel inventory.
  const loom_memory_channel_binding_t* binding_pointer_ = &binding_;
};

TEST_F(Aie2pChannelsTest, ExternalProtocolDoesNotConsumeLocalStorageOrLocks) {
  realization_.memory_bindings = &binding_pointer_;
  // External means outside this admitted native inventory. The parent can own
  // global storage or another device's local storage, or borrow caller storage.
  for (const auto space : {LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
                           LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP}) {
    for (const bool parent_allocated : {false, true}) {
      SCOPED_TRACE(::testing::Message() << space << ":" << parent_allocated);
      source_.storage.memory_space = space;
      occurrence_.allocation = parent_allocated ? &placement_ : nullptr;
      Select(true);
      const auto& channel = context_.channels[0];
      EXPECT_EQ(channel.external, &binding_);
      EXPECT_EQ(channel.pool_index, UINT32_MAX);
      EXPECT_EQ(channel.byte_stride, 96u);
      EXPECT_EQ(channel.record_byte_length, 64u);
      EXPECT_EQ(tile_.next_lock, 3u);
    }
  }
}

TEST_F(Aie2pChannelsTest, LocalStorageSelectsOffsetsAndOneSemaphorePair) {
  Select(true);
  const auto& channel = context_.channels[0];
  EXPECT_EQ(channel.external, nullptr);
  EXPECT_EQ(channel.pool_index, 0u);
  EXPECT_EQ(channel.byte_offset, 544u);
  EXPECT_EQ(channel.byte_stride, 96u);
  EXPECT_EQ(channel.record_byte_length, 64u);
  EXPECT_EQ(channel.free_lock, 3u);
  EXPECT_EQ(channel.ready_lock, 4u);
  EXPECT_EQ(tile_.next_lock, 5u);
}

TEST_F(Aie2pChannelsTest, GlobalAllocationRequiresAnExternalProtocol) {
  source_.storage.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
  Select(false);
  EXPECT_EQ(tile_.next_lock, 3u);
}

TEST_F(Aie2pChannelsTest, BorrowedStorageRequiresAnExternalProtocol) {
  occurrence_.allocation = nullptr;
  Select(false);
  EXPECT_EQ(tile_.next_lock, 3u);
}

}  // namespace
