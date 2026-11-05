// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/storage_arguments.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace {

class PipelineStorageArgumentsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("arguments"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 0));
  }
  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }
  loom_value_id_t Value(loom_type_t type) {
    loom_value_id_t value;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &value));
    return value;
  }
  void Buffer(loom_value_id_t value, loom_value_id_t root, uint64_t alignment) {
    loom_value_fact_buffer_reference_t reference = {
        .maximum_byte_extent = loom_value_facts_unknown(),
        .minimum_alignment = alignment,
        .root_value_id = root,
    };
    loom_value_facts_t facts;
    IREE_ASSERT_OK(loom_value_facts_make_buffer_reference(&facts_.context,
                                                          reference, &facts));
    IREE_ASSERT_OK(loom_value_fact_table_define(&facts_, value, facts));
  }
  void View(loom_value_id_t value, loom_value_id_t root,
            loom_value_facts_t offset, loom_value_facts_t length) {
    loom_value_fact_view_reference_t reference = {
        .base_byte_offset = offset,
        .footprint_byte_length = length,
        .minimum_alignment = 16,
        .root_minimum_alignment = 64,
        .root_value_id = root,
        .buffer_value_id = root,
    };
    loom_value_facts_t facts;
    IREE_ASSERT_OK(loom_value_facts_make_view_reference(&facts_.context,
                                                        reference, &facts));
    IREE_ASSERT_OK(loom_value_fact_table_define(&facts_, value, facts));
  }
  loom_pipeline_storage_arguments_t Build(
      const std::vector<loom_value_id_t>& values) {
    loom_pipeline_storage_arguments_t arguments;
    IREE_CHECK_OK(loom_pipeline_storage_arguments_build(
        module_, values.data(), values.size(), &facts_, &arena_, &arguments));
    return arguments;
  }
  // Fact extensions and retained argument projections share this arena.
  iree_arena_block_pool_t blocks_;
  // Storage for all analysis state in one test invocation.
  iree_arena_allocator_t arena_;
  // Core type and IR ownership; no parser or compiler stack is needed.
  loom_context_t context_;
  // Valid argument identities and types supplied to the projection API.
  loom_module_t* module_ = nullptr;
  // Inferred facts at the caller-to-callee boundary being projected.
  loom_value_fact_table_t facts_ = {};
};

TEST_F(PipelineStorageArgumentsTest,
       ParentRootsAndAliasedFormalsKeepSourcePositions) {
  const auto root = Value(loom_type_buffer());
  const auto first = Value(loom_type_buffer());
  const auto second = Value(loom_type_buffer());
  const auto unrelated = Value(loom_type_buffer());
  const auto count = Value(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  Buffer(first, root, 64);
  Buffer(second, root, 64);
  Buffer(unrelated, LOOM_VALUE_ID_INVALID, 4);
  const auto arguments = Build({count, second, unrelated, first});
  ASSERT_EQ(arguments.storage_count, 3u);
  EXPECT_EQ(arguments.storage[0].root, root);
  EXPECT_EQ(arguments.storage[0].argument_index, 1u);
  EXPECT_EQ(arguments.storage[2].root, root);
  EXPECT_EQ(arguments.storage[2].argument_index, 3u);
  EXPECT_EQ(loom_pipeline_storage_arguments_find_buffer(&arguments, root),
            &arguments.storage[0]);
  EXPECT_EQ(loom_pipeline_storage_arguments_find_buffer(&arguments, unrelated),
            &arguments.storage[1]);
  EXPECT_EQ(loom_pipeline_storage_arguments_find_buffer(&arguments, first),
            nullptr);
}

TEST_F(PipelineStorageArgumentsTest, ViewSubspansNeverGrantWholeRootAccess) {
  const auto root = Value(loom_type_buffer());
  const auto left = Value(loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_BF16, loom_dim_pack_static(64), 0));
  const auto right = Value(loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I1, loom_dim_pack_static(512), 0));
  View(left, root, loom_value_facts_exact_i64(256),
       loom_value_facts_exact_i64(128));
  View(right, root, loom_value_facts_exact_i64(1024),
       loom_value_facts_exact_i64(64));
  const auto arguments = Build({right, left});
  ASSERT_EQ(arguments.storage_count, 2u);
  EXPECT_FALSE(arguments.storage[0].buffer);
  EXPECT_FALSE(arguments.storage[1].buffer);
  EXPECT_EQ(arguments.storage[0].root, root);
  EXPECT_EQ(arguments.storage[0].byte_offset.range_lo, 1024);
  EXPECT_EQ(arguments.storage[0].byte_length.range_hi, 64);
  EXPECT_EQ(arguments.storage[1].byte_offset.range_lo, 256);
  EXPECT_EQ(arguments.storage[1].byte_length.range_hi, 128);
  EXPECT_EQ(arguments.storage[0].byte_alignment, 16u);
  EXPECT_EQ(loom_pipeline_storage_arguments_find_buffer(&arguments, root),
            nullptr);
}

TEST_F(PipelineStorageArgumentsTest, UnresolvedSpansRetainBounds) {
  const auto root = Value(loom_type_buffer());
  const auto extent = Value(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  const auto view = Value(loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(extent), 0));
  View(view, root, loom_value_facts_make(16, 128, 16),
       loom_value_facts_make(64, 512, 64));
  const auto arguments = Build({view});
  const auto& storage = arguments.storage[0];
  EXPECT_EQ(storage.byte_offset.range_lo, 16);
  EXPECT_EQ(storage.byte_offset.range_hi, 128);
  EXPECT_EQ(storage.byte_length.range_lo, 64);
  EXPECT_EQ(storage.byte_length.range_hi, 512);
}

TEST_F(PipelineStorageArgumentsTest, UnqualifiedStorageUsesItsOwnRoot) {
  const auto buffer = Value(loom_type_buffer());
  const auto view = Value(loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0));
  const auto arguments = Build({buffer, view});
  ASSERT_EQ(arguments.storage_count, 2u);
  EXPECT_EQ(arguments.storage[0].root, buffer);
  EXPECT_EQ(arguments.storage[1].root, view);
  EXPECT_EQ(arguments.storage[0].byte_alignment, 1u);
  int64_t length;
  EXPECT_FALSE(
      loom_value_facts_as_exact_i64(arguments.storage[0].byte_length, &length));
  EXPECT_EQ(loom_pipeline_storage_arguments_find_buffer(&arguments, buffer),
            &arguments.storage[0]);
}

TEST_F(PipelineStorageArgumentsTest, NonStorageArgumentsNeedNoBindingState) {
  const auto scalar = Value(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  const auto vector = Value(loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I64, loom_dim_pack_static(4), 0));
  const auto arguments = Build({scalar, vector});
  EXPECT_EQ(arguments.storage_count, 0u);
  EXPECT_EQ(arguments.storage, nullptr);
  EXPECT_EQ(arguments.buffers, nullptr);
  EXPECT_EQ(arguments.buffer_count, 0u);
  EXPECT_EQ(loom_pipeline_storage_arguments_find_buffer(&arguments, scalar),
            nullptr);
}

}  // namespace
