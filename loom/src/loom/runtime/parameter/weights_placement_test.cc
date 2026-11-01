// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/weights_test_base.h"

namespace loom_runtime::testing {

class WeightPlacementTest : public WeightsTest {
 protected:
  void SetUpPool() override {
    // Each block has 17 tensors with canonical 256-byte alignment and occupies
    // two 4 KiB slabs. The repeated first block shares those same slabs.
    AllocatePool(4 * SlabSize());
  }
};

TEST_P(WeightPlacementTest, SharedRootsSurviveAdditionalUniqueParameterBlocks) {
  // Two separately placed blocks grow the retained span storage. The last
  // consumer aliases the first block after that growth and reads its own data.
  constexpr int32_t kTensorCount = 34;
  constexpr size_t kTensorBytes = 8 * sizeof(int32_t);
  std::string header = "{";
  for (int32_t i = 0; i < kTensorCount; ++i) {
    if (i) {
      header += ",";
    }
    header += "\"tensor." + std::to_string(i) +
              "\":{\"dtype\":\"I32\",\"shape\":[8],\"data_offsets\":[" +
              std::to_string(i * kTensorBytes) + "," +
              std::to_string((i + 1) * kTensorBytes) + "]}";
  }
  header += "}";
  std::ofstream file(path_.path(), std::ios::binary);
  for (unsigned i = 0; i < 8; ++i) {
    file.put(static_cast<char>(uint64_t{header.size()} >> (8 * i)));
  }
  file.write(header.data(), header.size());
  for (int32_t i = 0; i < kTensorCount; ++i) {
    for (int32_t element = 0; element < 8; ++element) {
      const int32_t value = i * 10 + element;
      file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  file.close();
  ASSERT_TRUE(file.good());

  IREE_ASSERT_OK(Compile(0, "first_parameter_block"));
  IREE_ASSERT_OK(Compile(1, "second_parameter_block"));
  IREE_ASSERT_OK(Compile(2, "first_parameter_block"));
  IREE_ASSERT_OK(Load(0, 3));
  EXPECT_EQ(loom_runtime_weights_parameter_count(plans_[0]), kTensorCount);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).read_bytes,
            kTensorCount * kTensorBytes);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).prepared_bytes, 0u);
  const int32_t expected[] = {160, 330, 160};
  for (size_t stage = 0; stage < IREE_ARRAYSIZE(expected); ++stage) {
    ASSERT_NO_FATAL_FAILURE(ReadOutputOn(execution_, stage));
    for (int32_t element = 0; element < 8; ++element) {
      EXPECT_EQ(output_[element], expected[stage] + element);
    }
  }
}

TEST_P(WeightPlacementTest,
       PartiallySharedRootRejectsBeforeBackingOrPreparation) {
  IREE_ASSERT_OK(Compile(0, "first_parameter_block"));
  IREE_ASSERT_OK(Compile(1, "overlapping_parameter_block"));
  const loom_runtime_weight_root_t roots[] = {Root(0, 0), Root(1, 0)};
  // The second root mixes a previously placed tensor with new tensors. It
  // cannot alias the prior storage and must not silently duplicate that key.
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateRoots(0, IREE_ARRAYSIZE(roots), roots, {path_.path()}));
  ASSERT_NE(plans_[0], nullptr);
  EXPECT_EQ(loom_runtime_weights_parameter_count(plans_[0]), 17u);
  EXPECT_EQ(loom_runtime_weights_activity(plans_[0]).activations, 0u);
  EXPECT_EQ(loom_runtime_weights_statistics(plans_[0]).reserved_bytes, 0u);
  EXPECT_EQ(loom_runtime_weights_statistics(plans_[0]).committed_bytes, 0u);
  EXPECT_EQ(buffers_[0][0], nullptr);
  EXPECT_EQ(buffers_[1][0], nullptr);
}

TEST_P(WeightPlacementTest, SpanAllocationFailureLeavesAnOwnedRetirablePlan) {
  IREE_ASSERT_OK(Compile(0, "first_parameter_block"));
  const auto root = Root(0, 0);
  NativeAllocator allocator(host_allocator_);
  allocator.FailNextAllocation(IREE_ALLOCATOR_COMMAND_REALLOC);
  loom_runtime_weights_t* plan = nullptr;
  auto status = loom_runtime_weights_allocate(
      device_, transfer_, dispatch_, loom_runtime_pool_storage(storage_owner_),
      pool_, loom_runtime_pool_weight_geometry(storage_owner_), nullptr,
      loom_runtime_pool_workspace(storage_owner_), 0, 1, &root,
      allocator.allocator(), &plan);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_NE(plan, nullptr);
  if (plan) {
    EXPECT_EQ(loom_runtime_weights_parameter_count(plan), 0u);
    EXPECT_EQ(buffers_[0][0], nullptr);
    iree_hal_buffer_release(buffers_[0][0]);
    buffers_[0][0] = nullptr;
    IREE_EXPECT_OK(loom_runtime_weights_await_retirement(plan));
    IREE_EXPECT_OK(loom_runtime_weights_free(plan));
  }
}

TEST_P(WeightPlacementTest, IndexGrowthFailureRetiresPreviouslyPlacedSpans) {
  IREE_ASSERT_OK(Compile(0, "first_parameter_block"));
  IREE_ASSERT_OK(Compile(1, "second_parameter_block"));
  const loom_runtime_weight_root_t roots[] = {Root(0, 0), Root(1, 0)};
  NativeAllocator allocator(host_allocator_);
  struct IndexGrowthFailure {
    static iree_status_t Control(void* self, iree_allocator_command_t command,
                                 const void* params, void** inout_pointer) {
      auto* failure = static_cast<IndexGrowthFailure*>(self);
      if (command == IREE_ALLOCATOR_COMMAND_CALLOC && failure->grew) {
        failure->failed = true;
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
      const bool growing = command == IREE_ALLOCATOR_COMMAND_REALLOC &&
                           *inout_pointer != nullptr;
      auto status = failure->allocator.ctl(failure->allocator.self, command,
                                           params, inout_pointer);
      failure->grew |= growing && iree_status_is_ok(status);
      return status;
    }
    // Tracks all accepted ownership until the failed plan has been retired.
    iree_allocator_t allocator;
    // The span array has grown beyond its first published parameter block.
    bool grew = false;
    // The replacement index allocation exercised the selected failure.
    bool failed = false;
  } failure{allocator.allocator()};
  loom_runtime_weights_t* plan = nullptr;
  auto status = loom_runtime_weights_allocate(
      device_, transfer_, dispatch_, loom_runtime_pool_storage(storage_owner_),
      pool_, loom_runtime_pool_weight_geometry(storage_owner_), nullptr,
      loom_runtime_pool_workspace(storage_owner_), 0, IREE_ARRAYSIZE(roots),
      roots, {&failure, IndexGrowthFailure::Control}, &plan);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_TRUE(failure.grew);
  EXPECT_TRUE(failure.failed);
  EXPECT_NE(plan, nullptr);
  if (plan) {
    EXPECT_EQ(loom_runtime_weights_parameter_count(plan), 17u);
    EXPECT_EQ(loom_runtime_weights_statistics(plan).reserved_bytes, 0u);
    EXPECT_EQ(loom_runtime_weights_statistics(plan).committed_bytes, 0u);
    for (size_t stage = 0; stage < IREE_ARRAYSIZE(roots); ++stage) {
      EXPECT_EQ(buffers_[stage][0], nullptr);
      iree_hal_buffer_release(buffers_[stage][0]);
      buffers_[stage][0] = nullptr;
    }
    IREE_EXPECT_OK(loom_runtime_weights_await_retirement(plan));
    IREE_EXPECT_OK(loom_runtime_weights_free(plan));
  }
}

INSTANTIATE_TEST_SUITE_P(Placement, WeightPlacementTest,
                         ::testing::Values(false, true));

}  // namespace loom_runtime::testing
