// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_test_fixture.h"

namespace {

using loom_runtime::testing::memory::MemoryTest;

TEST_F(MemoryTest, ContiguousMappingsKeepIndependentPageOwnership) {
  mapping_queue.Initialize(loom_runtime_device_transfer_queue(device));
  mapping_queue.observe_mappings = true;
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 3 * kSlabSize, &mapping_queue));
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 4 * page, geometry.granularity, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 4 * page));
  ASSERT_EQ(mapping_queue.WaitForMappings(1).size(), 1u);
  ASSERT_EQ(mapping_queue.WaitForMappings(1)[0].mappings.size(), 1u);
  EXPECT_EQ(
      mapping_queue.WaitForMappings(1)[0].mappings[0].target_range.byte_length,
      4 * page);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), page, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 3 * page, 0x87654321);

  loom_runtime_virtual_buffer_begin_trim(first);
  Keep(first, page, page);
  Keep(first, 3 * page, page);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_trim(first));
  ASSERT_EQ(mapping_queue.WaitForMappings(1).size(), 2u);
  const auto removals = mapping_queue.WaitForMappings(2)[1].mappings;
  ASSERT_EQ(removals.size(), 2u);
  EXPECT_EQ(removals[0].target_range.byte_offset, 0u);
  EXPECT_EQ(removals[1].target_range.byte_offset, 2 * page);
  EXPECT_EQ(removals[0].target_range.byte_length, page);
  EXPECT_EQ(removals[1].target_range.byte_length, page);
  EXPECT_EQ(statistics[0].committed_bytes, 2 * page);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), page, 0x12345678);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 3 * page,
                0x87654321);

  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0xaabbccdd);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), page, 0x12345678);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 3 * page,
                0x87654321);
  EXPECT_EQ(statistics[0].committed_bytes, 3 * page);
}

TEST_F(MemoryTest, InterleavedBackingKeepsAdjacentTargetsSeparate) {
  mapping_queue.Initialize(loom_runtime_device_transfer_queue(device));
  mapping_queue.observe_mappings = true;
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 3 * kSlabSize, &mapping_queue));
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 2 * page, geometry.granularity, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 2 * page, geometry.granularity, &statistics[1], &second));
  const loom_runtime_memory_range_t ranges[] = {
      {first, 0, page},
      {second, 0, page},
      {first, page, page},
      {second, page, page},
  };
  loom_runtime_memory_operation_t* operation = nullptr;
  loom_runtime_memory_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_memory_pool_commit_async(
      pool, IREE_ARRAYSIZE(ranges), ranges, nullptr, 0, &admission,
      &operation));
  ASSERT_EQ(admission, LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE);
  loom_runtime_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  loom_runtime_memory_operation_free(operation);
  ASSERT_EQ(mapping_queue.WaitForMappings(1).size(), 1u);
  ASSERT_EQ(mapping_queue.WaitForMappings(1)[0].mappings.size(), 4u);
  const auto observed = mapping_queue.WaitForMappings(1);
  for (const auto& mapping : observed[0].mappings) {
    EXPECT_EQ(mapping.target_range.byte_length, page);
  }
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), page, 0x87654321);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), 0, 0xaabbccdd);
  RoundTrip(loom_runtime_virtual_buffer_handle(second), page, 0x11223344);
  loom_runtime_virtual_buffer_begin_trim(first);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_trim(first));
  ASSERT_EQ(mapping_queue.WaitForMappings(1).size(), 2u);
  EXPECT_EQ(mapping_queue.WaitForMappings(2)[1].mappings.size(), 2u);
  CheckRetained(loom_runtime_virtual_buffer_handle(second), 0, 0xaabbccdd);
  CheckRetained(loom_runtime_virtual_buffer_handle(second), page, 0x11223344);
}

TEST_F(MemoryTest, GrowthMovesCoalescedSpansWithoutChangingBacking) {
  mapping_queue.Initialize(loom_runtime_device_transfer_queue(device));
  mapping_queue.observe_mappings = true;
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 3 * kSlabSize, &mapping_queue));
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 4 * page, geometry.granularity, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 4 * page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 3 * page, 0x87654321);
  const auto span = loom_runtime_virtual_buffer_backing(first, 3);
  loom_runtime_memory_operation_t* operation = nullptr;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_grow_async(
      first, 8 * page, nullptr, 0, &operation));
  loom_runtime_memory_operation_await(operation);
  IREE_ASSERT_OK(loom_runtime_memory_operation_query_status(operation));
  loom_runtime_memory_operation_free(operation);
  ASSERT_EQ(mapping_queue.WaitForMappings(1).size(), 3u);
  for (const auto& observation : mapping_queue.WaitForMappings(3)) {
    const auto& batch = observation.mappings;
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].target_range.byte_length, 4 * page);
  }
  EXPECT_EQ(mapping_queue.WaitForMappings(2)[1].mappings[0].operation,
            IREE_HAL_MEMORY_MAPPING_OPERATION_BIND);
  EXPECT_EQ(mapping_queue.WaitForMappings(3)[2].mappings[0].operation,
            IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND);
  const auto grown = loom_runtime_virtual_buffer_backing(first, 3);
  EXPECT_EQ(grown.buffer, span.buffer);
  EXPECT_EQ(grown.offset, span.offset);
  EXPECT_EQ(statistics[0].committed_bytes, 4 * page);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 3 * page,
                0x87654321);
}

TEST_F(MemoryTest, FailedCoalescedMappingRetainsEveryPageUntilDetach) {
  mapping_queue.Initialize(loom_runtime_device_transfer_queue(device));
  mapping_queue.observe_mappings = true;
  ASSERT_NO_FATAL_FAILURE(UseBacking(false, 3 * kSlabSize, &mapping_queue));
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, 4 * page, geometry.granularity, &statistics[0], &first));
  mapping_queue.fail_next_map = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_runtime_virtual_buffer_commit(first, 0, 4 * page));
  ASSERT_EQ(mapping_queue.WaitForMappings(1).size(), 1u);
  ASSERT_EQ(mapping_queue.WaitForMappings(1)[0].mappings.size(), 1u);
  EXPECT_EQ(statistics[0].committed_bytes, 4 * page);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(loom_runtime_virtual_buffer_backing(first, i).length, page);
  }
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_free(first));
  first = nullptr;
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  loom_runtime_memory_pool_trim(pool, 0);
  IREE_EXPECT_OK(loom_runtime_memory_pool_await(pool));
  EXPECT_EQ(loom_runtime_memory_pool_statistics(pool).committed_bytes, 0u);
}

}  // namespace
