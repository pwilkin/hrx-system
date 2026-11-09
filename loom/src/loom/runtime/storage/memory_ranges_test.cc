// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_ranges.h"

#include <algorithm>
#include <vector>

#include "loom/runtime/storage/memory_test_fixture.h"

namespace {

using loom_runtime::testing::memory::MemoryTest;

TEST_F(MemoryTest, FootprintReadsUnorderedUnionWithoutAllocating) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  constexpr size_t kPageCount = 258;
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, kPageCount * page, 256, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, kPageCount * page, 256, &statistics[1], &second));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 1));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(second, 256 * page, 1));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);

  std::vector<loom_runtime_memory_range_t> ranges;
  bool touched[2][kPageCount] = {};
  for (size_t i = 0; i < 400; ++i) {
    const size_t owner = i % 2;
    const size_t index = (i * 73) % (kPageCount - 1);
    ranges.push_back({owner ? second : first, index * page + 1, page});
    touched[owner][index] = touched[owner][index + 1] = true;
  }
  uint64_t expected_total = 0;
  for (const auto& owner : touched) {
    for (bool present : owner) {
      expected_total += present ? page : 0;
    }
  }
  const uint64_t expected_backed =
      (touched[0][0] ? page : 0) + (touched[1][256] ? page : 0);
  native.ResetAllocationMetrics();
  for (size_t repeat = 0; repeat < 3; ++repeat) {
    const auto footprint =
        loom_runtime_memory_ranges_footprint(ranges.size(), ranges.data());
    EXPECT_EQ(footprint.total_bytes, expected_total);
    EXPECT_EQ(footprint.missing_bytes, expected_total - expected_backed);
    std::reverse(ranges.begin(), ranges.end());
  }
  EXPECT_EQ(native.allocation_metrics().request_count, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, page);
  EXPECT_EQ(statistics[1].committed_bytes, page);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
}

TEST_F(MemoryTest, AbsentVirtualExtentNeedsNoPageMetadata) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t length = iree_device_align(
      iree_min(kHighOffset, reservation_capabilities.max_byte_length / 2),
      geometry.mapping_size);
  ASSERT_GT(length, 4 * geometry.mapping_size);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, length, 256, &statistics[0], &first));
  loom_runtime_memory_range_t ranges[] = {
      {first, length - 1, 1},
      {first, 1, length - 2},
      {first, 0, 1},
  };
  native.ResetAllocationMetrics();
  const auto footprint =
      loom_runtime_memory_ranges_footprint(IREE_ARRAYSIZE(ranges), ranges);
  EXPECT_EQ(footprint.total_bytes, length);
  EXPECT_EQ(footprint.missing_bytes, length);
  EXPECT_EQ(native.allocation_metrics().request_count, 0u);
  EXPECT_EQ(statistics[0].committed_bytes, 0u);
  EXPECT_EQ(statistics[0].pending_bytes, 0u);
}

}  // namespace
