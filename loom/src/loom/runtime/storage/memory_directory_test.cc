// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_test_fixture.h"

namespace {

using loom_runtime::testing::memory::MemoryTest;

TEST_F(MemoryTest, DirectoryGrowthFailurePreservesMappedPages) {
  const auto geometry = loom_runtime_pool_state_geometry(storage_owner);
  const uint64_t page = geometry.mapping_size;
  const uint64_t length = 513 * page;
  ASSERT_NO_FATAL_FAILURE(
      UseBacking(false, length + geometry.backing_size, nullptr));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
      pool, geometry, length, geometry.granularity, &statistics[0], &first));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 0, 2 * page));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 256 * page, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 256 * page, 0x87654321);

  // A new sparse group and an existing group's payload growth can both fail.
  // Neither may change accepted page ownership or poison a retry.
  for (auto command :
       {IREE_ALLOCATOR_COMMAND_CALLOC, IREE_ALLOCATOR_COMMAND_REALLOC}) {
    native.FailNextAllocation(command);
    const uint64_t index = command == IREE_ALLOCATOR_COMMAND_CALLOC ? 512 : 257;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_runtime_virtual_buffer_commit(first, index * page, page));
    EXPECT_EQ(statistics[0].pending_bytes, 0u);
    EXPECT_EQ(statistics[0].committed_bytes, 3 * page);
    CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
    CheckRetained(loom_runtime_virtual_buffer_handle(first), 256 * page,
                  0x87654321);
  }
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 257 * page, page));
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(first, 512 * page, page));
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 257 * page, 0xaabbccdd);
  RoundTrip(loom_runtime_virtual_buffer_handle(first), 512 * page, 0xddccbbaa);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 0, 0x12345678);
  CheckRetained(loom_runtime_virtual_buffer_handle(first), 256 * page,
                0x87654321);
}

}  // namespace
