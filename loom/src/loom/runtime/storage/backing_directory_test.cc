// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/backing_directory.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct FailingAllocator {
  // Successful allocation calls allowed before a single injected failure.
  iree_host_size_t remaining = IREE_HOST_SIZE_MAX;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto& remaining = static_cast<FailingAllocator*>(self)->remaining;
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        remaining != IREE_HOST_SIZE_MAX) {
      if (!remaining) {
        remaining = IREE_HOST_SIZE_MAX;
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
      --remaining;
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, params, inout_pointer);
  }
};

TEST(BackingDirectoryTest, RootGrowthFailurePreservesAcceptedPayloads) {
  FailingAllocator allocator;
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(
      {&allocator, FailingAllocator::Control}, &directory);
  for (iree_host_size_t i = 0; i < 6; ++i) {
    loom_runtime_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, i * 256, &entry));
    entry->physical_offset = i * 16;
    entry->physical = reinterpret_cast<loom_runtime_backing_parent_t*>(i + 100);
    entry->mapped = true;
  }
  // Accept the new leaf allocation, then fail the root allocation. The
  // caller's accepted reservations must remain reachable exactly once.
  allocator.remaining = 1;
  loom_runtime_backing_entry_t* entry = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_runtime_backing_directory_insert(&directory, 6 * 256, &entry));
  EXPECT_EQ(directory.count, 6u);
  EXPECT_EQ(loom_runtime_backing_directory_find(&directory, 6 * 256), nullptr);
  for (iree_host_size_t i = 0; i < 6; ++i) {
    const auto* retained =
        loom_runtime_backing_directory_find_const(&directory, i * 256);
    ASSERT_NE(retained, nullptr);
    EXPECT_EQ(retained->physical_offset, i * 16);
    EXPECT_TRUE(retained->mapped);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(retained->physical), i + 100);
  }
  IREE_ASSERT_OK(
      loom_runtime_backing_directory_insert(&directory, 6 * 256, &entry));
  EXPECT_EQ(directory.count, 7u);
  loom_runtime_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest, ReusesRemovedSlotsWithoutGrowing) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  for (iree_host_size_t index = 0; index < 6; ++index) {
    loom_runtime_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, index, &entry));
  }
  ASSERT_EQ(directory.capacity, 8u);
  for (iree_host_size_t index = 0; index < 5; ++index) {
    loom_runtime_backing_directory_remove(
        &directory, loom_runtime_backing_directory_find(&directory, index));
  }
  ASSERT_EQ(directory.count, 1u);

  loom_runtime_backing_entry_t* replacement = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_backing_directory_insert(&directory, 100, &replacement));
  EXPECT_EQ(directory.capacity, 8u);
  EXPECT_EQ(directory.count, 2u);
  loom_runtime_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest, TrimSelectionPreservesLiveAndPendingOwnership) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  for (iree_host_size_t index : {0u, 1u, 2u, 512u}) {
    loom_runtime_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, index, &entry));
    entry->keep = true;
    if (index == 1) {
      entry->physical_offset = 1024;
      entry->physical = reinterpret_cast<loom_runtime_backing_parent_t*>(99);
      entry->mapped = true;
      entry->counted = true;
    } else if (index == 2) {
      entry->in_flight = true;
    }
  }
  // Finalizing a selection discards empty markers without clearing live marks.
  loom_runtime_backing_directory_discard_empty(&directory);
  EXPECT_EQ(directory.count, 2u);
  EXPECT_EQ(loom_runtime_backing_directory_find(&directory, 0), nullptr);
  EXPECT_EQ(loom_runtime_backing_directory_find(&directory, 512), nullptr);
  ASSERT_TRUE(loom_runtime_backing_directory_find(&directory, 1)->keep);
  ASSERT_TRUE(loom_runtime_backing_directory_find(&directory, 2)->keep);

  loom_runtime_backing_entry_t* marker = nullptr;
  IREE_ASSERT_OK(loom_runtime_backing_directory_insert(&directory, 0, &marker));
  marker->keep = true;
  loom_runtime_backing_directory_begin_trim(&directory);
  EXPECT_EQ(directory.count, 2u);
  EXPECT_EQ(loom_runtime_backing_directory_find(&directory, 0), nullptr);
  const auto* resident = loom_runtime_backing_directory_find(&directory, 1);
  ASSERT_NE(resident, nullptr);
  EXPECT_FALSE(resident->keep);
  EXPECT_TRUE(resident->counted);
  EXPECT_EQ(resident->physical_offset, 1024u);
  EXPECT_TRUE(resident->mapped);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(resident->physical), 99u);
  const auto* pending = loom_runtime_backing_directory_find(&directory, 2);
  ASSERT_NE(pending, nullptr);
  EXPECT_FALSE(pending->keep);
  EXPECT_TRUE(pending->in_flight);
  EXPECT_EQ(pending->physical, nullptr);
  loom_runtime_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest, RemovalDuringTraversalVisitsEveryDensePayload) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  const iree_host_size_t base = IREE_HOST_SIZE_MAX - 255;
  // Reverse insertion exercises compaction inside a fully populated group.
  for (iree_host_size_t i = 256; i > 0; --i) {
    loom_runtime_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(loom_runtime_backing_directory_insert(&directory,
                                                         base + i - 1, &entry));
    entry->physical_offset = i * 16;
    entry->physical = reinterpret_cast<loom_runtime_backing_parent_t*>(i + 100);
    entry->mapped = true;
  }
  for (iree_host_size_t i = 0; i < 256; ++i) {
    const auto* entry =
        loom_runtime_backing_directory_find_const(&directory, base + i);
    ASSERT_NE(entry, nullptr);
    EXPECT_TRUE(entry->mapped);
  }
  loom_runtime_backing_cursor_t cursor = {};
  iree_host_size_t count = 0;
  while (auto* entry =
             loom_runtime_backing_directory_next(&directory, &cursor)) {
    EXPECT_EQ(entry->index, base + count);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(entry->physical), count + 101);
    loom_runtime_backing_directory_remove(&directory, entry);
    ++count;
  }
  EXPECT_EQ(count, 256u);
  EXPECT_EQ(directory.count, 0u);
  loom_runtime_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest, GrowthAndShrinkPreserveSparseEntries) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);

  for (const iree_host_size_t index : {0u, 8u, 16u}) {
    loom_runtime_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, index, &entry));
    entry->keep = true;
  }
  EXPECT_EQ(directory.count, 3u);
  loom_runtime_backing_directory_remove(
      &directory, loom_runtime_backing_directory_find(&directory, 8));
  EXPECT_EQ(directory.count, 2u);
  EXPECT_NE(loom_runtime_backing_directory_find(&directory, 16), nullptr);

  loom_runtime_backing_entry_t* replacement = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_backing_directory_insert(&directory, 24, &replacement));
  EXPECT_EQ(directory.count, 3u);

  for (iree_host_size_t i = 1; i <= 128; ++i) {
    loom_runtime_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, i * 7, &entry));
  }
  loom_runtime_backing_entry_t* high = nullptr;
  IREE_ASSERT_OK(loom_runtime_backing_directory_insert(
      &directory, IREE_HOST_SIZE_MAX - 123, &high));
  high->in_flight = true;
  for (iree_host_size_t i = 1; i <= 128; ++i) {
    EXPECT_NE(loom_runtime_backing_directory_find_const(&directory, i * 7),
              nullptr);
  }
  EXPECT_EQ(loom_runtime_backing_directory_find_const(&directory,
                                                      IREE_HOST_SIZE_MAX - 123),
            high);

  loom_runtime_backing_directory_discard_empty(&directory);
  EXPECT_EQ(directory.count, 1u);
  EXPECT_EQ(directory.group_count, 1u);
  IREE_ASSERT_OK(loom_runtime_backing_directory_shrink(&directory));
  EXPECT_EQ(directory.capacity, 8u);
  const auto* shrunk_high = loom_runtime_backing_directory_find_const(
      &directory, IREE_HOST_SIZE_MAX - 123);
  ASSERT_NE(shrunk_high, nullptr);
  EXPECT_TRUE(shrunk_high->in_flight);
  loom_runtime_backing_directory_deinitialize(&directory);
}

TEST(BackingDirectoryTest,
     IncrementalGrowthPreservesPayloadsAcrossSparsePlanes) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  constexpr iree_host_size_t kCount = 4097;
  std::vector<iree_host_size_t> keys;
  for (iree_host_size_t i = 0; i < kCount; ++i) {
    // Nearby slots and distant planes exercise different hash bits without
    // allocating metadata for the untouched address interval between them.
    keys.push_back(i % 2 ? IREE_HOST_SIZE_MAX - i * 8192 : i);
    loom_runtime_backing_entry_t* entry = nullptr;
    IREE_ASSERT_OK(
        loom_runtime_backing_directory_insert(&directory, keys.back(), &entry));
    entry->physical_offset = i * 256;
    entry->physical = reinterpret_cast<loom_runtime_backing_parent_t*>(i + 100);
    entry->mapped = true;
    entry->keep = (i % 3) == 0;
  }
  ASSERT_EQ(directory.count, kCount);
  EXPECT_LT(directory.capacity, kCount + 256);
  auto check = [&](bool removed_even) {
    for (iree_host_size_t i = 0; i < kCount; ++i) {
      const auto* entry =
          loom_runtime_backing_directory_find_const(&directory, keys[i]);
      if (removed_even && i % 2 == 0) {
        EXPECT_EQ(entry, nullptr);
      } else {
        ASSERT_NE(entry, nullptr);
        EXPECT_EQ(entry->index, keys[i]);
        EXPECT_EQ(entry->physical_offset, i * 256);
        EXPECT_TRUE(entry->mapped);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(entry->physical), i + 100);
        EXPECT_EQ(entry->keep, i % 3 == 0);
      }
    }
  };
  ASSERT_NO_FATAL_FAILURE(check(false));
  for (iree_host_size_t i = 0; i < kCount; i += 2) {
    loom_runtime_backing_directory_remove(
        &directory, loom_runtime_backing_directory_find(&directory, keys[i]));
  }
  IREE_ASSERT_OK(loom_runtime_backing_directory_shrink(&directory));
  EXPECT_EQ(directory.count, kCount / 2);
  ASSERT_NO_FATAL_FAILURE(check(true));
  std::vector<bool> visited(kCount, false);
  loom_runtime_backing_cursor_t cursor = {};
  while (auto* entry =
             loom_runtime_backing_directory_next(&directory, &cursor)) {
    const size_t index = reinterpret_cast<uintptr_t>(entry->physical) - 100;
    ASSERT_LT(index, kCount);
    EXPECT_FALSE(visited[index]);
    visited[index] = true;
    loom_runtime_backing_directory_remove(&directory, entry);
  }
  EXPECT_EQ(directory.count, 0u);
  for (iree_host_size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(visited[i], i % 2 != 0);
  }
  IREE_ASSERT_OK(loom_runtime_backing_directory_shrink(&directory));
  EXPECT_EQ(directory.capacity, 0u);
  EXPECT_EQ(directory.buckets, nullptr);
  loom_runtime_backing_directory_deinitialize(&directory);
}

}  // namespace
