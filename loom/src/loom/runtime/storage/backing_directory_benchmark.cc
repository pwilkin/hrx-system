// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "benchmark/benchmark.h"
#include "loom/runtime/storage/backing_directory.h"

namespace {

static std::vector<iree_host_size_t> Populate(
    iree_host_size_t count, loom_runtime_backing_directory_t* directory) {
  std::vector<iree_host_size_t> indexes;
  indexes.reserve(count);
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t index = i * 17;
    loom_runtime_backing_entry_t* entry = nullptr;
    IREE_CHECK_OK(
        loom_runtime_backing_directory_insert(directory, index, &entry));
    entry->in_flight = true;
    indexes.push_back(index);
  }
  return indexes;
}

// Models repeated selection of already resident backing. Each item is one
// logical slot lookup and mark plus the corresponding cleanup lookup.
static void BM_MarkResidentSlots(benchmark::State& state) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  const auto indexes = Populate(state.range(0), &directory);
  for (auto _ : state) {
    for (const iree_host_size_t index : indexes) {
      loom_runtime_backing_entry_t* entry = nullptr;
      IREE_CHECK_OK(
          loom_runtime_backing_directory_insert(&directory, index, &entry));
      benchmark::DoNotOptimize(entry);
      entry->counted = true;
    }
    for (const iree_host_size_t index : indexes) {
      auto* entry = loom_runtime_backing_directory_find(&directory, index);
      benchmark::DoNotOptimize(entry);
      entry->counted = false;
    }
  }
  state.SetItemsProcessed(state.iterations() * indexes.size());
  loom_runtime_backing_directory_deinitialize(&directory);
}
BENCHMARK(BM_MarkResidentSlots)->Arg(16)->Arg(256)->Arg(4096);

// Models commitment that discovers missing slots and is then denied.
// Rollback preserves live entries and releases unused directory capacity.
static void BM_DeniedSparseGrowth(benchmark::State& state) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  auto live_indexes = Populate(state.range(0), &directory);
  const iree_host_size_t missing_count = state.range(1);
  const iree_host_size_t missing_origin = state.range(0) * 17 + 1;
  for (auto _ : state) {
    for (iree_host_size_t i = 0; i < missing_count; ++i) {
      loom_runtime_backing_entry_t* entry = nullptr;
      IREE_CHECK_OK(loom_runtime_backing_directory_insert(
          &directory, missing_origin + i * 17, &entry));
    }
    for (iree_host_size_t i = 0; i < missing_count; ++i) {
      auto* entry = loom_runtime_backing_directory_find(
          &directory, missing_origin + i * 17);
      loom_runtime_backing_directory_remove(&directory, entry);
    }
    IREE_CHECK_OK(loom_runtime_backing_directory_shrink(&directory));
  }
  benchmark::DoNotOptimize(live_indexes);
  state.SetItemsProcessed(state.iterations() * missing_count);
  loom_runtime_backing_directory_deinitialize(&directory);
}
BENCHMARK(BM_DeniedSparseGrowth)->Args({16, 16})->Args({256, 256});

static void RemoveDenseRange(loom_runtime_backing_directory_t* directory,
                             iree_host_size_t count) {
  for (iree_host_size_t index = 0; index < count; ++index) {
    auto* entry = loom_runtime_backing_directory_find(directory, index);
    loom_runtime_backing_directory_remove(directory, entry);
  }
  IREE_CHECK_OK(loom_runtime_backing_directory_shrink(directory));
}

// Models the two scalar page traversals of a successful dense commitment.
// Lifecycle cleanup is outside the timed region so the row isolates selection
// and publication work rather than allocator release behavior.
static void BM_CommitDenseScalar(benchmark::State& state) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  const iree_host_size_t count = state.range(0);
  for (auto _ : state) {
    for (iree_host_size_t index = 0; index < count; ++index) {
      loom_runtime_backing_entry_t* entry = nullptr;
      IREE_CHECK_OK(
          loom_runtime_backing_directory_insert(&directory, index, &entry));
      entry->pending_commit = true;
    }
    for (iree_host_size_t index = 0; index < count; ++index) {
      auto* entry = loom_runtime_backing_directory_find(&directory, index);
      entry->pending_commit = false;
      benchmark::DoNotOptimize(entry);
    }
    state.PauseTiming();
    RemoveDenseRange(&directory, count);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count);
  loom_runtime_backing_directory_deinitialize(&directory);
}
BENCHMARK(BM_CommitDenseScalar)->Arg(16)->Arg(256)->Arg(4096);

// Models the same successful commitment through grouped range insertion and
// traversal. Lifecycle cleanup remains outside the timed region.
static void BM_CommitDenseRange(benchmark::State& state) {
  loom_runtime_backing_directory_t directory;
  loom_runtime_backing_directory_initialize(iree_allocator_system(),
                                            &directory);
  const iree_host_size_t count = state.range(0);
  for (auto _ : state) {
    IREE_CHECK_OK(
        loom_runtime_backing_directory_insert_range(&directory, 0, count));
    loom_runtime_backing_range_cursor_t cursor = {.index = 0, .end = count};
    iree_host_size_t chunk_count = 0;
    loom_runtime_backing_entry_t* entries = nullptr;
    while ((entries = loom_runtime_backing_directory_range_next(
                &directory, &cursor, &chunk_count))) {
      for (iree_host_size_t index = 0; index < chunk_count; ++index) {
        entries[index].pending_commit = true;
      }
    }
    cursor = {.index = 0, .end = count};
    while ((entries = loom_runtime_backing_directory_range_next(
                &directory, &cursor, &chunk_count))) {
      for (iree_host_size_t index = 0; index < chunk_count; ++index) {
        entries[index].pending_commit = false;
        benchmark::DoNotOptimize(&entries[index]);
      }
    }
    state.PauseTiming();
    RemoveDenseRange(&directory, count);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * count);
  loom_runtime_backing_directory_deinitialize(&directory);
}
BENCHMARK(BM_CommitDenseRange)->Arg(16)->Arg(256)->Arg(4096);

}  // namespace
