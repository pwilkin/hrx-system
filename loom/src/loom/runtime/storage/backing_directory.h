// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_BACKING_DIRECTORY_H_
#define LOOM_RUNTIME_STORAGE_BACKING_DIRECTORY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct loom_runtime_backing_parent_t loom_runtime_backing_parent_t;

// Mutable state for one touched slot in a logical virtual reservation.
typedef struct loom_runtime_backing_entry_t {
  // Logical slot index within the virtual reservation.
  iree_host_size_t index;
  // Physical parent retained by this slot lease, or NULL when uncommitted.
  loom_runtime_backing_parent_t* physical;
  // Source byte origin within the parent, independent of the logical index.
  iree_device_size_t physical_offset;
  // Whether physical is currently mapped into its virtual slot.
  bool mapped;
  // Whether a current retained range requires this slab.
  bool keep;
  // Missing slab selected by the current serialized commitment transaction.
  bool pending_commit;
  // Temporary union mark scoped to one serialized backing selection.
  bool counted;
  // Accepted asynchronous operation excludes another editor of this slot.
  bool in_flight;
} loom_runtime_backing_entry_t;

typedef struct loom_runtime_backing_bucket_t loom_runtime_backing_bucket_t;

// Sparse slot directory whose storage scales with touched slots, not the
// reservation's logical extent. Mutations are serialized by its outer owner.
typedef struct loom_runtime_backing_directory_t {
  // Allocator owning entries through deinitialization.
  iree_allocator_t host_allocator;
  // Number of live entries.
  iree_host_size_t count;
  // Number of root buckets, independent of the virtual reservation extent.
  iree_host_size_t capacity;
  // Occupied root buckets, each owning one logical page group.
  iree_host_size_t group_count;
  // Deleted root buckets preserving probe chains until rehash.
  iree_host_size_t tombstone_count;
  // Owned sparse root indexing compact groups of adjacent logical slots.
  loom_runtime_backing_bucket_t* buckets;
} loom_runtime_backing_directory_t;

// A logical traversal position survives removal of the returned entry.
typedef struct loom_runtime_backing_cursor_t {
  // Root bucket containing the next logical slot to inspect.
  iree_host_size_t bucket;
  // Next logical offset within that group, including its end sentinel.
  uint16_t offset;
} loom_runtime_backing_cursor_t;

// Initializes an empty directory without allocating storage.
void loom_runtime_backing_directory_initialize(
    iree_allocator_t host_allocator,
    loom_runtime_backing_directory_t* out_directory);

// Frees directory storage. Entry payload ownership remains with the caller.
void loom_runtime_backing_directory_deinitialize(
    loom_runtime_backing_directory_t* directory);

// Finds an existing entry, returning NULL when the slot has never been touched.
loom_runtime_backing_entry_t* loom_runtime_backing_directory_find(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index);
const loom_runtime_backing_entry_t* loom_runtime_backing_directory_find_const(
    const loom_runtime_backing_directory_t* directory, iree_host_size_t index);

// Finds or inserts an empty entry for index. Payload growth moves at most one
// bounded logical group; root growth moves only group keys and pointers.
// Failure preserves existing entries and their ownership.
iree_status_t loom_runtime_backing_directory_insert(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index,
    loom_runtime_backing_entry_t** out_entry);

// Removes an entry and makes its payload slot reusable.
void loom_runtime_backing_directory_remove(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_entry_t* entry);

// Visits occupied payloads in root/group order. Initialize *cursor to zero;
// each call advances it and returns NULL at the end. Removing the returned
// entry is allowed. Insertion or shrink invalidates an active traversal.
loom_runtime_backing_entry_t* loom_runtime_backing_directory_next(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_cursor_t* cursor);
const loom_runtime_backing_entry_t* loom_runtime_backing_directory_next_const(
    const loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_cursor_t* cursor);

// Removes marker-only entries with neither physical nor in-flight ownership.
void loom_runtime_backing_directory_discard_empty(
    loom_runtime_backing_directory_t* directory);

// Removes marker-only entries and clears surviving keep flags in one pass,
// beginning a new retained-range selection without changing page ownership.
void loom_runtime_backing_directory_begin_trim(
    loom_runtime_backing_directory_t* directory);

// Compacts payloads at a cold reclamation boundary when at least half the
// storage is unused. Empty directories release all storage without allocating.
iree_status_t loom_runtime_backing_directory_shrink(
    loom_runtime_backing_directory_t* directory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_RUNTIME_STORAGE_BACKING_DIRECTORY_H_
