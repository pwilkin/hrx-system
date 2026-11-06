// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/backing_directory.h"

#include "iree/base/internal/math.h"
#include "iree/base/internal/prng.h"

#define LOOM_RUNTIME_BACKING_GROUP_SHIFT 8
#define LOOM_RUNTIME_BACKING_GROUP_CAPACITY 256
#define LOOM_RUNTIME_BACKING_BUCKET_EMPTY IREE_HOST_SIZE_MAX
#define LOOM_RUNTIME_BACKING_BUCKET_DELETED (IREE_HOST_SIZE_MAX - 1)

typedef struct loom_runtime_backing_group_t {
  // Logical occupancy determines compact payload positions by bitmap rank.
  uint64_t occupied[LOOM_RUNTIME_BACKING_GROUP_CAPACITY / 64];
  // Initialized payload count, equal to the occupancy population.
  uint16_t count;
  // Allocated payload slots, growing from one to the bounded group capacity.
  uint16_t capacity;
  // Unique caller-owned payloads in increasing logical-index order.
  loom_runtime_backing_entry_t entries[];
} loom_runtime_backing_group_t;

struct loom_runtime_backing_bucket_t {
  // Logical group index, or an empty/deleted sentinel above all valid keys.
  iree_host_size_t key;
  // Owned compact group; NULL for an empty or deleted bucket.
  loom_runtime_backing_group_t* group;
};

static iree_host_size_t loom_runtime_backing_hash(iree_host_size_t key) {
  iree_prng_splitmix64_state_t state = key;
  return (iree_host_size_t)iree_prng_splitmix64_next(&state);
}

// Returns an existing bucket or an insertion slot. The root always retains
// empty buckets, including after deletions, so every probe terminates.
static loom_runtime_backing_bucket_t* loom_runtime_backing_find_bucket(
    const loom_runtime_backing_directory_t* directory, iree_host_size_t key) {
  const iree_host_size_t mask = directory->capacity - 1;
  iree_host_size_t slot = loom_runtime_backing_hash(key) & mask;
  loom_runtime_backing_bucket_t* deleted = NULL;
  while (true) {
    loom_runtime_backing_bucket_t* bucket = &directory->buckets[slot];
    if (bucket->key == key) {
      return bucket;
    }
    if (bucket->key == LOOM_RUNTIME_BACKING_BUCKET_EMPTY) {
      return deleted ? deleted : bucket;
    }
    if (bucket->key == LOOM_RUNTIME_BACKING_BUCKET_DELETED && !deleted) {
      deleted = bucket;
    }
    slot = (slot + 1) & mask;
  }
}

static uint16_t loom_runtime_backing_group_rank(
    const loom_runtime_backing_group_t* group, uint16_t offset) {
  // Complete groups are dense: logical offset is already the payload index.
  if (group->count == LOOM_RUNTIME_BACKING_GROUP_CAPACITY) {
    return offset;
  }
  uint16_t position = 0;
  for (uint16_t i = 0; i < offset / 64; ++i) {
    position += (uint16_t)iree_math_count_ones_u64(group->occupied[i]);
  }
  position += (uint16_t)iree_math_count_ones_u64(
      group->occupied[offset / 64] & ((UINT64_C(1) << (offset % 64)) - 1));
  return position;
}

static iree_status_t loom_runtime_backing_rehash(
    loom_runtime_backing_directory_t* directory, iree_host_size_t capacity) {
  loom_runtime_backing_bucket_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      directory->host_allocator, capacity, sizeof(*buckets), (void**)&buckets));
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    buckets[i].key = LOOM_RUNTIME_BACKING_BUCKET_EMPTY;
  }
  for (iree_host_size_t i = 0; i < directory->capacity; ++i) {
    const loom_runtime_backing_bucket_t bucket = directory->buckets[i];
    if (!bucket.group) {
      continue;
    }
    iree_host_size_t slot =
        loom_runtime_backing_hash(bucket.key) & (capacity - 1);
    while (buckets[slot].group) {
      slot = (slot + 1) & (capacity - 1);
    }
    buckets[slot] = bucket;
  }
  iree_allocator_free(directory->host_allocator, directory->buckets);
  directory->buckets = buckets;
  directory->capacity = capacity;
  directory->tombstone_count = 0;
  return iree_ok_status();
}

static iree_status_t loom_runtime_backing_reserve_group(
    loom_runtime_backing_directory_t* directory) {
  if (directory->capacity &&
      directory->group_count + directory->tombstone_count + 1 <=
          directory->capacity - directory->capacity / 4) {
    return iree_ok_status();
  }
  iree_host_size_t capacity = directory->capacity;
  if (!capacity) {
    capacity = 8;
  } else if (directory->group_count + 1 > capacity - capacity / 4) {
    if (capacity > IREE_HOST_SIZE_MAX / sizeof(*directory->buckets) / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "sparse backing directory is too large");
    }
    capacity *= 2;
  }
  return loom_runtime_backing_rehash(directory, capacity);
}

static void loom_runtime_backing_release_storage(
    loom_runtime_backing_directory_t* directory) {
  for (iree_host_size_t i = 0; i < directory->capacity; ++i) {
    iree_allocator_free(directory->host_allocator, directory->buckets[i].group);
  }
  iree_allocator_free(directory->host_allocator, directory->buckets);
  directory->buckets = NULL;
  directory->capacity = 0;
  directory->group_count = 0;
  directory->tombstone_count = 0;
}

static void loom_runtime_backing_remove_group(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_bucket_t* bucket) {
  iree_allocator_free(directory->host_allocator, bucket->group);
  bucket->group = NULL;
  bucket->key = LOOM_RUNTIME_BACKING_BUCKET_DELETED;
  --directory->group_count;
  ++directory->tombstone_count;
}

void loom_runtime_backing_directory_initialize(
    iree_allocator_t host_allocator,
    loom_runtime_backing_directory_t* out_directory) {
  *out_directory = (loom_runtime_backing_directory_t){
      .host_allocator = host_allocator,
  };
}

void loom_runtime_backing_directory_deinitialize(
    loom_runtime_backing_directory_t* directory) {
  loom_runtime_backing_release_storage(directory);
  *directory = (loom_runtime_backing_directory_t){0};
}

const loom_runtime_backing_entry_t* loom_runtime_backing_directory_find_const(
    const loom_runtime_backing_directory_t* directory, iree_host_size_t index) {
  if (!directory->count) {
    return NULL;
  }
  const loom_runtime_backing_bucket_t* bucket =
      loom_runtime_backing_find_bucket(
          directory, index >> LOOM_RUNTIME_BACKING_GROUP_SHIFT);
  if (!bucket->group) {
    return NULL;
  }
  const uint16_t offset = index & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
  if (!(bucket->group->occupied[offset / 64] &
        (UINT64_C(1) << (offset % 64)))) {
    return NULL;
  }
  return &bucket->group
              ->entries[loom_runtime_backing_group_rank(bucket->group, offset)];
}

loom_runtime_backing_entry_t* loom_runtime_backing_directory_find(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index) {
  return (loom_runtime_backing_entry_t*)
      loom_runtime_backing_directory_find_const(directory, index);
}

iree_status_t loom_runtime_backing_directory_insert(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index,
    loom_runtime_backing_entry_t** out_entry) {
  const iree_host_size_t key = index >> LOOM_RUNTIME_BACKING_GROUP_SHIFT;
  const uint16_t offset = index & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
  loom_runtime_backing_bucket_t* bucket =
      directory->capacity ? loom_runtime_backing_find_bucket(directory, key)
                          : NULL;
  loom_runtime_backing_group_t* group = bucket ? bucket->group : NULL;
  if (!group) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
        directory->host_allocator, sizeof(*group), 1, sizeof(*group->entries),
        (void**)&group));
    iree_status_t status = loom_runtime_backing_reserve_group(directory);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(directory->host_allocator, group);
      return status;
    }
    group->capacity = 1;
    bucket = loom_runtime_backing_find_bucket(directory, key);
    if (bucket->key == LOOM_RUNTIME_BACKING_BUCKET_DELETED) {
      --directory->tombstone_count;
    }
    bucket->key = key;
    bucket->group = group;
    ++directory->group_count;
  }
  const uint64_t bit = UINT64_C(1) << (offset % 64);
  const uint16_t position = loom_runtime_backing_group_rank(group, offset);
  if (!(group->occupied[offset / 64] & bit)) {
    if (group->count == group->capacity) {
      const uint16_t capacity = group->capacity * 2;
      IREE_RETURN_IF_ERROR(iree_allocator_realloc(
          directory->host_allocator,
          sizeof(*group) + capacity * sizeof(*group->entries), (void**)&group));
      group->capacity = capacity;
      bucket->group = group;
    }
    memmove(&group->entries[position + 1], &group->entries[position],
            (group->count - position) * sizeof(*group->entries));
    group->entries[position] = (loom_runtime_backing_entry_t){.index = index};
    group->occupied[offset / 64] |= bit;
    ++group->count;
    ++directory->count;
  }
  *out_entry = &group->entries[position];
  return iree_ok_status();
}

void loom_runtime_backing_directory_remove(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_entry_t* entry) {
  loom_runtime_backing_bucket_t* bucket = loom_runtime_backing_find_bucket(
      directory, entry->index >> LOOM_RUNTIME_BACKING_GROUP_SHIFT);
  loom_runtime_backing_group_t* group = bucket->group;
  const uint16_t offset =
      entry->index & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
  group->occupied[offset / 64] &= ~(UINT64_C(1) << (offset % 64));
  const iree_host_size_t position = entry - group->entries;
  memmove(entry, entry + 1, (group->count - position - 1) * sizeof(*entry));
  group->entries[--group->count] = (loom_runtime_backing_entry_t){0};
  --directory->count;
  if (!group->count) {
    loom_runtime_backing_remove_group(directory, bucket);
  }
}

const loom_runtime_backing_entry_t* loom_runtime_backing_directory_next_const(
    const loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_cursor_t* cursor) {
  while (cursor->bucket < directory->capacity) {
    const loom_runtime_backing_group_t* group =
        directory->buckets[cursor->bucket].group;
    if (group) {
      while (cursor->offset < LOOM_RUNTIME_BACKING_GROUP_CAPACITY) {
        const uint16_t word = cursor->offset / 64;
        const uint64_t occupied =
            group->occupied[word] & (UINT64_MAX << (cursor->offset % 64));
        if (occupied) {
          const uint16_t offset =
              word * 64 +
              (uint16_t)iree_math_count_trailing_zeros_u64(occupied);
          cursor->offset = offset + 1;
          return &group
                      ->entries[loom_runtime_backing_group_rank(group, offset)];
        }
        cursor->offset = (word + 1) * 64;
      }
    }
    ++cursor->bucket;
    cursor->offset = 0;
  }
  return NULL;
}

loom_runtime_backing_entry_t* loom_runtime_backing_directory_next(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_cursor_t* cursor) {
  return (loom_runtime_backing_entry_t*)
      loom_runtime_backing_directory_next_const(directory, cursor);
}

void loom_runtime_backing_directory_discard_empty(
    loom_runtime_backing_directory_t* directory) {
  for (iree_host_size_t i = 0; i < directory->capacity; ++i) {
    loom_runtime_backing_group_t* group = directory->buckets[i].group;
    if (!group) {
      continue;
    }
    uint16_t count = 0;
    for (uint16_t j = 0; j < group->count; ++j) {
      const loom_runtime_backing_entry_t entry = group->entries[j];
      if (entry.physical || entry.in_flight) {
        if (count != j) {
          group->entries[count] = entry;
        }
        ++count;
      } else {
        IREE_ASSERT(!entry.pending_commit && !entry.counted);
        const uint16_t offset =
            entry.index & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
        group->occupied[offset / 64] &= ~(UINT64_C(1) << (offset % 64));
      }
    }
    directory->count -= group->count - count;
    group->count = count;
    if (!count) {
      loom_runtime_backing_remove_group(directory, &directory->buckets[i]);
    }
  }
  if (!directory->count) {
    loom_runtime_backing_release_storage(directory);
  }
}

iree_status_t loom_runtime_backing_directory_shrink(
    loom_runtime_backing_directory_t* directory) {
  if (!directory->count) {
    loom_runtime_backing_release_storage(directory);
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < directory->capacity && iree_status_is_ok(status); ++i) {
    loom_runtime_backing_group_t* group = directory->buckets[i].group;
    if (!group || group->count > group->capacity / 2) {
      continue;
    }
    const uint16_t capacity =
        (uint16_t)iree_math_round_up_to_pow2_u32(group->count);
    status = iree_allocator_realloc(
        directory->host_allocator,
        sizeof(*group) + capacity * sizeof(*group->entries), (void**)&group);
    if (iree_status_is_ok(status)) {
      group->capacity = capacity;
      directory->buckets[i].group = group;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t capacity = 8;
    while (directory->group_count > capacity - capacity / 4) {
      capacity *= 2;
    }
    if (capacity < directory->capacity ||
        directory->tombstone_count > directory->group_count) {
      status = loom_runtime_backing_rehash(directory, capacity);
    }
  }
  return status;
}
