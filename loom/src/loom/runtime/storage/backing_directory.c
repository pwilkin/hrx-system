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

static iree_status_t loom_runtime_backing_acquire_group(
    loom_runtime_backing_directory_t* directory, iree_host_size_t key,
    uint16_t minimum_capacity, loom_runtime_backing_bucket_t** out_bucket) {
  loom_runtime_backing_bucket_t* bucket =
      directory->capacity ? loom_runtime_backing_find_bucket(directory, key)
                          : NULL;
  loom_runtime_backing_group_t* group = bucket ? bucket->group : NULL;
  const uint16_t capacity =
      (uint16_t)iree_math_round_up_to_pow2_u32(minimum_capacity);
  if (!group) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
        directory->host_allocator, sizeof(*group), capacity,
        sizeof(*group->entries), (void**)&group));
    iree_status_t status = loom_runtime_backing_reserve_group(directory);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(directory->host_allocator, group);
      return status;
    }
    group->capacity = capacity;
    bucket = loom_runtime_backing_find_bucket(directory, key);
    if (bucket->key == LOOM_RUNTIME_BACKING_BUCKET_DELETED) {
      --directory->tombstone_count;
    }
    bucket->key = key;
    bucket->group = group;
    ++directory->group_count;
  } else if (capacity > group->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        directory->host_allocator,
        sizeof(*group) + capacity * sizeof(*group->entries), (void**)&group));
    group->capacity = capacity;
    bucket->group = group;
  }
  *out_bucket = bucket;
  return iree_ok_status();
}

static iree_status_t loom_runtime_backing_insert_entry(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_bucket_t* bucket, iree_host_size_t index,
    uint16_t position, loom_runtime_backing_entry_t** out_entry) {
  loom_runtime_backing_group_t* group = bucket->group;
  if (group->count == group->capacity) {
    IREE_RETURN_IF_ERROR(loom_runtime_backing_acquire_group(
        directory, bucket->key, group->count + 1, &bucket));
    group = bucket->group;
  }
  memmove(&group->entries[position + 1], &group->entries[position],
          (group->count - position) * sizeof(*group->entries));
  group->entries[position] = (loom_runtime_backing_entry_t){.index = index};
  const uint16_t offset = index & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
  group->occupied[offset / 64] |= UINT64_C(1) << (offset % 64);
  ++group->count;
  ++directory->count;
  *out_entry = &group->entries[position];
  return iree_ok_status();
}

static uint16_t loom_runtime_backing_group_occupied_count(
    const loom_runtime_backing_group_t* group, uint16_t begin, uint16_t end) {
  uint16_t count = 0;
  for (uint16_t word = begin / 64; word <= (end - 1) / 64; ++word) {
    const uint16_t word_begin = iree_max(begin, word * 64) - word * 64;
    const uint16_t word_end = iree_min(end, (word + 1) * 64) - word * 64;
    const uint64_t begin_mask = UINT64_MAX << word_begin;
    const uint64_t end_mask =
        word_end == 64 ? UINT64_MAX : (UINT64_C(1) << word_end) - 1;
    count += (uint16_t)iree_math_count_ones_u64(group->occupied[word] &
                                                begin_mask & end_mask);
  }
  return count;
}

static void loom_runtime_backing_group_mark_occupied(
    loom_runtime_backing_group_t* group, uint16_t begin, uint16_t end) {
  for (uint16_t word = begin / 64; word <= (end - 1) / 64; ++word) {
    const uint16_t word_begin = iree_max(begin, word * 64) - word * 64;
    const uint16_t word_end = iree_min(end, (word + 1) * 64) - word * 64;
    const uint64_t begin_mask = UINT64_MAX << word_begin;
    const uint64_t end_mask =
        word_end == 64 ? UINT64_MAX : (UINT64_C(1) << word_end) - 1;
    group->occupied[word] |= begin_mask & end_mask;
  }
}

// Reserves every missing payload before range mutation begins. Failure may
// retain capacity but leaves the logical entry set unchanged.
static iree_status_t loom_runtime_backing_directory_reserve_range(
    loom_runtime_backing_directory_t* directory, iree_host_size_t begin,
    iree_host_size_t end) {
  while (begin < end) {
    const iree_host_size_t key = begin >> LOOM_RUNTIME_BACKING_GROUP_SHIFT;
    const uint16_t offset = begin & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
    const iree_host_size_t group_end =
        begin + iree_min(end - begin,
                         (iree_host_size_t)LOOM_RUNTIME_BACKING_GROUP_CAPACITY -
                             offset);
    loom_runtime_backing_bucket_t* bucket =
        directory->capacity ? loom_runtime_backing_find_bucket(directory, key)
                            : NULL;
    const loom_runtime_backing_group_t* group = bucket ? bucket->group : NULL;
    const uint16_t range_count = (uint16_t)(group_end - begin);
    const uint16_t occupied_count =
        group ? loom_runtime_backing_group_occupied_count(group, offset,
                                                          offset + range_count)
              : 0;
    const uint16_t required_capacity =
        (group ? group->count : 0) + range_count - occupied_count;
    if (!group || required_capacity > group->capacity) {
      IREE_RETURN_IF_ERROR(loom_runtime_backing_acquire_group(
          directory, key, required_capacity, &bucket));
    }
    begin = group_end;
  }
  return iree_ok_status();
}

iree_status_t loom_runtime_backing_directory_insert_range(
    loom_runtime_backing_directory_t* directory, iree_host_size_t begin,
    iree_host_size_t end) {
  IREE_RETURN_IF_ERROR(
      loom_runtime_backing_directory_reserve_range(directory, begin, end));
  while (begin < end) {
    const iree_host_size_t key = begin >> LOOM_RUNTIME_BACKING_GROUP_SHIFT;
    const uint16_t offset = begin & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
    const iree_host_size_t group_end =
        begin + iree_min(end - begin,
                         (iree_host_size_t)LOOM_RUNTIME_BACKING_GROUP_CAPACITY -
                             offset);
    loom_runtime_backing_bucket_t* bucket =
        loom_runtime_backing_find_bucket(directory, key);
    loom_runtime_backing_group_t* group = bucket->group;
    const uint16_t range_count = (uint16_t)(group_end - begin);
    const uint16_t occupied_count = loom_runtime_backing_group_occupied_count(
        group, offset, offset + range_count);
    const uint16_t missing_count = range_count - occupied_count;
    if (!missing_count) {
      begin = group_end;
      continue;
    }
    const uint16_t position = loom_runtime_backing_group_rank(group, offset);
    memmove(
        &group->entries[position + range_count],
        &group->entries[position + occupied_count],
        (group->count - position - occupied_count) * sizeof(*group->entries));

    // Expand the compact occupied prefix backward so each existing payload is
    // read before a later logical destination can overwrite its source.
    uint16_t source = position + occupied_count;
    for (iree_host_size_t index = group_end; index > begin;) {
      --index;
      const uint16_t logical_offset =
          index & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
      loom_runtime_backing_entry_t* entry =
          &group->entries[position + (index - begin)];
      if (group->occupied[logical_offset / 64] &
          (UINT64_C(1) << (logical_offset % 64))) {
        *entry = group->entries[--source];
      } else {
        *entry = (loom_runtime_backing_entry_t){.index = index};
      }
    }
    loom_runtime_backing_group_mark_occupied(group, offset,
                                             offset + range_count);
    group->count += missing_count;
    directory->count += missing_count;
    begin = group_end;
  }
  return iree_ok_status();
}

iree_status_t loom_runtime_backing_directory_insert(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index,
    loom_runtime_backing_entry_t** out_entry) {
  loom_runtime_backing_bucket_t* bucket = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_backing_acquire_group(
      directory, index >> LOOM_RUNTIME_BACKING_GROUP_SHIFT, 1, &bucket));
  loom_runtime_backing_group_t* group = bucket->group;
  const uint16_t offset = index & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
  const uint64_t bit = UINT64_C(1) << (offset % 64);
  const uint16_t position = loom_runtime_backing_group_rank(group, offset);
  if (!(group->occupied[offset / 64] & bit)) {
    return loom_runtime_backing_insert_entry(directory, bucket, index, position,
                                             out_entry);
  }
  *out_entry = &group->entries[position];
  return iree_ok_status();
}

iree_status_t loom_runtime_backing_directory_keep(
    loom_runtime_backing_directory_t* directory, iree_host_size_t begin,
    iree_host_size_t end, iree_host_size_t* out_added_count) {
  iree_host_size_t added_count = 0;
  iree_status_t status = iree_ok_status();
  while (begin < end && iree_status_is_ok(status)) {
    loom_runtime_backing_bucket_t* bucket = NULL;
    status = loom_runtime_backing_acquire_group(
        directory, begin >> LOOM_RUNTIME_BACKING_GROUP_SHIFT, 1, &bucket);
    if (iree_status_is_ok(status)) {
      loom_runtime_backing_group_t* group = bucket->group;
      const uint16_t offset = begin & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
      const iree_host_size_t group_end =
          begin +
          iree_min(
              end - begin,
              (iree_host_size_t)LOOM_RUNTIME_BACKING_GROUP_CAPACITY - offset);
      uint16_t position = loom_runtime_backing_group_rank(group, offset);
      const uint16_t count = (uint16_t)(group_end - begin);
      // Rank selects the first index at or after begin. With sorted unique
      // entries, this endpoint proves the whole requested span is present.
      if (position + count <= group->count &&
          group->entries[position + count - 1].index == group_end - 1) {
        for (uint16_t i = position; i < position + count; ++i) {
          added_count += !group->entries[i].keep;
          group->entries[i].keep = true;
        }
        begin = group_end;
      } else {
        for (; begin < group_end && iree_status_is_ok(status);
             ++begin, ++position) {
          loom_runtime_backing_entry_t* entry =
              position < group->count ? &group->entries[position] : NULL;
          if (!entry || entry->index != begin) {
            status = loom_runtime_backing_insert_entry(directory, bucket, begin,
                                                       position, &entry);
            group = bucket->group;
          }
          if (iree_status_is_ok(status)) {
            added_count += !entry->keep;
            entry->keep = true;
          }
        }
      }
    }
  }
  *out_added_count = added_count;
  return status;
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

const loom_runtime_backing_entry_t*
loom_runtime_backing_directory_range_next_const(
    const loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_range_cursor_t* cursor, iree_host_size_t* out_count) {
  *out_count = 0;
  if (!directory->count) {
    cursor->index = cursor->end;
    return NULL;
  }
  while (cursor->index < cursor->end) {
    const iree_host_size_t begin = cursor->index;
    const iree_host_size_t key = begin >> LOOM_RUNTIME_BACKING_GROUP_SHIFT;
    const uint16_t offset = begin & (LOOM_RUNTIME_BACKING_GROUP_CAPACITY - 1);
    const iree_host_size_t group_end =
        begin + iree_min(cursor->end - begin,
                         (iree_host_size_t)LOOM_RUNTIME_BACKING_GROUP_CAPACITY -
                             offset);
    cursor->index = group_end;
    const loom_runtime_backing_bucket_t* bucket =
        loom_runtime_backing_find_bucket(directory, key);
    if (!bucket->group) {
      continue;
    }
    const loom_runtime_backing_group_t* group = bucket->group;
    const uint16_t count = loom_runtime_backing_group_occupied_count(
        group, offset, offset + (uint16_t)(group_end - begin));
    if (count) {
      *out_count = count;
      return &group->entries[loom_runtime_backing_group_rank(group, offset)];
    }
  }
  return NULL;
}

loom_runtime_backing_entry_t* loom_runtime_backing_directory_range_next(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_range_cursor_t* cursor, iree_host_size_t* out_count) {
  return (loom_runtime_backing_entry_t*)
      loom_runtime_backing_directory_range_next_const(directory, cursor,
                                                      out_count);
}

static void loom_runtime_backing_directory_compact(
    loom_runtime_backing_directory_t* directory, bool reset_keep) {
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
        if (reset_keep) {
          group->entries[count].keep = false;
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

void loom_runtime_backing_directory_discard_empty(
    loom_runtime_backing_directory_t* directory) {
  loom_runtime_backing_directory_compact(directory, false);
}

void loom_runtime_backing_directory_begin_trim(
    loom_runtime_backing_directory_t* directory) {
  loom_runtime_backing_directory_compact(directory, true);
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
