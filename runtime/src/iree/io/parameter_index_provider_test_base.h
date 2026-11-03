// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_PARAMETER_INDEX_PROVIDER_TEST_BASE_H_
#define IREE_IO_PARAMETER_INDEX_PROVIDER_TEST_BASE_H_

#include <cstring>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/task/registration/driver_module.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::io::testing {

template <typename T, void (*Release)(T*)>
class Ref {
 public:
  Ref() = default;
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  ~Ref() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) {
      Release(value_);
    }
    value_ = value;
  }

 private:
  // Owned reference released on reset or fixture teardown.
  T* value_ = nullptr;
};

inline iree_hal_device_group_t* CreateTaskDeviceGroup() {
  Ref<iree_async_proactor_pool_t, iree_async_proactor_pool_release>
      proactor_pool;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      proactor_pool.out()));

  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool.get();

  iree_hal_device_t* device = nullptr;
  iree_status_t status =
      iree_hal_create_device(registry, IREE_SV("task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  IREE_CHECK_OK(status);

  Ref<iree_async_frontier_tracker_t, iree_async_frontier_tracker_release>
      frontier_tracker;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      frontier_tracker.out()));

  iree_hal_device_group_t* device_group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker.get(), iree_allocator_system(), &device_group));
  iree_hal_device_release(device);
  return device_group;
}

inline iree_status_t AddSplatEntry(iree_io_parameter_index_t* index,
                                   iree_string_view_t key, uint64_t length,
                                   uint8_t pattern) {
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
  entry.storage.splat.pattern_length = 1;
  entry.storage.splat.pattern[0] = pattern;
  return iree_io_parameter_index_add(index, &entry);
}

inline iree_status_t AddFileEntry(iree_io_parameter_index_t* index,
                                  iree_string_view_t key, uint64_t offset,
                                  uint64_t length,
                                  iree_io_file_handle_t* handle) {
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
  entry.storage.file.handle = handle;
  entry.storage.file.offset = offset;
  return iree_io_parameter_index_add(index, &entry);
}

inline const iree_io_parameter_t* Resolve(
    iree_io_parameter_provider_t* provider, iree_string_view_t key) {
  const iree_io_parameter_t* parameter = nullptr;
  IREE_CHECK_OK(iree_io_parameter_provider_resolve(provider, IREE_SV("model"),
                                                   key, &parameter));
  return parameter;
}

typedef struct ParameterRequest {
  // Provider-owned identity resolved before transfer enumeration.
  const iree_io_parameter_t* parameter;
  // Parameter and buffer span.
  iree_io_parameter_span_t span;
} ParameterRequest;

inline iree_status_t EnumerateParameterRequest(
    void* user_data, iree_host_size_t i,
    const iree_io_parameter_t** out_parameter,
    iree_io_parameter_span_t* out_span) {
  const ParameterRequest* requests =
      static_cast<const ParameterRequest*>(user_data);
  *out_parameter = requests[i].parameter;
  *out_span = requests[i].span;
  return iree_ok_status();
}

inline iree_hal_buffer_t* AllocateTransferBuffer(
    iree_hal_pool_t* pool, iree_device_size_t byte_length) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(byte_length, 1),
      iree_infinite_timeout(), &buffer));
  return buffer;
}

inline void WriteBufferBytes(iree_hal_buffer_t* buffer,
                             iree_const_byte_span_t source) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_DISCARD, 0, source.data_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_WRITE, 0, source.data_length, &span));
  std::memcpy(span.data, source.data, source.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

inline void ExpectBufferBytes(iree_hal_buffer_t* buffer,
                              iree_device_size_t byte_length,
                              uint8_t expected_value) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, byte_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_READ, 0, byte_length, &span));
  for (iree_host_size_t i = 0; i < byte_length; ++i) {
    EXPECT_EQ(span.data[i], expected_value);
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

inline void ExpectBufferBytesEqual(iree_hal_buffer_t* buffer,
                                   iree_const_byte_span_t expected) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, expected.data_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_READ, 0, expected.data_length, &span));
  EXPECT_EQ(span.data_length, expected.data_length);
  EXPECT_EQ(0, std::memcmp(span.data, expected.data, expected.data_length));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

inline iree_status_t CreateTransferPool(iree_hal_device_group_t* device_group,
                                        bool admit_host_imports,
                                        iree_hal_pool_t** out_pool) {
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_pool_family_access_t family = {
      .family = iree_hal_device_queue_family(device, 0),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &family,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
              .properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                             IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK),
          },
  };
  const iree_hal_external_storage_class_t host_pointer_class =
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = host_pointer_class,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  if (admit_host_imports) {
    options.imports.class_count = 1;
    options.imports.classes = &import_class;
  }
  options.preferences.host = scope.host;
  return iree_hal_slab_pool_create(device_group, scope, &options,
                                   iree_allocator_system(), out_pool);
}

class ParameterIndexProviderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    device_group_ = CreateTaskDeviceGroup();
    IREE_ASSERT_OK(CreateTransferPool(device_group_,
                                      /*admit_host_imports=*/false,
                                      &transfer_pool_));
  }

  static void TearDownTestSuite() {
    iree_hal_pool_release(transfer_pool_);
    transfer_pool_ = nullptr;
    iree_hal_device_group_release(device_group_);
    device_group_ = nullptr;
  }

  static iree_hal_device_t* device() {
    return iree_hal_device_group_device_at(device_group_, 0);
  }

  static iree_hal_queue_t* queue() {
    return iree_hal_device_queue(device(), /*family_ordinal=*/0,
                                 /*queue_ordinal=*/0);
  }

  static iree_hal_device_group_t* device_group() { return device_group_; }

  static iree_hal_pool_t* transfer_pool() { return transfer_pool_; }

 private:
  // Shared task device whose queues retire every accepted operation.
  inline static iree_hal_device_group_t* device_group_ = nullptr;
  // Shared host-visible pool used by byte-accurate I/O witnesses.
  inline static iree_hal_pool_t* transfer_pool_ = nullptr;
};

}  // namespace iree::io::testing

#endif  // IREE_IO_PARAMETER_INDEX_PROVIDER_TEST_BASE_H_
