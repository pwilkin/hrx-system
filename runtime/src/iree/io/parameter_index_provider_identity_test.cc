// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/io/parameter_index_provider_test_base.h"

namespace iree::io::testing {

TEST_F(ParameterIndexProviderTest,
       ResolvedIdentitySurvivesGrowthAndCallerIndexRelease) {
  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  std::string key = "weights";
  std::string metadata = "eight bytes";
  const iree_io_parameter_index_entry_t entry = {
      .key = iree_make_cstring_view(key.c_str()),
      .metadata = iree_make_const_byte_span(metadata.data(), metadata.size()),
      .length = 8,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT,
      .storage = {.splat = {.pattern_length = 1, .pattern = {0xA7}}},
  };
  IREE_ASSERT_OK(iree_io_parameter_index_add(index.get(), &entry));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
      iree_allocator_system(), provider.out()));
  const auto* parameter = Resolve(provider.get(), IREE_SV("weights"));
  key.assign(key.size(), 'x');
  metadata.assign(metadata.size(), 'x');
  for (size_t i = 0; i < 1024; ++i) {
    const std::string other = "other" + std::to_string(i);
    IREE_ASSERT_OK(AddSplatEntry(
        index.get(), iree_make_cstring_view(other.c_str()), 16, 0xBC));
  }
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("weights"), 16, 0xBC));
  EXPECT_EQ(Resolve(provider.get(), IREE_SV("weights")), parameter);
  index.reset();
  const auto info = iree_io_parameter_provider_query(provider.get(), parameter);
  EXPECT_TRUE(iree_string_view_equal(info.key, IREE_SV("weights")));
  EXPECT_EQ(info.length, 8u);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(info.metadata.data),
                        info.metadata.data_length),
            "eight bytes");

  Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
  buffer.reset(AllocateTransferBuffer(transfer_pool(), 8));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate, done;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
  auto* gate_ptr = gate.get();
  auto* done_ptr = done.get();
  uint64_t value = 1;
  ParameterRequest request = {parameter, {0, 0, 8}};
  IREE_ASSERT_OK(iree_io_parameter_provider_gather(
      provider.get(), device(), queue(), {1, &gate_ptr, &value},
      {1, &done_ptr, &value}, buffer.get(), 1,
      {EnumerateParameterRequest, &request}));
  provider.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.get(), value, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      done.get(), value, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectBufferBytes(buffer.get(), 8, 0xA7);
}

TEST_F(ParameterIndexProviderTest, ResolutionChecksScopeAndMissingNames) {
  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("weights"), 8, 0xA7));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
      iree_allocator_system(), provider.out()));
  const auto* parameter = Resolve(provider.get(), IREE_SV("weights"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_io_parameter_provider_resolve(provider.get(), IREE_SV("other"),
                                         IREE_SV("weights"), &parameter));
  EXPECT_EQ(parameter, nullptr);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_io_parameter_provider_resolve(provider.get(), IREE_SV("model"),
                                         IREE_SV("missing"), &parameter));
  EXPECT_EQ(parameter, nullptr);
}

TEST_F(ParameterIndexProviderTest, ScalarResolutionFailureFailsCompletion) {
  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("weights"), 8, 0xA7));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
      iree_allocator_system(), provider.out()));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
  buffer.reset(AllocateTransferBuffer(transfer_pool(), 8));
  for (bool write : {false, true}) {
    for (bool missing_scope : {false, true}) {
      SCOPED_TRACE(write);
      SCOPED_TRACE(missing_scope);
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> done;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
      auto* done_ptr = done.get();
      uint64_t value = 1;
      const auto scope = missing_scope ? IREE_SV("other") : IREE_SV("model");
      const auto key = missing_scope ? IREE_SV("weights") : IREE_SV("missing");
      if (write) {
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_NOT_FOUND,
            iree_io_parameter_provider_write(
                provider.get(), device(), queue(),
                iree_hal_semaphore_list_empty(), {1, &done_ptr, &value},
                buffer.get(), 0, scope, key, 0, 8));
      } else {
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_NOT_FOUND,
            iree_io_parameter_provider_read(provider.get(), device(), queue(),
                                            iree_hal_semaphore_list_empty(),
                                            {1, &done_ptr, &value}, scope, key,
                                            0, buffer.get(), 0, 8));
      }
      uint64_t observed = 0;
      IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                            iree_hal_semaphore_query(done.get(), &observed));
    }
  }
}

TEST_F(ParameterIndexProviderTest, ScalarTransfersRetainResolvedPayload) {
  uint8_t bytes[8] = {};
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(bytes, sizeof(bytes)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      file.out()));
  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("weights"), 0, 8, file.get()));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
      iree_allocator_system(), provider.out()));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> source, target;
  source.reset(AllocateTransferBuffer(transfer_pool(), 8));
  target.reset(AllocateTransferBuffer(transfer_pool(), 8));
  const uint8_t expected[] = {1, 3, 5, 7, 9, 11, 13, 15};
  WriteBufferBytes(source.get(), iree_make_const_byte_span(expected, 8));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate, done;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
  auto* gate_ptr = gate.get();
  auto* done_ptr = done.get();
  uint64_t write_value = 1, read_value = 2;
  IREE_ASSERT_OK(iree_io_parameter_provider_write(
      provider.get(), device(), queue(), {1, &gate_ptr, &write_value},
      {1, &done_ptr, &write_value}, source.get(), 0, IREE_SV("model"),
      IREE_SV("weights"), 0, 8));
  IREE_ASSERT_OK(iree_io_parameter_provider_read(
      provider.get(), device(), queue(), {1, &done_ptr, &write_value},
      {1, &done_ptr, &read_value}, IREE_SV("model"), IREE_SV("weights"), 0,
      target.get(), 0, 8));
  provider.reset();
  index.reset();
  file.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.get(), write_value, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(done.get(), read_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectBufferBytesEqual(target.get(), iree_make_const_byte_span(expected, 8));
}

}  // namespace iree::io::testing
