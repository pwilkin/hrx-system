// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/driver/amdgpu/aql_command_buffer.h"
#include "iree/hal/driver/amdgpu/host_queue_command_buffer_profiling_test_util.h"
#include "iree/hal/driver/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using namespace test;

class IndirectDispatchTest : public ::testing::TestWithParam<bool> {
 protected:
  static void SetUpTestSuite() {
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        iree_allocator_system(), &libhsa_);
    if (iree_status_is_unavailable(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  // HSA runtime retained while the suite's devices are live.
  static iree_hal_amdgpu_libhsa_t libhsa_;
  // Physical agents used by each test's logical device.
  static iree_hal_amdgpu_topology_t topology_;
};

iree_hal_amdgpu_libhsa_t IndirectDispatchTest::libhsa_;
iree_hal_amdgpu_topology_t IndirectDispatchTest::topology_;

static void ExpectDispatchGeometry(
    const std::vector<uint32_t>& output,
    const std::array<uint32_t, 3>& expected_counts) {
  constexpr iree_host_size_t kRecordLength = 6;
  const iree_host_size_t record_count =
      expected_counts[0] * expected_counts[1] * expected_counts[2];
  ASSERT_EQ(output.size(), record_count * kRecordLength);
  for (iree_host_size_t record = 0; record < record_count; ++record) {
    SCOPED_TRACE(record);
    for (iree_host_size_t field = 0; field < kRecordLength; ++field) {
      SCOPED_TRACE(field);
      ASSERT_EQ(output[record * kRecordLength + field],
                expected_counts[field % expected_counts.size()]);
    }
  }
}

TEST_P(IndirectDispatchTest, EveryWorkgroupObservesDeviceGeneratedGeometry) {
  const bool profile_dispatches = GetParam();
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options, &libhsa_, &topology_,
                                        iree_allocator_system()));
  const iree_hal_queue_family_t* queue_family =
      iree_hal_queue_family(test_device.queue());
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadCtsExecutable(
      queue_family, IREE_SV("indirect_dispatch_geometry_test.bin"),
      executable.out()));
  iree_hal_executable_function_t write_counts;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("write_dispatch_counts"), &write_counts));
  iree_hal_executable_function_t read_geometry;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("read_dispatch_geometry"), &read_geometry));

  // These are the device producer's dimensions, not host dispatch parameters.
  // Keep every axis nontrivial while bounding simulator execution work.
  constexpr std::array<uint32_t, 3> kExpectedCounts = {32, 2, 3};
  constexpr iree_host_size_t kRecordCount =
      kExpectedCounts[0] * kExpectedCounts[1] * kExpectedCounts[2];
  constexpr iree_host_size_t kRecordLength = 6;
  constexpr iree_host_size_t kGuardLength = 16;
  constexpr uint32_t kSentinel = 0xCDCDCDCDu;
  std::vector<uint32_t> output(kRecordCount * kRecordLength + 2 * kGuardLength,
                               kSentinel);
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), output.size() * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output_buffer, 0, output.data(),
                                           output.size() * sizeof(uint32_t)));
  Ref<iree_hal_buffer_t> count_buffer;
  IREE_ASSERT_OK(CreateHostVisibleIndirectParameterBuffer(
      test_device.allocator(), sizeof(uint32_t[3]), count_buffer.out()));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_zero(count_buffer, 0, IREE_HAL_WHOLE_BUFFER));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  if (profile_dispatches) {
    IREE_ASSERT_OK(
        profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                        CommandBufferProfileSinkAsBase(&sink)));
  }
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      queue_family, IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0,
      command_buffer.out()));
  ASSERT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t count_ref =
      iree_hal_make_buffer_ref(count_buffer, 0, sizeof(uint32_t[3]));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, write_counts,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), {.count = 1, .values = &count_ref},
      IREE_HAL_DISPATCH_FLAG_NONE));
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ,
  };
  const iree_hal_barrier_t barrier = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
                           IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barrier_count = 1,
      .memory_barriers = &memory_barrier,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer, &barrier));
  iree_hal_buffer_ref_t output_ref =
      iree_hal_make_buffer_ref(output_buffer, kGuardLength * sizeof(uint32_t),
                               kRecordCount * kRecordLength * sizeof(uint32_t));
  const iree_hal_dispatch_config_t config = {
      .workgroup_size = {32, 1, 1},
      .workgroup_count_ref = count_ref,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, read_geometry, config,
      iree_const_byte_span_empty(), {.count = 1, .values = &output_ref},
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> completion;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), completion.out()));
  iree_hal_semaphore_t* completion_ptr = completion.get();
  uint64_t completion_value = 1;
  const iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &completion_ptr,
      .payload_values = &completion_value,
  };
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(), signal_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(completion, completion_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  if (profile_dispatches) {
    IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
    IREE_ASSERT_OK(profiling.End());
    EXPECT_EQ(sink.dispatch_events.size(), 2u);
  }

  std::array<uint32_t, 3> counts = {};
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(count_buffer, 0, counts.data(), sizeof(counts)));
  EXPECT_EQ(counts, kExpectedCounts);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer, 0, output.data(),
                                          output.size() * sizeof(uint32_t)));
  for (iree_host_size_t i = 0; i < kGuardLength; ++i) {
    EXPECT_EQ(output[i], kSentinel);
    EXPECT_EQ(output[output.size() - kGuardLength + i], kSentinel);
  }
  for (iree_host_size_t record = 0; record < kRecordCount; ++record) {
    SCOPED_TRACE(record);
    for (iree_host_size_t field = 0; field < kRecordLength; ++field) {
      SCOPED_TRACE(field);
      ASSERT_EQ(output[kGuardLength + record * kRecordLength + field],
                kExpectedCounts[field % 3]);
    }
  }
}

TEST_P(IndirectDispatchTest, HostVisibleStaticGeometryUsesDirectPackets) {
  const bool profile_dispatches = GetParam();
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&options, &libhsa_, &topology_,
                                        iree_allocator_system()));
  const iree_hal_queue_family_t* queue_family =
      iree_hal_queue_family(test_device.queue());
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadCtsExecutable(
      queue_family, IREE_SV("indirect_dispatch_geometry_test.bin"),
      executable.out()));
  iree_hal_executable_function_t read_geometry;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("read_dispatch_geometry"), &read_geometry));

  constexpr std::array<uint32_t, 3> kExpectedCounts = {32, 2, 3};
  constexpr iree_host_size_t kRecordLength = 6;
  constexpr iree_host_size_t kRecordCount =
      kExpectedCounts[0] * kExpectedCounts[1] * kExpectedCounts[2];
  constexpr iree_host_size_t kOutputLength =
      kRecordCount * kRecordLength * sizeof(uint32_t);
  Ref<iree_hal_buffer_t> count_buffer;
  IREE_ASSERT_OK(CreateHostVisibleIndirectParameterBuffer(
      test_device.allocator(), sizeof(kExpectedCounts), count_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(count_buffer, /*target_offset=*/0,
                                           kExpectedCounts.data(),
                                           sizeof(kExpectedCounts)));
  iree_hal_buffer_ref_t count_ref =
      iree_hal_make_buffer_ref(count_buffer, 0, sizeof(kExpectedCounts));
  const iree_hal_dispatch_config_t config = {
      .workgroup_size = {32, 1, 1},
      .workgroup_count_ref = count_ref,
  };

  Ref<iree_hal_buffer_t> direct_output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kOutputLength, direct_output_buffer.out()));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_zero(direct_output_buffer, 0, IREE_HAL_WHOLE_BUFFER));
  iree_hal_buffer_ref_t direct_output_ref = iree_hal_make_buffer_ref(
      direct_output_buffer, /*offset=*/0, kOutputLength);

  const iree_hal_dispatch_flags_t conflicting_indirect_flags =
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_dispatch(
          test_device.queue(), iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable, read_geometry, config,
          iree_const_byte_span_empty(),
          {.count = 1, .values = &direct_output_ref}, /*barriers=*/nullptr,
          conflicting_indirect_flags));
  Ref<iree_hal_command_buffer_t> invalid_command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      queue_family, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0,
      invalid_command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(invalid_command_buffer));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_dispatch(
                            invalid_command_buffer, executable, read_geometry,
                            config, iree_const_byte_span_empty(),
                            {.count = 1, .values = &direct_output_ref},
                            conflicting_indirect_flags));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(invalid_command_buffer));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  if (profile_dispatches) {
    IREE_ASSERT_OK(
        profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                        CommandBufferProfileSinkAsBase(&sink)));
  }

  Ref<iree_hal_semaphore_t> direct_completion;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), direct_completion.out()));
  iree_hal_semaphore_t* direct_completion_ptr = direct_completion.get();
  uint64_t direct_completion_value = 1;
  const iree_hal_semaphore_list_t direct_signal_list = {
      .count = 1,
      .semaphores = &direct_completion_ptr,
      .payload_values = &direct_completion_value,
  };
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      test_device.queue(), iree_hal_semaphore_list_empty(), direct_signal_list,
      executable, read_geometry, config, iree_const_byte_span_empty(),
      {.count = 1, .values = &direct_output_ref}, /*barriers=*/nullptr,
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      direct_completion, direct_completion_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> direct_output(kRecordCount * kRecordLength);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(direct_output_buffer,
                                          /*source_offset=*/0,
                                          direct_output.data(), kOutputLength));
  ExpectDispatchGeometry(direct_output, kExpectedCounts);

  Ref<iree_hal_buffer_t> recorded_output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kOutputLength, recorded_output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(recorded_output_buffer, 0,
                                          IREE_HAL_WHOLE_BUFFER));
  iree_hal_buffer_ref_t recorded_output_ref = iree_hal_make_buffer_ref(
      recorded_output_buffer, /*offset=*/0, kOutputLength);
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      queue_family, IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0,
      command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, read_geometry, config,
      iree_const_byte_span_empty(),
      {.count = 1, .values = &recorded_output_ref},
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->block_count, 1u);
  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  ASSERT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const auto* dispatch_command = reinterpret_cast<
      const iree_hal_amdgpu_command_buffer_dispatch_command_t*>(command);
  EXPECT_TRUE(iree_all_bits_set(
      dispatch_command->dispatch_flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS |
          IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS));
  EXPECT_FALSE(iree_any_bit_set(
      dispatch_command->header.flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER));
  const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
      iree_hal_amdgpu_command_buffer_block_binding_sources_const(
          program->first_block);
  ASSERT_LT(dispatch_command->payload.binding_source_count,
            program->first_block->binding_source_count);
  const iree_hal_amdgpu_command_buffer_binding_source_t& parameter_source =
      binding_sources[dispatch_command->payload.binding_source_count];
  EXPECT_EQ(
      parameter_source.flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER |
          IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS);

  Ref<iree_hal_semaphore_t> recorded_completion;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), recorded_completion.out()));
  iree_hal_semaphore_t* recorded_completion_ptr = recorded_completion.get();
  uint64_t recorded_completion_value = 1;
  const iree_hal_semaphore_list_t recorded_signal_list = {
      .count = 1,
      .semaphores = &recorded_completion_ptr,
      .payload_values = &recorded_completion_value,
  };
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      recorded_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      recorded_completion, recorded_completion_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  if (profile_dispatches) {
    IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
    IREE_ASSERT_OK(profiling.End());
    EXPECT_EQ(sink.dispatch_events.size(), 2u);
  }

  std::vector<uint32_t> recorded_output(kRecordCount * kRecordLength);
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(recorded_output_buffer, /*source_offset=*/0,
                               recorded_output.data(), kOutputLength));
  ExpectDispatchGeometry(recorded_output, kExpectedCounts);

  // Zero-grid indirect dispatches are still queue operations: they must signal
  // completion without executing the kernel body. Exercise both submission
  // paths because Loom uses zero-grid launches to predicate command programs.
  constexpr std::array<uint32_t, 3> kZeroCounts = {0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(count_buffer, /*target_offset=*/0,
                                           kZeroCounts.data(),
                                           sizeof(kZeroCounts)));

  direct_completion_value = 2;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      test_device.queue(), iree_hal_semaphore_list_empty(), direct_signal_list,
      executable, read_geometry, config, iree_const_byte_span_empty(),
      {.count = 1, .values = &direct_output_ref}, /*barriers=*/nullptr,
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      direct_completion, direct_completion_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  std::vector<uint32_t> direct_zero_output(kRecordCount * kRecordLength);
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(direct_output_buffer, /*source_offset=*/0,
                               direct_zero_output.data(), kOutputLength));
  EXPECT_EQ(direct_zero_output, direct_output);

  recorded_completion_value = 2;
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      recorded_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      recorded_completion, recorded_completion_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  std::vector<uint32_t> recorded_zero_output(kRecordCount * kRecordLength);
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(recorded_output_buffer, /*source_offset=*/0,
                               recorded_zero_output.data(), kOutputLength));
  EXPECT_EQ(recorded_zero_output, recorded_output);
}

INSTANTIATE_TEST_SUITE_P(Replay, IndirectDispatchTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Timestamps" : "Ordinary";
                         });

}  // namespace
}  // namespace iree::hal::amdgpu
