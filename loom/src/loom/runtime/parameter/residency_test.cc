// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/tooling/flags.h"
#include "loom/runtime/parameter/weights_test_base.h"

IREE_FLAG_LIST(string, residency_device,
               "Three explicit device URIs for physical-owner qualification. "
               "By default, independent pools use the target's test device.");

namespace loom_runtime::testing {
namespace {

class ResidencyTest : public WeightsTest {
 protected:
  void SetUpPool() override {
    IREE_ASSERT_OK(
        loom_runtime_test_host_create(1, host_allocator_, &owned_host_));
    IREE_ASSERT_OK(
        loom_runtime_test_driver_acquire(owned_host_, &owned_driver_));
    const auto uris = FLAG_residency_device_list();
    ASSERT_TRUE(uris.count == 0 || uris.count == 3);
    std::array<loom_runtime_device_options_t, 3> options = {};
    const iree_string_view_t names[] = {IREE_SV("first"), IREE_SV("second"),
                                        IREE_SV("third")};
    for (size_t i = 0; i < options.size(); ++i) {
      options[i] = {
          .driver = owned_driver_,
          .uri = uris.count ? uris.values[i]
                 : WeightDeviceUri()[0]
                     ? iree_make_cstring_view(WeightDeviceUri())
                     : loom_runtime_test_device()->device_uri,
          .name = names[i],
      };
    }
    IREE_ASSERT_OK(loom_runtime_device_group_allocate(
        owned_host_, options.size(), options.data(),
        native_allocator_.allocator(), &group_));
    for (size_t i = 0; i < options.size(); ++i) {
      auto* device = loom_runtime_device_group_at(group_, i);
      const auto* sanitizer = iree_hal_device_spec_sanitizer(
          iree_hal_device_spec(loom_runtime_device_handle(device)));
      const loom_runtime_pool_options_t pool_options = {
          .elastic = GetParam(),
          .weight_slab_size = kSlabBytes,
          .state_slab_size = kSlabBytes,
          .memory_limit = GetParam() ? 2 * kSlabBytes : 0,
          .asan = iree_any_bit_set(sanitizer->flags,
                                   IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN)
                      ? sanitizer->asan.pool_options
                      : iree_hal_asan_pool_options_t{},
          .placement =
              uris.count
                  ? loom_runtime_pool_placement_t{LOOM_RUNTIME_POOL_PLACEMENT_DEVICE,
                                                  static_cast<uint32_t>(i)}
                  : loom_runtime_pool_placement_t{},
      };
      loom_runtime_pool_t* pool = nullptr;
      IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
          group_, {names[i], UINT64_C(1) << i}, &pool_options, &pool));
      IREE_ASSERT_OK(native_allocator_.Observe(device, pool));
    }
    device_owner_ = loom_runtime_device_group_at(group_, 0);
    storage_owner_ = loom_runtime_device_group_pool_at(group_, 0);
    IREE_ASSERT_OK(
        AllocateWeightPool(device_owner_, false, 0, 0, &consumer_pool_));
  }

  // The group retains all three pools until every owner execution retires.
  void TearDownPool() override {
    IREE_EXPECT_OK(loom_runtime_pool_free(consumer_pool_));
    consumer_pool_ = nullptr;
    storage_owner_ = nullptr;
  }

  iree_hal_pool_t* ConsumerPool() override {
    return loom_runtime_pool_storage(consumer_pool_);
  }

  virtual void PrepareBacking() {}

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(WeightsTest::SetUp());
    ASSERT_NO_FATAL_FAILURE(PrepareBacking());
    WriteCheckpoint(adapter_path_.path(), 0);
    owner_executions_[0] = execution_;
    for (size_t i = 0; i < owner_executions_.size(); ++i) {
      auto* owner = loom_runtime_device_group_at(group_, i);
      if (i) {
        IREE_ASSERT_OK(loom_runtime_device_create_execution(
            owner, loom_runtime_device_group_pool_at(group_, i),
            &owner_executions_[i]));
      }
      IREE_ASSERT_OK(Compile(i, "copy_raw"));
      const auto root = Root(i, 0);
      IREE_ASSERT_OK(loom_runtime_weights_allocate(
          loom_runtime_device_handle(owner),
          loom_runtime_device_transfer_queue(owner),
          loom_runtime_device_dispatch_queue(owner),
          loom_runtime_pool_storage(
              loom_runtime_device_group_pool_at(group_, i)),
          Pool(i),
          loom_runtime_pool_weight_geometry(
              loom_runtime_device_group_pool_at(group_, i)),
          nullptr,
          loom_runtime_pool_workspace(
              loom_runtime_device_group_pool_at(group_, i)),
          0, 1, &root, host_allocator_, &plans_[i]));
      // Raw preparation records file ranges only. Each plan's device, queues,
      // and pool establish its physical owner independently of this shared
      // reflection.
      IREE_ASSERT_OK(
          Prepare(plans_[i], {i == 2 ? adapter_path_.path() : path_.path()}));
      IREE_ASSERT_OK(loom_runtime_residency_group_allocate(
          Cache(i), 1, &plans_[i], host_allocator_, &parameter_groups_[i]));
      IREE_ASSERT_OK(loom_runtime_residency_allocate(
          Cache(i), owner_executions_[i], 1, &parameter_groups_[i],
          host_allocator_, &residencies_[i]));
    }
  }

  void TearDown() override {
    if (activation_gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(activation_gate_, 2, nullptr));
    }
    if (activation_) {
      if (activation_phase_ != LOOM_RUNTIME_ACTIVATION_COMPLETE) {
        IREE_EXPECT_OK(loom_runtime_activation_advance(
            activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL,
            &activation_phase_));
        if (activation_phase_ == LOOM_RUNTIME_ACTIVATION_LOADING) {
          activation_wake_.Await();
          IREE_EXPECT_STATUS_IS(
              failed_owner_status_,
              loom_runtime_activation_advance(
                  activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL,
                  &activation_phase_));
        }
      }
      loom_runtime_activation_free(activation_);
    }
    iree_hal_semaphore_release(activation_gate_);
    if (deferred_backing_) {
      deferred_backing_->CompleteReturns();
    }
    FinishState();
    if (auxiliary_execution_) {
      IREE_EXPECT_OK(loom_runtime_execution_drain(auxiliary_execution_));
    }
    loom_runtime_residency_free(auxiliary_residency_);
    loom_runtime_execution_release(auxiliary_execution_);
    for (size_t i = 1; i < owner_executions_.size(); ++i) {
      if (owner_executions_[i]) {
        IREE_EXPECT_STATUS_IS(
            i == 2 ? failed_owner_status_ : IREE_STATUS_OK,
            loom_runtime_execution_drain(owner_executions_[i]));
      }
    }
    for (auto* residency : extra_residencies_) {
      loom_runtime_residency_free(residency);
    }
    for (auto* buffer : state_buffers_) {
      IREE_EXPECT_OK(loom_runtime_virtual_buffer_free(buffer));
    }
    for (auto* readiness : state_ready_) {
      iree_hal_semaphore_release(readiness);
    }
    WeightsTest::TearDown();
    for (size_t i = 1; i < owner_executions_.size(); ++i) {
      loom_runtime_execution_release(owner_executions_[i]);
    }
    IREE_EXPECT_OK(loom_runtime_device_group_free(group_));
    iree_hal_driver_release(owned_driver_);
    loom_runtime_host_release(owned_host_);
    native_allocator_.AwaitReleased();
  }

  loom_runtime_memory_pool_t* Pool(size_t owner) {
    return owner ? loom_runtime_pool_memory(
                       loom_runtime_device_group_pool_at(group_, owner))
                 : pool_;
  }

  loom_runtime_residency_cache_t* Cache(size_t owner) {
    return owner ? loom_runtime_pool_residency_cache(
                       loom_runtime_device_group_pool_at(group_, owner))
                 : ResidencyCache();
  }

  void AllocateState(size_t slot, size_t owner, uint64_t bytes) {
    IREE_ASSERT_OK(loom_runtime_virtual_buffer_allocate(
        Pool(owner),
        loom_runtime_pool_state_geometry(
            loom_runtime_device_group_pool_at(group_, owner)),
        bytes, 256, &state_usage_[slot], &state_buffers_[slot]));
  }

  void FinishState() {
    for (size_t i = 0; i < state_operations_.size(); ++i) {
      auto*& operation = state_operations_[i];
      if (operation) {
        loom_runtime_memory_operation_await(operation);
        IREE_EXPECT_OK(loom_runtime_memory_operation_query_status(operation));
        loom_runtime_memory_operation_free(operation);
        operation = nullptr;
      }
    }
  }

  void RoundTripState(size_t slot, size_t owner, uint32_t pattern) {
    auto* buffer = loom_runtime_virtual_buffer_handle(state_buffers_[slot]);
    uint32_t actual = 0;
    const iree_hal_transfer_operation_t operations[] = {
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill =
                {
                    .target_buffer = buffer,
                    .length = sizeof(pattern),
                    .pattern = &pattern,
                    .pattern_length = sizeof(pattern),
                },
        },
        {
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
            .download =
                {
                    .source_buffer = buffer,
                    .target = &actual,
                    .length = sizeof(actual),
                },
        },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_runtime_execution_transfer(owner_executions_[owner],
                                                   IREE_ARRAYSIZE(operations),
                                                   operations, &completion));
    IREE_ASSERT_OK(
        loom_runtime_execution_wait(owner_executions_[owner], completion));
    EXPECT_EQ(actual, pattern);
  }

  void ExpectUnactivated() {
    for (size_t i = 0; i < residencies_.size(); ++i) {
      EXPECT_EQ(loom_runtime_residency_snapshot(residencies_[i]).pin_count, 0u);
      EXPECT_EQ(loom_runtime_weights_activity(plans_[i]).activations, 0u);
      EXPECT_EQ(loom_runtime_weights_statistics(plans_[i]).committed_bytes, 0u);
      EXPECT_EQ(loom_runtime_weights_statistics(plans_[i]).pending_bytes, 0u);
    }
  }

  void CreateActivation() {
    // The caller's membership array need not outlive allocation.
    const auto members = residencies_;
    IREE_ASSERT_OK(loom_runtime_activation_allocate(
        members.size(), members.data(),
        loom_runtime_host_proactor_pool(owned_host_),
        loom_runtime_test_weight_activation_wake_t::Notify, &activation_wake_,
        host_allocator_, &activation_));
  }

  void HoldOwner(size_t owner) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_runtime_device_handle(loom_runtime_device_group_at(group_, owner)),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &activation_gate_));
    uint64_t value = 1;
    IREE_ASSERT_OK(loom_runtime_execution_barrier(
        owner_executions_[owner], {1, &activation_gate_, &value}));
  }

  void ReadOwner(size_t owner) {
    IREE_ASSERT_OK(loom_runtime_residency_await(residencies_[owner]));
    std::array<int32_t, 8> output = {};
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffers_[owner][0],
                .target = output.data(),
                .length = sizeof(output),
            },
    };
    uint64_t value = 0;
    IREE_ASSERT_OK(loom_runtime_execution_transfer(owner_executions_[owner], 1,
                                                   &download, &value));
    IREE_ASSERT_OK(loom_runtime_execution_drain(owner_executions_[owner]));
    for (int32_t i = 0; i < 8; ++i) {
      EXPECT_EQ(output[i], 30 + i);
    }
  }

  void AuxiliaryProgress() {
    auto* owner = loom_runtime_device_group_at(group_, 0);
    auto*& execution = auxiliary_execution_;
    IREE_ASSERT_OK(loom_runtime_device_create_execution(
        owner, loom_runtime_device_group_pool_at(group_, 0), &execution));
    auto*& residency = auxiliary_residency_;
    IREE_ASSERT_OK(loom_runtime_residency_allocate(
        Cache(0), execution, 0, nullptr, host_allocator_, &residency));
    loom_runtime_residency_admission_t admission;
    IREE_ASSERT_OK(loom_runtime_residency_try_admit(1, &residency, &admission));
    ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
    const uint32_t pattern = 0x13579bdf;
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = output_buffer_,
                .length = sizeof(output_),
                .pattern = &pattern,
                .pattern_length = sizeof(pattern),
            },
    };
    uint64_t value = 0;
    IREE_ASSERT_OK(
        loom_runtime_execution_transfer(execution, 1, &fill, &value));
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = output_buffer_,
                .target = output_.data(),
                .length = sizeof(output_),
            },
    };
    IREE_ASSERT_OK(
        loom_runtime_execution_transfer(execution, 1, &download, &value));
    IREE_ASSERT_OK(loom_runtime_execution_drain(execution));
    for (int32_t value : output_) {
      EXPECT_EQ(static_cast<uint32_t>(value), pattern);
    }
    loom_runtime_residency_cancel_admission(residency);
    loom_runtime_residency_release(residency);
    loom_runtime_residency_free(residency);
    residency = nullptr;
    loom_runtime_execution_release(execution);
    execution = nullptr;
  }

  // Native allocation dependency outliving the complete case-owned group.
  NativeAllocator native_allocator_;
  // Three explicitly selected owners, independent of the process-wide device.
  loom_runtime_device_group_t* group_ = nullptr;
  // Caller bindings scoped to this group's first device, outside residency.
  loom_runtime_pool_t* consumer_pool_ = nullptr;
  // One application worker shared by all physical admission owners.
  loom_runtime_host_t* owned_host_ = nullptr;
  // Native driver outliving the complete application group.
  iree_hal_driver_t* owned_driver_ = nullptr;
  // The first execution is borrowed from the base; the remaining two are owned.
  std::array<loom_runtime_execution_t*, 3> owner_executions_ = {};
  // Independent auxiliary work, owned through failed-assertion cleanup too.
  loom_runtime_execution_t* auxiliary_execution_ = nullptr;
  // Auxiliary admission released only after its independent execution retires.
  loom_runtime_residency_t* auxiliary_residency_ = nullptr;
  // Borrowers of existing groups, destroyed before their parameter owners.
  std::vector<loom_runtime_residency_t*> extra_residencies_;
  // Independent pressure and request state, retired before physical pools.
  std::array<loom_runtime_virtual_buffer_t*, 2> state_buffers_ = {};
  // Stable accounting through each state reservation's native release.
  std::array<loom_runtime_storage_statistics_t, 2> state_usage_ = {};
  // Independently accepted caller-owned state union operations.
  std::array<loom_runtime_memory_operation_t*, 2> state_operations_ = {};
  // State readiness timelines belonging to their explicit physical owners.
  std::array<iree_hal_semaphore_t*, 2> state_ready_ = {};
  // Expected terminal readiness error on the last owner after file truncation.
  iree_status_code_t failed_owner_status_ = IREE_STATUS_OK;
  // Actual asynchronous warmup retained through its final readiness callback.
  loom_runtime_activation_t* activation_ = nullptr;
  // Last owner-observed phase, bounding cleanup after an assertion failure.
  loom_runtime_activation_phase_t activation_phase_ =
      LOOM_RUNTIME_ACTIVATION_WAITING;
  // Joins production callback publication with the outer owner thread.
  loom_runtime_test_weight_activation_wake_t activation_wake_;
  // Explicit native predecessor released by the owner or terminal cleanup.
  iree_hal_semaphore_t* activation_gate_ = nullptr;
};

TEST_P(ResidencyTest, WarmupDenialWithdrawsEveryOwnerWithoutStartingLoads) {
  if (!GetParam()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(AllocateState(0, 2, 2 * kSlabBytes));
  IREE_ASSERT_OK(
      loom_runtime_virtual_buffer_commit(state_buffers_[0], 0, 2 * kSlabBytes));
  ASSERT_NO_FATAL_FAILURE(CreateActivation());
  IREE_ASSERT_OK(
      loom_runtime_activation_advance(activation_, 0, &activation_phase_));
  EXPECT_EQ(activation_phase_, LOOM_RUNTIME_ACTIVATION_WAITING);
  ASSERT_NO_FATAL_FAILURE(ExpectUnactivated());
  for (auto* residency : residencies_) {
    EXPECT_TRUE(loom_runtime_residency_snapshot(residency).admission.waiting);
  }
  IREE_ASSERT_OK(loom_runtime_activation_advance(
      activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &activation_phase_));
  EXPECT_EQ(activation_phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
  for (auto* residency : residencies_) {
    EXPECT_FALSE(loom_runtime_residency_snapshot(residency).admission.waiting);
  }
  ASSERT_NO_FATAL_FAILURE(ExpectUnactivated());
  ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
}

TEST_P(ResidencyTest, WarmupKeepsAllPinsUntilTheLastOwnerIsReady) {
  ASSERT_NO_FATAL_FAILURE(HoldOwner(2));
  ASSERT_NO_FATAL_FAILURE(CreateActivation());
  IREE_ASSERT_OK(
      loom_runtime_activation_advance(activation_, 0, &activation_phase_));
  ASSERT_EQ(activation_phase_, LOOM_RUNTIME_ACTIVATION_LOADING);
  ASSERT_NO_FATAL_FAILURE(ReadOwner(0));
  ASSERT_NO_FATAL_FAILURE(ReadOwner(1));
  IREE_ASSERT_OK(loom_runtime_activation_advance(
      activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &activation_phase_));
  EXPECT_EQ(activation_phase_, LOOM_RUNTIME_ACTIVATION_LOADING);
  for (auto* residency : residencies_) {
    EXPECT_EQ(loom_runtime_residency_snapshot(residency).pin_count, 1u);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(activation_gate_, 1, nullptr));
  activation_wake_.Await();
  IREE_ASSERT_OK(loom_runtime_activation_advance(
      activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &activation_phase_));
  EXPECT_EQ(activation_phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
  for (auto* residency : residencies_) {
    EXPECT_EQ(loom_runtime_residency_snapshot(residency).pin_count, 0u);
  }
  loom_runtime_residency_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_residency_try_admit(
      residencies_.size(), residencies_.data(), &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    EXPECT_EQ(loom_runtime_weights_activity(plans_[i]).activations, 1u);
    loom_runtime_residency_cancel_admission(residencies_[i]);
    loom_runtime_residency_release(residencies_[i]);
  }
}

TEST_P(ResidencyTest, FailedWarmupKeepsAllPinsWhileAnEarlierOwnerIsHeld) {
  ASSERT_NO_FATAL_FAILURE(HoldOwner(0));
  std::filesystem::resize_file(adapter_path_.path(), 0);
  ASSERT_NO_FATAL_FAILURE(CreateActivation());
  auto status =
      loom_runtime_activation_advance(activation_, 0, &activation_phase_);
  if (iree_status_is_ok(status)) {
    ASSERT_EQ(activation_phase_, LOOM_RUNTIME_ACTIVATION_LOADING);
    activation_wake_.Await();
    status =
        loom_runtime_activation_advance(activation_, 0, &activation_phase_);
    failed_owner_status_ = iree_status_code(status);
  }
  EXPECT_NE(iree_status_code(status), IREE_STATUS_OK);
  iree_status_free(status);
  EXPECT_EQ(activation_phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
  for (auto* residency : residencies_) {
    EXPECT_EQ(loom_runtime_residency_snapshot(residency).pin_count, 1u);
    EXPECT_FALSE(loom_runtime_residency_snapshot(residency).admission.waiting);
  }
  loom_runtime_activation_free(activation_);
  activation_ = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_signal(activation_gate_, 1, nullptr));
  ASSERT_NO_FATAL_FAILURE(ReadOwner(0));
  ASSERT_NO_FATAL_FAILURE(ReadOwner(1));
  ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
}

TEST_P(ResidencyTest, LateOwnerDenialStartsNoEarlierLoads) {
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 2, 2 * kSlabBytes));
    IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(state_buffers_[0], 0,
                                                      2 * kSlabBytes));
    loom_runtime_residency_admission_t denied;
    IREE_ASSERT_OK(loom_runtime_residency_try_admit(
        residencies_.size(), residencies_.data(), &denied));
    ASSERT_EQ(denied, LOOM_RUNTIME_RESIDENCY_ADMISSION_CAPACITY);
    ExpectUnactivated();
    for (auto* residency : residencies_) {
      EXPECT_TRUE(loom_runtime_residency_snapshot(residency).admission.waiting);
    }
    ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
    IREE_ASSERT_OK(loom_runtime_virtual_buffer_free(state_buffers_[0]));
    state_buffers_[0] = nullptr;
    IREE_ASSERT_OK(loom_runtime_memory_pool_await(Pool(2)));
  }
  loom_runtime_residency_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_residency_try_admit(
      residencies_.size(), residencies_.data(), &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    loom_runtime_residency_cancel_admission(residencies_[i]);
    loom_runtime_residency_release(residencies_[i]);
  }
}

TEST_P(ResidencyTest, InvocationAcquisitionDeniesTheCompleteSetBeforeLoading) {
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 2, 2 * kSlabBytes));
    IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(state_buffers_[0], 0,
                                                      2 * kSlabBytes));
    bool admitted = true;
    IREE_ASSERT_OK(loom_runtime_residency_try_acquire(
        residencies_.size(), residencies_.data(), &admitted));
    EXPECT_FALSE(admitted);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          loom_runtime_residency_acquire(residencies_.size(),
                                                         residencies_.data()));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          loom_runtime_residency_acquire_wait(
                              residencies_.size(), residencies_.data()));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          loom_runtime_residency_activate(residencies_.size(),
                                                          residencies_.data()));
    ExpectUnactivated();
    for (auto* residency : residencies_) {
      EXPECT_FALSE(
          loom_runtime_residency_snapshot(residency).admission.waiting);
    }
    ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
    IREE_ASSERT_OK(loom_runtime_virtual_buffer_free(state_buffers_[0]));
    state_buffers_[0] = nullptr;
    // The cold wrapper joins accepted return work before retrying the full set.
  }
  IREE_ASSERT_OK(loom_runtime_residency_activate(residencies_.size(),
                                                 residencies_.data()));
  for (auto* residency : residencies_) {
    EXPECT_EQ(loom_runtime_residency_snapshot(residency).pin_count, 0u);
  }
  IREE_ASSERT_OK(loom_runtime_residency_acquire_wait(residencies_.size(),
                                                     residencies_.data()));
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    EXPECT_EQ(loom_runtime_weights_activity(plans_[i]).activations, 1u);
    loom_runtime_residency_release(residencies_[i]);
  }
}

TEST_P(ResidencyTest, NestedInvocationProgressesPastWaitingNewWork) {
  IREE_ASSERT_OK(loom_runtime_residency_acquire_wait(residencies_.size(),
                                                     residencies_.data()));
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 1, kSlabBytes));
    IREE_ASSERT_OK(
        loom_runtime_virtual_buffer_commit(state_buffers_[0], 0, kSlabBytes));
    loom_runtime_residency_t* waiting = nullptr;
    IREE_ASSERT_OK(loom_runtime_residency_allocate(
        Cache(1), owner_executions_[1], 1, &parameter_groups_[1],
        host_allocator_, &waiting));
    extra_residencies_.push_back(waiting);
    IREE_ASSERT_OK(
        loom_runtime_residency_prepare_working_set(waiting, kSlabBytes, 0, 0));
    loom_runtime_residency_admission_t admission;
    IREE_ASSERT_OK(loom_runtime_residency_try_admit(1, &waiting, &admission));
    ASSERT_NE(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
    ASSERT_TRUE(loom_runtime_residency_snapshot(waiting).admission.waiting);
  }
  loom_runtime_memory_range_t range = {};
  loom_runtime_residency_state_t state = {
      .owner = 2,
      .ranges = &range,
      .range_count = GetParam() ? 1u : 0u,
      .value = 1,
      .operation = &state_operations_[1],
  };
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(1, 2, kSlabBytes));
    range = {state_buffers_[1], 0, kSlabBytes};
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_runtime_device_handle(loom_runtime_device_group_at(group_, 2)),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &state_ready_[1]));
    state.readiness = state_ready_[1];
  }
  bool admitted = false;
  IREE_ASSERT_OK(loom_runtime_residency_try_acquire_state(
      residencies_.size(), residencies_.data(), 1, &state, &admitted));
  ASSERT_TRUE(admitted);
  FinishState();
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(RoundTripState(1, 2, 0x71903512));
  }
  for (size_t i = 0; i < residencies_.size(); ++i) {
    EXPECT_EQ(loom_runtime_residency_snapshot(residencies_[i]).pin_count, 2u);
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    EXPECT_EQ(loom_runtime_weights_activity(plans_[i]).activations, 1u);
    loom_runtime_residency_release(residencies_[i]);
    loom_runtime_residency_release(residencies_[i]);
  }
}

TEST_P(ResidencyTest, NonzeroStateOwnerCommitsOneUnionAfterCompleteAdmission) {
  std::array<loom_runtime_memory_range_t, 2> ranges = {};
  loom_runtime_residency_state_t state = {
      .owner = 2,
      .ranges = ranges.data(),
      .range_count = GetParam() ? ranges.size() : 0,
      .value = 1,
      .operation = &state_operations_[1],
  };
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(1, 2, 2 * kSlabBytes));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_runtime_device_handle(loom_runtime_device_group_at(group_, 2)),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &state_ready_[1]));
    state.readiness = state_ready_[1];
    ranges = {
        {{state_buffers_[1], 0, 32}, {state_buffers_[1], kSlabBytes, 32}}};
    loom_runtime_residency_admission_t denied;
    IREE_ASSERT_OK(loom_runtime_residency_try_admit_state(
        residencies_.size(), residencies_.data(), 1, &state, &denied));
    ASSERT_EQ(denied, LOOM_RUNTIME_RESIDENCY_ADMISSION_CAPACITY);
    EXPECT_EQ(state_operations_[1], nullptr);
    EXPECT_EQ(state_usage_[1].committed_bytes, 0u);
    ExpectUnactivated();
    ranges[1].offset = 16;
  }
  for (uint64_t pin = 1; pin <= 2; ++pin) {
    state.value = pin;
    loom_runtime_residency_admission_t admission;
    IREE_ASSERT_OK(loom_runtime_residency_try_admit_state(
        residencies_.size(), residencies_.data(), 1, &state, &admission));
    ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
    FinishState();
    if (GetParam()) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(state_ready_[1], pin,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
      EXPECT_EQ(state_usage_[1].committed_bytes, kSlabBytes);
    }
    for (size_t i = 0; i < residencies_.size(); ++i) {
      EXPECT_EQ(loom_runtime_residency_snapshot(residencies_[i]).pin_count,
                pin);
      EXPECT_EQ(loom_runtime_weights_activity(plans_[i]).activations, 1u);
      ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    }
  }
  for (auto* residency : residencies_) {
    loom_runtime_residency_cancel_admission(residency);
    loom_runtime_residency_release(residency);
    loom_runtime_residency_release(residency);
  }
}

TEST_P(ResidencyTest, MinimumStateFitsWithParametersWithoutDuplicateCredit) {
  if (GetParam()) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          loom_runtime_residency_prepare_working_set(
                              residencies_[0], 0, 0, 2 * kSlabBytes));
  }
  IREE_ASSERT_OK(loom_runtime_residency_prepare_working_set(residencies_[0], 0,
                                                            0, kSlabBytes));
  loom_runtime_memory_range_t range = {};
  loom_runtime_residency_state_t state = {
      .owner = 0,
      .ranges = &range,
      .range_count = GetParam() ? 1u : 0u,
      .value = 1,
      .operation = &state_operations_[0],
  };
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 0, kSlabBytes));
    range = {state_buffers_[0], 0, kSlabBytes};
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_runtime_device_handle(device_owner_),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &state_ready_[0]));
    state.readiness = state_ready_[0];
  }
  bool admitted = false;
  IREE_ASSERT_OK(loom_runtime_residency_try_acquire_state(
      1, residencies_.data(), 1, &state, &admitted));
  ASSERT_TRUE(admitted);
  FinishState();
  ASSERT_NO_FATAL_FAILURE(ReadOwner(0));
  if (GetParam()) {
    EXPECT_EQ(state_usage_[0].committed_bytes, kSlabBytes);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_runtime_residency_check_private_peak(residencies_[0], 16));
  }
  loom_runtime_residency_release(residencies_[0]);
}

TEST_P(ResidencyTest, MultipleStateOwnersAdmitBeforeAnyBackingOrActivation) {
  std::array<loom_runtime_memory_range_t, 2> first_ranges = {};
  std::array<loom_runtime_memory_range_t, 2> last_ranges = {};
  std::array<loom_runtime_residency_state_t, 2> states = {{
      {
          .owner = 0,
          .ranges = first_ranges.data(),
          .range_count = GetParam() ? first_ranges.size() : 0,
          .value = 1,
          .operation = &state_operations_[0],
      },
      {
          .owner = 2,
          .ranges = last_ranges.data(),
          .range_count = GetParam() ? last_ranges.size() : 0,
          .value = 1,
          .operation = &state_operations_[1],
      },
  }};
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 0, kSlabBytes));
    ASSERT_NO_FATAL_FAILURE(AllocateState(1, 2, 2 * kSlabBytes));
    first_ranges = {{{state_buffers_[0], 0, 32}, {state_buffers_[0], 16, 32}}};
    last_ranges = {
        {{state_buffers_[1], 0, 32}, {state_buffers_[1], kSlabBytes, 32}}};
    for (size_t i = 0; i < states.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          loom_runtime_device_handle(
              loom_runtime_device_group_at(group_, states[i].owner)),
          IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &state_ready_[i]));
      states[i].readiness = state_ready_[i];
    }
    loom_runtime_residency_admission_t admission;
    IREE_ASSERT_OK(loom_runtime_residency_try_admit_state(
        residencies_.size(), residencies_.data(), states.size(), states.data(),
        &admission));
    ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_CAPACITY);
    for (size_t i = 0; i < states.size(); ++i) {
      EXPECT_EQ(state_operations_[i], nullptr);
      EXPECT_EQ(state_usage_[i].committed_bytes, 0u);
      EXPECT_EQ(state_usage_[i].pending_bytes, 0u);
      uint64_t value = UINT64_MAX;
      IREE_ASSERT_OK(iree_hal_semaphore_query(state_ready_[i], &value));
      EXPECT_EQ(value, 0u);
    }
    ExpectUnactivated();
    last_ranges[1].offset = 16;
  }
  // Nested admission reuses warm parameters but independently publishes every
  // state's readiness, including when all its ranges are already backed.
  for (uint64_t pin = 1; pin <= 2; ++pin) {
    for (auto& state : states) {
      state.value = pin;
    }
    loom_runtime_residency_admission_t admission;
    IREE_ASSERT_OK(loom_runtime_residency_try_admit_state(
        residencies_.size(), residencies_.data(), states.size(), states.data(),
        &admission));
    ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
    FinishState();
    if (GetParam()) {
      for (size_t i = 0; i < states.size(); ++i) {
        IREE_ASSERT_OK(iree_hal_semaphore_wait(state_ready_[i], pin,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
        EXPECT_EQ(state_usage_[i].committed_bytes, kSlabBytes);
        ASSERT_NO_FATAL_FAILURE(RoundTripState(
            i, states[i].owner, 0x11223344u + static_cast<uint32_t>(i)));
      }
    }
    for (size_t i = 0; i < residencies_.size(); ++i) {
      EXPECT_EQ(loom_runtime_residency_snapshot(residencies_[i]).pin_count,
                pin);
      EXPECT_EQ(loom_runtime_weights_activity(plans_[i]).activations, 1u);
      ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    }
  }
  for (auto* residency : residencies_) {
    loom_runtime_residency_cancel_admission(residency);
    loom_runtime_residency_release(residency);
    loom_runtime_residency_release(residency);
  }
}

TEST_P(ResidencyTest, LaterStateSubmissionFailureRetainsEarlierMappingOwner) {
  if (!GetParam()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(AllocateState(0, 0, kSlabBytes));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_runtime_device_handle(loom_runtime_device_group_at(group_, 2)),
      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      &state_ready_[1]));
  iree_hal_semaphore_fail(
      state_ready_[1],
      iree_make_status(IREE_STATUS_CANCELLED, "cancelled state"));
  const loom_runtime_memory_range_t range = {state_buffers_[0], 0, 32};
  const loom_runtime_residency_state_t states[] = {
      {
          .owner = 0,
          .ranges = &range,
          .range_count = 1,
          .operation = &state_operations_[0],
      },
      {
          .owner = 2,
          .readiness = state_ready_[1],
          .value = 1,
          .operation = &state_operations_[1],
      },
  };
  bool admitted = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_runtime_residency_try_acquire_state(
                            residencies_.size(), residencies_.data(),
                            IREE_ARRAYSIZE(states), states, &admitted));
  EXPECT_FALSE(admitted);
  EXPECT_NE(state_operations_[0], nullptr);
  EXPECT_EQ(state_operations_[1], nullptr);
  ExpectUnactivated();
  FinishState();
  EXPECT_EQ(state_usage_[0].committed_bytes, kSlabBytes);
  EXPECT_EQ(state_usage_[0].pending_bytes, 0u);
  ASSERT_NO_FATAL_FAILURE(RoundTripState(0, 0, 0x55667788));
  for (auto* residency : residencies_) {
    loom_runtime_residency_cancel_admission(residency);
  }
}

TEST_P(ResidencyTest, OverlappingSetsShareFifoOrderAndAllowAuxiliaryProgress) {
  loom_runtime_residency_t* shared = nullptr;
  IREE_ASSERT_OK(loom_runtime_residency_allocate(Cache(1), owner_executions_[1],
                                                 1, &parameter_groups_[1],
                                                 host_allocator_, &shared));
  extra_residencies_.push_back(shared);
  loom_runtime_residency_t* first[] = {residencies_[0], residencies_[1]};
  loom_runtime_residency_t* second[] = {residencies_[2], shared};
  loom_runtime_residency_admission_t admission;
  if (GetParam()) {
    ASSERT_NO_FATAL_FAILURE(AllocateState(0, 1, 2 * kSlabBytes));
    IREE_ASSERT_OK(loom_runtime_virtual_buffer_commit(state_buffers_[0], 0,
                                                      2 * kSlabBytes));
    IREE_ASSERT_OK(loom_runtime_residency_try_admit(2, first, &admission));
    ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_CAPACITY);
    IREE_ASSERT_OK(loom_runtime_residency_try_admit(2, second, &admission));
    ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_DEFERRED);
    ExpectUnactivated();
    EXPECT_TRUE(loom_runtime_residency_snapshot(shared).admission.waiting);
    ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
    IREE_ASSERT_OK(loom_runtime_virtual_buffer_free(state_buffers_[0]));
    state_buffers_[0] = nullptr;
    IREE_ASSERT_OK(loom_runtime_memory_pool_await(Pool(1)));
    IREE_ASSERT_OK(loom_runtime_residency_try_admit(2, second, &admission));
    ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_DEFERRED);
  }
  IREE_ASSERT_OK(loom_runtime_residency_try_admit(2, first, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  for (auto* residency : first) {
    loom_runtime_residency_cancel_admission(residency);
  }
  IREE_ASSERT_OK(loom_runtime_residency_try_admit(2, second, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
  }
  for (auto* residency : second) {
    loom_runtime_residency_cancel_admission(residency);
    loom_runtime_residency_release(residency);
  }
  for (auto* residency : first) {
    loom_runtime_residency_release(residency);
  }
}

TEST_P(ResidencyTest, AcceptedOwnersSurviveLaterSourceFailure) {
  std::filesystem::resize_file(adapter_path_.path(), 0);
  loom_runtime_residency_admission_t admission;
  auto status = loom_runtime_residency_try_admit(
      residencies_.size(), residencies_.data(), &admission);
  // The I/O failure can arrive during activation or on its imported readiness.
  // Either result must keep all accepted owners protected until retirement.
  const bool imported = iree_status_is_ok(status);
  if (imported) {
    ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
    status = loom_runtime_residency_await(residencies_[2]);
  }
  EXPECT_NE(iree_status_code(status), IREE_STATUS_OK);
  failed_owner_status_ = imported ? iree_status_code(status) : IREE_STATUS_OK;
  iree_status_free(status);
  for (auto* residency : residencies_) {
    EXPECT_EQ(loom_runtime_residency_snapshot(residency).pin_count, 1u);
  }
  ASSERT_NO_FATAL_FAILURE(ReadOwner(0));
  ASSERT_NO_FATAL_FAILURE(ReadOwner(1));
  ASSERT_NO_FATAL_FAILURE(AuxiliaryProgress());
  // The base owner releases all consumer roots, joins actual weight-view
  // retirement, and only then frees residencies and physical backing.
}

INSTANTIATE_TEST_SUITE_P(Owners, ResidencyTest, ::testing::Values(false, true));

class DeferredResidencyTest : public ResidencyTest {
 protected:
  void PrepareBacking() override { UseDeferredBacking(); }
};

TEST_P(DeferredResidencyTest, LateNativeReturnUndoesAllTemporaryProtection) {
  ASSERT_NO_FATAL_FAILURE(AllocateState(0, 0, 2 * kSlabBytes));
  IREE_ASSERT_OK(
      loom_runtime_virtual_buffer_commit(state_buffers_[0], 0, 2 * kSlabBytes));
  loom_runtime_virtual_buffer_begin_trim(state_buffers_[0]);
  IREE_ASSERT_OK(loom_runtime_virtual_buffer_trim(state_buffers_[0]));
  IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool_));
  deferred_backing_->DeferReturns();
  loom_runtime_memory_pool_trim(pool_, 0);
  deferred_backing_->WaitForReturn();
  IREE_ASSERT_OK(
      loom_runtime_residency_prepare_working_set(residencies_[1], 0, 16, 0));
  loom_runtime_residency_t* owners[] = {residencies_[1], residencies_[2],
                                        residencies_[0]};
  loom_runtime_residency_admission_t admission;
  IREE_ASSERT_OK(loom_runtime_residency_try_admit(3, owners, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_DEFERRED);
  ExpectUnactivated();
  // A parameter-free peer can use the first owner's entire workspace budget.
  // Any output credit retained by the denied attempt would reject it.
  loom_runtime_residency_t* independent = nullptr;
  IREE_ASSERT_OK(loom_runtime_residency_allocate(Cache(1), owner_executions_[1],
                                                 0, nullptr, host_allocator_,
                                                 &independent));
  extra_residencies_.push_back(independent);
  IREE_ASSERT_OK(loom_runtime_residency_prepare_working_set(
      independent, 2 * kSlabBytes, 0, 0));
  IREE_ASSERT_OK(loom_runtime_residency_try_admit(1, &independent, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  loom_runtime_residency_release(independent);
  deferred_backing_->CompleteReturns();
  IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_runtime_residency_try_admit(3, owners, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  for (size_t i = 0; i < residencies_.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(ReadOwner(i));
    loom_runtime_residency_cancel_admission(residencies_[i]);
    loom_runtime_residency_release(residencies_[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(Owners, DeferredResidencyTest,
                         ::testing::Values(true));

}  // namespace
}  // namespace loom_runtime::testing
