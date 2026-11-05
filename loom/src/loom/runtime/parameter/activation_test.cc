// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/activation.h"

#include <condition_variable>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/device/device.h"

namespace {

// Uses the real task HAL and proactor so CPU sanitizers observe the actual
// readiness publication. Weighted GPU output and pin pressure are exercised by
// weights_test; this parameter-free group isolates callback storage lifetime.
class ActivationTest
    : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t>> {
 protected:
  static void Wake(void* user_data) {
    auto* self = static_cast<ActivationTest*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->ready_ = true;
    self->condition_.notify_one();
  }

  void AwaitWake() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return ready_; });
  }

  void SetUp() override {
    IREE_ASSERT_OK(loom_runtime_host_create(std::get<0>(GetParam()), nullptr,
                                            iree_allocator_system(), &host_));
    const loom_runtime_device_options_t options = {
        .driver = loom_runtime_host_task_driver(host_),
        .uri = IREE_SV("task"),
    };
    std::vector<loom_runtime_device_options_t> devices(std::get<1>(GetParam()),
                                                       options);
    std::vector<std::string> names(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
      names[i] = "owner" + std::to_string(i);
      devices[i].name = iree_make_cstring_view(names[i].c_str());
    }
    IREE_ASSERT_OK(loom_runtime_device_group_allocate(
        host_, devices.size(), devices.data(), iree_allocator_system(),
        &group_));
    executions_.resize(devices.size());
    residencies_.resize(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
      auto* owner = loom_runtime_device_group_at(group_, i);
      const loom_runtime_pool_options_t pool_options = {};
      loom_runtime_pool_t* pool = nullptr;
      IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
          group_, {devices[i].name, UINT64_C(1) << i}, &pool_options, &pool));
      IREE_ASSERT_OK(
          loom_runtime_device_create_execution(owner, pool, &executions_[i]));
      IREE_ASSERT_OK(loom_runtime_residency_allocate(
          loom_runtime_pool_residency_cache(pool), executions_[i], 0, nullptr,
          iree_allocator_system(), &residencies_[i]));
    }
    auto* device = loom_runtime_device_group_at(group_, devices.size() - 1);
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_runtime_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
    uint64_t value = 1;
    IREE_ASSERT_OK(loom_runtime_execution_barrier(executions_.back(),
                                                  {1, &gate_, &value}));
    IREE_ASSERT_OK(loom_runtime_activation_allocate(
        residencies_.size(), residencies_.data(),
        loom_runtime_host_proactor_pool(host_), Wake, this,
        iree_allocator_system(), &activation_));
  }

  void TearDown() override {
    if (gate_ && expected_code_ == IREE_STATUS_OK) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    if (activation_) {
      if (phase_ != LOOM_RUNTIME_ACTIVATION_COMPLETE) {
        IREE_EXPECT_OK(loom_runtime_activation_advance(
            activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &phase_));
        if (phase_ == LOOM_RUNTIME_ACTIVATION_LOADING) {
          AwaitWake();
          IREE_EXPECT_OK(loom_runtime_activation_advance(
              activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &phase_));
        }
      }
      loom_runtime_activation_free(activation_);
    }
    for (size_t i = 0; i < executions_.size(); ++i) {
      if (executions_[i]) {
        IREE_EXPECT_STATUS_IS(
            i + 1 == executions_.size() ? expected_code_ : IREE_STATUS_OK,
            loom_runtime_execution_drain(executions_[i]));
      }
      loom_runtime_residency_free(residencies_[i]);
      loom_runtime_execution_release(executions_[i]);
    }
    iree_hal_semaphore_release(gate_);
    IREE_EXPECT_OK(loom_runtime_device_group_free(group_));
    loom_runtime_host_release(host_);
  }

  // Shared task workers and proactor pool, outliving every model borrower.
  loom_runtime_host_t* host_ = nullptr;
  // Native device and shared residency cache owner.
  loom_runtime_device_group_t* group_ = nullptr;
  // Independent model timelines; the final member has a held predecessor.
  std::vector<loom_runtime_execution_t*> executions_;
  // Complete parameter-free membership used by the real activation owner.
  std::vector<loom_runtime_residency_t*> residencies_;
  // Explicit dependency; only the outer owner publishes its completion.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Warmup operation whose final callback must precede destruction.
  loom_runtime_activation_t* activation_ = nullptr;
  // Last owner-observed phase, also bounding partial-test cleanup.
  loom_runtime_activation_phase_t phase_ = LOOM_RUNTIME_ACTIVATION_WAITING;
  // Synchronizes the outer test with the production callback's wake.
  std::mutex mutex_;
  // Outer test wait, never executed on the task pool.
  std::condition_variable condition_;
  // The warmup operation has requested an owner turn.
  bool ready_ = false;
  // Expected native timeline result after explicit dependency failure.
  iree_status_code_t expected_code_ = IREE_STATUS_OK;
};

TEST_P(ActivationTest, UnstartedCancellationHasNoCallbackOrActivation) {
  IREE_ASSERT_OK(loom_runtime_activation_advance(
      activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
  EXPECT_FALSE(loom_runtime_residency_snapshot(residencies_.back()).active);
  std::lock_guard<std::mutex> lock(mutex_);
  EXPECT_FALSE(ready_);
}

TEST_P(ActivationTest, SubmittedCancellationJoinsRealReadinessPublication) {
  IREE_ASSERT_OK(loom_runtime_activation_advance(activation_, 0, &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_LOADING);
  IREE_ASSERT_OK(loom_runtime_activation_advance(
      activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_LOADING);
  EXPECT_TRUE(loom_runtime_residency_snapshot(residencies_.back()).active);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    EXPECT_FALSE(ready_);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  AwaitWake();
  IREE_ASSERT_OK(loom_runtime_activation_advance(
      activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
}

TEST_P(ActivationTest, FailedReadinessJoinsTheCallbackAndPreservesFailure) {
  IREE_ASSERT_OK(loom_runtime_activation_advance(activation_, 0, &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_LOADING);
  expected_code_ = IREE_STATUS_DATA_LOSS;
  iree_hal_semaphore_fail(
      gate_, iree_make_status(expected_code_, "parameter predecessor failed"));
  AwaitWake();
  IREE_EXPECT_STATUS_IS(
      expected_code_, loom_runtime_activation_advance(activation_, 0, &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
  for (auto* residency : residencies_) {
    EXPECT_EQ(loom_runtime_residency_snapshot(residency).pin_count, 1u);
  }
  // Final advance joins the proactor publication before storage is released;
  // the failed native execution remains owned until ordinary owner draining.
  loom_runtime_activation_free(activation_);
  activation_ = nullptr;
}

TEST_P(ActivationTest, CompletedReadinessNotifiesReplacementObserver) {
  IREE_ASSERT_OK(loom_runtime_activation_advance(activation_, 0, &phase_));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  AwaitWake();
  // Detachment joins the old callback. Its payload can now be replaced even
  // though the owner has not consumed the completed operation yet.
  loom_runtime_activation_set_observer(activation_, nullptr, nullptr);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ready_ = false;
  }
  loom_runtime_activation_set_observer(activation_, Wake, this);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    EXPECT_TRUE(ready_);
  }
  IREE_ASSERT_OK(loom_runtime_activation_advance(activation_, 0, &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
}

TEST_P(ActivationTest, PendingReadinessUsesReplacementObserver) {
  IREE_ASSERT_OK(loom_runtime_activation_advance(activation_, 0, &phase_));
  loom_runtime_activation_set_observer(activation_, nullptr, nullptr);
  struct Observer {
    // Synchronizes the replacement owner's completion predicate.
    std::mutex mutex;
    // Wakes the outer test thread, never a task worker.
    std::condition_variable condition;
    // Published only by the new observer, under its mutex.
    bool ready = false;
  } observer;
  loom_runtime_activation_set_observer(
      activation_,
      [](void* payload) {
        auto* observer = static_cast<Observer*>(payload);
        std::lock_guard<std::mutex> lock(observer->mutex);
        observer->ready = true;
        observer->condition.notify_one();
      },
      &observer);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  {
    std::unique_lock<std::mutex> lock(observer.mutex);
    observer.condition.wait(lock, [&observer] { return observer.ready; });
  }
  loom_runtime_activation_set_observer(activation_, nullptr, nullptr);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    EXPECT_FALSE(ready_);
  }
  IREE_ASSERT_OK(loom_runtime_activation_advance(activation_, 0, &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
}

TEST_P(ActivationTest, NullableSlotCreatesAndRetiresOneWarmup) {
  IREE_ASSERT_OK(loom_runtime_activation_advance(
      activation_, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, &phase_));
  loom_runtime_activation_free(activation_);
  activation_ = nullptr;
  IREE_ASSERT_OK(loom_runtime_activation_poll(
      &activation_, residencies_.back(), loom_runtime_host_proactor_pool(host_),
      Wake, this, 0, iree_allocator_system(), &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_LOADING);
  auto* pending = activation_;
  ASSERT_NE(pending, nullptr);
  IREE_ASSERT_OK(loom_runtime_activation_poll(
      &activation_, residencies_.back(), loom_runtime_host_proactor_pool(host_),
      Wake, this, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, iree_allocator_system(),
      &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_LOADING);
  EXPECT_EQ(activation_, pending);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  AwaitWake();
  IREE_ASSERT_OK(loom_runtime_activation_poll(
      &activation_, residencies_.back(), loom_runtime_host_proactor_pool(host_),
      Wake, this, LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, iree_allocator_system(),
      &phase_));
  EXPECT_EQ(phase_, LOOM_RUNTIME_ACTIVATION_COMPLETE);
  EXPECT_EQ(activation_, nullptr);
}

INSTANTIATE_TEST_SUITE_P(Workers, ActivationTest,
                         ::testing::Combine(::testing::Values(1u, 4u),
                                            ::testing::Values(1u, 3u)));

TEST(ActivationSlotTest, EmptyCancellationNeedsNoResourcesOrAllocation) {
  loom_runtime_activation_t* activation = nullptr;
  loom_runtime_activation_phase_t phase = LOOM_RUNTIME_ACTIVATION_WAITING;
  IREE_ASSERT_OK(loom_runtime_activation_poll(
      &activation, nullptr, nullptr, nullptr, nullptr,
      LOOM_RUNTIME_ACTIVATION_FLAG_CANCEL, iree_allocator_null(), &phase));
  EXPECT_EQ(phase, LOOM_RUNTIME_ACTIVATION_COMPLETE);
  EXPECT_EQ(activation, nullptr);
}

TEST(ActivationSlotTest, AllocationFailurePreservesEmptySlot) {
  loom_runtime_activation_t* activation = nullptr;
  loom_runtime_activation_phase_t phase = LOOM_RUNTIME_ACTIVATION_WAITING;
  const iree_allocator_t failing = {
      nullptr, [](void*, iree_allocator_command_t, const void*, void**) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_runtime_activation_poll(&activation, nullptr, nullptr, nullptr,
                                   nullptr, 0, failing, &phase));
  EXPECT_EQ(phase, LOOM_RUNTIME_ACTIVATION_COMPLETE);
  EXPECT_EQ(activation, nullptr);
}

}  // namespace
