// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_REQUEST_POLICY_H_
#define LOOM_SERVE_SERVICE_TEXT_REQUEST_POLICY_H_

#include "iree/vm/buffer.h"
#include "loom/runtime/program/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable source policy shared by every request for one model. The linked
// program contains only request-local source functions and stateless native
// capabilities; mutable VM state belongs to request executions below.
typedef struct loom_serve_text_request_policy_t {
  // Canonical buffer identity borrowed from the model environment.
  iree_vm_ref_types_t types;
  // Owned request-only program shared by independent executions.
  loom_runtime_program_t* program;
  // Owned source-validated deployment defaults, opaque to native code.
  iree_vm_buffer_t* defaults;
  // Program-bound request configuration/validation shared by executions.
  iree_vm_function_ref_t request_options;
} loom_serve_text_request_policy_t;

// Mutable source execution owned and serialized by one request.
typedef struct loom_serve_text_request_execution_t {
  // Owned process and invocation for the policy program.
  loom_runtime_program_execution_t* program_execution;
  // Invocation borrowing program_execution storage.
  iree_vm_invocation_t* invocation;
  // Process-bound request configuration/validation function.
  iree_vm_function_t request_options;
} loom_serve_text_request_execution_t;

// Takes ownership of |program| on every outcome, resolves the request contract,
// and evaluates deployment defaults once. Known-input policy binds the token-
// validating prepare_request contract; chat binds request_options. Options has
// owned backing that source may retain. The environment outlives executions.
iree_status_t loom_serve_text_request_policy_initialize(
    iree_vm_environment_t* environment, loom_runtime_program_t* program,
    iree_vm_buffer_t* options, bool known_input,
    loom_serve_text_request_policy_t* out_policy);

// Releases frozen source values and the request-only linked program after every
// execution has retired. Partial initialization is accepted.
void loom_serve_text_request_policy_deinitialize(
    loom_serve_text_request_policy_t* policy);

// Allocates and binds one independent execution. The policy and its program
// remain live until the execution is deinitialized.
iree_status_t loom_serve_text_request_execution_initialize(
    const loom_serve_text_request_policy_t* policy,
    loom_serve_text_request_execution_t* out_execution);

// Releases a partial or complete request execution after its call retires.
void loom_serve_text_request_execution_deinitialize(
    loom_serve_text_request_execution_t* execution);

// Resolves one request against frozen defaults on an exclusive execution.
// Known-input callers pass immutable tokens; chat callers pass null. Source
// may retain the cloned body and tokens. Returned opaque options are owned.
iree_status_t loom_serve_text_request_options(
    const loom_serve_text_request_policy_t* policy,
    loom_serve_text_request_execution_t* execution, iree_string_view_t body,
    uint64_t request_id, iree_vm_buffer_t* tokens,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_REQUEST_POLICY_H_
