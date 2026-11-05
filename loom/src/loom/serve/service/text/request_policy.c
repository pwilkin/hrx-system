// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/request_policy.h"

#include <string.h>

#include "iree/vm/sync.h"

void loom_serve_text_request_policy_deinitialize(
    loom_serve_text_request_policy_t* policy) {
  iree_vm_buffer_release(policy->defaults);
  loom_runtime_program_free(policy->program);
  memset(policy, 0, sizeof(*policy));
}

iree_status_t loom_serve_text_request_policy_initialize(
    iree_vm_environment_t* environment, loom_runtime_program_t* program,
    iree_vm_buffer_t* options, bool known_input,
    loom_serve_text_request_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  out_policy->program = program;
  iree_status_t status = iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &out_policy->types);
  const iree_vm_signature_type_t buffer = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_REF,
      .value.ref = out_policy->types.buffer};
  const iree_vm_signature_type_t i64 = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR,
      .value.scalar = IREE_VM_SCALAR_TYPE_I64};
  const iree_vm_signature_type_t option_arguments[] = {buffer, buffer, i64,
                                                       buffer};
  iree_vm_function_ref_t defaults_ref = {0};
  const loom_runtime_program_export_binding_t bindings[] = {
      {IREE_SVL("request_defaults"), {&buffer, 1}, {&buffer, 1}, &defaults_ref},
      {known_input ? IREE_SV("prepare_request") : IREE_SV("request_options"),
       {option_arguments, known_input ? 4 : 3},
       {&buffer, 1},
       &out_policy->request_options},
  };
  if (iree_status_is_ok(status)) {
    status = loom_runtime_program_resolve_service_exports(
        program, IREE_ARRAYSIZE(bindings), bindings);
  }
  loom_runtime_program_execution_t* execution = NULL;
  iree_vm_function_t defaults = {0};
  if (iree_status_is_ok(status)) {
    status = loom_runtime_program_execution_allocate(program, &execution);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_runtime_program_execution_bind(execution, defaults_ref, &defaults);
  }
  iree_vm_variant_t argument = {0};
  iree_vm_variant_t result = {0};
  if (iree_status_is_ok(status)) {
    argument =
        iree_vm_buffer_variant_from_ptr_borrowed(&out_policy->types, options);
    status =
        iree_vm_invoke(loom_runtime_program_execution_invocation(execution),
                       defaults, iree_vm_variant_span_from_ptr(&argument, 1),
                       iree_vm_variant_span_from_ptr(&result, 1));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&out_policy->types, &result,
                                                  &out_policy->defaults);
  }
  if (iree_status_is_ok(status) && !out_policy->defaults) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source request defaults must not be null");
  }
  iree_vm_variant_reset(&result);
  iree_vm_variant_reset(&argument);
  loom_runtime_program_execution_free(execution);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_request_policy_deinitialize(out_policy);
  }
  return status;
}

void loom_serve_text_request_execution_deinitialize(
    loom_serve_text_request_execution_t* execution) {
  loom_runtime_program_execution_free(execution->program_execution);
  memset(execution, 0, sizeof(*execution));
}

iree_status_t loom_serve_text_request_execution_initialize(
    const loom_serve_text_request_policy_t* policy,
    loom_serve_text_request_execution_t* out_execution) {
  memset(out_execution, 0, sizeof(*out_execution));
  iree_status_t status = loom_runtime_program_execution_allocate(
      policy->program, &out_execution->program_execution);
  if (iree_status_is_ok(status)) {
    out_execution->invocation = loom_runtime_program_execution_invocation(
        out_execution->program_execution);
    status = loom_runtime_program_execution_bind(
        out_execution->program_execution, policy->request_options,
        &out_execution->request_options);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_text_request_execution_deinitialize(out_execution);
  }
  return status;
}

iree_status_t loom_serve_text_request_options(
    const loom_serve_text_request_policy_t* policy,
    loom_serve_text_request_execution_t* execution, iree_string_view_t body,
    uint64_t request_id, iree_vm_buffer_t* tokens,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_options) {
  *out_options = NULL;
  iree_vm_buffer_t* input = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                           iree_make_const_byte_span(body.data, body.size), 1,
                           host_allocator, &input));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&policy->types,
                                               policy->defaults),
      iree_vm_buffer_variant_from_ptr_move(&policy->types, &input),
      iree_vm_variant_from_i64((int64_t)request_id),
      iree_vm_buffer_variant_from_ptr_borrowed(&policy->types, tokens)};
  iree_vm_variant_t result = {0};
  iree_status_t status =
      iree_vm_invoke(execution->invocation, execution->request_options,
                     iree_vm_variant_span_from_ptr(arguments, tokens ? 4 : 3),
                     iree_vm_variant_span_from_ptr(&result, 1));
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(&policy->types, &result,
                                                  out_options);
  }
  if (iree_status_is_ok(status) && !*out_options) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source request options must not be null");
  }
  iree_vm_variant_reset(&result);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}
