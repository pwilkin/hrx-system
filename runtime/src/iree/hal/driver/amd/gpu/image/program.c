// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/image/program.h"

#include "amdf/gpu_target.h"

iree_status_t iree_hal_amd_gpu_program_select(
    amdf_gpu_instruction_set_t supported, uint32_t variant_count,
    const iree_hal_amd_gpu_program_variant_t* variants,
    iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel) {
  const uint32_t index = amdf_gpu_target_select_instructions(
      supported, variant_count, variants, sizeof(*variants));
  if (index == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no compatible native builtin instructions");
  }
  *out_code = variants[index].code;
  *out_kernel = variants[index].kernel;
  return iree_ok_status();
}
