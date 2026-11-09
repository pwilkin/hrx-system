// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_IMAGE_PROGRAM_H_
#define IREE_HAL_DRIVER_AMD_GPU_IMAGE_PROGRAM_H_

#include "amdf/gpu_target.h"
#include "iree/hal/driver/amd/gpu/image/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable compiler artifact; native selection owns ISA compatibility.
typedef struct iree_hal_amd_gpu_program_variant_t {
  // Instruction requirements of the bytes, in native selection vocabulary.
  amdf_gpu_instruction_set_t instructions;
  // Relocation-free instructions retained for the process lifetime.
  iree_const_byte_span_t code;
  // Complete native descriptor facts produced with those instructions.
  iree_hal_amd_gpu_image_kernel_t kernel;
} iree_hal_amd_gpu_program_variant_t;

// Selects and borrows one compatible artifact during cold program placement.
// The table is ordered by preference; libamdf owns compatibility. Failure
// leaves both outputs unchanged. This performs no allocation or native IO.
iree_status_t iree_hal_amd_gpu_program_select(
    amdf_gpu_instruction_set_t supported, uint32_t variant_count,
    const iree_hal_amd_gpu_program_variant_t* variants,
    iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_IMAGE_PROGRAM_H_
