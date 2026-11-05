// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P kernel realization and resident-pipeline lowering passes.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_PASS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_PASS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/registry.h"
#include "loom/pass/types.h"
#include "loom/transforms/pipeline/realization.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_aie2p_kernel_realization_pass_info(void);

iree_status_t loom_aie2p_kernel_realization_run(loom_pass_t* pass,
                                                loom_module_t* module,
                                                loom_func_like_t function);

const loom_pass_info_t* loom_aie2p_pipeline_lower_pass_info(void);

iree_status_t loom_aie2p_pipeline_lower_run(loom_pass_t* pass,
                                            loom_module_t* module,
                                            loom_func_like_t function);

// Creates target mechanics whose lifetime spans parent-wide selection and
// emission. The provider exposes this without exporting private native state.
iree_status_t loom_aie2p_pipeline_realization_create(
    loom_pass_t* pass, loom_module_t* module,
    const loom_target_function_version_t* version,
    loom_pipeline_resource_inventory_t* out_inventory,
    loom_pipeline_realization_callback_t* out_callback, bool* out_valid);

// Target pass registry containing AIE2P execution-realization passes.
extern const loom_pass_registry_t loom_aie2p_pipeline_pass_registry;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_PASS_H_
