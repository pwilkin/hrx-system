// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDHSA state consumed and modified by the compiler's device-library grid
// operations. Its allocation is independent of either kernel's arguments.
#include "amdf/gpu_device.h"

typedef struct {
  // Cross-grid state, absent for a single-device dispatch.
  unsigned long long multi_grid_sync;
  // Index of this grid among cooperating devices.
  unsigned int grid_ordinal;
  // Number of cooperating device grids.
  unsigned int grid_count;
  // Workitems in preceding device grids.
  unsigned long long preceding_workitem_count;
  // Workitems participating in this launch.
  unsigned long long total_workitem_count;
  // Mutable arrival and epoch words, also used by split GWS grid barriers.
  unsigned int single_grid_sync[2];
  // Number of workgroups participating in this grid.
  unsigned int workgroup_count;
  // Initialized ABI padding.
  unsigned int reserved;
} grid_state_t;

typedef struct {
  // Writable device grid state, retained through the following user grid.
  __attribute__((address_space(1))) grid_state_t* grid;
  // Total workitems in the admitted grid geometry.
  unsigned long long workitem_count;
  // Nonzero grid extent admitted by the host before publication.
  unsigned int workgroup_count;
  // Initialized ABI padding.
  unsigned int reserved;
} grid_arguments_t;

#if AMDF_GPU_DEVICE_GLOBAL_WAVE_SYNC
__attribute__((amdgpu_kernel, visibility("protected"), used, target("gws"),
               amdgpu_flat_work_group_size(1, 1))) void
initialize_grid_sync(grid_arguments_t arguments) {
  *arguments.grid = (grid_state_t){
      .grid_count = 1,
      .total_workitem_count = arguments.workitem_count,
      .workgroup_count = arguments.workgroup_count,
  };
  __builtin_amdgcn_ds_gws_init(arguments.workgroup_count - 1, 0);
}
#endif  // AMDF_GPU_DEVICE_GLOBAL_WAVE_SYNC
