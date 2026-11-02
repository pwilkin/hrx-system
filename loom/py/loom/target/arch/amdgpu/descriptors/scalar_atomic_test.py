# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

from loom.target.arch.amdgpu.descriptors.api import _with_storage_lease_rows
from loom.target.arch.amdgpu.descriptors.cdna import (
    _AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE,
    _AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE,
)
from loom.target.arch.amdgpu.descriptors.common import _COUNTER_SMEM
from loom.target.arch.amdgpu.descriptors.contracts import (
    _amdgpu_contract_descriptor_from_overlay,
)
from loom.target.arch.amdgpu.descriptors.finalization import (
    _finalize_amdgpu_descriptors,
)
from loom.target.arch.amdgpu.descriptors.sets import (
    _gfx11_core_overlays,
    _gfx12_core_overlays,
    _gfx90a_core_overlays,
    _gfx115x_core_overlays,
    _gfx940_core_overlays,
    _gfx950_core_overlays,
)
from loom.target.low_descriptors import (
    ConstraintKind,
    EffectKind,
    InstructionClass,
    MemorySpace,
    OperandRole,
    ScheduleClassFlag,
    StorageLeaseKind,
)


def test_cdna_scalar_atomic_family_preserves_payload_and_completion() -> None:
    for base, overlays in (
        (_AMDGPU_CDNA2_CORE_DESCRIPTOR_SET_BASE, _gfx90a_core_overlays()),
        (_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE, _gfx940_core_overlays()),
        (_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE, _gfx950_core_overlays()),
    ):
        rows = [
            row
            for row in overlays
            if row.instruction_name.startswith(("S_ATOMIC_", "S_BUFFER_ATOMIC_"))
        ]
        assert len(rows) == 208
        assert len({row.instruction_name for row in rows}) == 52
        descriptors = tuple(
            _amdgpu_contract_descriptor_from_overlay(row) for row in rows
        )
        descriptor_set = _with_storage_lease_rows(
            replace(base, descriptors=descriptors)
        )
        descriptor_set = _finalize_amdgpu_descriptors(
            descriptor_set, valu_sgpr_separation_cycles=0
        )
        classes = {row.name: row for row in descriptor_set.schedule_classes}
        for descriptor in descriptor_set.descriptors:
            assert descriptor.instruction_classes == (InstructionClass.SCALAR_MEMORY,)
            results = [x for x in descriptor.operands if x.role is OperandRole.RESULT]
            payload = next(x for x in descriptor.operands if x.field_name == "value")
            width_bits = 64 if "_x2" in descriptor.key else 32
            assert payload.unit_count == width_bits // 32 * (
                2 if "cmpswap" in descriptor.key else 1
            )
            assert [x.kind for x in descriptor.effects] == [
                EffectKind.READ,
                EffectKind.WRITE,
            ]
            assert all(x.memory_space is MemorySpace.GLOBAL for x in descriptor.effects)
            assert all(x.width_bits == width_bits for x in descriptor.effects)
            assert all(x.counter_id == _COUNTER_SMEM for x in descriptor.effects)
            schedule = classes[descriptor.schedule_class]
            assert set(schedule.flags) == {
                ScheduleClassFlag.MAY_LOAD,
                ScheduleClassFlag.MAY_STORE,
            }
            if results:
                assert results[0].unit_count == payload.unit_count
                assert {x.kind for x in descriptor.constraints} == {
                    ConstraintKind.TIED,
                    ConstraintKind.DESTRUCTIVE,
                }
                assert len(descriptor.storage_leases) == 1
                lease = descriptor.storage_leases[0]
                assert lease.kind is StorageLeaseKind.RESULT_WRITE
                assert lease.release_class_id == _COUNTER_SMEM
                assert lease.unit_count == payload.unit_count
            else:
                assert not descriptor.constraints
                assert not descriptor.storage_leases
        for row in rows:
            fixed = dict(row.fixed_encoding_fields)
            assert fixed["GLC"] == int("_rtn" in row.descriptor_key)
            assert fixed["SOFFSET_EN"] == int(
                not row.descriptor_key.endswith("_offset_only")
            )
            form = row.asm_forms[0]
            assert (form.native_assembly_mnemonic or form.mnemonic) == row.mnemonic


def test_rdna_descriptor_sets_do_not_advertise_scalar_atomics() -> None:
    for overlays in (
        _gfx11_core_overlays(),
        _gfx115x_core_overlays(),
        _gfx12_core_overlays(),
    ):
        assert not any(
            row.instruction_name.startswith(("S_ATOMIC_", "S_BUFFER_ATOMIC_"))
            for row in overlays
        )
