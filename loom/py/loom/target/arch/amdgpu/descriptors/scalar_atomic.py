# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ruff: noqa: F403, F405

"""CDNA scalar pointer and buffer atomic instruction families."""

from __future__ import annotations

from .common import *

# Native spelling, semantic operation, and value interpretation. Compare-exchange
# has two input values in SDATA; its returned memory value occupies the low half.
_SCALAR_ATOMIC_OPERATIONS = (
    ("swap", "exchange", "b"),
    ("cmpswap", "compare_exchange", "b"),
    ("add", "add", "u"),
    ("sub", "sub", "u"),
    ("smin", "min", "i"),
    ("umin", "min", "u"),
    ("smax", "max", "i"),
    ("umax", "max", "u"),
    ("and", "and", "b"),
    ("or", "or", "b"),
    ("xor", "xor", "b"),
    ("inc", "increment_wrap", "u"),
    ("dec", "decrement_wrap", "u"),
)


def _scalar_atomic_overlay(
    *,
    operation: str,
    semantic: str,
    value_kind: str,
    width_bits: int,
    buffer_address: bool,
    returns_old_value: bool,
    offset_only: bool,
) -> AmdgpuDescriptorOverlay:
    family = "s_buffer_atomic" if buffer_address else "s_atomic"
    native_suffix = "_x2" if width_bits == 64 else ""
    mnemonic = f"{family}_{operation}{native_suffix}"
    return_suffix = "_rtn" if returns_old_value else ""
    offset_suffix = "_offset_only" if offset_only else ""
    descriptor_key = f"amdgpu.{mnemonic}{return_suffix}{offset_suffix}"
    payload_units = width_bits // 32 * (2 if operation == "cmpswap" else 1)
    address_name = "resource" if buffer_address else "base"
    address_operand = (
        _sgpr_resource(address_name, units=4)
        if buffer_address
        else _sgpr_operand(address_name, units=2)
    )
    operands = (
        *(
            (AmdgpuOperandOverlay("SDATA", _sgpr_result(units=payload_units)),)
            if returns_old_value
            else ()
        ),
        AmdgpuOperandOverlay(
            "SDATA",
            _sgpr_operand("value", units=payload_units),
            role_exception_reason="xml-models-scalar-atomic-sdata-as-output-only",
        ),
        AmdgpuOperandOverlay("SBASE", address_operand),
    )
    operand_names = ("value", address_name)
    fixed_fields = (
        ("IMM", 1),
        ("SOFFSET_EN", int(not offset_only)),
        ("GLC", int(returns_old_value)),
    )
    if offset_only:
        fixed_fields += (("SOFFSET", 0),)
        operand_forms = ()
        native_offset = (_native_i64_immediate("offset"),)
    else:
        operands += (AmdgpuOperandOverlay("SOFFSET", _sgpr_operand("soffset")),)
        operand_names += ("soffset",)
        operand_forms = (
            _soffset_zero_operand_form(
                replacement_descriptor=f"{descriptor_key}_offset_only"
            ),
            _soffset_offset_operand_form(
                replacement_descriptor=f"{descriptor_key}_offset_only"
            ),
        )
        native_offset = (
            _native_operand("soffset"),
            _native_amdgpu_required_named_i64_immediate("offset"),
        )
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name=mnemonic.upper(),
        mnemonic=mnemonic,
        encoding_name="ENC_SMEM",
        semantic_tag=(
            f"memory.global.atomic.{semantic}.{value_kind}{width_bits}"
            + (".return" if returns_old_value else "")
        ),
        schedule_class=_SCHEDULE_SMEM_ATOMIC,
        operands=operands,
        immediate_fields=("OFFSET",),
        immediates=(_signed_offset_immediate(21),),
        fixed_encoding_fields=fixed_fields,
        effects=_global_atomic_effects(width_bits, counter_id=_COUNTER_SMEM),
        operand_forms=operand_forms,
        constraints=(
            (
                Constraint(ConstraintKind.TIED, 0, 1),
                Constraint(ConstraintKind.DESTRUCTIVE, 0, 1),
            )
            if returns_old_value
            else ()
        ),
        flags=(DescriptorFlag.SIDE_EFFECTING,),
        asm_forms=_asm(
            mnemonic=f"{mnemonic}{return_suffix}{offset_suffix}",
            native_assembly_mnemonic=(
                mnemonic if return_suffix or offset_suffix else None
            ),
            results=("dst",) if returns_old_value else (),
            operands=operand_names,
            immediates=("offset",),
            named_immediates=True,
            native_assembly_values=(
                _native_operand("value"),
                _native_operand(address_name),
                *native_offset,
                *((_native_modifier_literal("glc"),) if returns_old_value else ()),
            ),
        ),
    )


def _cdna_scalar_atomic_overlays() -> tuple[AmdgpuDescriptorOverlay, ...]:
    return tuple(
        _scalar_atomic_overlay(
            operation=operation,
            semantic=semantic,
            value_kind=value_kind,
            width_bits=width_bits,
            buffer_address=buffer_address,
            returns_old_value=returns_old_value,
            offset_only=offset_only,
        )
        for operation, semantic, value_kind in _SCALAR_ATOMIC_OPERATIONS
        for width_bits in (32, 64)
        for buffer_address in (False, True)
        for returns_old_value in (False, True)
        for offset_only in (False, True)
    )


__all__ = ["_cdna_scalar_atomic_overlays"]
