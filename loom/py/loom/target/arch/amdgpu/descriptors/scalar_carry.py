# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ruff: noqa: F403, F405

"""Scalar carry chains and PC-relative address arithmetic."""

from __future__ import annotations

from dataclasses import replace

from .alu import _s_add_u32_overlay, _s_add_u32_rhs_inline_overlay
from .common import *

_SYMBOL_BYTE_OFFSET_IMMEDIATE = Immediate(
    "byte_offset",
    ImmediateKind.UNSIGNED,
    flags=(ImmediateFlag.DEFAULT_VALUE,),
    bit_width=64,
    unsigned_max=(2**63) - 1,
    default_value=0,
)


def _symbol_rel32_immediate(field_name: str = "symbol") -> Immediate:
    return Immediate(
        field_name,
        ImmediateKind.ORDINAL,
        flags=(ImmediateFlag.SYMBOLIC, ImmediateFlag.RELATIVE),
        bit_width=32,
        unsigned_max=(2**32) - 1,
        encoding_field_id=amdgpu_encoding_field_id("LITERAL"),
    )


def _s_add_co_u32_overlay() -> AmdgpuDescriptorOverlay:
    # S_ADDK_I32 produces signed overflow instead of unsigned carry.
    return replace(
        _s_add_u32_overlay(),
        descriptor_key="amdgpu.s_add_co_u32",
        mnemonic="s_add_co_u32",
        semantic_tag="integer.add.carry_out.u32",
        implicit_operands=(_scc_output(_scc_result("carry")),),
        asm_forms=_asm(
            native_assembly_mnemonic="s_add_u32",
            results=("dst", "carry"),
            operands=("lhs", "rhs"),
        ),
        operand_forms=(
            _literal_operand_form(
                replacement_descriptor="amdgpu.s_add_co_u32.rhs_inline",
                source_operand="rhs",
            ),
        ),
    )


def _s_add_co_u32_rhs_inline_overlay() -> AmdgpuDescriptorOverlay:
    return replace(
        _s_add_u32_rhs_inline_overlay(),
        descriptor_key="amdgpu.s_add_co_u32.rhs_inline",
        mnemonic="s_add_co_u32",
        semantic_tag="integer.add.carry_out.u32",
        implicit_operands=(_scc_output(_scc_result("carry")),),
        asm_forms=_asm(
            mnemonic="s_add_co_u32_rhs_inline",
            native_assembly_mnemonic="s_add_u32",
            results=("dst", "carry"),
            operands=("lhs",),
            immediates=("imm32",),
        ),
    )


def _s_add_u32_rhs_symbol_rel32_lo_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_add_u32.rhs_symbol_rel32_lo",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        encoding_name="ENC_SOP2",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
        semantic_tag="address.add.pc_relative.lo.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result()),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        implicit_operands=(_scc_output(_scc_result("carry")),),
        immediates=(
            _symbol_rel32_immediate(),
            _SYMBOL_BYTE_OFFSET_IMMEDIATE,
        ),
        effects=(_PC_RELATIVE_EFFECT,),
        fixed_encoding_fields=(("SSRC1", _predefined("SRC_LITERAL", "OPR_SSRC")),),
        asm_forms=_asm(
            mnemonic="s_add_u32_rhs_symbol_rel32_lo",
            results=("dst", "carry"),
            operands=("lhs",),
            immediates=("symbol", "byte_offset"),
            named_immediates=True,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_addc_u32_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_addc_u32",
        instruction_name="S_ADDC_U32",
        mnemonic="s_addc_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.carry_in_out.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("sum")),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
            AmdgpuOperandOverlay("SSRC1", _sgpr_operand("rhs")),
        ),
        implicit_operands=(
            _scc_output(_scc_result("carry")),
            _scc_input(_scc_predicate("carry_in")),
        ),
        asm_forms=_asm(results=("sum", "carry"), operands=("lhs", "rhs", "carry_in")),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


def _s_addc_u32_rhs_symbol_rel32_hi_overlay() -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_addc_u32.rhs_symbol_rel32_hi",
        instruction_name="S_ADDC_U32",
        mnemonic="s_addc_u32",
        encoding_name="ENC_SOP2",
        encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
        semantic_tag="address.add.pc_relative.hi.u32",
        schedule_class=_SCHEDULE_SALU,
        operands=(
            AmdgpuOperandOverlay("SDST", _sgpr_result("sum")),
            AmdgpuOperandOverlay("SSRC0", _sgpr_operand("lhs")),
        ),
        implicit_operands=(
            _scc_output(_scc_result("carry")),
            _scc_input(_scc_predicate("carry_in")),
        ),
        immediates=(
            _symbol_rel32_immediate(),
            _SYMBOL_BYTE_OFFSET_IMMEDIATE,
        ),
        effects=(_PC_RELATIVE_EFFECT,),
        fixed_encoding_fields=(("SSRC1", _predefined("SRC_LITERAL", "OPR_SSRC")),),
        asm_forms=_asm(
            mnemonic="s_addc_u32_rhs_symbol_rel32_hi",
            results=("sum", "carry"),
            operands=("lhs", "carry_in"),
            immediates=("symbol", "byte_offset"),
            named_immediates=True,
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )


__all__ = [
    "_s_add_co_u32_overlay",
    "_s_add_co_u32_rhs_inline_overlay",
    "_s_add_u32_rhs_symbol_rel32_lo_overlay",
    "_s_addc_u32_overlay",
    "_s_addc_u32_rhs_symbol_rel32_hi_overlay",
]
