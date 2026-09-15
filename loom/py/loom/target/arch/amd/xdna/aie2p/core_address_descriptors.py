# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P address setup and memory operations with explicit pointer recurrence."""

from __future__ import annotations

from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.low_descriptors import DescriptorOpKind

_TARGET_KEY = "amd.xdna.aie2p"


def _address_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Selects modifier setup and native post-increment pointer updates."""

    result = []
    for form, mnemonic in (
        ("MOVA", "mova.modifier"),
        ("MOVXM", "mov.modifier.immediate"),
    ):
        result.append(
            _DescriptorSpec(
                form,
                f"{_TARGET_KEY}.constant.modifier.{form.lower()}",
                "integer.const.i20",
                f"II_{form}_eM",
                storage_overrides=(("dst", "eM"),),
                op_kind=DescriptorOpKind.CONST,
                asm_mnemonic=mnemonic,
            )
        )
    for destination, source, mnemonic in (
        ("eM", "eR", "mov.modifier"),
        ("eR", "eM", "mov.modifier-to-scalar"),
        ("eM", "eM", "mov.modifier.copy"),
    ):
        result.append(
            _DescriptorSpec(
                "MOV_alu_mv_mv_mv_scl",
                f"{_TARGET_KEY}.move.{source.lower()}.to.{destination.lower()}",
                f"register.move.{source.lower()}.to.{destination.lower()}",
                f"II_MOV_alu_mv_mv_mv_scl_{destination}_{source}",
                storage_overrides=(("dst", destination), ("src", source)),
                asm_mnemonic=mnemonic,
                allocation_move=destination == source,
            )
        )
    for immediate, suffix in ((False, ""), (True, "_imm")):
        addressing = "immediate" if immediate else "register"
        mnemonic_suffix = "" if immediate else ".modifier"
        for operation, form_prefix, shape, width in (
            ("load", "LDA_dms_lda", "i32", 32),
            ("load", "LDA_s8", "s8", 8),
            ("load", "LDA_u8", "u8", 8),
            ("load", "LDA_s16", "s16", 16),
            ("load", "LDA_u16", "u16", 16),
            ("store", "ST_dms_sts", "i32", 32),
            ("store", "ST_s8", "i8", 8),
            ("store", "ST_s16", "i16", 16),
        ):
            form = f"{form_prefix}_pstm_nrm{suffix}"
            mnemonic, operand = ("lda", "dst") if operation == "load" else ("st", "src")
            result.append(
                _DescriptorSpec(
                    form,
                    f"{_TARGET_KEY}.{operation}.scalar.{shape}.postincrement.{addressing}",
                    f"memory.{operation}.postincrement.{shape}",
                    f"II_{form}{'_eR' if width == 32 else ''}",
                    storage_overrides=((operand, "eR"),),
                    asm_mnemonic=f"{mnemonic}.{shape}.post{mnemonic_suffix}",
                    memory_width_bits=width,
                )
            )
        for lane in ("a", "b", "s"):
            form = f"PADD{lane.upper()}_pstm_nrm{suffix}"
            result.append(
                _DescriptorSpec(
                    form,
                    f"{_TARGET_KEY}.address.add.{lane}.{addressing}",
                    "address.add.i20",
                    f"II_{form}",
                    asm_mnemonic=f"padd{lane}{mnemonic_suffix}",
                )
            )
    return tuple(result)
