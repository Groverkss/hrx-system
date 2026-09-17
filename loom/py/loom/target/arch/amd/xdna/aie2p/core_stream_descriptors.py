# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P scalar and cascade stream protocol descriptors."""

from __future__ import annotations

from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.low_descriptors import Effect, EffectKind

_TARGET_KEY = "amd.xdna.aie2p"


def _scalar_stream_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Exposes scalar streams, packet headers, and their completion status."""

    # Stream traffic participates in a device protocol even when the payload
    # has no SSA consumer. Preserve its issue order with memory and other
    # protocol effects; this annotation does not add a hardware memory fence.
    effects = (Effect(EffectKind.BARRIER),)
    result = []
    for nonblocking in (False, True):
        form_infix = "nb_" if nonblocking else ""
        key_suffix = ".nonblocking" if nonblocking else ""
        mnemonic_suffix = ".nb" if nonblocking else ""
        result.append(
            _DescriptorSpec(
                f"MOV_{form_infix}lda",
                f"{_TARGET_KEY}.stream.read{key_suffix}.i32",
                f"stream.read{key_suffix}.i32",
                f"II_MOV_{form_infix}lda",
                asm_mnemonic=f"mov.ss{mnemonic_suffix}",
                effects=effects,
            )
        )
        for form_suffix, last_key, last_mnemonic in (
            ("mMStream_tlast_imm", "", ""),
            ("mMStream_tlast_reg", ".last.register", ".last.reg"),
            ("tlast", ".last", ".last"),
        ):
            for register_class in ("eR", "eP", "eDC", "eDJ", "eDN", "eM"):
                stem = "st_" if form_suffix.startswith("mMStream") else ""
                form = f"MOV_{form_infix}{stem}{form_suffix}"
                storage_key = (
                    "i32" if register_class == "eR" else register_class.lower()
                )
                storage_mnemonic = "" if register_class == "eR" else f".{storage_key}"
                key = f"stream.write{key_suffix}{last_key}.{storage_key}"
                result.append(
                    _DescriptorSpec(
                        form,
                        f"{_TARGET_KEY}.{key}",
                        key,
                        f"II_{form}_{register_class}",
                        storage_overrides=(("src", register_class),),
                        asm_mnemonic=f"mov.ms{mnemonic_suffix}{last_mnemonic}{storage_mnemonic}",
                        effects=effects,
                    )
                )
            for header, operation in (("PH", "packet"), ("CPH", "control-packet")):
                form = f"MOV_{header}_{form_infix}{form_suffix}"
                key = f"stream.write.{operation}{key_suffix}{last_key}"
                result.append(
                    _DescriptorSpec(
                        form,
                        f"{_TARGET_KEY}.{key}",
                        key,
                        f"II_{form}",
                        asm_mnemonic=f"mov.{header.lower()}{mnemonic_suffix}{last_mnemonic}",
                        effects=effects,
                    )
                )
    for direction, port, register_class in (
        ("read", "ss", "mSRSS0"),
        ("write", "ms", "mSRMS0"),
    ):
        result.append(
            _DescriptorSpec(
                "MOV_alu_mv_mv_mv_scl",
                f"{_TARGET_KEY}.stream.{direction}.status",
                f"stream.{direction}.status",
                f"II_MOV_alu_mv_mv_mv_scl_eR_{register_class}",
                storage_overrides=(("dst", "eR"), ("src", register_class)),
                implicit_inputs=("src",),
                asm_mnemonic=f"mov.{port}.status",
            )
        )
    return tuple(result)


def _cascade_descriptor_specs() -> tuple[_DescriptorSpec, ...]:
    """Exposes native 512-bit transfers and their enable state."""

    result = []
    for direction, port, native_stem, register_class in (
        ("read", "scd", "lda_mv_scd", "mCRSCDEn"),
        ("write", "mcd", "st_mv_mcd", "mCRMCDEn"),
    ):
        result.append(
            _DescriptorSpec(
                "MOV_alu_mv_mv_mv_cg",
                f"{_TARGET_KEY}.state.{port}-enable.immediate",
                f"state.write.{port}-enable",
                f"II_MOV_alu_mv_mv_mv_cg_{register_class}",
                storage_overrides=(("dst", register_class),),
                implicit_outputs=("dst",),
                asm_mnemonic=f"set.{port}-enable",
            )
        )
        for payload, suffix, mnemonic_suffix in (
            ("vector", "x", ""),
            ("accumulator", "bm", ".acc"),
        ):
            form = f"VMOV_{native_stem}_{suffix}"
            key = f"cascade.{direction}.{payload}.512"
            result.append(
                _DescriptorSpec(
                    form,
                    f"{_TARGET_KEY}.{key}",
                    key,
                    f"II_{form}",
                    storage_overrides=(
                        (("dst", "mBMs"),)
                        if direction == "read" and payload == "accumulator"
                        else ()
                    ),
                    asm_mnemonic=f"vmov.{port}{mnemonic_suffix}",
                    # Each transfer consumes or produces a distinct stream
                    # value, even when its payload has no SSA consumer.
                    effects=(Effect(EffectKind.BARRIER),),
                )
            )
    return tuple(result)
