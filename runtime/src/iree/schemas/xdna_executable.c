// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/schemas/xdna_executable.h"

void iree_xdna_elf_encode_header(const iree_xdna_elf_header_record_t* value,
                                 uint8_t* storage) {
  iree_unaligned_store_le_u32(storage + 0, value->magic);
  iree_unaligned_store_le_u16(storage + 4, value->version);
  iree_unaligned_store_le_u16(storage + 6, value->native_encoding);
  iree_unaligned_store_le_u32(storage + 8, value->target_generation);
  iree_unaligned_store_le_u32(storage + 12, value->device_profile_revision);
  iree_unaligned_store_le_u64(storage + 16, value->device_profile_id);
  iree_unaligned_store_le_u64(storage + 24, value->firmware_abi_id);
  iree_unaligned_store_le_u16(storage + 32, value->column_count);
  iree_unaligned_store_le_u16(storage + 34, value->row_count);
  iree_unaligned_store_le_u32(storage + 36, value->allocation_count);
  iree_unaligned_store_le_u32(storage + 40, value->allocation_use_count);
  iree_unaligned_store_le_u32(storage + 44, value->entry_count);
  iree_unaligned_store_le_u32(storage + 48, value->binding_count);
  iree_unaligned_store_le_u32(storage + 52, value->relocation_count);
  iree_unaligned_store_le_u32(storage + 56, value->invocation_count);
  iree_unaligned_store_le_u32(storage + 60, value->string_byte_length);
}

iree_xdna_elf_header_record_t iree_xdna_elf_decode_header(
    const uint8_t* storage) {
  return (iree_xdna_elf_header_record_t){
      .magic = iree_unaligned_load_le_u32(storage + 0),
      .version = iree_unaligned_load_le_u16(storage + 4),
      .native_encoding = iree_unaligned_load_le_u16(storage + 6),
      .target_generation = iree_unaligned_load_le_u32(storage + 8),
      .device_profile_revision = iree_unaligned_load_le_u32(storage + 12),
      .device_profile_id = iree_unaligned_load_le_u64(storage + 16),
      .firmware_abi_id = iree_unaligned_load_le_u64(storage + 24),
      .column_count = iree_unaligned_load_le_u16(storage + 32),
      .row_count = iree_unaligned_load_le_u16(storage + 34),
      .allocation_count = iree_unaligned_load_le_u32(storage + 36),
      .allocation_use_count = iree_unaligned_load_le_u32(storage + 40),
      .entry_count = iree_unaligned_load_le_u32(storage + 44),
      .binding_count = iree_unaligned_load_le_u32(storage + 48),
      .relocation_count = iree_unaligned_load_le_u32(storage + 52),
      .invocation_count = iree_unaligned_load_le_u32(storage + 56),
      .string_byte_length = iree_unaligned_load_le_u32(storage + 60),
  };
}

void iree_xdna_elf_encode_allocation(
    const iree_xdna_elf_allocation_record_t* value, uint8_t* storage) {
  iree_unaligned_store_le_u32(storage + 0, value->domain);
  iree_unaligned_store_le_u32(storage + 4, value->flags);
  iree_unaligned_store_le_u64(storage + 8, value->byte_length);
  iree_unaligned_store_le_u64(storage + 16, value->alignment);
  iree_unaligned_store_le_u32(storage + 24, value->first_load);
  iree_unaligned_store_le_u32(storage + 28, value->load_count);
}

iree_xdna_elf_allocation_record_t iree_xdna_elf_decode_allocation(
    const uint8_t* storage) {
  return (iree_xdna_elf_allocation_record_t){
      .domain = iree_unaligned_load_le_u32(storage + 0),
      .flags = iree_unaligned_load_le_u32(storage + 4),
      .byte_length = iree_unaligned_load_le_u64(storage + 8),
      .alignment = iree_unaligned_load_le_u64(storage + 16),
      .first_load = iree_unaligned_load_le_u32(storage + 24),
      .load_count = iree_unaligned_load_le_u32(storage + 28),
  };
}

void iree_xdna_elf_encode_entry(const iree_xdna_elf_entry_record_t* value,
                                uint8_t* storage) {
  iree_unaligned_store_le_u32(storage + 0, value->name_offset);
  iree_unaligned_store_le_u32(storage + 4, value->name_length);
  iree_unaligned_store_le_u32(storage + 8, value->first_allocation_use);
  iree_unaligned_store_le_u32(storage + 12, value->allocation_use_count);
  iree_unaligned_store_le_u32(storage + 16, value->first_binding);
  iree_unaligned_store_le_u32(storage + 20, value->binding_count);
  iree_unaligned_store_le_u32(storage + 24, value->first_static_relocation);
  iree_unaligned_store_le_u32(storage + 28, value->static_relocation_count);
  iree_unaligned_store_le_u32(storage + 32, value->first_dynamic_relocation);
  iree_unaligned_store_le_u32(storage + 36, value->dynamic_relocation_count);
  iree_unaligned_store_le_u32(storage + 40, value->first_invocation);
  iree_unaligned_store_le_u32(storage + 44, value->invocation_count);
}

iree_xdna_elf_entry_record_t iree_xdna_elf_decode_entry(
    const uint8_t* storage) {
  return (iree_xdna_elf_entry_record_t){
      .name_offset = iree_unaligned_load_le_u32(storage + 0),
      .name_length = iree_unaligned_load_le_u32(storage + 4),
      .first_allocation_use = iree_unaligned_load_le_u32(storage + 8),
      .allocation_use_count = iree_unaligned_load_le_u32(storage + 12),
      .first_binding = iree_unaligned_load_le_u32(storage + 16),
      .binding_count = iree_unaligned_load_le_u32(storage + 20),
      .first_static_relocation = iree_unaligned_load_le_u32(storage + 24),
      .static_relocation_count = iree_unaligned_load_le_u32(storage + 28),
      .first_dynamic_relocation = iree_unaligned_load_le_u32(storage + 32),
      .dynamic_relocation_count = iree_unaligned_load_le_u32(storage + 36),
      .first_invocation = iree_unaligned_load_le_u32(storage + 40),
      .invocation_count = iree_unaligned_load_le_u32(storage + 44),
  };
}

void iree_xdna_elf_encode_binding(const iree_xdna_elf_binding_record_t* value,
                                  uint8_t* storage) {
  iree_unaligned_store_le_u16(storage + 0, value->kind);
  iree_unaligned_store_le_u16(storage + 2, value->address_space);
  iree_unaligned_store_le_u16(storage + 4, value->access);
  iree_unaligned_store_le_u16(storage + 6, value->usage);
  iree_unaligned_store_le_u64(storage + 8, value->minimum_byte_length);
  iree_unaligned_store_le_u64(storage + 16, value->minimum_alignment);
  iree_unaligned_store_le_u64(storage + 24, value->minimum_byte_offset);
  iree_unaligned_store_le_u64(storage + 32, value->maximum_byte_offset);
}

iree_xdna_elf_binding_record_t iree_xdna_elf_decode_binding(
    const uint8_t* storage) {
  return (iree_xdna_elf_binding_record_t){
      .kind = iree_unaligned_load_le_u16(storage + 0),
      .address_space = iree_unaligned_load_le_u16(storage + 2),
      .access = iree_unaligned_load_le_u16(storage + 4),
      .usage = iree_unaligned_load_le_u16(storage + 6),
      .minimum_byte_length = iree_unaligned_load_le_u64(storage + 8),
      .minimum_alignment = iree_unaligned_load_le_u64(storage + 16),
      .minimum_byte_offset = iree_unaligned_load_le_u64(storage + 24),
      .maximum_byte_offset = iree_unaligned_load_le_u64(storage + 32),
  };
}

void iree_xdna_elf_encode_relocation(
    const iree_xdna_elf_relocation_record_t* value, uint8_t* storage) {
  iree_unaligned_store_le_u32(storage + 0, value->destination_use);
  iree_unaligned_store_le_u32(storage + 4, value->source_ordinal);
  iree_unaligned_store_le_u32(storage + 8, value->byte_offset);
  iree_unaligned_store_le_u32(storage + 12, value->kind);
  iree_unaligned_store_le_u64(storage + 16, (uint64_t)value->addend);
  iree_unaligned_store_le_u64(storage + 24, value->minimum_value);
  iree_unaligned_store_le_u64(storage + 32, value->maximum_value);
  iree_unaligned_store_le_u64(storage + 40, value->alignment);
}

iree_xdna_elf_relocation_record_t iree_xdna_elf_decode_relocation(
    const uint8_t* storage) {
  return (iree_xdna_elf_relocation_record_t){
      .destination_use = iree_unaligned_load_le_u32(storage + 0),
      .source_ordinal = iree_unaligned_load_le_u32(storage + 4),
      .byte_offset = iree_unaligned_load_le_u32(storage + 8),
      .kind = iree_unaligned_load_le_u32(storage + 12),
      .addend = (int64_t)iree_unaligned_load_le_u64(storage + 16),
      .minimum_value = iree_unaligned_load_le_u64(storage + 24),
      .maximum_value = iree_unaligned_load_le_u64(storage + 32),
      .alignment = iree_unaligned_load_le_u64(storage + 40),
  };
}

void iree_xdna_elf_encode_invocation(
    const iree_xdna_elf_invocation_record_t* value, uint8_t* storage) {
  iree_unaligned_store_le_u32(storage + 0, value->allocation_use);
  iree_unaligned_store_le_u32(storage + 4, value->byte_offset);
  iree_unaligned_store_le_u32(storage + 8, value->byte_length);
  iree_unaligned_store_le_u32(storage + 12, value->next_invocation);
}

iree_xdna_elf_invocation_record_t iree_xdna_elf_decode_invocation(
    const uint8_t* storage) {
  return (iree_xdna_elf_invocation_record_t){
      .allocation_use = iree_unaligned_load_le_u32(storage + 0),
      .byte_offset = iree_unaligned_load_le_u32(storage + 4),
      .byte_length = iree_unaligned_load_le_u32(storage + 8),
      .next_invocation = iree_unaligned_load_le_u32(storage + 12),
  };
}
