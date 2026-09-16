// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_BYTECODE_WRITER_TYPE_INDEX_H_
#define LOOM_FORMAT_BYTECODE_WRITER_TYPE_INDEX_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Invocation-owned projection from module type storage to wire equivalence.
// Structural wire identity omits scoped SSA dimension/encoding bindings. Each
// physically shared type is analyzed once; child equivalence is retained rather
// than recursively recovered during numbering. The immutable module outlives
// the index, and the supplied scratch arena owns all index storage.
typedef struct loom_bytecode_type_index_t {
  // Immutable module owning all borrowed type and attribute payloads.
  const loom_module_t* module;
  // Distinct by-value storage nodes and their retained wire representatives.
  struct loom_bytecode_type_node_t* nodes;
  // Number of populated storage nodes.
  iree_host_size_t count;
  // Allocated storage-node capacity.
  iree_host_size_t capacity;
  // Storage-node IDs indexed by exact representation hash, or UINT32_MAX.
  uint32_t* slots;
  // Power-of-two capacity of slots.
  iree_host_size_t slot_capacity;
} loom_bytecode_type_index_t;

// Builds the wire-equivalence projection without changing module or wire order.
// Fallibility is limited to scratch allocation and index-size representation.
iree_status_t loom_bytecode_type_index_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_bytecode_type_index_t* out_index);

// Returns the earliest module-table entry equivalent to a retained type, or
// LOOM_TYPE_ID_INVALID when the type is not in the module's retained closure.
// No structural traversal or allocation occurs during lookup.
loom_type_id_t loom_bytecode_type_index_lookup(
    const loom_bytecode_type_index_t* index, loom_type_t type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_TYPE_INDEX_H_
