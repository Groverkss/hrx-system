// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/value_refs.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"

static iree_status_t loom_value_walk_outgoing_type_refs(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_type_value_ref_callback_t callback, void* user_data) {
  for (loom_type_use_id_t use_id =
           loom_module_value_first_outgoing_type_use(module, value_id);
       use_id != LOOM_TYPE_USE_ID_INVALID;) {
    const loom_type_use_t* use = &module->type_uses.records[use_id];
    IREE_RETURN_IF_ERROR(callback(use->referenced_value_id, user_data));
    use_id = use->next_outgoing_use_id;
  }
  return iree_ok_status();
}

iree_status_t loom_op_walk_subtree_value_refs(
    const loom_module_t* module, const loom_op_t* op,
    loom_type_value_ref_callback_t callback, void* user_data) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  // Only symbol signatures define operand values. Ordinary users need no
  // interface metadata lookup on this per-operation path.
  if (op->operand_count > 0 &&
      iree_any_bit_set(op->traits, LOOM_TRAIT_SYMBOL_DEFINE) &&
      loom_op_vtable_owns_operands(loom_op_vtable(module, op))) {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_walk_outgoing_type_refs(
          module, operands[i], callback, user_data));
    }
  } else {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      if (operands[i] != LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(callback(operands[i], user_data));
      }
    }
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_walk_outgoing_type_refs(
        module, results[i], callback, user_data));
  }

  const loom_attribute_use_id_t* heads = loom_op_attribute_use_heads(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    for (loom_attribute_use_id_t use_id = heads[i]; use_id;) {
      const loom_attribute_use_t* use =
          &module->attribute_uses.records[use_id - 1];
      IREE_RETURN_IF_ERROR(callback(use->value_id, user_data));
      use_id = use->next_outgoing;
    }
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
        loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
        IREE_RETURN_IF_ERROR(loom_value_walk_outgoing_type_refs(
            module, arg_id, callback, user_data));
      }
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(loom_op_walk_subtree_value_refs(
            module, child_op, callback, user_data));
      }
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_module_walk_attribute_value_refs_impl(
    const loom_module_t* module, loom_attribute_t attr, uint8_t depth,
    loom_type_value_ref_callback_t callback,
    loom_type_value_ref_callback_t predicate_callback, void* user_data) {
  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_TYPE:
      if (attr.type_id == LOOM_TYPE_ID_INVALID ||
          attr.type_id >= module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u is out of range (module has %" PRIhsz
            " types)",
            (unsigned)attr.type_id, module->types.count);
      }
      return loom_type_walk_value_refs(
          module, module->types.entries[attr.type_id], callback, user_data);

    case LOOM_ATTR_PREDICATE_LIST:
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        for (uint8_t j = 0; j < predicate->arg_count; ++j) {
          if (predicate->arg_tags[j] != LOOM_PRED_ARG_VALUE) continue;
          IREE_RETURN_IF_ERROR(predicate_callback(
              (loom_value_id_t)predicate->args[j], user_data));
        }
      }
      return iree_ok_status();

    case LOOM_ATTR_DICT:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_walk_attribute_value_refs_impl(
            module, attr.dict_entries[i].value, (uint8_t)(depth + 1), callback,
            predicate_callback, user_data));
      }
      return iree_ok_status();

    case LOOM_ATTR_PARAMETERIZED:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_walk_attribute_value_refs_impl(
            module, attr.parameterized_slots[i], (uint8_t)(depth + 1), callback,
            predicate_callback, user_data));
      }
      return iree_ok_status();

    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      for (uint16_t i = 0; i < attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_module_walk_attribute_value_refs_impl(
            module, attr.parameterized_array[i], (uint8_t)(depth + 1), callback,
            predicate_callback, user_data));
      }
      return iree_ok_status();

    default:
      return iree_ok_status();
  }
}

iree_status_t loom_module_walk_attribute_value_refs(
    const loom_module_t* module, loom_attribute_t attr,
    loom_type_value_ref_callback_t callback, void* user_data) {
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "module is NULL");
  }
  if (!callback) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value reference callback is NULL");
  }
  return loom_module_walk_attribute_value_refs_impl(
      module, attr, /*depth=*/0, callback, callback, user_data);
}

//===----------------------------------------------------------------------===//
// Exact incoming and outgoing use lists
//===----------------------------------------------------------------------===//

static loom_attribute_use_id_t* loom_attribute_use_incoming_head(
    loom_module_t* module, const loom_attribute_use_t* use) {
  loom_value_segment_t* segment =
      loom_value_table_segment_for_id(&module->values, use->value_id);
  loom_value_attribute_use_heads_t* heads =
      &segment->attribute_use_heads[use->value_id & LOOM_VALUE_SEGMENT_MASK];
  return use->is_predicate ? &heads->predicate : &heads->type;
}

static void loom_attribute_use_link(loom_module_t* module,
                                    loom_attribute_use_id_t id) {
  loom_attribute_use_table_t* table = &module->attribute_uses;
  loom_attribute_use_t* use = &table->records[id - 1];
  loom_attribute_use_id_t* head = loom_attribute_use_incoming_head(module, use);
  use->previous_incoming = 0;
  use->next_incoming = *head;
  if (*head) {
    table->records[*head - 1].previous_incoming = id;
  }
  *head = id;
  loom_module_value(module, use->value_id)->flags |=
      LOOM_VALUE_FLAG_ATTRIBUTE_USES;
}

static void loom_attribute_use_unlink(loom_module_t* module,
                                      loom_attribute_use_id_t id) {
  loom_attribute_use_table_t* table = &module->attribute_uses;
  const loom_attribute_use_t* use = &table->records[id - 1];
  if (use->previous_incoming) {
    table->records[use->previous_incoming - 1].next_incoming =
        use->next_incoming;
  } else {
    *loom_attribute_use_incoming_head(module, use) = use->next_incoming;
  }
  if (use->next_incoming) {
    table->records[use->next_incoming - 1].previous_incoming =
        use->previous_incoming;
  }
  if (!loom_module_value_first_attribute_use(module, use->value_id)) {
    loom_module_value(module, use->value_id)->flags &=
        ~LOOM_VALUE_FLAG_ATTRIBUTE_USES;
  }
}

static iree_status_t loom_attribute_use_allocate(
    loom_module_t* module, loom_attribute_use_id_t* out_id) {
  loom_attribute_use_table_t* table = &module->attribute_uses;
  if (table->first_free) {
    *out_id = table->first_free;
    table->first_free = table->records[*out_id - 1].next_outgoing;
    return iree_ok_status();
  }
  if (table->count == table->capacity) {
    if (table->capacity == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "attribute use count exceeds maximum");
    }
    uint32_t capacity = 32;
    if (table->capacity > UINT32_MAX / 2) {
      capacity = UINT32_MAX;
    } else if (table->capacity) {
      capacity = table->capacity * 2;
    }
    loom_attribute_use_t* records = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, capacity, sizeof(*records), (void**)&records));
    if (table->count) {
      memcpy(records, table->records, table->count * sizeof(*records));
    }
    table->records = records;
    table->capacity = capacity;
  }
  *out_id = ++table->count;
  return iree_ok_status();
}

static void loom_attribute_use_recycle(loom_attribute_use_table_t* table,
                                       loom_attribute_use_id_t id) {
  loom_attribute_use_t* use = &table->records[id - 1];
  use->op = NULL;
  use->next_outgoing = table->first_free;
  table->first_free = id;
}

void loom_module_drop_attribute_uses(loom_module_t* module, loom_op_t* op,
                                     uint8_t attribute_index) {
  loom_attribute_use_table_t* table = &module->attribute_uses;
  loom_attribute_use_id_t id = loom_op_attribute_use_heads(op)[attribute_index];
  loom_op_attribute_use_heads(op)[attribute_index] = 0;
  while (id) {
    const loom_attribute_use_id_t next = table->records[id - 1].next_outgoing;
    loom_attribute_use_unlink(module, id);
    loom_attribute_use_recycle(table, id);
    id = next;
  }
}

void loom_module_drop_op_attribute_uses(loom_module_t* module, loom_op_t* op) {
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    loom_module_drop_attribute_uses(module, op, i);
  }
}

void loom_module_reset_attribute_uses(loom_module_t* module) {
  loom_attribute_use_table_t* table = &module->attribute_uses;
  for (uint32_t i = 0; i < table->count; ++i) {
    const loom_attribute_use_t* use = &table->records[i];
    if (!use->op) {
      continue;
    }
    *loom_attribute_use_incoming_head(module, use) = 0;
    loom_op_attribute_use_heads(use->op)[use->attribute_index] = 0;
    loom_module_value(module, use->value_id)->flags &=
        ~LOOM_VALUE_FLAG_ATTRIBUTE_USES;
  }
  table->count = 0;
  table->first_free = 0;
}

// New records are not linked to values until the entire payload walk succeeds.
typedef struct loom_attribute_use_build_t {
  // Module owning the replacement's storage and value identities.
  loom_module_t* module;
  // Stable owning operation.
  loom_op_t* op;
  // New outgoing list, unpublished until construction succeeds.
  loom_attribute_use_id_t first;
  // Ordinal of the attribute being replaced.
  uint8_t attribute_index;
} loom_attribute_use_build_t;

static iree_status_t loom_attribute_use_build_append(
    loom_attribute_use_build_t* build, loom_value_id_t value_id,
    bool is_predicate) {
  if (value_id >= build->module->values.count) {
    return iree_ok_status();
  }
  loom_attribute_use_id_t id = 0;
  IREE_RETURN_IF_ERROR(loom_attribute_use_allocate(build->module, &id));
  build->module->attribute_uses.records[id - 1] = (loom_attribute_use_t){
      .op = build->op,
      .value_id = value_id,
      .next_outgoing = build->first,
      .attribute_index = build->attribute_index,
      .is_predicate = is_predicate,
  };
  build->first = id;
  return iree_ok_status();
}

static iree_status_t loom_attribute_use_build_type(loom_value_id_t value_id,
                                                   void* user_data) {
  return loom_attribute_use_build_append(user_data, value_id, false);
}

static iree_status_t loom_attribute_use_build_predicate(
    loom_value_id_t value_id, void* user_data) {
  return loom_attribute_use_build_append(user_data, value_id, true);
}

iree_status_t loom_module_set_op_attribute(loom_module_t* module, loom_op_t* op,
                                           uint8_t attribute_index,
                                           loom_attribute_t attribute) {
  loom_attribute_use_build_t build = {
      .module = module,
      .op = op,
      .attribute_index = attribute_index,
  };
  iree_status_t status = loom_module_walk_attribute_value_refs_impl(
      module, attribute, 0, loom_attribute_use_build_type,
      loom_attribute_use_build_predicate, &build);
  loom_attribute_use_table_t* table = &module->attribute_uses;
  if (iree_status_is_ok(status)) {
    loom_module_drop_attribute_uses(module, op, attribute_index);
    loom_op_attrs(op)[attribute_index] = attribute;
    loom_op_attribute_use_heads(op)[attribute_index] = build.first;
    for (loom_attribute_use_id_t id = build.first; id;
         id = table->records[id - 1].next_outgoing) {
      loom_attribute_use_link(module, id);
    }
  } else {
    loom_attribute_use_id_t id = build.first;
    while (id) {
      const loom_attribute_use_id_t next = table->records[id - 1].next_outgoing;
      loom_attribute_use_recycle(table, id);
      id = next;
    }
  }
  return status;
}

iree_status_t loom_module_refresh_op_attribute_uses(loom_module_t* module,
                                                    loom_op_t* op) {
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0; i < op->attribute_count && iree_status_is_ok(status);
       ++i) {
    loom_attribute_t attribute = loom_op_attrs(op)[i];
    switch ((loom_attr_kind_t)attribute.kind) {
      case LOOM_ATTR_TYPE:
      case LOOM_ATTR_PREDICATE_LIST:
      case LOOM_ATTR_DICT:
      case LOOM_ATTR_PARAMETERIZED:
      case LOOM_ATTR_PARAMETERIZED_ARRAY:
        status = loom_module_set_op_attribute(module, op, i, attribute);
        break;
      default:
        // Scalar attributes carry no references. A bulk reader can replace a
        // formerly reference-carrying slot before refreshing its use records.
        if (loom_op_attribute_use_heads(op)[i]) {
          loom_module_drop_attribute_uses(module, op, i);
        }
        break;
    }
  }
  return status;
}

static iree_status_t loom_module_replace_attribute_value_refs_impl(
    loom_module_t* module, loom_attribute_t attr, loom_value_id_t old_id,
    loom_value_id_t new_id, uint8_t depth, loom_attribute_t* out_attr,
    bool* out_changed) {
  *out_attr = attr;
  *out_changed = false;

  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_ABSENT:
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_STRING:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
    case LOOM_ATTR_SCOPED_ENUM:
    case LOOM_ATTR_SYMBOL:
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET:
    case LOOM_ATTR_I64_ARRAY:
    case LOOM_ATTR_ENUM_ARRAY:
    case LOOM_ATTR_SIGNED_ENUM_SET:
    case LOOM_ATTR_ENCODING:
    case LOOM_ATTR_BYTES:
      return iree_ok_status();

    case LOOM_ATTR_TYPE: {
      if (attr.type_id == LOOM_TYPE_ID_INVALID ||
          attr.type_id >= module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute id %u is out of range (module has %" PRIhsz
            " types)",
            (unsigned)attr.type_id, module->types.count);
      }
      loom_type_t replaced_type = module->types.entries[attr.type_id];
      IREE_RETURN_IF_ERROR(loom_module_replace_type_value_references(
          module, replaced_type, old_id, new_id, &replaced_type, out_changed));
      if (!*out_changed) return iree_ok_status();
      return loom_module_intern_type_id(module, replaced_type,
                                        &out_attr->type_id);
    }

    case LOOM_ATTR_PREDICATE_LIST: {
      bool changed = false;
      for (uint16_t i = 0; i < attr.count; ++i) {
        const loom_predicate_t* predicate = &attr.predicate_list[i];
        for (uint8_t j = 0; j < predicate->arg_count; ++j) {
          if (predicate->arg_tags[j] == LOOM_PRED_ARG_VALUE &&
              (loom_value_id_t)predicate->args[j] == old_id) {
            changed = true;
          }
        }
      }
      if (!changed) return iree_ok_status();

      loom_predicate_t* predicates = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, attr.count,
                                                     sizeof(*predicates),
                                                     (void**)&predicates));
      memcpy(predicates, attr.predicate_list,
             (iree_host_size_t)attr.count * sizeof(*predicates));
      for (uint16_t i = 0; i < attr.count; ++i) {
        for (uint8_t j = 0; j < predicates[i].arg_count; ++j) {
          if (predicates[i].arg_tags[j] == LOOM_PRED_ARG_VALUE &&
              (loom_value_id_t)predicates[i].args[j] == old_id) {
            predicates[i].args[j] = (int64_t)new_id;
          }
        }
      }
      out_attr->predicate_list = predicates;
      *out_changed = true;
      return iree_ok_status();
    }

    case LOOM_ATTR_DICT: {
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      loom_named_attr_t* replaced_entries = NULL;
      for (uint16_t i = 0; i < attr.count; ++i) {
        loom_attribute_t replaced_value = attr.dict_entries[i].value;
        bool value_changed = false;
        IREE_RETURN_IF_ERROR(loom_module_replace_attribute_value_refs_impl(
            module, attr.dict_entries[i].value, old_id, new_id,
            (uint8_t)(depth + 1), &replaced_value, &value_changed));
        if (!value_changed) continue;
        if (!replaced_entries) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &module->arena, attr.count, sizeof(*replaced_entries),
              (void**)&replaced_entries));
          memcpy(replaced_entries, attr.dict_entries,
                 (iree_host_size_t)attr.count * sizeof(*replaced_entries));
        }
        replaced_entries[i].value = replaced_value;
      }
      if (!replaced_entries) return iree_ok_status();
      *out_attr = loom_make_canonical_attr_dict(replaced_entries, attr.count);
      *out_changed = true;
      return iree_ok_status();
    }

    case LOOM_ATTR_PARAMETERIZED: {
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      loom_attribute_t* replaced_slots = NULL;
      for (uint16_t i = 0; i < attr.count; ++i) {
        loom_attribute_t replaced_value = attr.parameterized_slots[i];
        bool value_changed = false;
        IREE_RETURN_IF_ERROR(loom_module_replace_attribute_value_refs_impl(
            module, attr.parameterized_slots[i], old_id, new_id,
            (uint8_t)(depth + 1), &replaced_value, &value_changed));
        if (!value_changed) continue;
        if (!replaced_slots) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &module->arena, attr.count, sizeof(*replaced_slots),
              (void**)&replaced_slots));
          memcpy(replaced_slots, attr.parameterized_slots,
                 (iree_host_size_t)attr.count * sizeof(*replaced_slots));
        }
        replaced_slots[i] = replaced_value;
      }
      if (!replaced_slots) return iree_ok_status();
      *out_attr = loom_make_parameterized_attr(
          (loom_parameterized_attr_kind_t)attr.reserved_1, replaced_slots,
          attr.count);
      *out_changed = true;
      return iree_ok_status();
    }

    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      loom_attribute_t* replaced_attributes = NULL;
      for (uint16_t i = 0; i < attr.count; ++i) {
        loom_attribute_t replaced_value = attr.parameterized_array[i];
        bool value_changed = false;
        IREE_RETURN_IF_ERROR(loom_module_replace_attribute_value_refs_impl(
            module, attr.parameterized_array[i], old_id, new_id,
            (uint8_t)(depth + 1), &replaced_value, &value_changed));
        if (!value_changed) continue;
        if (!replaced_attributes) {
          IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
              &module->arena, attr.count, sizeof(*replaced_attributes),
              (void**)&replaced_attributes));
          memcpy(replaced_attributes, attr.parameterized_array,
                 (iree_host_size_t)attr.count * sizeof(*replaced_attributes));
        }
        replaced_attributes[i] = replaced_value;
      }
      if (!replaced_attributes) return iree_ok_status();
      *out_attr =
          loom_attr_parameterized_array(replaced_attributes, attr.count);
      *out_changed = true;
      return iree_ok_status();
    }

    case LOOM_ATTR_ANY:
    case LOOM_ATTR_COUNT_:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown attribute kind %u", (unsigned)attr.kind);
}

iree_status_t loom_module_replace_attribute_value_references(
    loom_module_t* module, loom_attribute_t attr, loom_value_id_t old_id,
    loom_value_id_t new_id, loom_attribute_t* out_attr, bool* out_changed) {
  *out_attr = attr;
  *out_changed = false;
  if (old_id == new_id) return iree_ok_status();
  if (old_id >= module->values.count || new_id >= module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot replace attribute references from %%%u to %%%u in a module "
        "with %" PRIhsz " values",
        (unsigned)old_id, (unsigned)new_id, module->values.count);
  }
  return loom_module_replace_attribute_value_refs_impl(
      module, attr, old_id, new_id, /*depth=*/0, out_attr, out_changed);
}

iree_status_t loom_module_replace_op_attribute_value_references(
    loom_module_t* module, loom_op_t* op, uint8_t attribute_index,
    loom_value_id_t old_id, loom_value_id_t new_id) {
  loom_attribute_t replacement = {0};
  bool changed = false;
  IREE_RETURN_IF_ERROR(loom_module_replace_attribute_value_refs_impl(
      module, loom_op_attrs(op)[attribute_index], old_id, new_id, /*depth=*/0,
      &replacement, &changed));
  IREE_ASSERT(changed, "attribute use owner must contain the referenced value");

  // Identity substitution and structural interning preserve reference
  // multiplicity and type/predicate classification. Retain the owner's list;
  // only edges referencing old_id change their incoming list. All fallible
  // payload construction has completed before either representation changes.
  loom_attribute_use_table_t* table = &module->attribute_uses;
  for (loom_attribute_use_id_t id =
           loom_op_attribute_use_heads(op)[attribute_index];
       id; id = table->records[id - 1].next_outgoing) {
    loom_attribute_use_t* use = &table->records[id - 1];
    if (use->value_id != old_id) {
      continue;
    }
    loom_attribute_use_unlink(module, id);
    use->value_id = new_id;
    loom_attribute_use_link(module, id);
  }
  loom_op_attrs(op)[attribute_index] = replacement;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type-use replacement
//===----------------------------------------------------------------------===//

static bool loom_module_type_has_replaceable_dims(loom_type_t type) {
  return loom_type_is_shaped(type) || loom_type_is_pool(type);
}

static iree_status_t loom_module_replace_type_value_refs_impl(
    loom_module_t* module, loom_type_t type, loom_value_id_t old_id,
    loom_value_id_t new_id, loom_type_t* out_type, bool* out_changed);

static iree_status_t loom_module_replace_type_ref_sequence(
    loom_module_t* module, const loom_type_t* types,
    iree_host_size_t type_count, loom_value_id_t old_id, loom_value_id_t new_id,
    loom_type_t** out_types, bool* out_changed) {
  *out_types = NULL;
  *out_changed = false;
  if (type_count == 0) {
    return iree_ok_status();
  }
  if (!types) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "type sequence has %" PRIhsz " entries but a NULL payload", type_count);
  }

  loom_type_t* replaced_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, type_count,
                                                 sizeof(loom_type_t),
                                                 (void**)&replaced_types));
  for (iree_host_size_t i = 0; i < type_count; ++i) {
    bool element_changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_refs_impl(
        module, types[i], old_id, new_id, &replaced_types[i],
        &element_changed));
    *out_changed = *out_changed || element_changed;
  }
  *out_types = replaced_types;
  return iree_ok_status();
}

static iree_status_t loom_module_replace_type_value_refs_impl(
    loom_module_t* module, loom_type_t type, loom_value_id_t old_id,
    loom_value_id_t new_id, loom_type_t* out_type, bool* out_changed) {
  *out_type = type;
  *out_changed = false;

  loom_type_kind_t kind = loom_type_kind(type);
  if (!loom_type_kind_is_valid(kind)) return iree_ok_status();

  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) return iree_ok_status();
      iree_host_size_t type_count =
          (iree_host_size_t)data->arg_count + data->result_count;
      loom_type_t* replaced_types = NULL;
      IREE_RETURN_IF_ERROR(loom_module_replace_type_ref_sequence(
          module, data->types, type_count, old_id, new_id, &replaced_types,
          out_changed));
      if (!*out_changed) return iree_ok_status();
      return loom_module_intern_function_type(
          module, replaced_types, data->arg_count,
          replaced_types + data->arg_count, data->result_count, out_type);
    }

    case LOOM_TYPE_DIALECT: {
      uint16_t param_count = loom_type_dialect_param_count(type);
      loom_type_t* replaced_params = NULL;
      IREE_RETURN_IF_ERROR(loom_module_replace_type_ref_sequence(
          module, loom_type_dialect_params(type), param_count, old_id, new_id,
          &replaced_params, out_changed));
      if (!*out_changed) return iree_ok_status();
      loom_type_t replaced_type = loom_type_dialect(
          loom_type_dialect_name_id(type), param_count, replaced_params);
      return loom_module_intern_type(module, replaced_type, out_type);
    }

    case LOOM_TYPE_PARAMETERIZED: {
      const loom_parameterized_type_descriptor_t* descriptor =
          loom_type_parameterized_descriptor(type);
      uint8_t parameter_count = loom_type_parameterized_parameter_count(type);
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      loom_attribute_t replaced_parameters[UINT8_MAX];
      bool changed = false;
      for (uint8_t i = 0; i < parameter_count; ++i) {
        bool parameter_changed = false;
        IREE_RETURN_IF_ERROR(loom_module_replace_attribute_value_refs_impl(
            module, parameters[i], old_id, new_id, /*depth=*/1,
            &replaced_parameters[i], &parameter_changed));
        changed = changed || parameter_changed;
      }
      if (!changed) return iree_ok_status();
      *out_changed = true;
      return loom_module_make_parameterized_type(
          module, descriptor, replaced_parameters, parameter_count, out_type);
    }

    case LOOM_TYPE_REGISTER: {
      const loom_type_t* value_type = loom_type_register_value_type(type);
      if (!value_type) return iree_ok_status();
      loom_type_t replaced_value_type = *value_type;
      IREE_RETURN_IF_ERROR(loom_module_replace_type_value_refs_impl(
          module, *value_type, old_id, new_id, &replaced_value_type,
          out_changed));
      if (!*out_changed) return iree_ok_status();
      return loom_module_intern_register_type(
          module, loom_type_register_payload0(type),
          loom_type_register_payload1(type), replaced_value_type, out_type);
    }

    default:
      break;
  }

  loom_type_t replaced_type = type;
  if (loom_module_type_has_replaceable_dims(type)) {
    uint8_t rank = loom_type_rank(type);
    if (loom_type_has_inline_dims(type)) {
      for (uint8_t i = 0; i < rank; ++i) {
        if (!loom_dim_is_dynamic(replaced_type.dims[i])) continue;
        if (loom_dim_value_id(replaced_type.dims[i]) != old_id) continue;
        replaced_type.dims[i] = loom_dim_pack_dynamic(new_id);
        *out_changed = true;
      }
    } else if (rank > 0) {
      const loom_overflow_dim_t* old_dims =
          (const loom_overflow_dim_t*)(uintptr_t)type.dims[0];
      if (!old_dims) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rank-%u type has a NULL overflow dim payload",
                                rank);
      }
      bool dims_changed = false;
      for (uint8_t i = 0; i < rank; ++i) {
        if (!loom_dim_is_dynamic(old_dims[i])) continue;
        if (loom_dim_value_id(old_dims[i]) == old_id) dims_changed = true;
      }
      if (dims_changed) {
        loom_overflow_dim_t* new_dims = NULL;
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            &module->arena, rank, sizeof(loom_overflow_dim_t),
            (void**)&new_dims));
        for (uint8_t i = 0; i < rank; ++i) {
          new_dims[i] = old_dims[i];
          if (!loom_dim_is_dynamic(new_dims[i])) continue;
          if (loom_dim_value_id(new_dims[i]) == old_id) {
            new_dims[i] = loom_dim_pack_dynamic(new_id);
          }
        }
        replaced_type.dims[0] = (uint64_t)(uintptr_t)new_dims;
        replaced_type.dims[1] = 0;
        *out_changed = true;
      }
    }
  }

  if (old_id <= UINT16_MAX && loom_type_has_ssa_encoding(type) &&
      loom_type_encoding_value_id(type) == old_id) {
    if (new_id > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "cannot store value %%%u in a 16-bit SSA encoding reference",
          (unsigned)new_id);
    }
    replaced_type.encoding_id = (uint16_t)new_id;
    *out_changed = true;
  }

  if (!*out_changed) return iree_ok_status();
  return loom_module_intern_type(module, replaced_type, out_type);
}

iree_status_t loom_module_replace_type_value_references(
    loom_module_t* module, loom_type_t type, loom_value_id_t old_id,
    loom_value_id_t new_id, loom_type_t* out_type, bool* out_changed) {
  *out_type = type;
  *out_changed = false;
  if (old_id == new_id) return iree_ok_status();
  if (old_id >= module->values.count || new_id >= module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot replace type references from %%%u to %%%u in a module with "
        "%" PRIhsz " values",
        (unsigned)old_id, (unsigned)new_id, module->values.count);
  }
  return loom_module_replace_type_value_refs_impl(module, type, old_id, new_id,
                                                  out_type, out_changed);
}

iree_status_t loom_module_replace_value_type_uses(loom_module_t* module,
                                                  loom_value_id_t old_id,
                                                  loom_value_id_t new_id) {
  if (old_id == new_id) return iree_ok_status();
  if (old_id >= module->values.count || new_id >= module->values.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "cannot replace type references from %%%u to %%%u in a module with "
        "%" PRIhsz " values",
        (unsigned)old_id, (unsigned)new_id, module->values.count);
  }

  // Each iteration rewrites one carrier value reached from the incoming-use
  // head. The helper removes all outgoing type-use records for that carrier
  // before inserting the replacement records, so the next head is always a
  // still-unprocessed carrier without rescanning the table.
  while (loom_module_value_has_type_uses(module, old_id)) {
    loom_type_use_id_t use_id =
        loom_module_value_first_incoming_type_use(module, old_id);
    loom_value_id_t user_value_id =
        module->type_uses.records[use_id].user_value_id;
    loom_type_t old_type = loom_module_value_type(module, user_value_id);
    loom_type_t new_type = old_type;
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_refs_impl(
        module, old_type, old_id, new_id, &new_type, &changed));
    if (!changed) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "type-use table says value %%%u references %%%u, but the type does "
          "not contain that reference",
          (unsigned)user_value_id, (unsigned)old_id);
    }
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, user_value_id, new_type));
  }
  return iree_ok_status();
}
