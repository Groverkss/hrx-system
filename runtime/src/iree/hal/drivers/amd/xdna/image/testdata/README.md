# XDNA execution fixtures

`mul_i32.loom` produces both native images used by the loader and execution
CTS. The `mul_i32` export multiplies sixteen little-endian i32 pairs, retaining
each product's low 32 bits. Its three buffer bindings are read-only lhs,
read-only rhs and write-only output, each 64 bytes.

Both images use one column and six rows with a resident compute worker. The
establishing invocation loads code and configures the array; its continuation
reuses that state. Each finite invocation completes at the output DMA wait.
Load ranges splice shared file bytes into final command backing, and declared
binding relocations update both invocation ranges.

Regenerate from the repository root after building `loom-compile`:

```sh
iree-bazel-run //loom/src/loom/tools/loom-compile -- \
  runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32.loom \
  --root=@mul_i32 --target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11 \
  --format=xdna \
  --output=runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32.xdna
iree-bazel-run //loom/src/loom/tools/loom-compile -- \
  runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32.loom \
  --root=@mul_i32 --target=amd.xdna.aie2p:amd.xdna.strix.17f0_10 \
  --format=xdna \
  --output=runtime/src/iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.xdna
```

Strix and Krackan share the NPU4 execution-profile identity; Halo uses its own
profile. The native CTS selects the matching intact image from the endpoint,
checks exact numerical results with changing inputs, and exercises independent
context lifetimes and caller-owned shared data. Host loader tests exercise
segmented source ownership and malformed external metadata through the actual
image reader. They do not decode or emulate tile instructions.
