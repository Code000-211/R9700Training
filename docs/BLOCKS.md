# The blocks

Each adjoint family has its own check: `docs/adjoints/elementwise.md`, `linear.md`, `qk.md` and
`attention.md` check one op against finite differences, in f64, on shapes of their own choosing.
This is the check that the four of them fit together: one full-attention block and one MLP block,
run as the bf16 fixture issues them, through the registered twins and adjoints.

`tests/host/block_attention_mlp.cpp` is the test, `tests/support/block_check.h` the machinery
around the registry it needs (`twin_run`, `adjoint_run`, `kernel_run`, one random direction and a
central difference). It adds no adjoint and patches no family: a mismatch it finds is reported to
the family that owns the op.

## The two blocks

Both take the fixture's residual in and give the residual out, so the block's own cotangent is the
residual's, and the check is a gradient check through every op between them.

The attention block is layer 3's, the only layer whose `(l + 1) % interval == 0` makes it a full
attention layer:

| # | op | what it is here |
|---|---|---|
| 1 | `rmsnorm` | `input_layernorm`, Gemma-style: `wadd` 1, eps 1e-6 |
| 2 | `gemm_nt` | `q_proj`, `[T, 128] → [T, 2·heads·head_dim]` |
| 3 | `rmsnorm` | `q_norm`, over the `T·heads` query rows of `head_dim` 32 |
| 4 | `gemm_nt` | `k_proj`, `[T, 128] → [T, kv_heads·head_dim]` |
| 5 | `rmsnorm` | `k_norm`, over the `T·kv_heads` key rows |
| 6 | `gemm_nt` | `v_proj` |
| 7 | `rope` | `q` and the keys, `rotary_dim` 16 of 32, theta 10000, gptj halves |
| 8 | `kv_store` | the keys and values into the paged pool |
| 9 | `attn_paged` | gqa 2, causal, no window, `block_size` 16, scale `1/√32` |
| 10 | `sigmoid` | the gate half of `q_proj`'s row |
| 11 | `mul` | the gate against the attention output |
| 12 | `gemm_nt` | `o_proj` |
| 13 | `add` | the residual |

and the MLP block is layer 0's, the manifest's `mlp_sublayer` case:

| # | op | what it is here |
|---|---|---|
| 1 | `rmsnorm` | `post_attention_layernorm` |
| 2 | `gemm_nt` | the fused gate-and-up projection, `[T, 128] → [T, 2·ffn]` |
| 3 | `silu_mul` | `silu(gate) · up`, the gate in the low half of each row |
| 4 | `gemm_nt` | `down_proj`, `[T, ffn] → [T, 128]` |
| 5 | `add` | the residual |

Two details of the fixture the blocks have to reproduce, both of them the model's and not this
file's:

- `q_proj`'s row is `[heads · (query 32, gate 32)]`: each head's query and its gate are adjacent.
  `q_view`/`gate_view` take the two halves; the gate is not a second matrix.
- `q_norm` and `k_norm` run **before** `rope`, on the head's 32 values, with `wadd` 1.

The pool is `[n_blocks, kv_heads, block_size, 2·head_dim]`, the key at the slot's first `head_dim`
and the value at the next, which is where `attn_paged` reads them (`src/adjoint/host/attention.cpp`
says so at the top) and where `kv_store` writes them. The slot of a row is `kv_slot`'s: from the
sequence's row of the block table and the position in the sequence. That agreement is what makes
the two families chain, and it is what the batch case exercises.

## The three checks

| check | what it compares | tolerance | measured worst |
|---|---|---|---|
| finite difference | `⟨adjoint(dx), d⟩` against a central difference of `⟨forward(x + t·d), cot⟩`, `t` = 1e-6 | 1e-6 relative | 3.1e-8 |
| the reference | output against `{attention,mlp}_sublayer.{single,batch}.output`, composed adjoint against `input_cotangent` | 1e-8 relative | 4.2e-16; the attention batch's padded sequences are the documented exception |
| the host kernels | `forward`'s f64 twin chain against the same op sequence run through the registered host kernels in bf16 | 1e-2 relative | 4.6e-3 |

The finite difference is one random direction and a central difference, not one direction per
input element: one forward pair then touches every element of the block's input, and the per-op
checks in `tests/host/adjoint_*.cpp` are the ones that take the gradients apart element by element.
The tolerance is 1e-6 and the measured error 3e-8, so the difference is the difference scheme's.

The reference comparison is the independent one: it shares no code and no derivation with the
twins. `docs/ORACLE.md` says what the cases hold. The single cases are compared whole. The batch
cases are compared on their packed valid rows, because the reference's batch is padded and no valid
row reads a padded one; the attention batch's *input cotangent* is the exception, since a padded row
does read the valid ones, and `docs/ORACLE.md` has the cause, the measurement and the tolerance that
follows from it (`kPaddedRowTolerance` in the test).

The kernel chain is the same op sequence issued the other way: the records above, run through
`libref`'s host kernels by the registry's kernel rows rather than through the twins, in bf16 as the
fixture runs them. It is what the twins are a twin *of*, so a mismatch means the family's twin and
the family's kernel disagree, and the measured 4.6e-3 is bf16's rounding through the block's
thirteen or five ops, not a bug.

## The batch, and the pool cotangent's addressing

The single case is one sequence of 23 tokens: two pool blocks of 16, the second holding 7. The batch
is three sequences of 17, 11 and 6, packed, with its own blocks: 2 for the first sequence (17
crosses the 16-token boundary), 1 each for the other two, 4 pool blocks with the block table
`[[0, 1], [2, -1], [3, -1]]` and a stride of 2. `slot_mapping` is per packed row, from `kv_slot`, so
the second sequence starts at slot 32 and the third at 48.

The adjoint of `attn_paged` writes the pool's cotangent, and the adjoint of `kv_store` gathers out of
it: each row's key and value cotangent is the pool's cotangent at that row's slot, which is the sum
over every row that read it. With three sequences of different lengths that sum has to land on the
right sequence and the right position, and the finite-difference check on the whole batch is what
says it does — the pool is 4 blocks wide with a `-1` in the block table, so a pool cotangent that
lands one block or one position off changes the check's answer.

The rope positions are the packed row's index, not the position in its sequence: that is what the
reference's block case hands the block (`make_reference.py`'s `arange` over the packed rows), and it
is what both sides are compared on.

## Two things the records make sharp

`attn_paged`'s operands keep their schema positions: `block_table`, `seqused`, `k_descale`,
`v_descale`, `cu_seqlens`. A record that writes `cu_seqlens` but leaves the two descales out has the
twins read the `cu_seqlens` as the k descale (`attention.cpp`'s `listed` reads operand 4), and the
scores come out flat: the forward then differs by 0.18 relative, which is how this file found it.
The descales are in the record at one, before the `cu_seqlens`. Nothing is wrong with the family,
which documents the positions as fixed; but a record that means to pass a ragged batch has to carry
them.

`rope` and `kv_store` have no twin and no adjoint row in the registry (`src/adjoint/host/qk.cpp`
registers neither; it exports `rope_forward_f64`, `rope_adjoint_f64`, `kv_store_forward_f64` and
`kv_store_adjoint_f64`). The block check calls those four functions directly, so what it checks is
the qk family's f64 code and the slot agreement with `attn_paged`, not a walk through the registry.
Until the qk family registers the two ops, the composed walk cannot bridge attention and linear
through the registry. That is reported here rather than patched: the rows belong to `qk`, and this
issue adds no adjoint.

## Where the weights come from

The block cases' weights are the fixture checkpoint's own layer 0 and layer 3 tensors, read the way
`tests/support/fixture.h` reads them: the manifest's `cases.blocks` names the layers, and the
checkpoint the writer writes is the one the reference was computed from.

The checkpoint keeps `mlp.gate_proj.weight` and `mlp.up_proj.weight` separate; no tensor holds the
fused projection. The converter concatenates their rows, gate first, into the one
`[2·ffn, hidden]` parameter the block reads (radiance's `rad_block_mlp.h`, `map_concat`), so the
test builds the same weight the same way and the block's single `gemm_nt` sees exactly what the
fixture's single record sees. The block issues one projection, as the table above says, not two
that the test then joins.

The test writes that checkpoint itself, with `r97::write_checkpoint` into a directory of its own,
instead of calling `r97::fixture()`. Its sha256 is the manifest's `fixture.sha256`, so the bytes are
the fixture's; what a `mkdtemp` buys is that the directory outlives the test. See the next section.

## Two failures this file met and did not fix

Both are in the harness and predate this test; both are reported, not worked around in shared code.

- `tests/support/fixture.h`'s cache and `radtest::tmpdir()` are function-local statics, and on this
  tree the first of them is created during static initialisation, by chatdata's `R97_SELF_CHECK`. A
  `R97_TEST` runs from an exit-time handler (`tests/support/test.h`), which runs after those statics
  have been destroyed, so every fixture-based test that runs there sees empty paths: on an untouched
  tree the host binary reports 60 tests and 14 failures, all of them missing fixture, plugin or
  oracle files. This test does not depend on them.
- `test_loss_ce_objective_loss` fails two `CHECK`s: `tests/host/loss_ce_objective_loss.cpp:109`
  (`got.loss == want.loss` for a window of the objective) and `:116` (the sum over the windows),
  and `R97_SELF_CHECK`'s `CHECK` calls `std::abort()`. It runs during static initialisation, so the
  whole host binary aborts there and no `R97_TEST` in it runs at all. Run by hand with that one
  self-check skipped, this tree's host binary reports 64 tests and 14 failures, the same 14 as
  above; the four block tests are among the 50 that pass.

## Reproducing

```sh
cmake -B build -G Ninja -DR97_WITH_HIP=OFF
cmake --build build --target r97_host_tests
./build/r97_host_tests        # or: ctest --test-dir build -L host
```

On this tree the binary aborts in a self-check before any `R97_TEST` runs, as the section above
says, so the four tests are read out of it with that one self-check skipped locally.
