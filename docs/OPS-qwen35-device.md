# The device step on qwen3.8-27b-iq4r-s-v4

`docs/OPS-qwen35-host.md` lists the ops of the tiny bf16 and int8 fixtures on the host tier. This
file is its device counterpart: the ops `qwen3.8-27b-iq4r-s-v4.rad` really issues on an R9700 when
the training forward runs on `libr4d`, and for each fused device form the host chain it stands in
for.

The container is `qwen3.8-27b-iq4r-s-v4.rad` (15.4 GiB, architecture `qwen35`, 64 layers, hidden
5120, 262144-token context, a 248320-row vocabulary of which 248077 are real tokens, IQR-V4 coded
weights). The tier is
`libr4d`; the key-value pool is bf16; speculation, the drafter, the MTP head and the vision tower
are off (`n_spec=0`, `mm_max_patches=0`), which is what the step's graph does not contain.

## The forward-only StepFn on the card

The device forward is the same `ForwardDriver` the host tier runs (`r97/forward.h`): a
`ForwardConfig` with `kernels = {"libr4d"}`, `kv_dtype` bf16, `max_tok`/`max_ctx` sized for the
step's sequences and `max_seqs` its sequence count; the driver itself turns speculation and the
vision tower off (`n_spec=0`, `mm_max_patches=0`) and asks for a linear-attention checkpoint
interval of at least `max_ctx`, since a prefill is chunked at most that far and a second chunk
would be a second step. `driver->step(ce_model())` returns the forward-only `StepFn` (`r97/step.h`):
it runs the batch as one prefill step on the card, ignores `grads`, and leaves the step in
`driver->last()`, whose `logits` boundary `CeObjective` scores. `tools/loss.cpp` is that StepFn
behind a command, `src/loss_run.cpp` is the batched loop the tests use, and
`tests/gpu/device_forward.cpp` compares it with the host tier and with `radiance --kld-record`.

## The inventory

```
ROCR_VISIBLE_DEVICES=0 r97train ops --device 0
# /srv/models/qwen3.8-27b-iqr-v4/qwen3.8-27b-iq4r-s-v4.rad: one prefill step, 758 issues, 22 op forms
```

One 8-token prefill step of this container at `max_tok` 512 issues 758 launches over 22 op forms:

| op | block | form | count | twin | host | device | seed | on path | adjoint |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `embed_lookup` | embedding | dtype=bf16 | 1 | yes | yes | - | - | yes | linear |
| `rope_table` | head | dtype=f32 | 1 | - | - | - | - | yes | qk |
| `rmsnorm_had_quant_i8g` | gated-delta-net | dtype=bf16 | 48 | - | - | yes | - | yes | quant |
| `gemm_nt` | gated-delta-net | dtype=bf16 | 48 | yes | yes | - | - | yes | linear |
| `gemm_nt_q` | gated-delta-net | dtype=i8a8h | 96 | yes | yes | - | - | yes | linear |
| `gdn_conv_prep` | gated-delta-net | - | 48 | yes | yes | yes | - | yes | gdn_prep |
| `gdn_kkt_solve` | gated-delta-net | - | 48 | yes | yes | yes | - | yes | gdn_kkt |
| `gdn_chunk_scan` | gated-delta-net | - | 48 | yes | yes | yes | - | yes | gdn_scan |
| `gdn_gated_rmsnorm` | gated-delta-net | act=silu | 48 | yes | yes | yes | - | yes | gdn_prep |
| `had_quant_act_i8g` | gated-delta-net | dtype=bf16 | 48 | yes | yes | yes | - | yes | quant |
| `rmsnorm_had_quant_i8g` | mlp | dtype=bf16 | 64 | - | - | yes | - | yes | quant |
| `gemm_nt_q_gated` | mlp | dtype=i8a8h act=silu | 64 | - | - | - | - | yes | none |
| `gemm_nt_q` | mlp | dtype=i8a8h | 64 | yes | yes | - | - | yes | linear |
| `rmsnorm_had_quant_i8g` | full-attention | dtype=bf16 | 16 | - | - | yes | - | yes | quant |
| `gemm_nt_q` | full-attention | dtype=i8a8h | 64 | yes | yes | - | - | yes | linear |
| `qk_norm_rope_gate_kv_store` | full-attention | q_dtype=bf16 kv_dtype=bf16 | 16 | - | - | yes | - | yes | qk |
| `attn_paged` | full-attention | q_dtype=bf16 kv_dtype=bf16 | 16 | yes | yes | yes | - | yes | attention |
| `gate_had_quant_i8g` | full-attention | dtype=bf16 act=sigmoid | 16 | - | - | yes | - | yes | quant |
| `add` | mlp | dtype=bf16 | 1 | yes | yes | yes | - | yes | elementwise |
| `rmsnorm` | head | dtype=bf16 | 1 | yes | yes | yes | - | yes | elementwise |
| `gather_rows` | head | dtype=bf16 | 1 | yes | yes | - | - | yes | linear |
| `logits_gemm_vocab` | head | dtype=bf16 | 1 | - | - | - | logits | yes | none |

Read it as `docs/OPS-qwen35-host.md` reads the host table: `count` is launches in this one step,
`twin` whether the host and device adjoints agree, and `host`/`device` which of r97's own two
adjoint sides exist for the form. The `-` in the `host` column is expected for the fused forms:
the host tier's chain is the unfused ops, so an adjoint for the fused form exists on the device
side only. The run also prints how many radiance ops have no kernel at all on the host tier,
"38 of 1140 declared ops have no host kernel", which includes `rmsnorm_had_quant_i8g`; that is
why a host step cannot issue this path, and against this build the host tier refuses the
container up front, `qwen35_w8a8: rad_arch_declare returned invalid argument`.

The block split is the architecture's: 48 gated-delta-net layers, 16 full-attention layers, one
MLP and one attention norm a layer, one MLP gate/up and one down/out GEMM a layer each side.

## Fused device forms and the host chain each replaces

These are the forms on this container's path that do more than one host op's work. The chain is
the host op sequence radiance's own row text and `core/docs/OPS.md` say the form is byte-identical
to; on the host tier that sequence is what an int8 container of this architecture issues.

| device form | host chain |
| --- | --- |
| `rmsnorm_had_quant_i8g` | `rmsnorm` (residual added, bf16 out) → `had_quant_act_i8g` (FWHT rotate, int8 group codes) |
| `gemm_nt_q_gated` | `gemm_nt_q` → `gated_had_quant_i8g` (silu product, rotate, int8 group codes) |
| `gate_had_quant_i8g` | `sigmoid` → `mul` (the attention gate) → `quant_act_fp8` (rotate, int8 codes) |
| `qk_norm_rope_gate_kv_store` | `rmsnorm` (q), `rmsnorm` (k), `rope` (q), `rope` (k), `kv_store` |
| `qk_norm_rope_gate` | the same four launches without the store |
| `qk_norm_rope_gate_kv_store_streamed` | the four launches, then `kv_store_streamed` |

Per form:

- **`rmsnorm_had_quant_i8g`** (128 launches here: 48 + 64 + 16) — the residual add, RMS norm and
  the int8 quantiser of `had_quant_act_i8g` in one launch. The row text: "one launch for a
  residual add, RMS norm and quantiser ... byte-identical to `rmsnorm` + `had_quant_act_i8g`".
  Every block on this container opens this way, and the MLP's norm is the same form.
- **`gemm_nt_q_gated`** (64 launches, the MLP's gate/up projection) — the GEMM writes the gated
  product already rotated and int8-coded, so the separate product and quantiser launches go away.
  Its device row is `gemm_iqr_gated`, declared for `M <= 16` only (see the IQR rows below). The
  plain `gemm_nt_q`
  (224 launches: 96 GDN, 64 MLP down, 64 attention) has no such fold.
- **`gate_had_quant_i8g`** (16 launches) — `gate_quant_fp8` (`act=sigmoid`) with the rotate and
  int8 group codes folded in; radiance's row text calls `act` none exactly `mul` +
  `quant_act_fp8`, and sigmoid is the gated attention's gate. It saves the launch before the
  attention's rotated out projection.
- **`qk_norm_rope_gate_kv_store`** (16 launches, all of this container's full-attention layers) —
  the gated-attention QKV preamble: split `[q|gate]` per head, RMS-norm q and k over the head,
  partial NeoX RoPE and the gate copy (a column read of the projection, which needs no op of its
  own), and the store into the bf16 paged cache, one launch. The unfused chain is the four
  norm/rope launches plus `kv_store`; radiance's selftest checks the fused output byte for byte
  against the stored-bf16 reference.
- **`qk_norm_rope_gate`** and **`_kv_store_streamed`** are the same prologue without the store,
  and with `kv_store_streamed`'s host-memory path. Neither is on this step: the pool fits in
  VRAM, and the head's `kv_store` is the resident one. They are the forms a decode step or a
  pool streamed to host memory would issue.

Two fused forms the issue names are not in the inventory's 8-token step; the same table and the
architecture's own declaration text say why:

- **`gated_had_quant_i8g`** — the MLP's silu product and quantiser. At `M <= 16` the architecture
  folds it into `gemm_nt_q_gated` and the standalone form is not issued, which is why the
  inventory's 8-token step does not show it. A prefill step past 16 rows issues it, right after
  the `gemm_nt_q` that produces the `[M, 2n]` bf16 plane; `gemm_iqr_gated`'s own text says it is
  byte-identical to `gemm_iqr_nt_m64` followed by this form.
- **`attn_paged_gate_quant`** — the single-launch attention plus gate plus quantiser exists in
  `libr4d` only for the decode head shapes (`h256` `gqa12`/`gqa6`, `bs4`), where its gate is a
  slice of the attention output. Here the gate is its own tensor, so the step issues
  `attn_paged` and `gate_had_quant_i8g` as two launches; folding them is a decode optimisation,
  not a prefill one.

## The IQR GEMMs and the head

The container's weights are IQR-V4 (trellis-coded with a per-group affine pair; `docs/IQR.md`).
The op forms in the inventory are the same op names a non-IQR int8 container uses — `gemm_nt_q`,
`gemm_nt_q_gated`, `logits_gemm_vocab` — and the decode happens inside the device row that serves
the form, where the weight is read. Which row serves which shape:

| device row | op form it serves | shape it is picked for | host op it replaces |
| --- | --- | --- | --- |
| `gemm_iqr_nt_m64` | `gemm_nt_q` (i8a8h) | M ≤ 64 (a short step, a decode row) | `gemm_nt_q` |
| `gemm_iqr_tiled` | `gemm_nt_q` (i8a8h) | any M; the prefill rows | `gemm_nt_q` |
| `gemm_iqr_gated` | `gemm_nt_q_gated` (i8a8h, act=silu) | M ≤ 16, byte-identical to `gemm_iqr_nt_m64` + `gated_had_quant_i8g` | `gemm_nt_q` → `gated_had_quant_i8g` |
| `embed_lookup_iqr` | `embed_lookup` | every token | `embed_lookup` |
| `logits_gemm_vocab_iqr` | `logits_gemm_vocab` | the head's input rows, over the vocabulary's width classes | `logits_gemm` |
| `logits_gemm_iqr` | `logits_gemm` | the head when the vocabulary is not classed | `logits_gemm` |
| `logit_rerank_iqr` | `logit_rerank` | the drafter's rerank; off here (`n_spec=0`) | `logit_rerank` |

A 4096-token prefill step is `M = 4096`, past the 16-row window `gemm_iqr_gated` is declared for:
the architecture asks for the form at `M` up to `min(rows, 16)` (`decl_gated_fold_iqr` in
`arch/common/rad_fp8.h`, "a CPU-only engine takes the pair"), so a prefill issues `gemm_nt_q` and
`gated_had_quant_i8g` -- the pair the fused form is byte-identical to -- and the GEMM inside it is
served by `gemm_iqr_tiled`. The inventory's 8-token step is inside that window, which is why the
table above shows the fused form; so is the 1-token warm-up a load runs. `logits_gemm_vocab_iqr` groups the head's rows into width
classes so one launch decodes the whole vocabulary; it is the same product as the host's
`logits_gemm`, with each class's trellis decoded at its own width.

The step's logits rows are one f32 a vocabulary row, and `CeObjective` sums over all of them, so
the loss covers exactly the terms radiance's `--kld-record` sums for the same positions. A 4096-token sequence has 4095 scored positions, and
`r97/head.h` walks them in chunks of positions (64 rows a chunk by default, 256 in the driver's
window), so the head's `[positions, vocab]` work is chunked rather than materialised at once.

## Reproducing

```
# the inventory (one 8-token prefill step)
ROCR_VISIBLE_DEVICES=0 r97train ops --device 0

# the loss of a chat file on the card, with the VRAM after loading and the peak
ROCR_VISIBLE_DEVICES=0 r97train loss tests/fixtures/chat_32.jsonl \
    --device 0 --max-tokens 4096 --max-seqs 32 --window 256 --load-vram
```

`--device N` hides every card but the chosen HIP ordinal from the process before the device layer
initialises (`HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`), so radiance's rank 0 is that card.
Under a container whose `/dev/kfd` is group `render`, run the command in that group (`sg render -c
'...'`). The host tier cannot start this container (see the inventory section); the tiny int8
fixture is what compares the two tiers, in `tests/gpu/device_forward.cpp`.
