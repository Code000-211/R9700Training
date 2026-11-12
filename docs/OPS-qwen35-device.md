# The device kernel tier on the card: `r97train loss --device N` / `r97train ops --device N`

This is the card-side companion to `docs/OPS-qwen35-host.md`. The host tier (libavx + libref) is
the CPU **fallback**, and it runs *beside* the device set rather than instead of it: the w8a8 27B
trunk declares ops only libr4d implements, so a pure host kernel set cannot declare this container
(measured; see `docs/reports/2026-11-12-device-forward.md`). The **device tier** runs the 27B on one
card through libr4d's fused HIP kernels, with libavx/libref as the per-op fallback for anything
libr4d does not serve.

`--device N` does exactly two things, both before any device call:

1. `setenv("ROCR_VISIBLE_DEVICES", "N", 1)` -- the engine picks the card only through
   `ROCR_VISIBLE_DEVICES` (`third_party/radiance/core/engine_bringup.cpp`, `usable_devices`),
   which the ROCR runtime reads once, at init.
2. sets the kernel hierarchy to `{libr4d, libavx, libref}` when `--kernels` was not given.

The container below is the one on disk: `/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad` (`qwen35`,
`quant=w8a8_fwht128`, 64 layers, 262144 ctx). It is the **w8a8** container: its trunk linears are
`i8*bf16[1x128]/fwht128` served by `gemm_nt_q` at `i8a8h`, so its GEMM rows are the plain-int8
`gemm_i8a8_*` family, **not** `gemm_iqr_*` (those fire only for the narrower IQR recipes).

## Device op inventory

`r97train ops <container> --device 0` over one prefill step: 758 issues, 22 op forms. The
`device` column is libr4d's; `host` is libavx/libref's. `twin` means a same-named op exists in
both tiers.

| op | block | form | count | twin | host | device | seed | on path | adjoint |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `embed_lookup` | embedding | dtype=bf16 | 1 | yes | yes | - | - | yes | linear |
| `rope_table` | head | dtype=f32 theta=10000000 scale=1 | 1 | - | - | - | - | yes | qk |
| `rmsnorm_had_quant_i8g` | gated-delta-net | dtype=bf16 eps=1e-06 wadd=1 | 48 | - | - | yes | - | yes | quant |
| `gemm_nt` | gated-delta-net | dtype=bf16 | 48 | yes | yes | - | - | yes | linear |
| `gemm_nt_q` | gated-delta-net | dtype=i8a8h | 96 | yes | yes | - | - | yes | linear |
| `gdn_conv_prep` | gated-delta-net | - | 48 | yes | yes | yes | - | yes | gdn_prep |
| `gdn_kkt_solve` | gated-delta-net | - | 48 | yes | yes | yes | - | yes | gdn_kkt |
| `gdn_chunk_scan` | gated-delta-net | scale=0.088388347648318433 | 48 | yes | yes | yes | - | yes | gdn_scan |
| `gdn_gated_rmsnorm` | gated-delta-net | eps=1e-06 act=silu | 48 | yes | yes | yes | - | yes | gdn_prep |
| `had_quant_act_i8g` | gated-delta-net | dtype=bf16 | 48 | yes | yes | yes | - | yes | quant |
| `rmsnorm_had_quant_i8g` | mlp | dtype=bf16 eps=1e-06 wadd=1 | 64 | - | - | yes | - | yes | quant |
| `gemm_nt_q_gated` | mlp | dtype=i8a8h act=silu | 64 | - | - | - | - | yes | none |
| `gemm_nt_q` | mlp | dtype=i8a8h | 64 | yes | yes | - | - | yes | linear |
| `rmsnorm_had_quant_i8g` | full-attention | dtype=bf16 eps=1e-06 wadd=1 | 16 | - | - | yes | - | yes | quant |
| `gemm_nt_q` | full-attention | dtype=i8a8h | 64 | yes | yes | - | - | yes | linear |
| `qk_norm_rope_gate_kv_store` | full-attention | q_dtype=bf16 eps=1e-06 wadd=1 kv_dtype=bf16 | 16 | - | - | yes | - | yes | qk |
| `attn_paged` | full-attention | q_dtype=bf16 kv_dtype=bf16 | 16 | yes | yes | yes | - | yes | attention |
| `gate_had_quant_i8g` | full-attention | dtype=bf16 act=sigmoid | 16 | - | - | yes | - | yes | quant |
| `add` | mlp | dtype=bf16 | 1 | yes | yes | yes | - | yes | elementwise |
| `rmsnorm` | head | dtype=bf16 eps=1e-06 wadd=1 | 1 | yes | yes | yes | - | yes | elementwise |
| `gather_rows` | head | dtype=bf16 | 1 | yes | yes | - | - | yes | linear |
| `logits_gemm_vocab` | head | dtype=bf16 | 1 | - | - | - | logits | yes | none |

Three op forms are **device-only** — `libavx` and `libref` carry no implementation of
`rmsnorm_had_quant_i8g`, `qk_norm_rope_gate_kv_store` or `gate_had_quant_i8g` at all — and so are
`gate_quant_fp8`, `gated_had_quant_i8g`, `gdn_conv_recurrent_update_had`, `logits_gemm_vocab` and
`ar_rmsnorm_had_quant_i8g`, which the same graph reaches. The arch marks each fused form with
`rad_op_declared()` and declares an unfused chain in its place; that chain is not a host-only route
for this container, which fails to declare on a pure host kernel set (measured):
`r97train ops <container> --kernels libavx,libref` returns
`rad_arch_declare returned invalid argument` (see
`docs/reports/2026-11-12-device-forward.md`). Radiance's model is host-site execution *beside* the
device set (`core/engine_bringup.cpp:159`: "Whole-model host execution needs every op covered"),
not host *instead of* device.
`gemm_nt_q_gated` shows no same-named kernel in either tier in the inventory: the arch issues it
as a fused gate_up form (its device serving row is named differently). `gemm_nt_q`, `attn_paged`,
the `gdn_*` set, `add`, `rmsnorm` have same-named host rows.

## The fused ops the container issues, and the host chain each replaces

The arch (`qwen35_w8a8`) declares a fused op only after `rad_op_declared()` sees libr4d declare
it (`arch/qwen35_fp8/qwen35_fp8.cpp:871-890`). The arch declares the unfused chain below in its
place when it is not (`rad_op_declared`); the ops listed above have no host implementation, and a
pure host kernel set fails to declare this container (measured). Where the arch's byte-identity
reference names a primitive chain, that is given too.

| fused op (libr4d) | what it computes | host chain it replaces (in order) |
| --- | --- | --- |
| `rmsnorm_had_quant_i8g` | residual add (opt.) + RMS norm + width-128 Hadamard + per-(row,128) int8 quantiser, one pass; writes rotated int8 codes + f32 scale (+ opt. bf16 norm out, + residual INOUT) | `rmsnorm_quant_fp8` (norm+add, E4M3 codes, bf16 out) → `had_quant_act_i8g` (bf16 → rotated int8). Primitive chain: `rmsnorm` + `had_quant_act_i8g` |
| `gate_had_quant_i8g` | `sigmoid(gate) * attn_out` (gate = strided slice of `[q\|gate]`) + width-128 Hadamard + int8 quantiser, one pass; writes bf16 gated product (in place) + rotated int8 codes + f32 scale | `sigmoid` → `mul` → `quant_act_fp8` (dead fp8 copy) → `had_quant_act_i8g`. Primitive chain: `gate_quant_fp8` (= `sigmoid`+`mul`+`quant_act_fp8`) then `had_quant_act_i8g` |
| `qk_norm_rope_gate_kv_store` | gated-attention QKV preamble: split `[q\|gate]` per head, Q RMS norm, K RMS norm (in place), partial NeoX RoPE (q,k), **plus the paged k/v cache store**; writes `q` (de-interleaved, normed, rotated), `k`, and the cache | `rmsnorm` (q, strided) → `rmsnorm` (k) → `rope` (q) → `rope` (k) → `kv_store` |

`gemm_nt_q_gated` (the fused gate_up form) is `gemm_nt_q` (gate_up) → `gated_had_quant_i8g`
folded into the GEMM epilogue; the `[M,2n]` bf16 gate_up plane is never written. `attn_paged` and
the `gdn_*` ops are issued on both tiers by the same name (the device rows are different kernels,
not fusions).

### Notes

- **`attn_paged_gate_quant` is not issued here.** It needs `!w.attn.q8_rot`; on the w8a8
  rotated-int8 trunk the model sets `m.a_attn.q8_rot = true`, so the block issues `attn_paged`
  then `gate_had_quant_i8g` instead (matching the inventory above).
- **`gemm_iqr_gated` / `gemm_iqr_nt_m64` / `gemm_iqr_tiled` are not issued here.** They fire only
  for the narrower IQR recipes; this w8a8 container's GEMM rows are `gemm_i8a8_nt_m16` /
  `gemm_i8a8_tiled`, and its gate_up fold is `gemm_nt_q_gated`.
- **`qk_norm_rope_gate_kv_store` is issued only when `T <= qk_fuse_rows(g)`** and a cos/sin table
  operand exists (`arch/common/rad_block_attn_gated_fp8.h:306-316`).
- **`rmsnorm_had_quant_i8g` is chosen only when the model sets `q8_fed && q8_rot`** after
  `rad_op_declared("rmsnorm_had_quant_i8g")` (`arch/qwen35_fp8/qwen35_fp8.cpp:879-881`).
