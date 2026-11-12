# The reference step's rows

`r97train train --reference` runs the tiny fixture's whole op graph on the f64 twins, live and in
process. That graph shows shapes a single family's check never does: a flat `attn_q` where
`attn_paged` expects `[rows, heads, head_dim]`, a module reading its slice of a fused buffer, a
rank-3 adapter input, and a record whose buffers carry the fixture's names rather than the
schema's. Rather than widen the family rows every other walk uses, the reference step tags its
records with `r97ref=1` and `src/adjoint/host/reference_<family>.cpp` registers rows for the same
op names behind that tag. The host join's records that split an activation by head carry
`r97join=1` and reach the same rows, because the engine records them in the reference-shaped
form; the record's `fused_slots` option decides whether a slot's factor is named by its module or
written into one fused gradient. The registry prefers a predicated row to a catch-all, so no other
walk reaches them; `docs/adjoints/<op>.md` stays the description of the family row.

| file | what it adds |
| --- | --- |
| `reference_attention.cpp` | `attn_paged` over a flat `[rows, heads * head_dim]` `q`, as the fixture records it. |
| `reference_elementwise.cpp` | `rmsnorm` and `sigmoid` over a column slice of a fused `[rows, groups, 2 * n]` buffer; `sigmoid` recomputes its output instead of reading a snapshot. |
| `reference_gdn_prep.cpp` | `gdn_conv_prep` over a fused gate buffer whose `a` and `b_gate` are the last two `H` columns. |
| `reference_linear.cpp` | `gemm_nt*` with the fixture's buffer names, resolved through the schema's read order. |
| `reference_lora.cpp` | a rank-3 `lora_shrink` input and per-slot gradient names for a fused weight, for the reference step and the host join's head-split records. |

The reference step also has two private ops of its own, `ref_add_cols` and `ref_select_rows`,
because it composes fused LoRA blocks into their columns and the full logits row space into the
objective's scored rows; `docs/adjoints/reference_add_cols.md` and
`docs/adjoints/reference_select_rows.md` say what each does.
