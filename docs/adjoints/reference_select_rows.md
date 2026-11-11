# `ref_select_rows`

A private op of the reference step, which composes the tiny fixture's graph. It reads `x`, a
`[rows, V]` buffer, and `idx`, an `i32` row list, and writes `y[r, :] = x[idx[r], :]`. The adjoint
scatters the output cotangent back into the selected source rows. The forward writes one logits row
per token, the BOS row included, while the objective's boundary has one row per scored token; this
op is where the two row spaces meet.
