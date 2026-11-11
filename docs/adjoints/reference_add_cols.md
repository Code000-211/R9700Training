# `ref_add_cols`

A private op of the reference step, which composes the tiny fixture's graph. It reads `y`, a
`[M, row]` buffer, and `inc`, an `[M, W]` block, and writes `y` with `inc` added into columns
`off .. off + W`; `off` is a parameter. The adjoint passes the output cotangent straight through to
`y` and takes the block's cotangent out of the same column range. It exists because a LoRA expand
of one module in a fused weight must land in that module's columns without widening the module's
`B`.
