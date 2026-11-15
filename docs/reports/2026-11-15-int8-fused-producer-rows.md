# The int8 fused producers: host rows, and their own finite differences

2026-11-15. Closes the gap `docs/reports/2026-11-14-device-backward.md` names as its
next step, and removes the last op group on the backward path that had no finite
difference on either side.

## The gap

`rmsnorm_had_quant_i8g`, `gated_had_quant_i8g` and `gate_had_quant_i8g` had **device rows only**
(`src/adjoint/device/quant.hip:874-876`). The GPU tier checked them against a host *composition* of
the same parts, written inside the test TU (`tests/gpu/adjoint_quant.cpp`).

That is a relative comparison, and a relative comparison cancels any factor the two sides share --
which is exactly how the `gemm_nt_q` activation-scale omission survived every green test. **208 of
the walk's 1459 records are these three ops** (14%), and they produce every code plane the 288
`gemm_nt_q` consumers read.

## What was added

### The rows (`src/adjoint/host/quant.cpp`)

The three producers are an elementwise part composed with a quantiser, so their host twin is that
composition and nothing else:

1. the part's twin (`rmsnorm` / `silu_mul` / `act_mul`, in `src/adjoint/host/elementwise.cpp`) over
   the fused record's read operands;
2. the `had_quant_act_i8g` quantiser's twin over a **synthetic** record built from the fused
   record's `n` and `group`;
3. on the way back, the quantiser's adjoint, whose output cotangent the part's adjoint then folds
   into the read slots.

The rows are keyed by the same predicates the device side keys its rows with, so one record selects
one row on both sides: `n` and `group` numeric, `n` a multiple of `group`, `group` a power of two no
larger than 1024; `gated_had_quant_i8g` serves only `act` absent or `"silu"`, and
`gate_had_quant_i8g` only `act == "sigmoid"`. `rmsnorm_had_quant_i8g`'s gain is frozen at schema
position 1, as the device row freezes it.

The test TU's own copies of that composition are deleted, so the registry is the only definition.

### The linearised rows (`<op>_ste`)

These ops are **quantisers**: their forward rounds, so a difference quotient against the plain twin
would differentiate a step rather than measure a slope. Each therefore gets a linearised row, as the
fp8 producers in the same family already have.

The base point -- the part's own output, the activation the quantiser saw -- is appended to the schema
after the codes and the scale, and the frozen list names that position:

| op | schema | base at |
|---|---|---|
| `rmsnorm_had_quant_i8g_ste` | `x, w, codes, scale, base` | 4 |
| `gated_had_quant_i8g_ste` | `gate_up, codes, scale, base` | 3 |
| `gate_had_quant_i8g_ste` | `a, b, codes, scale, base` | 4 |

Unlike the fp8 producers, whose twins do the linearisation themselves, these reach the quantiser's
own `had_quant_act_i8g_ste` row, so the base point is the second operand of the **quantiser's**
record rather than of the fused one. `quant_twin` already implements that (`src/adjoint/host/quant.cpp`,
the `is_frozen` branch), so the composition only has to propagate `is_linearised(rec.op)` into the
synthetic record's name and pass the base point through.

One consequence worth recording: `fused_adjoint` used to locate the codes and the scale by counting
the record's reads and indexing `fwd[reads]` / `fwd[reads + 1]`. That is correct only while the
writes are contiguous after the reads, and an appended base point breaks it. The row now finds them
by **dtype and shape**, as the device row does (`src/adjoint/device/quant.hip:492-523`).

## The measurement

Both tiers build, and all six rows are held to their own finite differences at two shapes:

| op | trunk-128 | two-groups |
|---|---|---|
| `rmsnorm_had_quant_i8g_ste` | 1.39e-09 | 2.65e-10 |
| `gated_had_quant_i8g_ste` | 3.54e-11 | 1.38e-10 |
| `gate_had_quant_i8g_ste` | 3.61e-10 | 6.15e-10 |

The fp8 fused producers' own finite differences run in the same test binary and read 1e-12 to 1e-9,
so the harness was not the thing being measured.

The full host tier is green: **100% tests passed, 0 failed out of 24**.

## The defect the first run found -- in the new row

The first attempt froze only the base point, and `rmsnorm_had_quant_i8g_ste` came back
`adjoint -664.832955, finite differences -812.442756`: **18%** out.

The cause is that the plain row freezes the norm's **gain** at schema position 1, and
`rmsnorm_host_adjoint` correspondingly leaves that cotangent at zero -- the walk never asks for it.
With the `_ste` row freezing only the base point, `check_adjoint` perturbed the gain too and measured
a `dw` the adjoint deliberately does not produce.

`check_adjoint` skips frozen slots (`tests/support/adjoint_check.h:369-371`, `:426`) and
`R97_REGISTER_ADJOINT_IF_F` takes the frozen list as `{__VA_ARGS__}`
(`include/r97/adjoint.h:232-235`), so the fix is to freeze both:

```c++
R97_REGISTER_ADJOINT_IF_F("rmsnorm_had_quant_i8g_ste", host, rmsnorm_had_quant_i8g_adjoint, rotated, 1, 4)
```

Worth recording because it is the same shape of mistake the family exists to catch, one level up: a
comparison that measures a slot only one of the two sides fills proves the wrong thing, exactly as a
comparison of two sides that share a factor proves nothing.

## What this does not do

It does **not** name the ~2e33 amplification of `docs/reports/2026-11-14-device-backward.md` (#55). It
removes the last op group on the backward path that had no finite difference on either side, and it
establishes that these three producers' **host** rows are the exact derivative of their own forward --
so a factor shared between the host and device rows is now excluded there too.

What remains unprobed on this path is the **device** side: `src/adjoint/device/quant.hip` computes the
codes' cotangent as `cot / s` in f32, and `gdn_o`'s cotangents reach 1e35..1e38 against a live scale as
small as 5.96e-05, i.e. 1.7e42, which is over f32's 3.4e38 ceiling. That is issue **#140**, and an
overflow gives `Inf` rather than a multiplier, so it is a separate range defect that makes `all_finite`
fail for its own reason.
