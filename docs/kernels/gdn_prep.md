# `gdn_prep` on the card: `gdn_conv_prep` and `gdn_gated_rmsnorm`

The device adjoints of the `gdn_prep` family live in `src/adjoint/device/gdn_prep.hip`, are held to
the host adjoints by `tests/gpu/adjoint_gdn_prep.cpp`, and this is the third file the family's
device version owes. The host adjoints and the formulas they differentiate are
`src/adjoint/host/gdn_prep.cpp` and `docs/adjoints/gdn_prep.md`; nothing here re-derives them, and
the arithmetic below is the same arithmetic the host does, in f32 instead of f64.

## What the two ops compute

- `gdn_conv_prep` folds the gated delta net's pre-recurrent work into one record: the short causal
  convolution over the projected row (`width` taps, a zero cold-start state), the silu that follows
  it, the l2 normalisation of q and k, and the per-chunk gate `G = cumsum(g)` and `beta` of the
  recurrence. It reads x, the conv weight and bias, `A_log`, `dt_bias`, the sequence offsets, the
  gate and beta projections, the state slots and the warm-start flags; it writes q, k, v, the gate
  increments and beta, and the tail of the conv state.
- `gdn_gated_rmsnorm` is the gated RMS norm on the recurrence's output: `o_i = x_i * sc * w_i *
  g(z_i)` for one row of `channels`, `sc = rsqrt(mean(x^2) + eps)`, `g` the op's `act` (`silu` by
  default, `sigmoid` when the record says so). One row is one head of one token.

## The kernels

| kernel | one thread is | what it computes |
|---|---|---|
| `conv_acc_kernel` | (token, channel) | the `width` taps of the convolution, summed in f32, plus the bias; the phase's `acc`, kept for the two phases that need it |
| `l2_backward_kernel` | (token, head), one block | `ss = Σ u_d^2`, `dot = Σ u_d do_d`: a shared-memory reduction over the head's `K` channels, then `du_d = inv*do_d - inv^3*u_d*dot` and the silu's derivative |
| `v_backward_kernel` | (token, channel of v) | the v half of the conv row: the activation's derivative alone |
| `conv_backward_kernel` | (source row, channel) | the `width`-tap window that read this row, summed in f32, plus the cold-start state's cotangent on the last `width-1` rows; one rounding to bf16 |
| `gate_backward_kernel` | (sequence, head, chunk) | the within-chunk tail of `d_g`, in f32; `da = tail * (-exp(A_log)) * softplus'(a + dt_bias)`; `d_beta = d_beta_in * s(1-s)` |
| `gated_rmsnorm_adjoint_kernel` | (row), one block | the row's `Σ x^2` and `A = Σ x_e w_e g(z_e) do_e` as one shared-memory reduction, then `dx_e = w_e g(sc do_e) - x_e sc^3/ch A` and `dz_e = do_e x_e sc w_e g'(z_e)` |

Every channel of the conv row belongs to exactly one of q, k and v, so the three backward phases
store into `dacc` instead of adding to it; nothing in the conv op's backward needs an atomic. The
gate's tail is a suffix sum inside a chunk and the chunks are independent, which is why one thread
per chunk serves the whole (token, head) plane and why the tail's length (64 tokens at the real
geometry) costs nothing but the loop.

The conv op is served at a prefill from position 0, which is the only shape this family's records
describe: the carried state is zero, so `conv_state` is a written operand and no history is read,
`has_init` is frozen at 0 and nothing varies it, and what the op writes into the state is the tail
of `x`, whose cotangent lands on the same rows. The kernel serves that shape, as the host row does;
a warm start would need the record to read its state, which no caller writes yet.

### The fusion it is the reference for

What the card runs for this step is one kernel, not four: `r4d_gdn_conv_prep` in
`third_party/radiance/libr4d/r4d_gdn_conv_w4_h128_bf16.hip` does the convolution, the split of the
packed row, both l2 norms, the gate and its cumsum together, and keeps the conv output in
registers instead of writing it to HBM and reading it back. The record the trainer appends names
the whole prep as one op, so "the composition of the host adjoints of its parts" is exactly the
one host row this file compares against, and no `HostRef` chain is needed. One detail of the
fusion does not reach the adjoint: the forward rounds the conv output to bf16 before the norm (the
unfused pair does the same), while the host adjoint and this kernel both differentiate the f32
composition. That is a rounding of the forward's value, not of the cotangent, and it sits inside
the tolerance the comparison is held to.

## Precision

Gradients are bf16 on the card, as the ops' dtypes are. Operands and cotangents are staged at f32;
every sum that runs over more than a handful of terms -- a chunk's gate tail, the convolution's
window, a head's l2 dot, a row's two reductions -- accumulates in f32, and each element is rounded
to bf16 once, when it is stored. The comparison against the host adjoint therefore holds to the
harness's bf16 tolerance (`1e-3 + 0.02*|host|` per element, `tests/support/device_check.h`), not
bit for bit: the host sums the same terms in f64.

## Scratch

The call owns every buffer it touches, because the contract hands a device adjoint f64 operands
rather than device memory; all of it is scratch, it goes back before the call returns, and the
test fails a case that leaves any of it live. The host-side f32 staging is a thread-local pool
reused across calls (up to 16 buffers a call, and no two operands of a call share one, because a
copy may still be in flight when the next operand is staged): it keeps the largest size it has
seen, about 0.4 GiB after a 4096-token conv call, so a call in a loop does not allocate and free
hundreds of megabytes on the host each time.

At the real geometry (16 key heads of 128, 48 value heads of 128, convolution width 4, a conv row
of 10240, 4096 tokens) the conv op's phases take:

| live at the end of the phase | 4096 tokens | 1024 tokens |
|---|---|---|
| the convolution: `x` + `acc` + the small operands | 320 MiB | 80 MiB |
| the norms: `+ dq`, `dk`, `dv` and `dacc` (`x` has gone back) | 480 MiB | 120 MiB |
| the convolution's backward: `dacc` + the bf16 `dx` | 240 MiB | 60 MiB |
| the gate: a few megabytes | 4 MiB | 1 MiB |

`acc` and `dacc` are the two 160 MiB buffers at 4096 tokens; they are what a fused conv-norm-silu
epilogue would save, and they are kept because the three backward phases each need the activation
they share. The gated RMS norm holds `x`, `z` and `do` at f32 plus the two bf16 results: 384 MiB at
196608 rows (4096 tokens of 48 heads), 96 MiB at 49152 rows.

The measured high-water marks agree with the table: 491784 KiB at 4096 conv tokens, 123096 KiB at
1024, 393216 KiB at 196608 norm rows, 98304 KiB at 49152.

## Measured time

One card, an AMD Radeon AI PRO R9700 (gfx1201, 32 GiB), `R97_DEVICE=1`, from

```
R97_DEVICE=1 sg render -c './build/r97_gpu_tests --bench'
```

(`sg render` only because this machine's card is in the `render` group's hands; on a machine where
the user may open `/dev/kfd`, `R97_DEVICE=1 ./build/r97_gpu_tests --bench` is the same run, and
`ctest --test-dir build -L gpu` runs the same cases without `--bench`.) The right-hand number is the
forward op's own time from the same run: the twin, which is the whole op in f64 on the host -- the
registry holds device adjoints and host twins, and no device forward, so the twin is the forward
of record for now, and its f64 arithmetic is what the ratios are against.

```
gdn_conv_prep 1024 tok, 1 seq, 16x128+48x128, w4   56.224 ms adjoint   373.422 ms forward twin    0.15x forward  scratch peak 123096 KiB
gdn_conv_prep 4096 tok, 1 seq, 16x128+48x128, w4  208.769 ms adjoint  1552.807 ms forward twin    0.13x forward  scratch peak 491784 KiB
gdn_conv_prep 4096 tok, 3 seq, 16x128+48x128, w4  215.859 ms adjoint  1533.020 ms forward twin    0.14x forward  scratch peak 491784 KiB
gdn_gated_rmsnorm 49152 x 128 silu                 49.378 ms adjoint    87.851 ms forward twin    0.56x forward  scratch peak 98304 KiB
gdn_gated_rmsnorm 196608 x 128 silu               194.200 ms adjoint   352.906 ms forward twin    0.55x forward  scratch peak 393216 KiB
```

(the lines are the test's, with the columns' spacing collapsed. The two 4096-token cases were
timed twice, the 4096-token norm three times, the 1024-token cases five times and the fixture
cases twenty times. The machine's other runs share the box, and the same lines moved by a factor
of two to three between runs -- the 4096-token conv case came out at 580 ms on one run and 209 ms
on the next, the 1024-token one at 268, 82 and 56 ms -- so read them as the order of magnitude a
shared machine gives and re-measure on a quiet one before comparing.)

What the adjoint's number is, and is not: it is the whole call, as the test times it, from f64
operands to f64 cotangents. At 4096 tokens the call converts about 84 million f64 values to f32
for the card (roughly 1 GiB of host memory traffic in the staging alone), sends some 320 MiB down
the bus and brings some 80 MiB back, then adds the bf16 results into f64 cotangents (another
0.7 GiB on the host). Most of the 209 ms is that staging and accumulation, not the kernels: the
two 160 MiB buffers the norm phases share are written and read a handful of times each, which is a
few tens of milliseconds of a 32 GiB card's bandwidth. A caller that handed the op device buffers,
or kept the adjoint's results in f32 on the card, would not pay it, and a device-only comparison
would not see it. The fixture's ~3 ms is the other end of the same picture: twenty-one device
allocations, six launches and seven stream drains, with nothing in them. Nothing here is tuned
against a target; the number is what the family costs in the shape the contract gives it.

## What the test covers

`tests/gpu/adjoint_gdn_prep.cpp`, in the `gpu` tier behind the guard (it needs a card, not the
model container, so it carries no `model` label):

| case | geometry |
|---|---|
| fixture, one sequence | hidden 128: 2 key heads and 4 value heads of 16, width 4, 8 tokens |
| fixture, unequal lengths | the same, 3 sequences of 5, 3 and 2 tokens |
| fixture, wide row | the same, 7 tokens, a row of 160 (the conv's 128 plus 32 the op does not read) |
| real, one sequence | 16x128 key and 48x128 value heads, width 4, 1024 tokens |
| real, one sequence | the same, 4096 tokens |
| real, unequal lengths | the same in total, 3 sequences of 1500, 1400 and 1196 |
| norm, fixture | 32 rows (8 tokens of 4 value heads) of 16 channels, `silu` and `sigmoid` |
| norm, real | 49152 and 196608 rows of 128 channels |

and three things that are not shapes: the registered device rows freeze exactly what the host rows
freeze (the conv weight, its bias, `A_log`, `dt_bias`, the offsets, the state slots and the
warm-start flags, and the norm's weight), so the walk asks neither side for a weight's cotangent; a
kernel that drops the gate's cotangent is rejected in the gate's slot, and one that drops the
norm's `z` is rejected in `z`'s; and every case's scratch high-water mark is non-zero, so the
comparison is against a kernel that really ran off the allocator.

The test assembles its own calls by schema position (`make_schema_call` in
`tests/gpu/adjoint_gdn_prep.cpp`) rather than appending the twin's writes to the reads, the way the
shared harness's `make_call` does. It has to: `gdn_conv_prep` writes its INOUT `conv_state` at
position 5 and reads its gate at 7, and both the host adjoint and the kernel index the forward by
schema position. The layout is local to this family's test so that the shared harness, which every
device kernel's test uses, is left alone; the same fix belongs in a change of its own that can say
so for every family.
