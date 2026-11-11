# The `quant` device family

The card's straight-through adjoint for the quantiser and for the three fused producers the card
issues: `had_quant_act_i8g`, `rmsnorm_had_quant_i8g`, `gated_had_quant_i8g`, `gate_had_quant_i8g`.

| file | what |
|---|---|
| `src/adjoint/device/quant.hip` | the kernels and the four `Side::device` rows |
| `tests/gpu/adjoint_quant.cpp` | the device-versus-host check, label `gpu` |
| `docs/adjoints/quant.md` | the host family: the forward, the estimator, the rejected alternatives |

## What the rows serve

Every record writes the quantiser's two outputs and nothing else: the int8 codes and one f32 scale
per group. The reads are the fused producer's own operands, and the row pins their order, because a
device adjoint has no schema beyond the record it is handed: the codes and the scale are the last
two operands, the reads are everything before them.

| op | reads | writes |
|---|---|---|
| `had_quant_act_i8g` | `x` `[M, n]` | `q` `i8` `[M, n]`, `scale` `f32` `[M, n/group]` |
| `rmsnorm_had_quant_i8g` | `x` `[M, n]`, `w` `[n]` | as above |
| `gated_had_quant_i8g` | `gate_up` `[M, 2n]` | as above |
| `gate_had_quant_i8g` | `gate` `[M, n]`, `x` `[M, n]` | as above |

Parameters, read the way the host adjoint reads them (`n` and `group` as an int or a decimal
string, the rest as strings):

| parameter | ops | meaning |
|---|---|---|
| `n` | all | the width; required |
| `group` | all | the scale's group width; a power of two, a divisor of `n`, at most 1024 (one block) |
| `act` | `gated`, `gate` | `gated`: absent or `"silu"`; `gate`: `"sigmoid"`, required |
| `eps`, `wadd`, `mult` | `rmsnorm` | the RMS norm's epsilon, its bias on the gain, its output multiplier |
| `dtype`, `M` | all | not read; `M` is the codes' first dimension |

The rotated value, the codes and the scale are laid out in plain column order -- element `(r, j)` of
a group-major `[M, n]` buffer -- the order `docs/ADAPTER.md` describes for both the host and the
card. The real shapes use `group = 128`; the tiny fixture is `hidden = 128`, so the width and the
group are the same there and `group = 64` is the other case the test runs.

## The mask, and what "read from the codes" means

`docs/adjoints/quant.md` states the estimator: the cotangent of the codes is divided by the group's
scale and dropped where `|u_j| > 127`, `u = v / s`, `v` the group's rotated activation and `s` its
scale. The kernel applies exactly that:

```
m_j = 0   ⟺   |c_j| = 127  and  |v_j · (1/s)| > 127
```

Both halves are the reference's rule, and the second one cannot be dropped. In the forward every
value of a group has `|u_j| ≤ 127` in exact arithmetic, and only the rounding of the reciprocal can
put one outside -- so the group's amax, which rounds to `±127` by construction, is the element the
mask can engage on -- while a code of `127` is *not* by itself evidence of a clamp, since every
value in `(126.5, 127]` rounds to it. A mask of `|c_j| = 127` alone is the reading
`docs/adjoints/quant.md` §5 rejects: it would drop the amax's cotangent in every group, including
the groups where the reciprocal rounded inward and nothing was clamped. The codes are read -- they
sit at the top of the grid for every element the clamp could have reached, and for no other, since
`|c_j| ≤ 126` implies `|u_j| ≤ 126.5` -- and the value decides, which is the reference's
`|u_j| ≤ 127` exactly.

That equivalence needs `v` to be the reference's `v` to the bit, so the kernel forms it in f64 with
the host `rotated()`'s stage order. The butterflies of one stage are independent, so a whole group
shares a stage across its threads without changing a digit.

## Precision, and what is on the tape

- **f64:** the rotated activation `v` and the scale, and therefore the mask.
- **f32:** the division by `s`, the inverse rotation, and the elementwise backprop's operands. The
  RMS norm's `Σ x²` and `Σ r x c` accumulate in f32 for the row, per the family's tolerance.
- **bf16:** a gradient is rounded once, on the way out, and then added to `dc` as f64 -- the tape's
  cotangents are shared and never overwritten.
- **The intermediate is not on the tape.** For a fused producer the value the quantiser rotated is
  the elementwise part's output, so the call rebuilds it through the parts' registered twin
  (`rmsnorm`, `silu_mul`, `act_mul` in `src/adjoint/host/elementwise.cpp`) -- the same definition
  the host reference composes, so both sides rotate identical numbers. It is not rounded to bf16 on
  the way: the twin writes f64, and the 2⁻⁹ that a bf16 round would add is inside the tolerance
  rather than worth a second rounding on one side only.
- **The gain is frozen.** `rmsnorm_had_quant_i8g` registers as `R97_REGISTER_ADJOINT_IF_F(..., 1)`,
  matching `R97_REGISTER_ADJOINT_F("rmsnorm", host, ..., 1)`: `w`'s slot stays zero, and the test
  checks the row's frozen list is `{1}`.
- **No throw.** A record whose operands, widths or parameters do not fit the row returns without
  touching `dc`; so does a part whose twin refuses the record (the parts' twins throw on a record
  they cannot serve, and the call catches that).

`plugin.hip` gains a `twin_find` fallback beside its `adjoint_register` and scratch ones: the plugin
references a registry function for the first time, and without the fallback it would fail to load
in a radiance process that does not link r97 (`rad-convert` scans the same `kernels/` directory).
That process never runs an adjoint, so the fallback is never reached; in `r97train` and the gpu
tests the executable's definition wins, as `ENABLE_EXPORTS` intends.

## What is not served

- `quant_act_i8g` without the rotation, the fp8 quantisers, and the `out_bf16`/`residual` forms of
  these records: no row, so a lookup finds nothing rather than a wrong kernel.
- `group` above 1024, a group that is not a power of two, and a group that does not divide `n`: the
  predicates refuse the record at the lookup.
- `gated_had_quant_i8g` with `act = "sigmoid"` (a second, different half) and
  `gate_had_quant_i8g` without `act = "sigmoid"` (the schema's default is not this row's to guess).

## The test

`tests/gpu/adjoint_quant.cpp` runs the four ops over the tiny fixture's shape and the real one:

| case | what it adds |
|---|---|
| `4 x 128`, group 128 | the fixture's width, one group per row |
| `4 x 128`, group 64 | two groups per row, and a group narrower than a warp |
| `1024 x 5120` | the real hidden size, one sequence |
| `2024 x 5120` | a batch of unequal lengths (300 + 700 + 1024) |
| `4096 x 5120` | the real hidden size, the longest sequence |

The reference is `docs/kernels/README.md`'s `HostRef`: the quantiser's host adjoint on the
intermediate, then the elementwise part's host adjoint, in the order the kernel undoes them, and the
comparison is the harness's bf16 tolerance, `1e-3 + 0.02·|host|` per element. The
test registers the fused producers' twins (the composition of their parts' twins) because the host
family has no fused row of its own yet. It also checks that the four device rows are registered,
that the RMS norm's row freezes the gain, that `gated_had_quant_i8g` with `act = "gelu"` finds no
row, and that the harness rejects a wrong kernel -- one that hands the cotangent back without the
mask and the inverse rotation.

## Timing and scratch

Measured with `./build/r97_gpu_tests --bench` on one R9700 (`gfx1201`, 32 GB), card 1, under
the `gpu` guard, 20 repetitions per case. The adjoint's time is the whole call -- upload,
launch, download -- and the forward is the op's f64 host twin, because these ops have no device
forward. The correctness comparison ran in the same process and passed every case.

```sh
cmake -B build -G Ninja && cmake --build build
ctest --test-dir build -L gpu             # the checks
R97_DEVICE=1 sg render -c './build/r97_gpu_tests --bench'   # the same, plus one line per case:
# <op> <shape>  <adjoint> ms adjoint  <forward> ms forward twin  <ratio>x forward  scratch peak <K> KiB
```

`had_quant_act_i8g`:

| case | adjoint | forward twin | ratio | scratch |
|---|---|---|---|---|
| `4 x 128`, group 128 | 0.699 ms | 0.006 ms | 111.71x | 8 KiB |
| `4 x 128`, group 64 | 0.684 ms | 0.006 ms | 117.61x | 8 KiB |
| `1024 x 5120` | 124.106 ms | 174.564 ms | 0.71x | 87 360 KiB |
| `2024 x 5120` | 271.044 ms | 333.682 ms | 0.81x | 172 672 KiB |
| `4096 x 5120` | 591.478 ms | 576.584 ms | 1.03x | 349 440 KiB |

`rmsnorm_had_quant_i8g`:

| case | adjoint | forward twin | ratio | scratch |
|---|---|---|---|---|
| `4 x 128`, group 128 | 0.790 ms | 0.010 ms | 77.85x | 8 KiB |
| `4 x 128`, group 64 | 0.828 ms | 0.010 ms | 82.35x | 8 KiB |
| `1024 x 5120` | 144.886 ms | 222.116 ms | 0.65x | 87 360 KiB |
| `2024 x 5120` | 347.345 ms | 520.241 ms | 0.67x | 172 672 KiB |
| `4096 x 5120` | 1100.972 ms | 1440.513 ms | 0.76x | 349 440 KiB |

`gated_had_quant_i8g`:

| case | adjoint | forward twin | ratio | scratch |
|---|---|---|---|---|
| `4 x 128`, group 128 | 1.042 ms | 0.016 ms | 63.19x | 8 KiB |
| `4 x 128`, group 64 | 1.249 ms | 0.016 ms | 77.54x | 8 KiB |
| `1024 x 5120` | 500.946 ms | 590.103 ms | 0.85x | 87 360 KiB |
| `2024 x 5120` | 584.051 ms | 668.512 ms | 0.87x | 172 672 KiB |
| `4096 x 5120` | 1696.857 ms | 2401.680 ms | 0.71x | 349 440 KiB |

`gate_had_quant_i8g`:

| case | adjoint | forward twin | ratio | scratch |
|---|---|---|---|---|
| `4 x 128`, group 128 | 1.299 ms | 0.016 ms | 83.25x | 8 KiB |
| `4 x 128`, group 64 | 1.308 ms | 0.014 ms | 95.56x | 8 KiB |
| `1024 x 5120` | 404.743 ms | 711.269 ms | 0.57x | 87 360 KiB |
| `2024 x 5120` | 929.031 ms | 847.672 ms | 1.10x | 172 672 KiB |
| `4096 x 5120` | 937.293 ms | 850.661 ms | 1.10x | 349 440 KiB |

The tiny cases are dominated by the call overhead (a launch is ~0.7 ms on this card), so their
ratio says nothing about throughput. The real shapes are the interesting ones.

The scratch the call takes is fixed by the code and can be read off it. With `T = M · n` elements of
codes, the mask pass holds `8T` (the f64 rotated value), `T` (the codes), `4T` (the f32 cotangent)
and `4T` (the output) plus `8·M·n/group` for the scales; the backprop adds the elementwise part's
operands, `4T` for `quant`/`rmsnorm` (plus `4n` for the gain) and `8T` for `gated`/`gate`, and
`2T`/`4T` of bf16 out. The pass is the high-water mark, near `17T`; the peaks the run reported at
those cases were:

| case | `T` | measured scratch |
|---|---|---|
| `4 x 128` | 512 | 8 KiB |
| `1024 x 5120` | 5.24 M | 87 360 KiB (85 MiB) |
| `2024 x 5120` | 10.4 M | 172 672 KiB (169 MiB) |
| `4096 x 5120` | 21.0 M | 349 440 KiB (341 MiB) |

The f64 upload of `v` is what makes the largest case cost that much; it buys the mask the
reference's exact decision. The call also copies that much through the host, since the contract
hands a device adjoint host values and not a device buffer -- so if anything dominates the time at
5120 it is the transfer, not the two kernels, which is why the ratio to the forward twin is worth
reading beside the two absolute numbers.
