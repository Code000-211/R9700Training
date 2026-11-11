# The `elementwise` family on the card

The device adjoints of the eight ops in `src/adjoint/host/elementwise.cpp`:

| op        | schema                              | the kernel writes                                    |
| --------- | ----------------------------------- | ---------------------------------------------------- |
| `mul`     | `a, b -> y`, `y = a * b`            | `da = dy * b` and `db = dy * a`; `b` may be one row   |
| `scale`   | `x, s -> y`, `y = x * s[0]`         | `dx = dy * s[0]` (`s` frozen)                         |
| `sigmoid` | `x -> y`, `y = sigmoid(x)`          | `dx = dy * y * (1 - y)`, from the forward `y`         |
| `silu`    | `x -> y`, `y = x * sigmoid(x)`      | `dx = dy * silu'(x)`                                  |
| `silu_mul` | `gate_up -> y`, `y = silu(g) * u`   | both halves of `d gate_up` in one pass, `[rows, 2n]`  |
| `act_mul` | `a, b -> y`, `y = act(a) * b`       | `da = dy * act'(a) * b`, `db = dy * act(a)`           |
| `cast`    | `x -> y`, between float types       | `dx = dy`: a cast only rounds                         |
| `rmsnorm` | `x, w -> y` (`w` frozen)            | `dx = mult * (s*c*dy - s³*x*t/n)` in f32              |

One source, `src/adjoint/device/elementwise.hip`, in the `libr97` plugin; one test,
`tests/gpu/adjoint_elementwise.cpp`, in the `gpu` tier. `docs/kernels/README.md` is the rule both
follow; `docs/adjoints/elementwise.md` is the host side.

## The fused gated MLP activation

`silu_mul` is the one member the forward fuses on the card: the gated MLP activation issues `silu`
and `mul` as one op over `[rows, 2n]`, so its adjoint is one pass over the fused row -- `d gate` from the
gate's slope and the up half, `d up` from the silu of the same gate -- not a silu adjoint followed
by a mul adjoint.

Both forms are on the card. `mul` and `silu` are each registered as well, so a graph that issues the
pair separately has its rows; and the fused kernel is held to their composition, as a fused op must
be: the test gives `check()` a `HostRef` that runs `mul`'s host adjoint over `(silu(g), u)` and then
`silu`'s over `g` -- the order the fused pass undoes them -- and compares the fused row against what
those two write. That is the reference at the fixture's width and at the real one (256 rows by
17408), and the fused kernel's own real shape (4096 rows) is compared against the host row.

## The tolerance

Each kernel is held to the host adjoint on the same random-in-`[-1, 1)` inputs element for element
within

    |device - host| <= 1e-3 + 0.02 * |host|

(`kAbsTol` and `kRelTol` in `tests/support/device_check.h`), for every slot of every cotangent.
The 2 % is the room bf16's eight-bit mantissa needs: the device computes in f32 -- forward values
and cotangents are read at f32, every reduction is a sum of f32 -- and rounds once, at the store,
to the dtype the record gives the operand. No element of any case is outside that, at any shape
below.

## Measured

One R9700 (gfx1201, 32 GB), card 0 of a two-card machine, in the container of
`docs/TESTING.md`; every line is the harness's own, from

    ./r97_gpu_tests --bench

with the real model's shapes: hidden 5120 and the MLP's 17408 (`n_ff` in the container of
`StillDeadcode/qwen3.8-27b-iqr-v4`), 1024 tokens as one sequence and 4096 as a batch of unequal
lengths, `[1024, 512, 2560]`. `forward twin` is the op's own f64 forward over the same operands in
the same run; `times forward` is the adjoint's cost over it.

```
mul 1024 x 5120                48.883 ms adjoint    32.395 ms forward twin    1.51x forward  scratch peak 81920 KiB
mul 1024 x 5120 row b          31.876 ms adjoint    54.629 ms forward twin    0.58x forward  scratch peak 51250 KiB
scale 1024 x 5120              22.672 ms adjoint    31.452 ms forward twin    0.72x forward  scratch peak 30720 KiB
sigmoid 1024 x 5120            31.267 ms adjoint    65.925 ms forward twin    0.47x forward  scratch peak 51200 KiB
silu 1024 x 5120               29.962 ms adjoint    67.725 ms forward twin    0.44x forward  scratch peak 51200 KiB
cast 1024 x 5120 bf16->f32     21.482 ms adjoint    28.936 ms forward twin    0.74x forward  scratch peak 30720 KiB
rmsnorm 1024 x 5120 gain wadd 1.0 mult 1.0   30.181 ms adjoint    35.119 ms forward twin    0.86x forward  scratch peak 51220 KiB
act_mul 1024 x 5120 silu       48.616 ms adjoint    70.454 ms forward twin    0.69x forward  scratch peak 81920 KiB
act_mul 1024 x 5120 gelu_tanh  48.497 ms adjoint   150.478 ms forward twin    0.32x forward  scratch peak 81920 KiB
silu_mul 1024 x 17408         316.512 ms adjoint   218.587 ms forward twin    1.45x forward  scratch peak 278528 KiB
mul 1024 x 17408 row b        200.003 ms adjoint   187.572 ms forward twin    1.07x forward  scratch peak 174250 KiB
act_mul 1024 x 17408 silu     318.159 ms adjoint   242.797 ms forward twin    1.31x forward  scratch peak 278528 KiB
mul 1024+512+2560 x 5120 batch  373.822 ms adjoint   130.787 ms forward twin    2.86x forward  scratch peak 327680 KiB
mul 1024+512+2560 x 5120 batch row b  232.424 ms adjoint   227.572 ms forward twin    1.02x forward  scratch peak 204850 KiB
silu 1024+512+2560 x 5120 batch  234.153 ms adjoint   272.926 ms forward twin    0.86x forward  scratch peak 204800 KiB
rmsnorm 1024+512+2560 x 5120 batch gain wadd 1.0 mult 1.0  234.358 ms adjoint   143.010 ms forward twin    1.64x forward  scratch peak 204820 KiB
silu_mul 1024+512+2560 x 17408 batch 1292.509 ms adjoint   878.990 ms forward twin    1.47x forward  scratch peak 1114112 KiB
```

At the fixture's 4 x 128 the same lines read about 0.3 ms an adjoint against 0.001-0.01 ms of
forward: at that size a call is one launch and three copies, and nothing it measures is the kernel.

What the adjoint column holds is the whole call, and the call is not the kernel: the contract hands
a device adjoint f64 operands, so each call converts the tape's f64 to f32 (5.2 M values at 1024 x
5120), copies them up, copies the bf16 cotangents down and adds them back as f64. `cast`, whose
kernel only copies, costs 21.5 ms at 1024 x 5120 and `scale`, one multiply, 22.7 ms: the work that
scales with elements is the conversion and the copy, not the arithmetic. The column still answers
the question a trainer asks of it -- what a call costs at this shape -- and it is the same number
the forward twin is measured against. A repeat on the shared card differs by up to about 50 % on
the largest cases (`mul` at 4096 x 5120 read 251 ms on one run and 374 ms on the next), so read the
column as an order and not as a constant.

The kernel's own work is that scratch: one f32 read of every forward value, one f32 read of the
cotangent, one write of the cotangent in the slot's dtype. For the fused gated activation at
4096 x 17408 that is 570 MB read of `gate_up`, 285 MB read of `dy`, 285 MB written back.

## Scratch

The scratch a call takes is the buffers its kernel needs, freed before it returns; the harness's
`scratch peak` is the allocator's high-water mark over the call. Every buffer is one per operand,
at f32 for a read and at the slot's own width for a write:

| op                   | device buffers                                                   | at 1024 x 5120, measured |
| -------------------- | ---------------------------------------------------------------- | ------------------------ |
| `mul` elementwise    | `a`, `b`, `dy` f32; `da`, `db` bf16                              | 80 MB (81920 KiB)        |
| `mul` one-row `b`    | `a`, `dy` f32 at `na`, `b` at `nb`, staging `nb` f32; `da` bf16, `db` bf16 | 50 MB (51250 KiB) |
| `scale`, `cast`      | `dy` f32; `dx` bf16                                              | 30 MB (30720 KiB)        |
| `sigmoid`            | `y`, `dy` f32; `dx` bf16                                         | 50 MB (51200 KiB)        |
| `silu`               | `x`, `dy` f32; `dx` bf16                                         | 50 MB (51200 KiB)        |
| `act_mul`            | `a`, `b`, `dy` f32; `da`, `db` bf16                              | 80 MB (81920 KiB)        |
| `rmsnorm`            | `x`, `w`, `dy` f32; `dx` bf16                                    | 50 MB (51220 KiB)        |
| `silu_mul` 1024 rows | `gate_up` (2n) and `dy` (n) f32; `d gate_up` bf16                | 272 MB (278528 KiB)      |
| `silu_mul` 4096 rows | the same at 4 x, `n` = 17408                                     | 1088 MB (1114112 KiB)    |

`f32_bytes(n)` is 4 n and `slot_bytes(slot, n)` is 4 n for an f32 slot and 2 n for a bf16 one, so
the fused activation's three buffers over `[rows, 2n]` are 4 x 2n + 4 x n + 2 x 2n = 16 n per row --
the largest thing the family asks for.

## How the kernels are built

Each is one launch over its shape, with no intermediate on the tape:

- **The pointwise six** (`scale`, `sigmoid`, `silu`, `cast`, and both halves of `mul` and
  `act_mul`) are one thread per element, 256 to a block. `sigmoid` reads `y` off the tape rather
  than recomputing the sigmoid.
- **`silu_mul`** launches a 2-D grid: one block column per 256 of `n`, one block row per row of
  the batch, so the gate and the up projection of a row are written by the same thread from the
  same read of the cotangent.
- **`mul` with a one-row `b`** is the family's only reduction over rows. One thread takes a
  column, and the rows of that column are split across a few blocks of rows (`grid.y`), whose
  partial sums meet in f32 staging through one `atomicAdd` each -- a block per column would leave
  most of the card idle at 5120 columns, and an atomic per element would queue 4096 writers on one
  address. `chunks = min(rows, max(1, 2048 / column_blocks))`.
- **`rmsnorm`** is one block of 256 threads per row. A single pass over the row sums both `x²` (for
  the rms) and `dy*x*c` (the shared term, which does not need the rms), the two meet in a shared
  tree reduction in f32, the gain `s` is formed there, and a second pass writes
  `mult * (s*c*dy - s³*x*t/n)`. The 4096-row case is 4096 blocks: enough to fill the card.
- **The store** is one place. `OutStore::set` rounds to the record's dtype -- bf16 round-to-nearest-
  even, or a plain f32 write for the one member that hands back an f32 cotangent -- so a kernel's
  output is rounded once, however many values it wrote.

The two weights (`scale`'s `s`, `rmsnorm`'s `w`) are registered frozen, as on the host, so the walk
never asks the device for their cotangents.

## Running it

```sh
cmake --build build -j 32
cd build && R97_DEVICE=0 ./r97_gpu_tests            # correctness, ~2 min
cd build && R97_DEVICE=0 ./r97_gpu_tests --bench    # and the lines above
ctest -L gpu --output-on-failure                    # the same through ctest, under the card guard
```

The test reads no container: the real shapes are the design's numbers, so it carries the `gpu`
label and not `model`. `R97_DEVICE` picks the card.
