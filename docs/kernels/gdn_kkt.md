# `gdn_kkt_solve`: the device adjoint

`gdn_kkt_solve` is the gated delta net's in-chunk solve, and 48 of the model's 64 layers are delta
net layers, so this adjoint and the chunk scan's decide what the backward costs. The kernel is one
`.hip` in `libr97`, registered as the `device` side of the op's adjoint; the arithmetic it runs is
the host adjoint's, `src/adjoint/host/gdn_kkt.cpp`, which `docs/adjoints/gdn_kkt.md` derives and
`tests/host/adjoint_gdn_kkt.cpp` checks at 1e-8 against finite differences.

| | |
|---|---|
| source | `src/adjoint/device/gdn_kkt.hip` |
| test | `tests/gpu/adjoint_gdn_kkt.cpp` |
| host adjoint | `src/adjoint/host/gdn_kkt.cpp`, `docs/adjoints/gdn_kkt.md` |
| reads | `k` bf16 `[rows, key_heads, head_k]`, `beta` f32 `[rows, heads]`, `g` f32 `[rows, heads]`, `cu_seqlens` i32 `[n_seq + 1]` (`head_k`, `chunk` are parameters) |
| adds to | `dk` bf16 `[rows, key_heads, head_k]`, `dbeta` f32 `[rows, heads]`, `dg` f32 `[rows, heads]`; nothing for the frozen `cu_seqlens` |
| frozen | operand 3, `cu_seqlens` |

## 1. What it undoes

Per (sequence, value head, chunk), the forward forms the strict lower block and inverts it,

    phi[i, l] = exp(gc[i] - gc[l]),   M[i, l] = beta[i] <kc[i], kc[l]> phi[i, l]   (l < i)
    A = (I + M)^-1

and writes `A` into the tape, so the adjoint reads it back rather than forming it again. With
`Ac = dloss/dA` and only the strict lower part of `dE` free, `Et = -(A^T Ac A^T)` restricted to
`l < i`, evaluated as two triangular products over the half-triangles they need:

    Y[p, l]  = sum_{j<=l} Ac[p, j] A[l, j]          l <= p
    Et[i, l] = -sum_{p>=i} A[p, i] Y[p, l]          l < i

and from `Et`, for every `l < i` with `e = Et[i, l]` and `m = beta[i] <kc[i], kc[l]> phi[i, l]`:

    dbeta[i]   += e <kc[i], kc[l]> phi[i, l]
    dg[i]      += e m              dg[l] -= e m
    dk[i, d]   += e beta[i] phi[i, l] kc[l, d]
    dk[l, d]   += e beta[i] phi[i, l] kc[i, d]

`beta` enters `M` linearly, so `dbeta` needs no division by it — but it also carries no `beta[i]`,
while both `dg` terms do: a kernel that uses `m` for `dbeta` is wrong by that factor and the
fixture catches it in the `beta` slot.

## 2. The kernel

One block owns one **(sequence, key head, chunk)** and walks the `grp = heads / key_heads` value
heads that read that key head. That blocking is what keeps the kernel free of atomics: a key row's
cotangent is written by the block that owns it, with the group's value heads summed inside the
block, and no two blocks touch the same `dk` element. `dk` is `key_heads * head_k` per token — 2048
of the 2144 gradient values a token carries at the model's geometry, 19 in 20 — so writing it once
matters.

128 threads per block, in four phases (shared memory in floats, `head_k = K`, chunk `= C`):

| phase | what | where it lives |
|---|---|---|
| 0 | zero the block's `dk` rows | `dks` `C × K` |
| 1 | `Y` for one value head, a row at a time, one thread per column | `Et` `C × C`, then overwritten in place |
| 2 | `Et` over `Y`, ascending `i`, one thread per column | `Et` |
| 3 | per row `i`: read `k_i`, then for each `l < i` the dot, `coeff = e beta_i phi`, `cij = e dot phi`, `dgm = beta_i cij`; one thread per lane updates `dk[i][d]` and `dk[l][d]` from `coeff`; thread 0 sums `cij`/`dgm` into `dbeta`/`dg` | `krow` `K`, `coeff`/`cij`/`dgm` `C` |
| store | `dbeta`, `dg` per head; the `dks` rows once per block | output |

Phase 2 writes `Et` over `Y` row by row in ascending `i`, so a row is never read after its `Et` has
been written; each thread owns one column, so no two threads write one value. Phase 3 re-reads
`k_l` from the card's memory, which has the chunk of one key head in cache for the life of the
block, and keeps `k_i` in `krow` so a thread needs it once. `g` is the chunk's cumsum from the
forward, so `phi = exp(gc[i] - gc[l]) <= 1`; nothing here recomputes it.

The solve is **f32**: `A`, `Ac` and `k` are uploaded at f32, `Y` and `Et` are formed and held at
f32, and the whole sum over the chunk's 64 rows is f32. The one rounding is the store, to the dtype
the schema gives each cotangent: `dk` bf16, `dbeta` and `dg` f32. That is the contract a device
adjoint runs under, and §4 measures what it costs.

Shared memory is `C² + C·K + 7C + K` floats — 50.25 KiB at the model's chunk 64 and 128 lanes,
21.8 KiB at the fixture's 16 — so it is past the default 48 KiB opt-in and the call asks for it with
`hipFuncSetAttribute`. A record whose chunk and lanes do not fit in 64 KiB, or whose shapes or
`cu_seqlens` do not hold up, is declined before anything is allocated: the adjoint adds nothing and
a later, differently blocked kernel can take it. Everything else is scratch: `k`, `beta`, `g`, `A`
and `Ac` at f32, the chunk-prefix and `cu_seqlens` tables, and one buffer per output at its schema
dtype. The measured high-water mark is in §5.

## 3. The test

`tests/gpu/adjoint_gdn_kkt.cpp` builds the same random case the host test does — unit-norm keys on
bf16's grid, `beta` a sigmoid of the gate, `g` the log of a per-chunk cumsummed decay — runs the
registered device adjoint and the host adjoint on the same f64 forward values and cotangent, and
holds every element of the three gradients to `1e-3 + 0.02·|host|`. Five shapes come from the tiny
fixture's delta-net geometry (2 key heads over 4 value heads, 16 lanes, chunk 64) with lengths
below one chunk, exactly one, two chunks with a short last, a three-sequence batch, and one
geometry without grouping (one key head per value head, 32 lanes); four more are the real model's
(16 key heads over 48 value heads, 128 lanes, chunk 64, `docs/DESIGN.md`): 1024 tokens as one
sequence, 1024 as a batch of unequal lengths (`250+750+24`, the third shorter than one chunk), 4096
as one sequence, and 4096 as a batch of unequal lengths (`1000+3000+96`). The batches are there
because `cu_seqlens` addresses rows: a chunk must not take the next sequence's keys for its own,
and a sequence's first chunk starts one row past the previous sequence's last short chunk.

The test also runs the harness against a deliberately wrong kernel (the right cotangents with
`dg`'s sign thrown away) and requires the harness to reject it in `dg`'s slot — otherwise a
comparison that accepted anything would accept this one too. It reads no model container: the real
shapes are design numbers, so the test carries the `gpu` label and not `model`.

## 4. Error against the f64 host adjoint

Measured on the 4096-token single sequence, 8,781,824 gradient elements (8,388,608 key elements,
196,608 `beta` and 196,608 `g` values):

| solve | max abs | relative L2 | outside `1e-3 + 0.02·\|host\|` | tightest |
|---|---|---|---|---|---|
| f32, one bf16 store (this kernel) | 9.8e-4 | 1.2e-3 | 0 of 8,781,824 | 0.16 of the tolerance |
| `Y` and `Et` rounded to bf16 as they are formed | 4.6e-3 | 2.7e-3 | 1591 | 3.5× the tolerance |

The f32 row's error is dominated by the one rounding the contract requires, the bf16 store of `dk`:
its unit roundoff is 2^-8, and the normwise error lands at 1.2e-3, an order under the 2% tolerance.
**bf16 would not suffice**, and the second row is the measurement rather than a guess: the test
re-runs the host adjoint's two triangular products with `Y` and `Et` rounded to bf16 as they are
formed (everything else f64, so this is the *best* a bf16 solve could do — a kernel that also
rounded its products would be worse), and 1591 elements come out past the tolerance the family is
held to, the worst of them 3.5× past it. The delta rule's `Et` is a 64-deep transposed solve whose
rows feed each other, so the 8-bit mantissa compounds where a single dot product would not.

The test prints both rows, and another run gives them again: the kernel's accumulation order is
fixed, so only `__expf` and the reference's f64 rounding could move the last digit.

## 5. Time per call

One run of `./build/r97_gpu_tests --bench` on the R9700 (gfx1201, 32 GB). The adjoint's time is the
whole call — upload of `A`, `Ac` and `k`, launch, download of the three gradients; the forward is
the op's f64 host twin over the same operands in the same run, which `docs/kernels/README.md` makes
the honest reference until an op has a device forward. `grp = 3` at the real geometry.

```
gdn_kkt_solve 5 x 4 heads of 16, chunk 64    1.190 ms adjoint     0.003 ms forward twin  435.28x forward  scratch peak 11 KiB
gdn_kkt_solve 64 x 4 heads of 16, chunk 64    4.236 ms adjoint     0.383 ms forward twin   11.06x forward  scratch peak 144 KiB
gdn_kkt_solve 101 x 4 heads of 16, chunk 64    3.962 ms adjoint     0.285 ms forward twin   13.88x forward  scratch peak 227 KiB
gdn_kkt_solve 5+64+150 x 4 heads of 16, chunk 64 batch    4.026 ms adjoint     0.937 ms forward twin    4.30x forward  scratch peak 492 KiB
gdn_kkt_solve 90 x 2 heads of 32, chunk 64    3.239 ms adjoint     0.179 ms forward twin   18.13x forward  scratch peak 126 KiB
gdn_kkt_solve 1024 x 48 heads of 128, chunk 64   36.782 ms adjoint   206.739 ms forward twin    0.18x forward  scratch peak 37632 KiB
gdn_kkt_solve 250+750+24 x 48 heads of 128, chunk 64 batch   33.970 ms adjoint   193.507 ms forward twin    0.18x forward  scratch peak 37632 KiB
gdn_kkt_solve 4096 x 48 heads of 128, chunk 64  200.562 ms adjoint   908.599 ms forward twin    0.22x forward  scratch peak 150528 KiB
gdn_kkt_solve 1000+3000+96 x 48 heads of 128, chunk 64 batch  201.601 ms adjoint   877.472 ms forward twin    0.23x forward  scratch peak 150528 KiB
```

A batch costs what one sequence costs for the same tokens: 4096 tokens is 201 ms whether it is one
sequence or three, 1000+3000+96 is 202 ms, and 1024 tokens is 37 ms as one sequence and 34 ms as
250+750+24 — a few percent apart, which is noise, and the batch is not cheaper in principle: it
carries one more chunk block than the single sequence. A chunk's work depends on its own rows, and
the block count is the number of (sequence, chunk, key head) triples. The call is 4.4× to 5.7×
cheaper than the f64 host twin at the real shapes. The per-token cost is not flat between 1024
and 4096 tokens (37 ms for 1024 against 201 for 4096): the 4096 call moves four times the f32
operand volume through the same upload path. At the fixture's geometric sizes the number is launch
and copy overhead, not the kernel, and the ratio means nothing there.

The scratch peak is the whole call's device footprint, which is dominated by `A` and `Ac` at f32
(48 MiB each at 4096 tokens), then `k` (32 MiB) and `dk` (16 MiB); 37,632 KiB at 1024 tokens and
150,528 KiB at 4096 is that same list scaled (36,864 and 147,456 KiB of it), plus the `beta` and
`g` operands with their gradients. It is the peak the harness's allocator ledger records, not the
kernel's shared memory (§2: 50.25 KiB per block).

Numbers from a shared card: the block above is one run, taken while other contributors were
building (load average 15 to 27 of 48 cores); it put the 1024-token adjoint at 37 ms where an
earlier, quieter run had 33 ms. The same binary on the same card moved the 4096-token adjoint from
201 to 578 ms and its twin from 899 to 3575 ms while the machine sat at a load average of 110, so
treat the ratios as what this machine measured at the time, not as a specification, and do not
compare them across cards.

## 6. Running it

```sh
cmake --build build
./build/r97_gpu_tests --bench      # the same checks plus the timing lines
ctest --test-dir build -L gpu      # the card, serially, under the GPU test guard
```

The test binary needs a card, not a container: `R97_DEVICE` selects the card, and the build passes
the plugin it built as `R97_DEVICE_PLUGIN`.
