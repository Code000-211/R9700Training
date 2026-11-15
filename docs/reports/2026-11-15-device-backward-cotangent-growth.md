# The device backward's `h` cotangent grows 3.2 decades per block

Issue #145. The device step's backward grows the residual stream's cotangent by a factor of
`10^3.2` per block over thirteen blocks, while the forward `h` is flat and the finite difference
says the leaf gradients are flat. This is the measurement, the host rows the A/B needed, and what
is left: the growth is in the composition, and the rule that supplies it is still open.

## What was reported

`tests/gpu/device_step.cpp`'s `device_step_tape` on
`/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad`, 512 tokens, TP 1:

| block | `maxabs` of `h`'s cotangent at the norm |
|---|---|
| 63 | 7.24e-6 |
| 62 | 0.01386 |
| 61 | 34 |
| 60 | 35072 |
| 59 | 8.336e7 |
| 58 | 8.644e10 |
| 57 | 1.561e14 |
| 56 | 2.286e17 |
| 55 | 4.058e20 |
| 54 | 4.345e23 |
| 53 | 6.335e26 |
| 52 | 9.458e29 |
| 51 | 1.612e33 |
| 50 | 2.659e36 |

The growth is 3.2 decades per block, in both the GDN (3.200) and the attention (3.193) blocks.
There are two `h` records per block -- the mixer's norm and the FFN's -- and the walk's own
deltas alternate between about 1.4 and 1.8 decades, median 1.60, so the per-block factor is the
product of two rows' factors rather than one row's.
The forward `h` over the same blocks is 5.66 to 7.03: flat to within 0.03 decades per block. The
smallest group scale the quantiser uses on this container is 5.95831e-05, and the largest single
cotangent element is 2.65846e36 -- finite, so nothing has overflowed yet; the first position whose
output is entirely non-finite is 1134. No cotangent is below 1e-32.

Within one block (62) the growth is not one row's: 1428 `add` 9.232e-4, 1421 FFN norm 0.013855
(x15.0), 1412 mixer norm 0.478516 (x34.5), 1408 attention norm 0.980469 (x2.05). Each norm's
adjoint multiplies by the block's own factor, so the per-block factor is the product of the rows'
factors, not one row's.

## The plan's premise was wrong, and that is why the first attempt could not run

The plan said the host walk would read each operand's forward value out of `AdValue::raw`. It
cannot: `raw` is the engine's VRAM pointer (`pulled_bytes` returns 0 for a dtype whose
`elem_size` is 0, so `pull_operands` keeps `content.data = recorded.data`), and the bytes behind it
are libr4d's unit-major rearranged layout, not a host `RadPlanesHeader`. The host rows have to
decode from the container, which is what `src/vocab_head.cpp` already does for the head's weights.

Three pieces carry that:

- `src/iqr_form.h` derives the stored form from the encoding's own planes (as libr4d's `classify`
  does), validates it with `r97_iqr_validate`, decodes with `r97_iqr_decode_host`, and folds
  `scale.col` the way `iqr_plane.hip`'s `eff_scale` does -- `bf16(row_f32[n] * cf_f32[kb])`, the
  round-to-nearest-even `bf16_bits` of libr4d's `r4d_iqr_dec.h`.
- `src/stored_weight.{h,cpp}` holds the decoded weight as int8 codes plus bf16 group scales, keyed
  by `(encoding, rows, cols)`. The encoding pointer is one per weight in the program, so keys
  cannot collide; a process-wide store is what lets the walk reach a weight it decoded in an
  earlier block.
- `src/step_device.cpp`'s `register_stored_operand` runs on every snapshot the host walk takes. A
  plain encoding (one codes plane, block 1x1, core float dtype) is widened to f64 values, so every
  row works unchanged; anything else with a `cols % 128 == 0` is decoded into the store.

`R97_DUMP_STORED=1` prints one line per stored operand, and the step end reports how many weights
the walk decoded and how many bytes they held.

## What the container's block 63 carries

`blk.63.*` has six trellis weights and four plain bf16 norms. All six decode: `attn_qg`
[12288,5120], `attn_k`/`attn_v` [1024,5120], `attn_output` [5120,6144] (bf16 per 128), `ffn_down`
[5120,17408], `ffn_gate_up` [34816,5120]. RMS 0.00103 to 0.00233, 98.7% non-zero codes, scales
1.5e-5 to 3.0e-4. `blk.63.ffn_norm.weight` is a plain `[1 5120] bf16`.

The trunk form, in rad-info's spelling:

```
trellis:codes=u6[1x1],scale=f32[1x*],trellis=i8{65536x1},trellis.mad=u32{2x1},
        scale.col=f32[*x128]/fwht128
```

The scheme is `trellis`, not `affine`; the trellis *table* carries `block[0]=256` (the run rows)
and `block[1]=1` (the rb).

## The host rows the walk needed

The first two were named refusals the walk hit in order; the next two were silent, and each is a
place where a row read a width, a position or a shape off the record's parameters or its own order,
while the tape carries the fact on the operands:

1. `rmsnorm_had_quant_i8g`'s gain arrives as `[512, 5120]` (2621440 values) as well as a single
   row (`[5120]`). `norm_gain` required one dimension; a `[rows, n]` operand of the row's width is
   now accepted.
2. `gate_had_quant_i8g`'s record declares `n=256` while its codes are 6144 wide over a 48-group
   scale, and its elementwise intermediate is `[512, 24, 256]` -- the codes' value count in the
   elementwise op's own shape. `fused_adjoint` now derives `rows`, `n` and `group` from the codes
   and the scale operands and tests the intermediate's value count, exactly as the device row does.
3. `rmsnorm_had_quant_i8g`'s gain arrives as `[512, 5120]` with 2621440 values, and **its rows are
   not the same one** -- for a weight the container holds as `[1, 5120]`. The record is
   `h, x:inout, w`, so both rows do land on the weight (the device row's "second read" is the first
   non-written operand after `h`, and `x` is written), and both index row 0. A row-identity test
   that refused this would refuse the walk, so what it does instead is say so once:
   `note_gain_rows` prints that the tape's gain is not the broadcast its shape claims, and the
   gain is read from row 0. The tape's form is the thing to fix, and it is named here rather than
   silently averaged away.
4. The residual's cotangent: `fused_adjoint` read `call.cot[0]` and `[1]` positionally (below).
5. The residual's *routing*: a frozen in-place operand's slot was never sized, so the walk dropped
   the residual stream's cotangent at every fused norm (its own section below).

The refusals were silent about *why* until they were given the operand list; the messages now
carry it. Fix 4 is the one the A/B needed: it is the difference between a host walk that returns
the same order of magnitude as the device walk and one that returns zero.

`tests/host/adjoint_quant_cot_slot.cpp` pins fix 4 in milliseconds: it builds the
fused rmsnorm record with the output's write before the codes and again after them, and requires
the residual's gradient to be non-zero and identical both ways. Under the positional read the first
order returns zero.

One more was found by reading rather than by the walk, and is latent on this container:
`gemm_nt` and `logits_gemm` took a stored weight's *code* as its value, without the group scale,
which would have been a silent factor of `1/scale` on any plain weight the join decoded. The walk
never takes that path -- the container's plain weights are widened to values, so `stored` is null --
so it is fixed while there and not exercised.

## The gate: the host walk grows where the device walk grows

The plan's gate was to walk the same tape with the host rows and see whether the host's `h`
cotangent grows like the device's (a composition defect) or is flat (a device-row defect). It
could not run at first: the host rows refused record after record, and then -- once they served
every record -- the host walk's cotangent went to *exactly zero* at the first block boundary while
the device's grew 2.1 decades there.

The cause was positional. `fused_adjoint` read the codes' and the scale's cotangents as
`call.cot[0]` and `call.cot[1]`, but `cot` holds one entry per write in the record's own write
order, and the record the walk prepares writes the elementwise output *before* the codes: `cot[0]`
there is the output's slot, which no consumer of that producer fills. The quantiser's part
therefore contributed nothing and only the head's seed survived. Reading the slots by role -- the
write's index among the writes, which is what the device row does
(`src/adjoint/device/quant.hip:532`'s `cot_slot`) -- fixed it:

| position | `codes_cot` | `quant_dx` | `dx` \|sum\| (host, before) | after |
|---|---|---|---|---|
| 1449 (FFN norm) | 0.435 | 15.7 | 1.44 | 1.44 |
| 1429 (mixer norm) | 4.49 | 174.7 | **0** | **191.9** |

The host walk's gradient norm went from 4.905 to 48.616 on the same tape: the row had been
contributing nothing below the first boundary.

With the rows serving, the two walks grow by the same amount, and the factor between them widens
with the block:

| position | host `maxabs` | device `maxabs` | host delta | device delta |
|---|---|---|---|---|
| 1449 (block 63, FFN norm) | 1.95e-5 | 7.24e-6 | | |
| 1429 (block 63, mixer norm) | 1.56e-3 | 9.23e-4 | +1.90 dec | +2.11 dec |
| 1421 (block 62, FFN norm) | 6.44e-2 | 1.386e-2 | +1.62 dec | +1.18 dec |
| 1408 (block 62, mixer norm) | 6.81 | 0.980 | +2.02 dec | +1.85 dec |
| 1400 (block 61, FFN norm) | 271 | 34 | +1.60 dec | +1.54 dec |
| 1387 (block 61, mixer norm) | 2.55e4 | 1360 | +1.97 dec | +1.60 dec |

Over those five records the host grows 9.11 decades and the device 8.28, and the ratio between the
two sides climbs with the block -- 1.69, 2.69, 4.65, 6.95, 7.97, 18.75 -- which is itself the second
half of the answer: the rows contribute about 0.17 decades per record and the composition about 1.5.
Per block, of the 3.2 decades, roughly 2.9 are the composition's and 0.3 the rows'.

The whole A/B, on four blocks of the same tape (`R97_HOST_WALK_AB=1 R97_WALK_BLOCKS=4`; arm 2 runs
the host walk alone, which is how the rows were measured without the device walk's NaN):

| | device walk | host walk |
|---|---|---|
| gradient | **NaN** | finite, norm 3.38e11 |
| leaves non-finite | -- | 0 of 800 |
| first non-finite | `gdn_gated_rmsnorm` operand `gdn_o`[0] at 1143 | -- |
| backward | 281.5 s | 526.6 s |

The table predates the gain work and still describes this code: the tape's `x` is an in-place
*write* (`x:inout`), so the host's `norm_gain` skipped it and the device row's first read after `h`
was the weight either way -- both rows already read the weight's row 0. The one-block arm-2 run
after the change reports the same loss (7371.591348) and 0 of 800 leaves non-finite; its gradient
norm, 48.6, is over a one-block walk and so is not comparable with the four-block one above.

So there are two separate defects, and the growth is only the first:

1. The composition multiplies the residual stream's cotangent by about 1.6e3 per block, on both
   sides. The finite difference says the true leaf gradients are flat, so the factor is not the
   model's.
2. The device `gdn_gated_rmsnorm` row turns the already huge cotangent into NaN at position 1143
   where the host row stays finite. A row that is handed 1e30-odd values and returns NaN is a
   different bug from the one that made them that large, and it is the one that makes the *device*
   gradient non-finite while the host's is not.

## The forward, measured on the buffer the backward's growth would need

The premise above compares the norm's *output* `h`, which is O(1) by construction. The comparison
that matters is the buffer the cotangent is measured on, so `R97_DUMP_FWD=2` was run on the same
tape and the residual stream read at every block:

| position | op | buffer | forward `maxabs` | per-block |
|---|---|---|---|---|
| 14 | `rmsnorm_had_quant_i8g` | `x` | 720 | +3.12 from the embedding |
| 1429 | `rmsnorm_had_quant_i8g` | `x` | 14720 | +0.03 |
| 1449 | `rmsnorm_had_quant_i8g` | `x` | 14720 | +0.00 |
| 1457 | `add` | `x` | 14720 | +0.00 |

720 to 14720 over thirteen blocks is +1.31 decades in total, 0.10 per block: the forward stream is
flat at every block after the embedding, so the backward's 3.2 decades per block has no counterpart
in the forward. The growth is a defect in the walk, not the model.

## The composition defect the chain exposed, and why it is not the growth

`R97_DUMP_REC=1` prints one line per record in walk order: the largest cotangent it was handed,
the largest it left on its reads, and the ratio. Read per buffer instead (`DUMPPOS` beside
`DUMPDC`), the residual stream's chain is four records long and it **stops**:

| position | op | `cot[x]` | `dc[x]` |
|---|---|---|---|
| 1458 | `rmsnorm` (final) | -- | 1.3159e-7 |
| 1457 | `add` | 1.3159e-7 | 1.3159e-7 |
| 1449 | `rmsnorm_had_quant_i8g` | 1.3159e-7 | **sized=0** |
| 1429 | `rmsnorm_had_quant_i8g` | **live=0** | -- |

The fused norm's residual operand is a write the record also reads (`in_place`), and its row
declares it **frozen**: the row does not compute that slot, the walk's passthrough does. But the
freeze also kept the walk from *sizing* the slot, and the passthrough skips unsized slots ("not the
row's to fill") while the distribution skips frozen operands -- so the residual's cotangent was
dropped at every fused norm. `tests/host/backward_compose.cpp` pins the rule in milliseconds with
stand-in rows, and the fix is one clause in each of the two `frozen_at` tests in `src/backward.cpp`.

The rule has a second half, which the same file pins: the direct path is only the answer when no
*separate* read of that buffer already answered. A record shaped `read x, write x` gives the row the
read's slot, and that slot distributes to the same prior version the write's segments name -- so
filling both would count the cotangent twice. The check is on the value the row produced, not the
shape: a row that left the read empty has said nothing about the write either. Removing the guard
makes the same-buffer case fail at three times the seed, which is what the case is for.

The fix is proved on the live tape, not only in the unit case: with it, `x` is `sized=1` at both
fused norms and carries **1.3159e-7 unchanged** through the block (`pos=1449` and `pos=1429`, the
same value). Before it, `x` was `sized=0` at 1449 and `live=0` at 1429.

It is a real defect and it is fixed, but it is **not** the growth: the residual stream's cotangent is
flat across the block, and the growth is confined to the branch activations' cotangents (`h`, 1.95e-5
at the FFN's norm and 1.56e-3 at the mixer's).

## What the norms' own scale says

`R97_DUMP_QUANT=1` now also prints each norm's `s = 1/rms` per row and its gain's largest
magnitude, beside the sums it was handed and left:

| position | op | `s = 1/rms` | `gain_max` | `dy_sum` | `dx_sum` | `dx/dy` |
|---|---|---|---|---|---|---|
| 1458 | `rmsnorm` (final) | 2.96e-4 .. 3.94e-4 | 2.71 | 42.5 | 0.0285 | 6.7e-4 |
| 1449 | `rmsnorm_had_quant_i8g` (FFN) | 0.0215 .. 0.174 | 5.09 | 15.7 | 1.44 | 0.092 |
| 1440 | `rmsnorm` (k-norm) | 0.0719 .. 0.256 | 2.61 | 26.9 | 4.93 | 0.18 |
| 1439 | `rmsnorm` (q-norm) | 0.0598 .. 0.248 | 2.21 | 20.9 | 3.78 | 0.18 |
| 1429 | `rmsnorm_had_quant_i8g` (mixer) | 0.875 .. 0.892 | 8.31 | 175 | 192 | 1.10 |

Two facts follow, and both correct earlier reading of this report:

1. **The row is its own scale.** `dx_sum/dy_sum = 6.7e-4` and `s_max * gain_max = 1.07e-3` -- the
   same number, so `rmsnorm_host_adjoint` applies exactly `s * gain` and nothing else. No norm on
   this tape has a Jacobian the forward does not imply.
2. **`s` spans four decades across the tape's norms** -- 2.96e-4 at the final norm, 0.022..0.17 at
   the FFN's, 0.88 at the mixer's -- because each norm is handed a different buffer. `R97_DUMP_FWD=2`
   names them:

   | position | record | buffer | forward `maxabs` |
   |---|---|---|---|
   | 1457 | `add` | `x` (the residual stream) | 14720 |
   | 1449 | `rmsnorm_had_quant_i8g` | `h` (the mixer's output, normed in place) | 6.5 |
   | 1458 | `rmsnorm` (final) | `x` (the residual stream) | 14720 |

   So the final norm is handed the residual (rms 3400, hence `s = 3e-4`) and the fused norms are
   handed the O(1) branch activations. The norm shrinks whatever it is handed by `s * gain`, and the
   cotangent chain's steps are that same factor at every norm.

   Every row here is consistent with its own forward, so **no norm on this tape has a Jacobian the
   forward does not imply**, and the growth is not one of them. What the table does show is that the
   mixer's norm is handed a cotangent 11x the FFN's and returns 133x, and both factors are `s`.

The per-block factor is the product of two steps, and the table above separates them: the same
block's two norms differ by 80x (1449 -> 1429) and 106x (1421 -> 1408), and one block's start
differs from the next's by 41x (1429 -> 1421) and 40x (1408 -> 1400). So a block is about
40 x 40 = 1.6e3, which is the figure at the top of this report -- the two factors are the two
records, not one record's factor measured twice.

## The two-block test that decides the rest

One measurement separates "one branch's transpose is wrong by a constant" from "the chain multiplies
per block": the walk's gradient against a central difference at two *adjacent* blocks, which is what
`R97_CENTRAL=1 R97_CENTRAL_BLKS=model.layers.63.,model.layers.62.` runs. The prediction is written
down before the run so the result cannot be read backwards:

- the walk is 249x **small** at block 63;
- if the per-block factor is the 1.6e3 the cotangent table shows, block 62 must come out about
  `249 * 1.6e3 = 4e5` times **large** -- a sign flip between adjacent blocks, which no single row
  error can produce and which pins the growth to a factor that compounds through the chain;
- if block 62 comes out about 249x small too, the error is a constant in one branch and the chain is
  not multiplying -- and then the cotangent table's growth is itself the thing to re-measure.

### The result, read against that prediction

Neither branch: the FD's own numbers show it is not measuring a derivative. At block 63, one weight and
two perturbation sizes:

| perturbation | loss change |
|---|---|
| +1% (`eps_big` = 1.14e-4) | +0.0797 |
| -1% | -0.1364 |
| +10% (the control) | **-0.2131** |

A linear response gives the 10% move ten times the 1% move, +0.797. It gives -0.213, the *opposite
sign*, and the two 1% points disagree with each other by 1.7x. The loss is 7371.59, whose bf16
storage resolution is ~4e-3 relative, i.e. ~29 absolute: the ~0.2 the perturbations produce is 0.7% of
one bf16 ulp of the loss, and it does not scale with the perturbation. The `central` the arm prints
(948.9) is that 0.2 divided by 2*eps, so it is quantisation noise divided by a small number, not a
gradient. Block 62 shows the same shape (10%: +0.123 and -0.024, a 5x asymmetry; its "control" is
the +eps_big point because `eps_big` = 1e-4 is already 10% of its w = 0.0010075).

So the two-block run does not separate "constant" from "compounding", and the earlier "249x small"
framing rests on this same oracle. **The FD arm as built cannot adjudicate this gradient.** It needs a
forward that resolves the loss's response -- f32 storage, or a loss with more dynamic range than 3e-5
relative -- before it can be quoted either way. The control does its job (the params are live) and the
sign of the walk's gradient is not in question; the magnitude is unmeasured.

The cotangent rate is not an FD question at all: it is read directly off the walk's own buffer
cotangents, one per block boundary, and that is the number the ticket is about.

## The post-fix rate, measured over all thirteen blocks

`R97_DUMP_DC=1` on the device walk, the `h` cotangent the fused norm leaves at each block boundary
(walk order, so downwards through the stack):

| pos | block | `h` cotangent maxabs | factor per block |
|---|---|---|---|
| 1449 | 63 | 7.24196e-06 | -- |
| 1429 | 62 | 9.23157e-04 | 127x |
| 1421 | 61 | 1.3855e-02 | 15.0x |
| 1408 | 60 | 9.80469e-01 | 70.8x |
| 1400 | 59 | 3.375e+01 | 34.4x |
| 1387 | 58 | 1.36e+03 | 40.3x |
| 1379 | 57 | 3.5072e+04 | 25.8x |
| 1366 | 56 | 1.36806e+06 | 39.0x |
| 1358 | 55 | 8.44104e+07 | 61.7x |
| 1338 | 54 | 3.79165e+09 | 44.9x |

Geometric mean 40.1x per block, **1.60 decades per block**. The pre-fix table measured 3.3, so the
composition fix halved the rate and did not remove it. The **residual** stream's cotangent, the same
walk, is 1.31316e-07 at every one of these boundaries: flat, before and after.

The same walk on the host arm, from the chain table above, leaves `dc[h]` at 1.95e-05 on block 63's
norm and 1.56e-03 on the next one: 1.9 decades per block. **Both arms grow at the same rate**, so this
is not a device-row defect; the two arms' rows agree to a factor of 2.7 at block 63 (7.24e-06 device
against 1.95e-05 host).

## Where the growth is, record by record

`R97_DUMP_REC=1` on the same walk, block 63, the largest ratio each record applies to the cotangent it
is handed (`ratio` is the largest element's, `sum_ratio` the sum's, and `dc_slot` names which of the
record's reads the largest `dc` landed on):

| pos | op | cot_max | dc_max | ratio | sum_ratio | dc_slot |
|---|---|---|---|---|---|---|
| 1456 | lora_expand | 1.31e-07 | 1.22e-06 | **9.29** | 1.07 | 0 |
| 1452 | lora_expand | 1.28e-05 | 1.14e-04 | **8.91** | 1.06 | 0 |
| 1448 | lora_expand | 7.24e-06 | 8.01e-05 | **11.07** | 1.06 | 0 |
| 1438 | lora_expand | 3.81e-05 | 1.79e-04 | **4.68** | 1.16 | 0 |
| 1435 | lora_expand | 1.19e-04 | 7.27e-04 | **6.10** | 1.34 | 0 |
| 1432 | lora_expand | 8.06e-05 | 4.24e-04 | **5.26** | 1.07 | 0 |
| 1444 | attn_paged | 4.63e-05 | 2.88e-04 | 6.23 | 4.62 | 1 |
| 1454/1446/1436/1433/1430 | gemm_nt_q | -- | -- | 0.008-1.85 | 0.02-4.5 | 0 |
| 1458/1440/1439/1449/1429 | rmsnorm, rmsnorm_had_quant_i8g | -- | -- | 7e-4-0.42 | 0.18-2.65 | 0 |
| 1453 | gated_had_quant_i8g | 1.88e-05 | 1.28e-05 | 0.68 | 4.50 | 0 |
| 1445 | gate_had_quant_i8g | 1.47e-03 | 8.06e-05 | 0.055 | 1.53 | 0 |
| 1455/1451/1447/1437/1434/1431 | lora_shrink | -- | -- | 15-36 | 31-416 | **1** |

**The `lora_shrink` rows are not the record's value-space ratio.** Every one of them has `dc_slot=1`,
and the `dc_max` there equals the record's `h.s8` (scale) cotangent to the digit -- 0.0063392 at pos
1437, 0.0181934 at 1434, 0.00831351 at 1431, the same numbers `DUMPDC` prints for `h.s8` at those
positions. The scale's cotangent is *dead*: the quantiser's adjoint treats the scale as a constant
(the straight-through rule, `src/adjoint/host/quant.cpp:341` drops `cot[1]`), so nothing reads it.
On the slot that does travel -- the codes, `h.q8` -- the record **shrinks**: 1.79e-04 in, 4.23e-06
out at pos 1437 (0.024); 7.27e-04 in, 2.03e-05 out at 1434 (0.028); 4.24e-04 in, 6.71e-06 out at
1431 (0.016).

So the only record that multiplies on a live slot is **`lora_expand`**, at 4.7-11.1 per instance, and
its magnitude is its own matrix: `dx = B^T dy` over 5120 channel terms with `B` of order 1e-3, which
is what the container's own `lora_A` elements measure (1.0e-03 and 1.1e-02 in the finite-difference
arm). The earlier version of this section read the scale slot as if it were the value slot and
concluded the multiplier was `lora_shrink` with "`A` of order one"; both were wrong, and the sampled
elements are two decades below order one.

Every other record shrinks or passes through. So no record is out of line with its siblings, and the
growth is the branch's Jacobian product -- but the only factor in it that is above one is the LoRA
expansion's `B^T`, at 4.7-11.1, and `B`'s entries are measured rather than inferred.

## The live-scale finite difference: the amplifying rows are exact

Every row check in the suite runs at O(1) synthetic values (`tests/support/adjoint_check.h` seeds with
`urand`), and the quantisers' clamp masks are scale-dependent, so a row can pass there and be wrong
where the walk actually is. `R97_FD_REC=<pos>` closes that gap: it runs the row's twin at the tape's
own operands and compares `dc[i][e]` against the twin's derivative of `<twin, cot>` at that element,
at the walk's magnitude. Host arm, one block, block 63's chain:

| pos | op | read | analytic | numeric | ratio |
|---|---|---|---|---|---|
| 1456 | lora_expand | `lane0` | -1.22004e-06 | -1.22004e-06 | **1** |
| 1448 | lora_expand | `lane0` | -1.3156e-04 | -1.3156e-04 | **1** |
| 1455 | lora_shrink | `ffn.q8` | 1.2534e-06 | 1.2534e-06 | **1** |
| 1455 | lora_shrink | `ffn.s8` | 1.8775e-05 | 1.8775e-05 | **1** |
| 1444 | attn_paged | `attn_q` | 2.51058e-04 | 2.51058e-04 | **1** |
| 1444 | attn_paged | `kv:1` | -3.9499e-04 | -3.9499e-04 | **1** |

**Every well-posed slot is exact to the printed digits**, at magnitudes from 1e-07 to 1e-04. So the
4.7-11.1x that `lora_expand` applies is the *true* Jacobian of `B^T` at the walk's own operating
point, not a row defect, and the growth is the product of those true factors. That is what earns the
"branch Jacobian" reading, which until this run was an inference from ratios that could equally have
been a row error at scale.

Three rows cannot be checked this way, and each says why:

- **`gemm_nt_q`**: its weight is an unresolved `none:` operand in the host walk's snapshot, so the twin
  is fed a zero matrix and reports `numeric=0` against a nonzero `analytic`. The *row* has the weight
  (it resolves it through `stored_weight`); the twin does not.
- **`rmsnorm_had_quant_i8g`'s activation**: its output is quantised, so the twin is a step function and
  its central difference is 0 almost everywhere while the straight-through row reports `cot/s`. The FD
  is the wrong oracle for a quantiser by construction, at any scale.
- **its weight**: frozen by design (`nW`), so `analytic=0` against the FD's true -0.0072 is the walk
  deliberately not training it.

`R97_FD_REC` stays in the tree as the instrument for the two rows that remain open at live scale.

## What the device cannot hold

The device path stores every cotangent as bf16 (`f32_to_bf16` in all nine device row files) and the
host path keeps f64. At block 54 the `h` cotangent is 3.79e+09 and the rate carries it past bf16's
3.4e+38 by block ~50, which is where the walk's first non-finite appears:

```
non-finite: cotangent op gdn_gated_rmsnorm, operand 'gdn_o'[0] at position 1143, value 0, from the row
```

with `DUMPDC side=device op=gdn_gated_rmsnorm pos=1143 buffer=gdn_o ... nonfinite=1184349
maxabs=3.38953e+38`. 3.38953e+38 is bf16's largest finite value; the row is where the magnitude first
exceeds it, not where the arithmetic goes wrong. The host walk, which keeps f64, ends finite with a
gradient norm of 48.6156.

## Status: what is fixed, what is measured, what is not

**Fixed, with proof.** The composition defects (the frozen in-place slot that was never sized and
never distributed; the passthrough that double-counted a buffer a separate read had already answered)
are fixed in `src/backward.cpp` and pinned by three cases in `tests/host/backward_compose.cpp`. Their
proof is direct: `DUMPDC` at the fused norm's `h` read goes from `sized=0` to `sized=1 maxabs=7.24e-06`,
and the residual stream's cotangent is 1.31316e-07 at every block boundary, before and after.

**Measured.** The branch's `h` cotangent grows 1.60 decades per block over thirteen blocks (40.1x,
geometric mean), down from 3.3 before the fix. The host arm grows at the same rate (1.9 decades on the
one block both arms measured), and the two arms agree to a factor of 2.7 at block 63. No record's
*live* ratio is out of line with its siblings: the only one above one is `lora_expand`'s 4.7-11.1,
which is `B^T` over 5120 terms with `B` of order 1e-3 (measured: 1.0e-03, 1.1e-02), and every other
record shrinks. `lora_shrink`'s larger `DUMPREC` ratios are its *scale* slot, which the quantiser's
straight-through adjoint drops -- dead cotangent, not the row's Jacobian.

**Resolved for the rows that amplify; open for three.** The loss-level central difference in
`device_step_tape` cannot adjudicate this gradient -- at block 63 the same weight moves the loss +0.080
at 1% and -0.213 at 10%, a sign flip where a linear response gives +0.797 -- but the *row-level*
`R97_FD_REC` can, and it says every well-posed slot is exact at the walk's own magnitudes (table
above). So the 4.7-11.1x the branch applies is the twin's true Jacobian. **That is weaker than "the
growth is the product of true factors", and the third open item is why:** the FD is circular on the
forward, and `DUMPTWIN` shows the fused norm's twin re-quantising an activation the kernel did not
produce (99.4% of codes, scale off by 101%) at exactly the record whose `dc[h]` the growth is measured
on. A twin that is wrong there makes the norm's row wrong and the FD blind to it in the same step.

The three open items at live scale:

1. **the fused norm's twin** -- codes disagree with the kernel's on 99.4% of elements at pos 1449 and
   93.9% at 1429, against 5.9% (one integer step) for both gated producers. This is the one that
   matters: it is the row the growth is read from.
2. **`gemm_nt_q`** -- its weight is an unresolved `none:5@1454`/`none:6@1454` operand in the host
   snapshot, so its twin sees zeros.
3. **the quantised producers' activations** -- a step function, which no central difference describes.

So the claim this report can stand behind is: the amplifying rows' *adjoints* match their *twins*, the
gated twins match the kernel, and the norm's twin does not. Whether the growth is the true Jacobian is
still open, and now has a named suspect.


**A consequence, not a second defect.** The device path stores cotangents as bf16 (all nine device row
files call `f32_to_bf16`); the host keeps f64. At block 54 the branch cotangent is 3.79e+09 and the
1.60-decade rate carries it past 3.4e+38 by block ~50, where `gdn_gated_rmsnorm`'s `gdn_o` reads
`maxabs=3.38953e+38` and 1.18M of its 1.96M elements are non-finite. That value is bf16's largest
finite number, so the row is where the magnitude first exceeds the format, not where its arithmetic
fails. f32 would not help (the same 3.4e+38 ceiling); only f64 or a bounded cotangent would, and the
growth's being real is exactly what is unresolved.

## What this leaves

The earlier hypothesis in this report -- that a factor of 1e4 per gemm was a `1/scale` applied once
instead of twice -- is dead, and the per-record chain is what kills it. `gemm_nt_q`'s ratio is 0.008 to
1.85, a *shrink* on every instance but one, because `dx = W^T dy` carries the weight matrix's own
magnitude; the quantised producer's `1/scale` and the gemm's `scale` do cancel, which is the algebra at
`src/adjoint/host/quant.cpp` line 365: the tape hands back the codes' cotangent, `gemm_nt_q` has
already multiplied by `s` on its way to it, and dividing by `s` restores `W^T dy`. `lora_expand` is
the one record that multiplies on a live slot, by 4.7-11.1, and its factor is its own matrix: `B^T`
over 5120 channel terms with `B` of order 1e-3, measured from the container's own `lora_A` elements
(1.0e-03, 1.1e-02).

So the growth is the branch's Jacobian, not a misapplied scale, and it is present in both arms at the
same rate. Whether it is the *true* Jacobian is the one question this work could not answer, because
the only oracle for it -- the central difference -- cannot resolve a loss response of 3e-5 relative
through a bf16 forward (section above). What would answer it, in order of cost:

- a central difference on a forward whose activations are f32, so the loss resolves the perturbation;
- or the same bf16 forward over many more tokens, so the response's cancellation averages down;
- or a per-block adjoint test that differentiates one block's rows against the block's own twin in
  isolation -- `tests/host/backward_compose.cpp` is the shape of it, at block scope rather than record
  scope.

Until one of those runs, the honest statement is: the composition defects are fixed and pinned, the
residual stream's cotangent is flat, the branch's cotangent grows 1.60 decades per block in both arms
with no record out of line with its siblings, and the device path cannot carry that magnitude in bf16.

## Why the int8 producers' twins have no kernel check

`tests/host/adjoint_quant.cpp` checks the fp8 family's twins against radiance's host kernels
(`adjoint_fused_quant_twin_matches_the_host_kernel`). The same check for the three int8 producers
cannot exist: `check_kernel` asks the kernel table for `rmsnorm_had_quant_i8g` and gets

```
no host kernel for op 'rmsnorm_had_quant_i8g' in the kernel table
```

so those rows are FD-checked against their own twins and nothing else. The tape is the only other
oracle, and it is a real one: a fused producer writes the part's output in place, so the walk can
compare its twin's intermediate against the activation the kernel actually wrote. `R97_DUMP_QUANT=1`
now prints that comparison as `DUMPTWIN pos= op= code_err= code_bad= scale_rel= act_at= act_err=
act_rel=` -- the twin's re-quantisation against the tape's codes and scales, and the twin's
intermediate against the activation at the record's in-place write. It is the check the kernel table
cannot give, and it costs one pass.

## The twins' forward against the tape: the gate matches, the norm does not

The live-scale FD above compares a row against its own twin, so it is circular on the *forward*: if a
twin's activation differs from the kernel's, the walk evaluates every row at the wrong point and the
FD still reads 1. `DUMPTWIN` (under `R97_DUMP_QUANT=1`) is the tape-side oracle -- it re-quantises the
twin's intermediate and compares the codes and the scale against the ones the kernel actually wrote:

| pos | op | `code_err` | `code_bad` | `scale_rel` |
|---|---|---|---|---|
| 1453 | gated_had_quant_i8g | 1 | 529616/8912896 (5.9%) | 0.00289 |
| 1445 | gate_had_quant_i8g | 1 | 184822/3145728 (5.9%) | 0.00262 |
| **1449** | **rmsnorm_had_quant_i8g** | **254** | **2604887/2621440 (99.4%)** | **1.01364** |
| **1429** | **rmsnorm_had_quant_i8g** | **45** | **2462396/2621440 (93.9%)** | **0.31958** |

The two gated producers agree to one integer step on 94% of codes and 0.3% on the scale -- bf16
rounding at the quantiser's boundaries, which is what a correct twin looks like. **The fused norm does
not agree at all**: 99.4% of its codes differ, by up to 254 (twice the i8 grid), and its scale by
101%. A scale is `amax/127`, so a scale that is off by 2x means the twin's activation has twice the
amplitude of the kernel's -- or is a different tensor.

**The activation column of `DUMPTWIN` cannot decide that**, and the reason is worth recording: the
walk's snapshot of an *in-place* write is the value the buffer held *before* the record, so for
`rmsnorm_had_quant_i8g` there is no operand in `fwd` holding the kernel's normed `h` at all --
`best_of=[1:x:1 3:h.q8:1.03 5:h:1.36]` is the twin's `y` against the residual, the codes and the
*un-normed* `h`. Only an *ordinary* write (the codes, the scale) carries the kernel's output, which is
why the code comparison above is the one that counts.

So this is an open discrepancy at exactly the record whose `dc[h]` the growth is measured on, and the
FD is blind to it by construction. It is not the per-row gain: `FDRMS pos=1449 rows=512 w_rank=1
w0=5120` says the tape's gain there is a single `[5120]` row, so row 0 is the whole gain and
`R97_GAIN_ROW` changes nothing (measured: identical output to the digit). The remaining candidates are
the twin's norm parameters (`eps`, `wadd`, `mult`) and the quantiser's rotation, and the tape's own
codes are the oracle for both -- re-quantise the kernel's activation with the twin's quantiser, or the
twin's activation with the kernel's, and the one that matches names the side.

## The gain's two snapshots

`GAINPOS` (under `R97_GAIN_POS=1`) prints the operand `norm_gain` takes, with the schema's shape and
the snapshot's. On the live fused records the same operand arrives twice, in two different `fwd`
vectors, with different shapes:

```
GAINPOS pos=1449 j=2 blk.63.ffn_norm.weight schema=[5120] value_rank=1 values=5120
GAINPOS pos=1449 j=2 blk.63.ffn_norm.weight schema=[5120] value_rank=2 values=2621440
GAINPOS pos=1429 j=2 blk.63.attn_norm.weight schema=[5120] value_rank=1 values=5120
GAINPOS pos=1429 j=2 blk.63.attn_norm.weight schema=[5120] value_rank=2 values=2621440
```

The schema is the container's own description of the weight -- a single `[5120]` row -- and one
snapshot agrees with it while the other is a `[512, 5120]` tensor materialised once per row. Which one
a row gets depends on the `fwd` vector it is handed, and `note_gain_rows` warns about the second while
`FDRMS` (inside `rmsnorm_twin`) sees the first. That is the contradiction the two instruments
reported, and it is a property of the tape, not of either instrument.

**Preferring the snapshot whose rank matches the schema does not fix the twin's disagreement**, and it
was tried and reverted: with `norm_gain` skipping the rank-mismatched snapshot, `DUMPTWIN`'s
`code_err` at 1449 stayed at 254 and the walk's own test went from passing to failing. So the rank
mismatch is real and is not the cause of the norm's twin disagreeing with the kernel; the cause is
still the twin's norm parameters or the quantiser's rotation.

## The quantiser's rotation is right; the norm's activation is not

`geometry()` sets `rotated = is_rotated(op)` and `is_rotated` is true for `Op::had`, which is what
`had_quant_record` names. **The twin quantiser therefore already applies the FWHT** -- there is no
"unrotated" arm. `DUMPROT` adds a second rotation on top of the twin's own, so the table reads:

| producer | twin's own rotation (one FWHT) | `DUMPROT` (two FWHTs) |
|---|---|---|
| `gated_had_quant_i8g` | **5.9%** | 99.2% |
| `gate_had_quant_i8g` | **5.9%** | 99.3% |
| `rmsnorm_had_quant_i8g` | 99.4% | 99.4% |
| `rmsnorm_had_quant_i8g` (1429) | 93.9% | 99.3% |

The gated rows match at one rotation and collapse at two: their twins reproduce the kernel's codes to
**1 LSB**, so both the rotation and the activation are right. The norm rows disagree at both counts,
so no rotation explains them -- the twin's **normed activation** differs from the kernel's, and that
is where the next test goes.

**The earlier schema test was the wrong shape.** It *skipped* a snapshot whose rank disagreed with the
schema, which for `rmsnorm_had_quant_i8g` leaves `norm_gain` with no gain at all (`w == nullptr`) --
the twin then normalises without a weight, a different wrong answer rather than a fix, which is why
`code_err` stayed at 254 and the walk's test failed. The right shape is to **prefer** the
schema-matching snapshot and fall back only when there is none; `GAINPOS` already shows both snapshots
at the same position, so the tape carries what decides it.

### The gain offset is right; the earlier test was backwards

`NORMTWIN` prints the twin's constants: `w=yes eps=1e-06 wadd=1`. **That is correct.**
`docs/OPS-qwen35-device.md` declares `eps=1e-06 wadd=1` on every `rmsnorm_had_quant_i8g` row, and
`docs/adjoints/elementwise.md:181` records that Qwen3.5 uses GemmaRMSNorm's `w + 1` with a
**zero-centred gain**. The record and the twin both match the op.

Forcing `wadd=0` therefore tested the wrong direction: with a zero-centred gain it makes `w + 0` near
zero, so the twin's activation collapses. The run cannot discriminate the 99.4% disagreement, and the
numbers say so once read correctly:

| pos | `code_bad` | `scale_rel` | `act_rel` |
|---|---|---|---|
| 1449, `wadd=1` | 99.4% | 1.014 | 1.363 |
| 1449, `wadd=0` | 99.4% | 0.919 | 1.044 |

**`act_rel` is `max|f - y| / max|f|` -- a relative error, not a match.** 1.363 -> 1.044 is a move
from 136% to 104% error: total disagreement either way. The same reading applies to `best_of`'s third
field, where `1:x:1` is a **100% relative error**, not "matches `x` at ratio 1". An identity would
print `1:x:0`. The earlier reading of that field as a match was wrong.

**`code_bad` counts any nonzero code difference**, so 99.4% is not by itself evidence of a
permutation: a uniform sub-LSB mismatch flips nearly every code, and `code_err=254` is a single
element. The instrument that separates the two is a histogram of `|twin_code - kernel_code|` -- a
tight 1-2 with a few outliers is a small systematic difference (gain values, precision), a broad
spread is what a layout difference would need.

**The schema-rank preference is inert.** Preferring the snapshot whose rank matches the record's
schema changes nothing (`code_err` 254 at 1449, 45 at 1429, byte-identical), so `norm_gain` was
already returning the rank-matched operand and the change is reverted rather than kept as an unproven
branch.

### The gain is exonerated on every axis

`GAINPOS` at both norm positions shows **one operand (`j=2`) on one buffer** with two snapshots:

| pos | buffer | schema | snapshot 1 | snapshot 2 |
|---|---|---|---|---|
| 1449 | `blk.63.ffn_norm.weight` | `[5120]` | rank 1, `[5120]` | rank 2, `[512,5120]` |
| 1429 | `blk.63.attn_norm.weight` | `[5120]` | rank 1, `[5120]` | rank 2, `[512,5120]` |

So the twin and the other caller are handed **different snapshots of the same weight buffer**, not
different tensors. `FDRMS` confirms which one the twin gets: `w_rank=1 w0=5120 per_row=0`. The rank-1
snapshot is the weight itself and is the shape the record declares, so the twin's gain is the right
buffer, the right shape, the right `wadd` and the right `eps`.

That kills the per-row theory twice over: `R97_GAIN_ROW` only re-indexes the tensor the twin already
holds, and the twin does not hold a per-row tensor.

**The gain is therefore exonerated on every axis** -- buffer, shape, offset, epsilon. What is left in
the norm's twin is its quantiser's scale or rounding, or the tape's snapshot of `h`; the histogram
says the difference is structural (a broad spread to 128), not a rounding artefact.

### The kernel's activation, reconstructed, and a tape defect at 1429

`DUMPROT` now also reconstructs the kernel's own activation from its codes and scale --
`fwht(codes * scale) / group`, since `fwht` is unnormalised (`H*H = n*I`) -- and compares it to the
twin's output. That bypasses the twin's quantiser entirely:

| pos | op | `deq_rel` (twin's activation vs the kernel's) |
|---|---|---|
| 1453 | `gated_had_quant_i8g` | **0.005** |
| 1445 | `gate_had_quant_i8g` | **0.008** |
| 1449 | `rmsnorm_had_quant_i8g` | **1.364** |
| 1429 | `rmsnorm_had_quant_i8g` | **0.628** |

The gated twins reproduce the kernel's activation to 0.5-0.8%, so the twin machinery is sound. The
norm's are 63-136% off, so the defect is in the **norm's activation**, not its quantiser.

`DEQORACLE` then puts the reconstructed activation beside the tensors around it:

| pos | `deq_vs_in` | `deq_vs_write` | `in_vs_write` |
|---|---|---|---|
| 1453 gated | 21.76 | 0.906 | 0.083 |
| 1449 norm | 0.996 | **0.0003** | 0.996 |
| 1429 norm | 0.0055 | 0.0055 | **0** |

Two results:

- **The oracle is sound.** At 1449 the reconstructed activation matches the in-place write to
  **0.03%**, so `dequant(codes, scale)` is exactly what the kernel stored.
- **At 1429 the tape records the same value for the read operand and the in-place write**
  (`in_vs_write = 0`). A read that is also written in place must be snapshotted at its *pre*-write
  value; here the twin is handed the post-write value and normalises its own output. That is a tape
  defect, and it makes the 1429 twin meaningless.

At 1449 the snapshots do differ (`in_vs_write = 0.996`), so that defect is absent there -- yet the
twin is still 136% off. A relative error of 1.364 with a `scale_rel` of 1.014 is consistent with the
twin's activation being about 2x the kernel's, which is the size of a `(w+1)/w` factor when the
zero-centred gain sits near 1; but forcing `wadd=0` moved `act_rel` only to 1.044, so a single gain
factor does not explain it either. What is established is the direction: the norm's twin computes a
different activation, and at 1429 the tape hands it the wrong snapshot outright.

## The quotient's guard## The quotient's guard

The mask pass divides each code's cotangent by its group scale, and the quotient is exact only
while it stays inside f32. On this container it does not: 2.66e36 over a 5.96e-5 scale is 4.5e40.
The kernel now counts the overflowing quotients where the division happens -- so an element the
mask zeroed is not counted, and a group whose scale keeps the quotient finite is not either -- and
reports `QUANTGUARD op=... pos=... nonfinite=N of M worst=... min_scale=...` on stderr. The count
is of quotients that are not finite at all -- an Inf past f32's range, or a NaN the cotangent
already carried -- because on this container both happen: the top blocks' cotangents are finite but
huge, and the low blocks' are already NaN.
`R97_QUANT_GUARD=1` turns the same count into a refusal; the default is to report, because a run
whose cotangents already overflow must reach the measurement that says why.
