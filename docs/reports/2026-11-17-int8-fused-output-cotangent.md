# The int8 fused producers' output cotangent: the host row is now the device row's function

Issue #142. The device row for the three int8 fused producers (`rmsnorm_had_quant_i8g`,
`gated_had_quant_i8g`, `gate_had_quant_i8g`) folds the **fused bf16 output's own cotangent** into
its result; the host row did not. They were therefore **different functions**, which breaks the
transitivity argument that clears these ops: the device-vs-host check agrees with a reference that
is not the row the walk runs, and the host-vs-finite-difference check agrees with a function whose
term is zero in every fixture, so a factor the device adds and the host lacks is invisible to both
halves -- the same shape of blind spot as the `gemm_nt_q` activation scale that survived every
green test on the way to #55.

## What the change does

`fused_adjoint` (`src/adjoint/host/quant.cpp`) now mirrors the device row's post-pass over the
record's writes (`src/adjoint/device/quant.hip:817-880`) write for write:

- a write of the part's own width (`total` values) joins the quantiser's output cotangent, in
  `ecot[0]`, before the elementwise part's adjoint runs -- the term the device adds into its
  rotated cotangent;
- a write over a buffer the record also reads is the **direct path**: the engine writes the
  elementwise input back, so the write's cotangent is the read's and is added to the read's `dc`
  slot at the identity;
- a write whose cotangent slot holds no values, or one scalar, is the engine's no-value
  placeholder and is skipped, as the device skips it;
- anything else refuses by name, as the device refuses it.

Before, the host folded only the in-place write's cotangent (the #147/#150 fold), threw where the
device skips the placeholder, and silently ignored every other write. `fused_twin` is the forward
half of the same rule: it now produces the output's write wherever the schema puts it -- the part's
output over the read buffer it covers (in place), the part's output to a fresh buffer, the read's
own value on the direct path, and no values on a placeholder.

## Why the fixture was the hard part

The term was zero in every fixture that exists. The producers' `_ste` records wrote only the codes
and the scale; the walk hands the row one cotangent per write, so the output's slot was never
there, and a row that dropped the term and the exact row agreed at zero. The fixture now writes the
part's output beside the codes (`out` in `i8g_rec`), and `check_adjoint`'s random cotangent for
that write is what makes the term live: a row that drops it, or routes it through the part instead
of onto the part's output, fails the finite difference. The in-place half of the rule already had
its aliased check (`in_place_rmsnorm_had_quant_i8g`, `tests/host/adjoint_in_place_fd.cpp`, landed
with #147/#150), and it is unchanged: an in-place write of the part's width takes the same branch
it always did.

The branches the fixture does not reach are pinned in
`adjoint_i8g_producers_fold_the_fused_writes` (`tests/host/adjoint_quant.cpp`): the direct path is
checked as an exact identity (`dc[read]` with the term equals `dc[read]` without it plus the
cotangent, value for value, which neither routing through the part nor dropping satisfies), the
refusal of a write the row does not serve names the op, and the two placeholder widths skip rather
than refuse.

## The GPU tier now compares the same function

`tests/gpu/adjoint_quant.cpp` carried a second, inline reference -- the composition of the parts'
host adjoints written inside the test TU. It dropped the same term the old host row dropped, and it
matched the device only because the fixture never made the term non-zero. It is deleted: the
harness now looks up the registered host row (`src/adjoint/host/quant.cpp` registers the
composition under each producer's name), so device and host are the same function, and
`quant_case` gives the fused producers a consumed `out_bf16` write with a random cotangent, so the
term is non-zero on both sides of the comparison. `direct_path_case` pins the direct path where
both sides are the registered rows.

## Measured

The host tier (`r97_host_tests`, the whole binary, exit 0) with the output's cotangent live -- the
term the fixture used to leave at zero:

| producer (linearised) | shape | FD rel err |
|---|---|---|
| `rmsnorm_had_quant_i8g_ste` | trunk-128 | 5.0e-10 |
| `gated_had_quant_i8g_ste` | trunk-128 | 9.7e-11 |
| `gate_had_quant_i8g_ste` | trunk-128 | 1.7e-09 |
| `rmsnorm_had_quant_i8g_ste` | two-groups | 1.5e-10 |
| `gated_had_quant_i8g_ste` | two-groups | 6.4e-11 |
| `gate_had_quant_i8g_ste` | two-groups | 2.1e-10 |

A row that dropped or misrouted the term would miss by O(1) here, not by 1e-10: the finite
difference moves the part's output and the random cotangent on its write is what carries that into
the loss. The in-place aliased check (`in_place_rmsnorm_had_quant_i8g`, the branch the row always
took) is unchanged at 9.6e-10 (trunk-128) and 5.2e-10 (two-groups). The new pins
(`adjoint_i8g_producers_fold_the_fused_writes`) hold: the direct path's exact identity ("the
write's cotangent is the read's"), the refusal of an unserved write by name ("the write 'extra'
carries 1280 values, not the part's 640 or the direct path's 640"), and both placeholder widths (0
and 1 values) skipped with the row served.

Device tier (`adjoint_quant_device`, in a held window): all three producers agree with the
registered host row (tolerance `0.001 + 0.02*|host|`) at the five shapes -- 4x128, 4x128 two
groups, 1024x5120, 2024x5120 batch 300+700+1024, 4096x5120 -- with the fused producer carrying its
consumed `out_bf16` write; the wrong kernel is still refused in slot 0; the tape's phantom
shapes, the live gate shape and the four malformed-record refusals hold; and the direct path
reads "device == host, the write's cotangent on the read's slot".

## Relation to #148 and its open PR #157

#157 fixes the device step walk's forward snapshot: an in-place read with no earlier writer kept
the live pull, i.e. the op's own post-op output, so the row differentiated at its output (63 live
reads, the first at position 1429, the position #145 independently identified). That is the
forward's half -- the value a row reads. This is the backward's half -- the cotangent a row folds
in. Different files (`step_device.cpp` against `adjoint/host/quant.cpp` and the two quant test
files), no shared contract, no rebase interaction.

What neither alone establishes, and what the pair does: that the device row and the host row are
the same function **at the same point** on the live tape. The fixture-based claims here do not
depend on #157 at all -- the GPU test builds its records directly and never walks -- but the
transitivity, taken over a real step, additionally needs #157's forward fix, and a finite gradient
additionally needs #140's f32 resolution (the fixed-tree walk still reports the overflowing
`cot / s` at `rmsnorm_had_quant_i8g` position 1, which is #140's, not this change's).

## What this does not prove

The magnitude caveat of the 2026-11-14 report stands: these checks run on synthetic fixtures with
O(1) values, and a defect that engages only at the live cotangent's 1e34..1e37 would not fire on
them. On the measurement at hand the term is small -- one of 78 `rmsnorm_had_quant_i8g` records in
a contiguous 895-record dump carries a cotangent on the bf16 output, and there it is one part in
8.0e5 of the codes' path in value space. This closes the "different functions" breakage, which is
where a factor could hide; it is not a claim about the term's size.
