# Device kernels

This repository's device kernels live in `libr97.so`, a radiance kernel plugin built from
`src/adjoint/device/` and loaded next to `libr4d` in radiance's home. A kernel is an op's device
adjoint: an `r97::AdjointFn` registered with `R97_REGISTER_ADJOINT(op, device, fn)` from the same
`.hip` file that defines it. The worked example is `add` in `src/adjoint/device/add.hip`, its test
in `tests/gpu/adjoint_add.cpp`, and the harness both use in `tests/support/device_check.h`.

Nothing lists the kernels: the CMake glob collects every `.hip` under `src/adjoint/device/` into
the plugin, and each file registers itself into r97's registry when the plugin loads. To add one,
add a file; no shared file changes.

## Adding a device kernel

1. **Write `src/adjoint/device/<op>.hip`.** Include `"r97/adjoint.h"` for the contract,
   `"scratch.h"` for device buffers, and `"rad_device.h"` for radiance's device layer. The
   contract (`include/r97/adjoint.h`) hands a `const AdjointCall&`:

   - `call.rec` -- the op's record; `params` carries the op's options as the host adjoint reads
     them.
   - `call.fwd` / `call.cot` -- the forward values and the cotangents, in schema order, as f64
     `AdValue`s. A device adjoint receives host values, not device buffers: it owns every upload
     and download.
   - `call.dc` -- one `AdValue` per operand in schema order, zeroed by the caller. A kernel
     **adds to** these values (the host adjoint does the same): a cotangent can be shared by
     several ops.
   - `call.grads` -- unused by an op with no trainable leaves; leave it alone.

   Read only the operands the schema says the op reads, decide broadcasting from the cotangent
   sizes exactly as the host adjoint does (so both spellings of the op agree on what a shape
   means), and return without touching `dc` when the shapes cannot be served: an adjoint must not
   throw.

2. **Take device memory from `scratch_alloc()` and return it to `scratch_free()`**, or hold it in
   a `ScratchBlock`, which frees on every return path. The allocator is in
   `src/adjoint/device/scratch.h`. It charges the same VRAM pool as the engine, keeps a live-bytes
   ledger, and records a process-wide high-water mark; the harness reads the peak for the kernel's
   doc and fails a case whose kernel leaked. A failed allocation returns null: clean up and
   return.

3. **Launch on a cached `RadStream`** (`rad_stream_create`), upload with `rad_memcpy_async`,
   launch with `hipLaunchKernelGGL`, and `rad_stream_sync` before downloading. The stream belongs
   to the device `rad_dev_set()` chose, so a process that will run kernels selects its device
   first.

4. **Register it** at the bottom of the same file:

   ```cpp
   R97_REGISTER_ADJOINT("add", device, add_device_adjoint)
   ```

5. **Test it** in `tests/gpu/<op>.cpp` (see below).

The plugin's radiance-side table (`src/adjoint/device/plugin.hip`) is deliberately empty: the
plugin exports no radiance op rows, it is the vehicle that gets r97's kernel code objects into the
engine. That file also carries fallback definitions of `adjoint_register` and the scratch calls,
so the plugin loads in a radiance process that does not link r97 -- `rad-convert`, which scans the
same `kernels/` directory -- and simply serves it no adjoints. A process that links r97 must build
with `ENABLE_EXPORTS` (`r97train` and the gpu tests do); its definitions win over the fallbacks and
the kernels register for real.

## The harness

`tests/support/device_check.h` is the device-versus-host comparison every device kernel uses.

```cpp
using namespace r97;

device_check::Case c;
c.rec = /* the op's OpRecord, operands in schema order */;
c.in  = { /* one AdValue per read operand */ };
c.cot = { /* one AdValue per written operand, same shape */ };
c.label = "16 x 5120 bias";

device_check::require(device_check::check(c));
```

`check()` builds the forward values with the op's f64 twin, runs the registered `Side::device`
adjoint, runs the registered `Side::host` adjoint on the same values, and compares every `dc` slot
within `kAbsTol + kRelTol * |host|` (`1e-3 + 0.02 * |host|`). The tolerance is the room a kernel
that rounds to bf16 and accumulates in f32 needs: bf16 has 8 significand bits, so one unit
roundoff is 2^-8, and a few hundred terms move a reduction's result by a few of those. It is not a
license to compute another function; the test for the wrong kernel below holds the harness to
that.

A device kernel that fuses several ops has no single host row to compare against: pass the host
reference as the third argument, a `HostRef` that runs host adjoints in the order the kernel
undoes them.

```cpp
device_check::check(c, device_fn, [](AdjointCall& call) { host_b(call); host_a(call); });
```

`check()` resets the scratch ledger before the device run, reports `scratch_peak` (bytes) and
fails the case if the kernel left any scratch live. It reports a lookup failure, a leak and a
mismatch as a `Report`; `require()` turns a failed report into a thrown `std::runtime_error` with
the value, the slot and the index in its message, so a failing `ctest` says what disagreed.

Every device test checks that the harness rejects a deliberately wrong kernel, too -- usually the
real device adjoint with one operand's contribution dropped. The point is that the comparison has
teeth: a harness that accepted that kernel would accept anything.

## The bench

`tests/gpu/main.cpp` accepts `--bench`; with it, each case also prints one line per shape, in
this form:

```
<op> <shape>            <adjoint> ms adjoint    <forward> ms forward twin    <ratio>x forward  scratch peak <KiB> KiB
```

For example the `add` bias case prints its own `add 16 x 5120 bias` label followed by the times,
the ratio and its scratch peak.

The adjoint's time is the whole call -- upload, launch, download -- because that is what the
contract makes a device adjoint pay. The forward is the op's f64 host twin: until an op has a
device forward, the twin is the honest reference for "the forward op it belongs to", and the line
says so. Quote the line in the kernel's doc, with the card it was measured on; numbers are not
comparable across cards.

## Build and test

```sh
cmake -B build -G Ninja            # finds the card's target (default gfx1201)
cmake --build build                # builds libr97.so into the radiance home
ctest --test-dir build -L gpu      # the card, serially, under the GPU test guard
./build/r97_gpu_tests --bench      # the same checks plus the timing lines
```

- `R97_DEVICE` selects the card; the guard is in `tests/support/gpu_guard.h` and its rules are in
  `docs/TESTING.md`.
- `R97_DEVICE_SO` overrides the plugin path. The build normally passes the file it built as
  `R97_DEVICE_PLUGIN`, and `RADIANCE_HOME/kernels/libr97.so` is the fallback the engine uses.
- `-DR97_WITH_HIP=OFF` (or a machine without `hipcc`) builds the host-only configuration: `r97`,
  `r97train` and the host tests, and no `libr97.so`. The gpu test binary still builds and links;
  without a card its tests fail by name, as the gpu tier is meant to.
- A device kernel's test needs a card, not a model container, when its shapes come from the
  fixture (`docs/FIXTURES.md`) and the design (`docs/DESIGN.md`). Only a test that reads the real
  container carries the `model` label and sets `R97_MODEL_DIR`.
