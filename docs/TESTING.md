# Testing

Three test labels, and the variables that run them. ctest runs each tier's binary; the sources are
globs of their directories (a new test file needs no edit anywhere), and the tests inside one
binary register themselves with `R97_TEST(name)` (`tests/support/r97_test.h`) and run in
registration order.

## The three labels

| Label | Binary | Needs | When it cannot run |
|---|---|---|---|
| `host` | `r97_host_tests` | no card | never — runs anywhere |
| `gpu` | `r97_gpu_tests` | one card, under the guard | never — a missing card is a failure |
| `model` | `r97_model_tests` | the real container | skipped, exit 77, naming where it looked |

```sh
cmake -B build -G Ninja && cmake --build build
ctest --test-dir build -L host    # no card needed
ctest --test-dir build -L gpu     # the card, serially
ctest --test-dir build -L model   # the container, exit 77 (skip) when absent
```

## The GPU test guard

Several people and several agents run GPU tests on one machine. Every `gpu` test takes the guard
(`tests/support/gpu_guard.h`) before it touches a card:

- **Card selection.** `R97_DEVICE` names the card, default `0`. A test reads it with
  `r97test::parse_device`, which fails a bad value at once instead of silently reading card 0.
- **A lock per card.** The test's `r97test::DeviceGuard` holds an exclusive `flock` on
  `<lock_dir>/r97-card<N>.lock` for its lifetime. Two runs started together queue: the second
  blocks in the constructor until the first's guard is destroyed. The tier also runs serially
  under ctest (`RUN_SERIAL`); the lock is for runs ctest does not start, such as two agents
  building and testing at once.
- **The free-VRAM check.** A test declares what it needs in `GuardOptions::need_vram`, in bytes.
  Before anything happens, the guard reads
  `/sys/class/drm/card*/device/mem_info_vram_{total,used}` for the selected card and, when free
  VRAM (total − used) is less than needed, fails at once naming the free and the needed amounts
  and the processes that hold the card (from `amd-smi process`, whose CSV nests one
  per-process record per line under `process_info`, so its pid, name and `vram_mem` are read out
  of that record), instead of dying with an allocation error halfway:

  ```
  card1: free VRAM 70 MiB, the test needs 1024 MiB. Processes that hold the card:
    gpu 0 pid 1543186 radiance 32500 MiB
  ```

  A card nothing holds reads `0,No running processes detected`; the guard then says so instead of
  listing holders.

The guard's logic itself is tested in the `host` tier (`tests/host/gpu_guard.cpp`) against a fake
sysfs tree, with a stub `amd-smi`, so it runs with no card at all.

### Using the guard in a gpu test

```cpp
#include "support/gpu_guard.h"

r97test::GuardOptions o;
o.device    = r97test::parse_device(getenv("R97_DEVICE") ? getenv("R97_DEVICE") : "0");
o.need_vram = int64_t(1) << 30;                // what this test needs, in bytes
o.bdf       = r97test::hip_bdf(o.device);      // HIP device -> PCI bus id -> sysfs card
r97test::DeviceGuard guard(o);                 // locks (queues), then checks VRAM; throws with detail

// ... open the card through radiance's device layer (rad_dev_set, rad_dev_alloc, ...);
// the guard is released when it leaves scope.
```

`o.bdf` is how the guard maps the HIP device to its `/sys/class/drm` card: two cards with equal
VRAM cannot be told apart by size. A single-card machine is unambiguous without it. The guard
finds HIP's `hipDeviceGetPCIBusId` at run time rather than linking to it, so a host-only
configuration (`-DR97_WITH_HIP=OFF`) still builds and links the `gpu` binary — it has no card to
run it on, but the `host` tests build and run there.

## The model container

A `model` test resolves the container through `tests/support/model_dir.h`: `R97_MODEL_DIR` when
set, else the build's `R97_MODEL_RAD`, else `/srv/models/qwen3.8-27b-iqr-v4`. When the container is
absent the test throws `r97test::Skip{why}` with a message naming what it looked for; the registry
prints it, runs the tests that follow, and the binary exits 77 when nothing failed and something
skipped. Present and passing, the test exits 0.

## Variables

| Variable | Meaning |
|---|---|
| `R97_DEVICE` | the card ordinal the tests use; default `0` |
| `R97_LOCK_DIR` | where the guard keeps the per-card locks; default `/tmp` (set it when `/tmp` is not writable) |
| `R97_MODEL_DIR` | where the model tests find the container: a `.rad` file, or a directory that holds one (the first `*.rad` inside; the `qwen3.8-27b-iq4r-s-v4.rad` name wins if present) |
| `R97_MODEL_RAD` | the same thing, resolved by the build at configure time; `R97_MODEL_DIR`, when set, wins over it |

The default directory is `/srv/models/qwen3.8-27b-iqr-v4`. Fetch the container (15.4 GiB):

```sh
hf download StillDeadcode/qwen3.8-27b-iqr-v4 qwen3.8-27b-iq4r-s-v4.rad
```

## Notes

- A `gpu` test that cannot fit the card stops before opening it and says so; it does not retry.
- `tests/gpu/gpu_guard_smoke.cpp` is the tier's smoke test: it opens the card through radiance's
  device layer, allocates 1 GiB and frees it. Two copies started together run one after the other,
  each waiting for the other's lock.
- The guard holds a lock, never ownership: it never stops a process it did not start. If a stale
  holder is wedged, kill only processes you started.
- The guard's `host` tests create their files under `/tmp/…` and clean up after themselves.
