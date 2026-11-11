GPU tests go here. They need the card and run serially (ctest label `gpu`).

The sources build into one binary whose `main` is `main.cpp`; each source registers its checks
with `R97_TEST` (the runner is `tests/support/test.h`). A device kernel's test uses
`tests/support/device_check.h`, which compares it with the host adjoint within a bf16 tolerance;
`adjoint_add.cpp` is the worked example. `--bench` adds a timed pass and prints the line a
kernel's doc quotes. `docs/kernels/README.md` has the rules.
