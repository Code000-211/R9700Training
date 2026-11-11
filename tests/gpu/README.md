GPU tests go here. They need the card and run serially (ctest label `gpu`).

The sources build into one binary whose `main` is `main.cpp`; each source registers its checks
with `R97_TEST` (the runner is `tests/support/test.h`). Unlike the host and model tiers, this
binary runs those checks from `main` rather than at exit: a device test `dlopen`s `libr97.so`,
whose rows register into r97's registry as it loads, and at exit that registry is destroyed before
a handler armed during static initialisation would run. So a gpu source just registers its check;
`main.cpp` runs them while every static is still alive. A device kernel's test uses
`tests/support/device_check.h`, which compares it with the host adjoint within a bf16 tolerance;
`adjoint_add.cpp` is the worked example. `--bench` adds a timed pass and prints the line a
kernel's doc quotes. `docs/kernels/README.md` has the rules.
