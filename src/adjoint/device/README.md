# Device adjoints (HIP)

The `.hip` sources here build into `libr97.so`, the radiance kernel plugin loaded next to
`libr4d`; `scratch.cpp` is host-side and builds into `libr97.a`, which both the plugin (through
the process that loads it) and the tests use. The CMake glob collects every `.hip` file, and each
kernel registers its adjoint with `R97_REGISTER_ADJOINT(op, device, fn)`; `add.hip` is the worked
example, `plugin.hip` the plugin's radiance-side exports and fallbacks, and `scratch.h` the device
allocator a kernel takes its buffers from. `docs/kernels/README.md` explains how to add a kernel.
