# Adjoints: the contract and how to add an op

The adapter trains through a walk over the op tape: the forward appends one
`OpRecord` per op (`include/r97/op.h`), the walk replays them backwards. The
contract every op implements lives in `include/r97/adjoint.h`. A **twin** is the
forward of one op in f64; an **adjoint** takes the op's record, the forward
values of its operands, and the cotangents of the operands it wrote, and adds to
the cotangents of the operands it read. An adjoint of an op with trainable
leaves also adds a gradient to a `ParamSet`. Host adjoints compute in f64.

Registration is a static object per op, in the op's own file, keyed by op name:
`R97_REGISTER_TWIN(op, fn)` and `R97_REGISTER_ADJOINT(op, side, fn)`, with side
`host` or `device`. A row can take a predicate on the op's parameters (for
example the `dtype` parameter): `R97_REGISTER_TWIN_IF`, `R97_REGISTER_ADJOINT_IF`,
`R97_REGISTER_ADJOINT_F` (which marks operand positions frozen -- a weight, a
scale: the walk does not ask for their cotangent), `R97_REGISTER_ADJOINT_IF_F`.
A new op needs no edit to any shared file.

`r97train ops --registered` lists every op with a twin, a host adjoint or a
device adjoint. The checks every adjoint test runs live in
`tests/support/adjoint_check.h`: `check_kernel` calls radiance's host kernel for
the op through the kernel table, as `rad-kbench` does, and compares with the
twin within a bf16 tolerance; `check_adjoint` takes central finite differences
of the twin against the adjoint (random inputs, random output cotangent, the
relative error of the directional derivative). The kernel check needs the
reference plugin, which the build puts beside radiance in
`build/radiance-build/radiance_home/kernels/`; the checker finds it there (or
under `$RADIANCE_HOME`, or wherever `R97_LIBREF_SO` points). A missing plugin or
kernel says so instead of passing quietly.

## Adding an op, in four steps

`add` (`src/adjoint/host/add.cpp`) is the worked example; copy its shape.

1. **Write the twin**: a function
   `void op_twin(const r97::OpRecord& rec, const std::vector<r97::AdValue>& operands,
   std::vector<r97::AdValue>* out)`.
   It reads the op's parameters from `rec.params` and the forward values from
   `operands` (the read operands, in schema order), and pushes what the op would
   have written had it computed in f64 -- one `AdValue` per written operand, in
   schema order.
2. **Write the host adjoint**: a function
   `void op_adjoint(const r97::AdjointCall& call)`.
   It reads the cotangent of each written operand from `call.cot` (the writes'
   order, one entry per operand the op wrote), adds its share to the cotangent of
   each read operand in `call.dc->at(i)` (add, never overwrite; frozen slots stay
   untouched), and, if the op has trainable leaves, adds a `Param` to
   `call.grads` named `<prefix>.<leaf>` with the prefix `call.prefix` supplies.
   Host adjoints compute in f64. `call.fwd` and `call.dc` are the record's schema
   positions throughout -- every operand where the record puts it -- so an op
   whose schema writes before it reads (an INOUT operand) keeps that order.
3. **Register both**: at the bottom of the op's file,
   `R97_REGISTER_TWIN("op", op_twin)` and
   `R97_REGISTER_ADJOINT("op", host, op_adjoint)` -- plus `_IF` / `_F` variants
   when a predicate or a frozen operand applies. Nothing else edits: sources and
   tests are collected by directory glob, and the binaries that ask the registry
   link `r97` whole, so a row in a new file is in the registry on the next
   build. That link is what carries the row: `r97` is an archive, and an archive
   drops the members nothing references, so a binary that links it plainly sees
   only the ops it already calls. `r97train` and the host and gpu test tiers link
   it whole (`R97_LINK_WHOLE` in `CMakeLists.txt`); a build that drops that says
   `no twin registered for op 'add'` in `adjoint_add` rather than losing the op
   quietly.
4. **Test it**: a file under `tests/host/` -- say what the op asks the checker
   (`check_kernel` and `check_adjoint` at the shapes that matter) and include one
   case it must reject if the op has a failure mode worth pinning. The host tests
   are one binary: write `R97_TEST(name) { ... }` (`tests/support/test.h`),
   so the test joins the run by being in the directory.
   Run them with `ctest --test-dir build -L host`. A doc page under
   `docs/adjoints/<op>.md` says in one paragraph what the op's walk does.

## Where the registry ends

The op registry is keyed by op and does not know the loss, the leaves, or the
walk. An `Objective` (`include/r97/objective.h`) gives the seeds the walk starts
from -- the cotangents of the output the loss scores; a `Delta`
(`include/r97/delta.h`) names the trainable leaves an adjoint adds gradients
for, and its `grad_prefix` is the prefix under which those gradients are named.
Neither is registered here.
