# The walk's time per op, and the cotangent it does not allocate

`include/r97/backward.h`, `src/backward.cpp`, `src/step_device.cpp`, `include/r97/step_device.h`.

## The dc slot the row may not write holds no payload

The walk used to zero a full f64 copy of *every* operand of the record it was adjointing,
before it consulted which of them the row might write. On the qwen35 tape that is the fold's
`[34816, 5120]` weight -- 178M doubles, 1.4 GB -- and the head's four class weights -- 248320 x
5120 = 1.27G doubles, about 10 GB -- zeroed on every call, for slots no row writes (they are
frozen) or that are discarded (the written operands). At 16 s a block that fills the 17-minute
walk; a stream sync in the tens of microseconds cannot.

The walk now resolves the serving row before it sizes the slots, and zeroes a dc slot only when
the row may write it: the operand is not in the serving row's frozen list, and it is a read or
an in-place write the record also reads. A slot the row may not write keeps its dtype and shape
and holds no payload. Two details ride with the change:

- the frozen list is the one of the row that *served*, not of the requested side: with
  `allow_host_fallback`, a record a device row does not serve is adjointed by the host row, and
  its frozen operands are the host row's;
- `peak_cotangent_bytes` counts the payload the walk actually held, so it stays the walk's
  working set; an unallocated slot contributes zero to the sum, as the formula always did.

The row contract that follows: a row never writes a frozen slot or a plain write's slot. Every
row registered today honours it -- they write the non-frozen read slots only, and the ones that
guard on a slot's size either decline the call or no-op on the empty slot -- so no row needed
touching for the change. `tests/host/backward_alloc.cpp` pins it: a record whose frozen operand
is a giga-byte of weights walks with a peak cotangent working set under 16 MiB.

## The report's per-op timing

`BackwardReport::ops` splits the walk's time across the rows it called: one line per op name
(the name the row was resolved under), the row calls made for it, and the wall time around
them, sorted by op name. The line a block gives is the report of the `backward_block` call that
walked it; a row the walk never reached (off the path, behind a stop) names nothing.

The device join sums the head walk and every block walk into `DeviceStepReport::row_seconds`,
and times the replay's pull-back -- the copy-and-sync of each op's operands back from the card,
one `rad_stream_sync` per op, outside the walk -- into `DeviceStepReport::pull_seconds`. Both
lists are sorted by op name. The 512-token probe's next run reads the walk's seconds off those
two lists: the rows' share against the pull-back's.

## What this did not build

The per-block staging pass -- one upload, one sync, one download per block, behind a new row
entry point -- is not in. The hypothesis it replaced, that the walk's roughly 1280 stream round
trips a step are the cost, was never measured, and the zero-fill above is a concrete cost the
staging pass would have left in place. The per-row path is the only row path until the probe's
split says the bus is the term.
