# The device forward on the real container, 2026-11-11

The forward-only step of the training tier on one card, on `qwen3.8-27b-iq4r-s-v4.rad` (IQR-V4
codes, architecture `qwen35`, 64 layers, hidden 5120, 262144-token context, a 248320-row
vocabulary): `r97train loss --device 0`, a 4096-token prefill step on `libr4d`, the loss of 32
conversations against `radiance --kld-record`, the tiny int8 fixture's device loss against the
host's, and the inventory of the ops the step issues. The inventory itself is
`docs/OPS-qwen35-device.md`; this file is the measurements.

Everything below ran on an R9700 (gfx1201, 32 GB, `0000:43:00.0`) in this build tree, on
2026-11-11, against the container above. The machine has two of the cards; the second one was
busy with another worktree's benchmark, so the runs took the first (see "Two cards, one lock").

## The commands

```
sg render -c 'ROCR_VISIBLE_DEVICES=0 R97_DEVICE=0 build/r97train loss tests/fixtures/chat_32.jsonl \
    --device 0 --max-tokens 4096 --max-seqs 32 --window 256 --load-vram'
sg render -c 'R97_LOCK_DIR=/tmp/r97-lock ROCR_VISIBLE_DEVICES=1 R97_DEVICE=0 \
    R97_MODEL_CASES=1 build/r97_gpu_tests'
sg render -c 'ROCR_VISIBLE_DEVICES=0 R97_DEVICE=0 build/r97train ops --device 0'
sg render -c 'ROCR_VISIBLE_DEVICES=0 build/radiance-build/bin/radiance --model \
    /srv/models/qwen3.8-27b-iqr-v4/qwen3.8-27b-iq4r-s-v4.rad \
    --radiance-home build/radiance-build/radiance_home --kernels libr4d \
    --max-num-batched-tokens 4096 --max-num-seqs 1 --max-model-len 4096 \
    --kld-corpus /tmp/prefill.jsonl --kld-record /tmp/prefill.kld'
```

`sg render` because the shell's groups do not carry `render`, and `/dev/kfd` is group-read only;
`ROCR_VISIBLE_DEVICES` because radiance ranks the devices HIP shows it, so a test's `R97_DEVICE`
card has to be the process's device 0 (`docs/TESTING.md`, "Card selection"), and the test's
`Card` refuses when `hip_bdf(0)` and `hip_bdf(R97_DEVICE)` name different cards rather than
measuring the wrong one. The test run above took the second card (`ROCR_VISIBLE_DEVICES=1`,
`R97_DEVICE=0`), with a lock directory of its own because the guard's lock keys on the ordinal --
see "Two cards, one lock".

## The 32 conversations

`tests/fixtures/chat_32.jsonl` holds 32 one-exchange conversations. `ChatData` renders each of
them with the container's own chat template and tokenizer: 32 kept, 0 skipped, 2344 tokens, 436 of
them scored. One step takes all 32 sequences (`--max-seqs 32`, `--max-tokens 4096`, a 256-token
window), so the step is a single prefill of 2344 tokens over 32 sequences:

```
vram after loading 22717 MiB used of 32624 MiB
loss 2.22157668
sum 968.607433
scored 436 tokens 2344 sequences 32 steps 1 windows 10
forward 42.975 s  54.5 tokens/s
data conversations 32 kept 32 shortened 0 skipped 0
vram peak 22716 MiB used of 32624 MiB (20 ms samples)
```

The same file through the test (`device_loss_matches_kld`), the device loss against the NLL
`radiance --kld-record` wrote for the same token sequences on the same container:

```
32 conversations: device loss 2.22157668, radiance kld 2.2215767, relative 9.63e-09,
                  436 scored, 37.66 s, 62.2 tokens/s, 22750 MiB used after the step of 32624 MiB
```

9.6e-9 relative against the 1e-3 the issue asks for. The test also asserts that the ids the engine
tokens and the ids radiance recorded are equal, and that the recorded row count is the scored
count, so the two sides score the same positions and not merely a similar number of them.

The 42.975 s is the engine's prefill of the 32 sequences and the host-side loss together; the two
were not timed apart in this run. For scale, the container alone on this card: `radiance` with
`--kld-corpus` of one 1-token document takes 20.4 s wall (nearly all of it loading the weights),
and with one 2401-token document 27.6 s, so a 2401-token prefill is about 7 s of wall time. The
loss is the f64 log-sum-exp of `src/loss.cpp` over each scored row's whole vocabulary: two passes
over 436 x 248320 f32, 433 MB each. A device kernel for the objective would have to beat that
loop's share, which this run does not measure.

## The tiny int8 fixture

The host tier cannot start this container (the architecture declares the host backend invalid for
w8a8, `qwen35_w8a8: rad_arch_declare returned invalid argument`, and 38 of the 1140 declared ops
have no host kernel at all), so the tier comparison has to be a tiny container both tiers start.

It cannot be the repository's tiny fixture, `tests/support/fixture.h`'s hidden-128 checkpoint: no
libr4d row covers its shapes. A step of it on the card resolves
`gdn_chunk_scan`, `gdn_conv_prep`, `gdn_conv_update`, `gdn_conv_recurrent_update_had`,
`gdn_gated_rmsnorm` (channels=16), `gdn_kkt_solve`, `gdn_recurrent_update` (head_k=16, head_v=16),
`attn_paged_*` (4 query heads of 32 over 2 of 32) and `had_quant_act_i8g` (n=64) to no kernel at
all -- libr4d's GDN rows are declared for `head_k = 128`, its quantiser for a multiple of 128
columns, its paged attention for `head_dim = 256` with 6 query heads per key-value head. (That
list is what `r97train loss <chat> --model <the w8a8 fixture> --device 0` prints, and what the
case reported before this change: "no kernel for the device band".) The engine refuses the whole
step, so no tolerance can be measured on that container.

So `tests/support/fixture_device.h` writes the same `tiny_model.h` checkpoint at the shapes those
rows are declared for -- 2 layers (1 GDN, 1 attention), hidden 1536 as 6 query heads of 256 over 1
key-value head, 1 GDN key head of 128 over 4 value heads of 128, feedforward 1024, vocabulary 320,
ctx 4096 -- and converts it with the same `w8a8_fwht128` recipe. Every band of its step resolves.
The case puts the same three-conversation chat file through it twice, through `r97train loss`:

```
device_forward_tiny_fixture  a chat file through the device-shaped tiny w8a8 container, once on
                             libavx,libref and once on libr4d; the means agree to 1e-3
```

```
tiny fixture: 135 tokens, 26 scored; host loss 8.3043117, device loss 8.3099589, relative 0.00068
```

6.8e-4 relative. Both sides read the same int8 codes, so what is left is the kernels' arithmetic:
the card's f32 tiles in their own order against the host reference's, and the bf16 round trip of
the step's hidden rows. The tolerance is 1e-3 and the margin is small, so it is stated as the
measured figure rather than treated as slack.

The host side of that comparison runs in a **child process** with the card out of its environment
(`ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=`), not in the test's own. radiance plans the device
bands from the devices a process can see, and `libavx`/`libref` declare no device kernel, so an
engine that sees a card refuses to start on the host tier ("no kernel for the device band"). A
card machine's host tier is therefore a process whose card is hidden, and `r97train loss` is what
runs there; the test checks that both runs kept the same conversations, scored the same 26 of 135
tokens, and ran them as one step.

## A 4096-token sequence

`device_forward_4096`: one sequence of exactly 4096 tokens (4095 scored), one prefill step with
`max_tok = max_ctx = 4096` and `max_seqs = 1`:

```
4096 tokens: loss 0.780208 over 4095 scored, 112.57 s (36.4 tokens/s),
             vram after loading 18309 MiB, peak 18273 MiB of 32624 MiB
```

18.3 GiB after loading and at the peak of the step, both of 32624 MiB, at 36.4 tokens per second
for the step the loss and the prefill share. The after-loading figure comes from a one-token step
with the same `ForwardConfig` before the real one (`--load-vram`'s measurement); the peak is the
20 ms sampler's over the real step, which starts before that engine's own load, so the two agree
to the few MiB that the two engines' scratch differs by.

The case did not run before the change that came with it. A prefill is cut into chunks of at most
the linear-attention checkpoint interval -- 2048 by default -- and this driver's step is the
batch's own rows and block brackets, which a second chunk breaks ("block 0's entry holds no
residual of the step's 4096 rows: the step did not run the batch in one piece"). The driver now
asks radiance for an interval of at least `max_ctx`/`max_tok`, so a sequence at `max_ctx` runs in
one piece and the batch stays one step (`src/forward.cpp`).

The headroom is the other half: 18.3 GiB of the 32.6 GiB card for one 4096-token sequence, and
22.75 GiB for the 32-conversation step with its 32-sequence pool.

The three cases are what `R97_MODEL_CASES=1 build/r97_gpu_tests` ran, together with the rest of
the gpu tier: 20 tests, 0 failed.

## Two cards, one lock

The card guard's lock is `/tmp/r97-card<N>.lock` keyed on `R97_DEVICE`, not on the card's PCI
address, and this machine has two cards. A second worktree held card 0's lock for a benchmark, so
these runs used the guard's `R97_LOCK_DIR` to lock a private directory while `R97_DEVICE=0` named
the only device the process could see (card 1 through `ROCR_VISIBLE_DEVICES`). The guard's VRAM
check still consulted the physical card the device resolves to. A run that does not do this waits
on the other worktree's lock; that is the guard working as designed, not a fault.

## What the step issues

`r97train ops --device 0` records one 8-token prefill step of the container and prints the op
forms it issued: 758 issues over 22 forms, 4 of them fused device forms (`rmsnorm_had_quant_i8g`,
`gemm_nt_q_gated`, `gate_had_quant_i8g`, `qk_norm_rope_gate_kv_store`), each with a device adjoint
and no host one -- the host tier's chain is the unfused ops. The table, and for each fused form
the chain of host ops it replaces, is `docs/OPS-qwen35-device.md`.

## What it leaves alone

- The host tier still refuses this container's w8a8 architecture; nothing here changes that. The
  device-shaped tiny fixture is what compares the tiers instead, and it carries no reference of
  its own: the comparison it feeds is device against host, not against a stored loss.
- The repository's own tiny fixture (`tests/support/fixture.h`) still cannot run on the card, and
  the host tests that use it are unchanged. `tests/support/fixture_device.h` adds a second shape
  beside it rather than changing the one the oracle was made from.
- The per-card lock keys on the HIP ordinal, so on a two-card machine a run that remaps
  visibility to the second card locks the first card's file. Not fixed here.
- `build/r97_host_tests` is red in this tree (29 of 129 fail) both before and after this change;
  see the summary of the issue's pull request.
