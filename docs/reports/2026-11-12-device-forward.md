# Device forward on the real container — 2026-11-12

The training forward (`r97::ForwardDriver`) on one card, over the container on disk:
`/srv/models/rad/qwen3.8-27b-iq5r-s-v4.rad` (`qwen35`, `quant=w8a8_fwht128`, 64 layers,
262144 ctx). Card: AMD Radeon AI PRO R9700 (gfx1201, 31.86 GiB VRAM), `ROCR_VISIBLE_DEVICES=0`.

The device tier is `--kernels libr4d,libavx,libref` (libr4d's fused HIP kernels, with libavx/libref
as the per-op fallback). `r97train loss --device N` / `r97train ops --device N` select it by
setting `ROCR_VISIBLE_DEVICES=N` before any device call and defaulting the kernel hierarchy to
that list.

The device-tier test runs are under the shared GPU lock (`flock /tmp/gpu.lock`), with prod
`radiance` already stopped, so their numbers are not contended. Of the `ops` probes below, the
host-tier ones die in declare (1.2 s, nothing loaded on a card); the `--kernels libavx,libr4d`
probe does load the container on one card, since `r97train ops` runs a prefill step
(`tools/ops.cpp`, `inventory_main`).

Every number below comes from the revision this report ships with: the four tests and `ctest -L
host` were re-run on that build (the test binary is newer than every source file it links).

## device_forward_device_tier — the device tier is the one running

The forward loss over `tests/fixtures/chat_small.jsonl` (three short conversations), driven on
the device tier; the engine's own kernel-selection report is captured and checked to name
`libr4d` (a silent host fallback would not).

| | |
| --- | --- |
| kernel selection | `libr4d` (device), `libavx` (host) 1075 + `libr4d` 2484 op bands |
| scored tokens | 32 |
| loss (sum over scored) | 87.881445 |
| mean NLL | 2.746295 |
| `device_selected_libr4d` | true |

## device_forward_model_loss — the device forward against radiance's own KL record

32 single-turn conversations ("What is the square of N?" / "The square of N is N²."), scored
through the forward driver on the device tier and recorded through radiance's `kld` mode on the
same device tier. The token streams must match, and the two mean losses must agree.

| | |
| --- | --- |
| rows (scored tokens) | 541 |
| device forward, sum | 696.238629 |
| device forward, mean | 1.286948 |
| radiance KL record, mean | 1.286948 |
| relative difference | < 1e-12 (identical to double precision) |
| token streams identical | true |

The device forward and radiance's served forward agree to full double precision on the same
541 tokens: they are the same libr4d kernels on the same graph.

## device_forward_seq4096 / device_forward_seq2048 — one prefill piece of each size

The ticket asks for **the loss of 4096-token sequences on one card**. `ForwardDriver::run` requires
the whole batch in one prefill piece (`src/forward.cpp:177-181`), and radiance caps that piece at its
**linear-state checkpoint interval** (`core/sched/geometry.cpp:209`), whose default is 2048
(`core/rad_core.h:942`). The driver had never set it, so a 4096-token step was cut in two and the
driver refused it:

```
I chunk geometry ... checkpoint interval 2048, max_tok 4096 -> prefill chunks of at most 2048
  (the checkpoint interval, not the budget)
device_forward_seq4096: threw: r97: forward: block 0's entry holds no residual of the step's 4096
  rows: the step did not run the batch in one piece
```

`ForwardDriver::run_engine` now sets `cfg.checkpoint_interval = 0`, beside the other serving-only
switches it turns off for training (`cfg.n_spec`, `cfg.mm_max_patches`): training never resumes from
a checkpoint, so the interval buys it nothing, and the prefill piece becomes the step's `max_tok`.
The step is one piece by the driver's own contract (`forward.h:3-8`), so the fixed 0 enforces it
rather than exposing it as an option a caller could set back to 2048.
**That is a generalisation, not a swap**: 2048 runs exactly as it did, and 4096 now fits. Radiance's
own default is untouched, so the KL record above still runs at interval 2048. The engine now reports
`checkpoint interval 0` and the chunk at the budget for both:

```
I chunk geometry ... checkpoint interval 0, max_tok 4096 -> prefill chunks of at most 4096
I chunk geometry ... checkpoint interval 0, max_tok 2048 -> prefill chunks of at most 2048
```

Both sizes are registered as tests. Each records **the loss over every scored token** — the number
the ticket is named for — the throughput, and the card's VRAM at the step's peak (a 500 µs
background poller on `mem_info_vram_used`):

| | 4096 tokens | 2048 tokens |
| --- | ---: | ---: |
| scored tokens | 4095 | 2047 |
| loss (sum over scored) | 34682.089228 | 18156.120479 |
| mean NLL | 8.469375 | 8.869624 |
| engine's own prefill time | `1 prompts, 4096 positions in 2 s` | `1 prompts, 2048 positions in 1 s` |
| step wall time (load + forward) | ≈105.8 s | ≈71.4 s |
| `tok_s` as the test reports it | 38.7 | 28.7 |
| VRAM after load | 20706 MiB | 18446 MiB |
| VRAM at the step's peak | 20706 MiB | 18446 MiB |
| card | `card0` (device 0) | `card0` (device 0) |

The loss is high because the prompt is synthetic (`1 + (i*7) % 299`: the sequence is not text), not
because the forward is wrong — `device_forward_model_loss` is the check on the numbers.

`tok_s` is the test's `n / step_seconds`, and `step_seconds` covers the whole engine run the driver
performs — `start()` (the model's load) **and** `serve()` (the forward) — so it is n tokens over
load-plus-prefill, not a prefill rate. That is why the 4096 figure looks *better* per token (38.7 vs
28.7): the ~70-104 s load is amortised over twice the tokens. The engine's own accounting separates
the two: the prefill of the 4096-token piece takes 2 s and of the 2048-token piece 1 s — the same
order, since a prefill's cost is roughly linear in tokens (the log rounds to whole seconds, so
attention's slight super-linearity at 4096 is not visible). Neither is a 2048-token ceiling: the
chunk-geometry lines above show one chunk of 4096 and one chunk of 2048.

## Device-versus-host is not runnable on this host

The issue's device-versus-host comparison needs one model to declare and run on **both** tiers. On
this host no model does.

- **The real 27B container** runs on the **device** tier (758 issues, 22 op forms) and not on the
  **host** tier. `r97train ops <container> --kernels libavx,libref` — and `--kernels libavx` or
  `--kernels libref` alone — exit 1:

  ```
  E qwen35_w8a8: rad_arch_declare returned invalid argument
  ```

  with **no** builder error lines after it: the refusal is a bare `RAD_E_INVAL` (a
  `RAD_ARCH_TRY`-propagated code), not a recorded declare error, so the engine prints no reason.

  Reproduced, and stated as such:

  1. `--kernels libavx,libref` fails to declare this container; so do `--kernels libavx` alone and
     `--kernels libref` alone. `--kernels libavx,libr4d` declares the full graph (758 issues, 22 op
     forms, 35.6 s).
  2. The graph's op forms include eight that **no host library implements**: neither `libavx` nor
     `libref` references the name anywhere in its tree, while `libr4d` does.

     | op form | libr4d | libavx | libref |
     | --- | ---: | ---: | ---: |
     | `rmsnorm_had_quant_i8g` | 40 | 0 | 0 |
     | `gate_had_quant_i8g` | 10 | 0 | 0 |
     | `gated_had_quant_i8g` | 27 | 0 | 0 |
     | `gate_quant_fp8` | 79 | 0 | 0 |
     | `gdn_conv_recurrent_update_had` | 10 | 0 | 0 |
     | `qk_norm_rope_gate_kv_store` | 55 | 0 | 0 |
     | `logits_gemm_vocab` | 14 | 0 | 0 |
     | `ar_rmsnorm_had_quant_i8g` | 15 | 0 | 0 |

     Counts are occurrences of the op name in each library's sources (whole tree); a zero means the
     library carries no implementation of that op. `ar_rmsnorm_had_quant_i8g` is reached only at
     `--tp 2`; the rest are in the single-card graph. `rmsnorm_quant_fp8` — the fp8 sibling of
     `rmsnorm_had_quant_i8g` — is the contrast: it has host kernels in both `libavx` (9) and
     `libref` (9).

  So a host-only run of this container has no complete kernel set. **Which** op the declare refuses
  on is not pinned here: the refusal is silent, and radiance's own design note
  (`core/engine_bringup.cpp:159`) says what that means — *"Whole-model host execution needs every op
  covered"*, i.e. a host kernel set is meant to run beside the device one, with the ops it lacks
  pinned to the card. Naming the single triggering op would need an instrumented build.

- **The tiny int8 w8a8 fixture** runs on the host tier (`tests/host/loss_cli.cpp`) but the device
  tier refuses its geometry: libr4d's GDN kernels require `head_k == 128` (the fixture has 16),
  `had_quant_act_i8g` requires `n % 128 == 0` (the fixture has 64), and `attn_paged` requires
  `head_dim ∈ {64,128,256,512}` (the fixture has 32). The engine reports
  `19 device band(s) resolved to no kernel ... not starting`.

So the device forward's correctness is checked against **radiance's own served forward** (the KL
record above) — the same libr4d kernels and graph production serves — rather than against a host
run of the same container, which this host cannot perform.

## Device op inventory

`r97train ops <container> --device 0`: one prefill step, 758 issues, 22 op forms. The full table
and the per-fused-op host chain are in `docs/OPS-qwen35-device.md`.
