# Serving on the card: the serving-only build serves the 27B container

2026-11-13. Card 0 of the two-card machine (gfx1201, 31.86 GiB), the container
`/srv/models/qwen3.8-27b-iqr-v4/qwen3.8-27b-iq4r-s-v4.rad` (`qwen35`, `w8a8_fwht128`, 64 layers,
262144 ctx, 15.4 GiB on disk). `docs/DEPLOYMENT.md` section 5 recorded the host's pair of numbers
and said no card had taken the card's. This is that run: a build with only `patches/serving/`
serves an adapter on the card, and its record holds the trainer's card number at `2.2e-10`
relative, where `docs/EVAL.md` asks for `1e-4`.

## The two builds

```sh
cmake -B build -G Ninja
cmake -B build-serving -G Ninja -DR97_PATCH_SET=serving
cmake --build build
cmake --build build-serving
cmake --build build-serving/radiance-build --target radiance      # the server
```

Both against the pinned submodule (`451b136`, v1.17.0-58), HIP from `/opt/rocm`, and
`-DRAD_GPU_TARGETS=gfx1201` (that build's default). The host tiers pass in both:
`ctest --test-dir build -L host` 23/23 (282 s), `ctest --test-dir build-serving -L host` 22/22
(5.7 s). The serving tree carries no `r97train`; its server is
`build-serving/radiance-build/bin/radiance`, which starts the engine with `libr4d`, `libavx` and
`libref` and says what it left alone:

```
I kernel libraries on the search path that --kernels (or RADIANCE_KERNELS) does not name, and so not loaded: r97
```

## The card's serve, through the label the card runs

```sh
R97_DEVICE=0 R97_MODEL_CASES=1 ./build-serving/r97_gpu_tests
```

`23 tests, 0 failed` in about 25 minutes: the tree's gpu cases and its model cases in one start,
the model ones reading the container (`replay_layers_card` is `skipped (the issue observer patch
0101 is not in this build)`). The engine comes up on the card through AQL, and the serving case says:

```
I device: AQL backend, 2 device(s) [0:gfx1201 1:gfx1201]
I   --kernels                          libr4d@0.5.0,libavx@0.1.0,libref@0.1.0      [auto]
I lora: 'trained' in slot 0 -- 64 modules, rank 4, alpha 8, base /srv/models/qwen3.8-27b-iqr-v4/qwen3.8-27b-iq4r-s-v4.rad
serving_container: base loss 1.91553624021, served loss 1.92551851738
```

That adapter is not a fixture: the case writes it with `write_adapter`
(`include/r97/adapter_io.h`), the writer a training run uses, for this container's own linears --
rank 4, alpha 8, 64 `down_proj` modules, `radiance_input_basis: fwht128`, and the container's path
as its base. The case serves the container's prefill with no adapter, then the same prefill
through it, and requires the two losses to differ, so a build that dropped the adapter cannot pass.

## The card's number, held to the trainer's

The held-out corpus is eight UltraChat `test_sft` conversations, `./build/r97train data --model
<container> --data heldout.jsonl --kld-corpus heldout.kld.jsonl` reading `tokens: 3759, scored:
1840` and writing 16 kld documents over those same 1840 positions.
The trainer's number is `loss --device`, which names `libr4d,libavx,libref` and puts the card in
`ROCR_VISIBLE_DEVICES` (`tools/loss.cpp`); the served number is the serving-only build's
`--kld-record`, turned into a loss by `tools/kld_loss.py`. The corpus, the kld corpus and the
adapter are the files the run above left behind (under `/tmp` on this machine); they are shortened
here to `heldout.jsonl`, `heldout.kld.jsonl` and `adapter`.

```sh
M=/srv/models/qwen3.8-27b-iqr-v4/qwen3.8-27b-iq4r-s-v4.rad
A=/tmp/run/r97-served-card-serving_container-HyEteV/adapter
./build/r97train loss --device 0 --model "$M" --data heldout.jsonl --lora "$A"
# loss: 1.37629250328      scored: 1840

./build-serving/radiance-build/bin/radiance --model "$M" \
    --radiance-home build-serving/radiance-build/radiance_home --kernels libr4d,libavx,libref \
    --max-num-seqs 8 --max-num-batched-tokens 8192 --max-model-len 4096 \
    --lora trained="$A" --kld-lora trained \
    --kld-corpus heldout.kld.jsonl --kld-record served
# kld: 16 documents in 3 s, 0 preemption(s)
python3 tools/kld_loss.py served
# loss: 1.37629250298      scored: 1840
```

`3.0e-10` apart, `2.2e-10` relative. The two sides score the same 1840 positions on the same card
through the same kernels; the trainer's `loss` and the server's KL mode agree to the digits they
print.

The limits in the second command are not slack: the engine's default is 256 sequences of worst
case, which for this container asks for `36.70 GiB` of linear and conv state against the card's
`16.17 GiB` of cache, and it stops at startup rather than serve (`short by 20.53 GiB`). The
trainer's own reading of this corpus is 8 sequences at 582 tokens (`loss` sets `max_seqs = 8` and
`max_ctx` to the longest document), which is inside the limits named above, so the two sides batch
the corpus the same way.

## What the card did not take

**A run's own adapter for this container.** `tools/train.cpp` gives a run the host kernels
(`opts.kernels = {"libavx", "libref"}`), and the host tier cannot declare this container -- the
qwen35 `w8a8` plugin hands back `rad_arch_declare returned invalid argument`
(`docs/reports/2026-11-12-device-forward.md`) -- so no run of it can be made here at all, and no
host number exists for it to be held to. What the card took is the pair for the adapter the
trainer's writer makes for the container it serves, which is the deployment `docs/DEPLOYMENT.md`
section 4 describes.

**`ci/serving_build.sh` as a script.** It rebuilds both trees from scratch under `build-ci/`; its
commands were run by hand in `build/` and `build-serving/` instead, which is what the two builds
above are. Its card labels are the run above (serving tree) and the gpu and model labels of the
default tree, which were not run to completion here: the gpu cases alone cost about 20 minutes on
this machine (`adjoint_attention_device`'s `2048/1300/748` case compares against a host reference
that is minutes of `f64`, and `adjoint_linear_device_measures` spends 10 to 15 s on a single row's
host reference), and nothing in this change touches them.

## Machine notes

The card opens `/dev/kfd` only for a process in the `render` group here, so every card command
above ran under `sg render -c`. The engine warns that card 0's power cap is 210 W against the
card's 300 W default; the numbers above are losses, not timings, and do not depend on the clock.
