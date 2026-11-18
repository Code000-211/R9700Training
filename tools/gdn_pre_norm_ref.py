#!/usr/bin/env python3
"""gdn-pre-norm reference: the native-model answer to issue #149.

Runs the same 512 tokens the engine's walk stepped (device_step_tape's id
scheme, `1 + (i*7) % 299`) through the checkpoint in the framework it was
trained in -- Hugging Face `transformers`, the native bf16 weights -- and
hooks the Gated DeltaNet layer's gated RMS norm, capturing everything the
engine's `gdn_gated_rmsnorm` record names:

  - `gdn_o`    schema nX, the chunk scan's output before the norm;
  - `gdn_in`   schema nZ, the gate `z` (the engine's buffer name for the
               gate operand, src/adjoint/host/gdn_prep.cpp);
  - `gdn_norm` schema nO, the norm's output;
  - `normed_in` the GDN module's own input, i.e. input_layernorm's output
               (modeling_qwen3_5.py applies the layernorm before calling
               the mixer), an extra sanity quantity the tape does not name
               at this op.

`row_ss` is a row's mean square, the same statistic the tape's FDREC line
prints; one row is one (token, value head) of 128 channels, as in the
engine's `gdn_gated_rmsnorm`.  The check that the hooks sit on the
model's own path recomputes the norm from the captured x and z in the
model's own order (f32 variance, the weight multiply in bf16, the silu
gate in f32) and compares it to the norm's own output: they must agree,
or the captured values are from a path the model did not take.

The comparison is against the engine's base container: the walk that
measured #149 ran a seeded LoRA adapter over `in_proj_qkv`/`in_proj_z`
(docs/ADAPTER.md), so the engine side was re-measured with
R97_DSTEP_NO_SEED=1 before the two were set beside each other.

Usage:
  python3 gdn_pre_norm_ref.py <model-dir> [--layers 61,60,62] [--device auto]
  --device auto  the 27B: device_map=auto, 28 GiB pinned per card, no cpu
                 key so an under-sized map fails loudly;
  --device cpu   the dry run (0.8B): plain CPU, no device map.
"""

import argparse
import json
import os
import sys

if os.environ.get("GDN_REF_PURE_TORCH") == "1":
    # Block fla so transformers' kernel dispatch falls back to the pure
    # torch delta-rule body: the fla Triton kernel cannot run on CPU, which
    # is where the dry run goes.  On the GPU run the default (fla) stands.
    sys.modules["fla"] = None

import torch
from transformers import AutoConfig
from transformers import Qwen3_5ForConditionalGeneration

K_EPS = 1e-6  # the engine's kRmsEps, cross-checked against the config


def tokens(n):
    return [1 + (i * 7) % 299 for i in range(n)]  # device_step.cpp's scheme


def row_ss(x, rows, cols):
    """Mean square of each row of a rows x cols matrix, f64."""
    x = x.to(torch.float64).reshape(rows, cols)
    return x.pow(2).mean(-1)


def stats(name, ss):
    """ss: a 1-D tensor of row means-squares.  Prints min, p01, p25,
    median, p75, p99 and max -- the row the issue's table is made of."""
    ss = ss.flatten()
    q = ss.quantile(torch.tensor([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0],
                                 dtype=torch.float64, device=ss.device))
    print(f"{name}: n={ss.numel()}  row_ss "
          f"min={q[0]:.4e} p01={q[1]:.4e} p25={q[2]:.4e} med={q[3]:.4e} "
          f"p75={q[4]:.4e} p99={q[5]:.4e} max={q[6]:.4e}")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", default="61,60,62")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = ap.parse_args()
    layers = [int(v) for v in args.layers.split(",")]
    cfg = AutoConfig.from_pretrained(args.model)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    print(f"config: hidden={tc.hidden_size} layers={tc.num_hidden_layers} "
          f"rms_norm_eps={tc.rms_norm_eps} "
          f"(engine kRmsEps={K_EPS}, "
          f"{'match' if tc.rms_norm_eps == K_EPS else 'MISMATCH'})")
    for l in layers:
        print(f"layer {l}: {tc.layer_types[l]}")

    if args.device == "cpu":
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            attn_implementation="eager")
        model.to("cpu")
    else:
        # Integer keys (accelerate rejects strings); no cpu key: if the
        # weights do not fit the two cards the load fails loudly instead of
        # offloading to a CPU that cannot hold them.
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory={0: 28 * 2**30, 1: 28 * 2**30},
            attn_implementation="eager")
    model.config.text_config.use_cache = False
    model.eval()
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    captured = {}

    def hook_layer(idx):
        gdn = None
        for m in model.modules():
            if isinstance(m, M.Qwen3_5GatedDeltaNet) and m.layer_idx == idx:
                gdn = m
                break
        if gdn is None:
            sys.exit(f"no GatedDeltaNet module at layer {idx}")
        state = {}
        captured[idx] = state

        # The decoder layer calls linear_attn entirely by keyword
        # (modeling_qwen3_5.py:876-881), so the positional tuple of a plain
        # pre-hook is empty and kwargs must be asked for.
        def layer_pre(mod, inp, kw, st=state):
            st["normed_in"] = inp[0] if inp else kw["hidden_states"]
        gdn.register_forward_pre_hook(layer_pre, with_kwargs=True)

        def norm_pre(mod, inp, st=state):
            st["gdn_o"] = inp[0]
            st["z"] = inp[1]
            st["w"] = mod.weight.detach()
            st["eps"] = mod.variance_epsilon
        gdn.norm.register_forward_pre_hook(norm_pre)
        gdn.norm.register_forward_hook(
            lambda mod, inp, out, st=state: st.__setitem__("gdn_norm", out))

    for l in layers:
        hook_layer(l)

    # A dispatched model has no reliable .device; the embedding's weight is
    # where the ids have to go for the hooks to move them on.
    dev = model.get_input_embeddings().weight.device
    ids = torch.tensor([tokens(args.tokens)], device=dev)
    with torch.no_grad():
        model(input_ids=ids)

    out = {"layers": {}}
    for l in layers:
        st = captured[l]
        x, z = st["gdn_o"], st["z"]
        t, h = st["normed_in"].shape[1], st["normed_in"].shape[2]
        hdim = x.shape[-1]
        assert x.shape == z.shape, f"norm operands differ: {x.shape} vs {z.shape}"
        nheads = x.shape[0] // t
        print(f"\nlayer {l}: normed_in {tuple(st['normed_in'].shape)}  "
              f"gdn_o {tuple(x.shape)} ({t} tokens x {nheads} heads x {hdim})")

        # The hook is on the model's own path only if the norm's output is
        # the norm of the captured inputs, in the model's own order: f32
        # variance, the weight multiply in bf16, the silu gate in f32.
        xx = x.to(torch.float32)
        var = xx.pow(2).mean(-1, keepdim=True)
        hh = (xx * torch.rsqrt(var + st["eps"])).to(torch.bfloat16)
        hh = (st["w"] * hh).to(torch.bfloat16)
        ref = (hh * torch.nn.functional.silu(z.to(torch.float32))).to(torch.bfloat16)
        d = (ref.float() - st["gdn_norm"].float()).abs().max().item()
        r = st["gdn_norm"].float().abs().max().item()
        ok = d <= 4e-6 * max(r, 1e-6)
        print(f"norm recompute: max|diff|={d:.3e} vs max|out|={r:.3e} "
              f"({'hook on the model path' if ok else 'HOOK DISAGREES'})")
        if not ok:
            sys.exit(1)

        qr = stats(f"  {l} normed_in", row_ss(st["normed_in"], t, h))
        xo = row_ss(x, t * nheads, hdim)
        qo = stats(f"  {l} gdn_o    ", xo)
        xz = row_ss(z, t * nheads, hdim)
        qz = stats(f"  {l} gdn_in   ", xz)
        qn = stats(f"  {l} gdn_norm ", row_ss(st["gdn_norm"], t * nheads, hdim))
        out["layers"][str(l)] = {
            "normed_in": qr.tolist(), "gdn_o": qo.tolist(),
            "gdn_in": qz.tolist(), "gdn_norm": qn.tolist(),
            # The full per-(token, head) row_ss, so the exact rows the tape's
            # FDREC measured (flat elem = token * nheads * 128 + head * 128
            # + channel) can be pulled out of the matrix post-hoc.
            "gdn_o_rows": xo.tolist(), "gdn_in_rows": xz.tolist(),
            "norm_recompute_max_abs_diff": d, "eps": st["eps"],
        }

    print("\nvs the engine's FDREC rows, unseeded base container (this report):")
    print("  engine 62: gdn_o row_ss=1.21188e-07 / 6.06567e-04  "
          "gdn_in(gate) row_ss=7.19 / 5.03")
    print("  engine 61: gdn_o row_ss=9.32871e-07 / 1.07165e-04  "
          "gdn_in(gate) row_ss=5.48 / 5.11")
    print("  engine 60: gdn_o row_ss=3.06421e-06 / 2.19576e-04  "
          "gdn_in(gate) row_ss=4.94 / 4.05")
    with open("gdn_pre_norm_ref.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
