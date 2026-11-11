#!/usr/bin/env python3
"""Generate the PyTorch reference under tests/fixtures/oracle/.

PyTorch is a tool here, not a dependency of this repository: this script runs once, from a virtual
environment with tools/oracle/requirements.txt, and writes files that are checked in. The build and
every test read those files and never import Python.

The reference is Hugging Face's own Qwen3.5 implementation, differentiated by torch.autograd. It
shares no code and no derivation with radiance's adjoints: it is a second opinion on the same
checkpoint.

    ./build/r97_write_fixture /tmp/tiny_qwen35_bf16
    tools/oracle/make_reference.py --checkpoint /tmp/tiny_qwen35_bf16

See docs/ORACLE.md for the command, the run time and every place where this script has to override
a dtype cast so the model really runs in f64.

Nothing is random without a seed; every tensor that is not in the checkpoint comes from one
torch.Generator seeded with --seed, so a rerun on the same machine reproduces the files bit for
bit. torch runs on one thread for the same reason.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F

# The checkpoint's hashes are of bytes, not of tensors; the same reader is on the C++ side.
DEFAULT_SEED = 20261111
DEFAULT_CHUNK = 64  # the delta rule's chunk size, asserted against the installed implementation


def package_versions() -> dict[str, str]:
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": __import__("numpy").__version__,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_pure_pytorch(modeling) -> None:
    """Fail unless the fallback, pure-PyTorch delta rule and convolution are the ones in use.

    transformers takes a fast path whenever `fla`, `causal_conv1d` or a hub kernel is installed.
    The reference must not: it has to be the code in modeling_qwen3_5.py that the issue names, and
    the same code must be what the f64 overrides below edit. This checks the packages are absent,
    then checks the decorated functions still wrap the pure-PyTorch bodies.
    """
    for package in ("fla", "causal_conv1d", "kernels"):
        if importlib.util.find_spec(package) is not None:
            raise SystemExit(
                f"oracle: {package} is installed; the reference must take the pure-PyTorch path. "
                "Remove it from the environment and run again."
            )

    for name, marker in (
        ("torch_chunk_gated_delta_rule", "torch.linalg.solve_triangular"),
        ("torch_recurrent_gated_delta_rule", "last_recurrent_state"),
    ):
        fallback = getattr(modeling, name)
        wrapped = getattr(fallback, "__wrapped__", None)
        if wrapped is None or marker not in inspect.getsource(wrapped):
            raise SystemExit(f"oracle: {name} is not the pure-PyTorch fallback")

    conv = getattr(modeling, "causal_conv1d_fn")
    wrapped = getattr(conv, "__wrapped__", None)
    if wrapped is None or "F.conv1d" not in inspect.getsource(wrapped):
        raise SystemExit("oracle: causal_conv1d_fn is not the pure-PyTorch fallback")

    # The longest sequence has to cross more than two chunks. Read the chunk size from the
    # installed fallback rather than trusting DEFAULT_CHUNK, so a changed default stops the
    # script instead of silently making the sequence shorter than the case claims.
    chunk = inspect.signature(getattr(modeling, "torch_chunk_gated_delta_rule").__wrapped__)
    default = chunk.parameters["chunk_size"].default
    if default != DEFAULT_CHUNK:
        raise SystemExit(
            f"oracle: the installed chunk size is {default}, not {DEFAULT_CHUNK}; update "
            "DEFAULT_CHUNK and the manifest's longest_chunks"
        )


def patch_source(modeling, obj, replacements, name=None):
    """Exec a copy of `obj`'s source with dtype-only replacements, in the modeling module.

    The replacements remove casts that force activations to f32. Executing into the module's own
    dict (rather than a copy) means a patched function sees the other patched globals, as the stock
    code sees the stock ones. Every replacement must match exactly once: a transformers update
    that moves the line stops the script instead of silently producing a different reference.
    """
    src = textwrap.dedent(inspect.getsource(obj))
    for old, new in replacements:
        count = src.count(old)
        if count != 1:
            raise SystemExit(
                f"oracle: expected one {old!r} in {getattr(obj, '__qualname__', obj)}, found {count}"
            )
        src = src.replace(old, new)
    exec(compile(src, f"<oracle override {getattr(obj, '__qualname__', obj)}>", "exec"), modeling.__dict__)
    if name is not None:
        return modeling.__dict__[name]
    return obj


def patch_dtype_casts(modeling) -> dict[str, object]:
    """Replace every f32 activation cast in the text model with its input's dtype.

    Returns the stock implementations, so prove_dtype_only() can show, in f32, that each override
    changes the dtype and nothing else.
    """
    stock = {
        "rms_norm": modeling.Qwen3_5RMSNorm.forward,
        "rms_norm_gated": modeling.Qwen3_5RMSNormGated.forward,
        "rotary": modeling.Qwen3_5TextRotaryEmbedding.forward,
        "eager_attention": modeling.eager_attention_forward,
        "chunk": modeling.torch_chunk_gated_delta_rule,
        "recurrent": modeling.torch_recurrent_gated_delta_rule,
        "gdn_forward": modeling.Qwen3_5GatedDeltaNet.forward,
    }

    modeling.Qwen3_5RMSNorm.forward = patch_source(
        modeling,
        stock["rms_norm"],
        [
            ("output = self._norm(x.float())", "output = self._norm(x)"),
            ("output = output * (1.0 + self.weight.float())", "output = output * (1.0 + self.weight)"),
        ],
        name="forward",
    )
    modeling.Qwen3_5RMSNormGated.forward = patch_source(
        modeling,
        stock["rms_norm_gated"],
        [
            ("hidden_states = hidden_states.to(torch.float32)", "hidden_states = hidden_states"),
            (
                "hidden_states = self.weight * hidden_states.to(input_dtype)",
                "hidden_states = self.weight * hidden_states",
            ),
            (
                "hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))",
                "hidden_states = hidden_states * ACT2FN[self.activation](gate)",
            ),
            ("return hidden_states.to(input_dtype)", "return hidden_states"),
        ],
        name="forward",
    )
    # The rotary forward's final casts are its only f32 rounding for f64 inputs; the decorators
    # (torch.no_grad and dynamic_rope_update) stay, as they are part of the stock behaviour.
    modeling.Qwen3_5TextRotaryEmbedding.forward = patch_source(
        modeling,
        stock["rotary"],
        [
            (
                "freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)",
                "freqs = position_ids[..., None].to(x.dtype) * self.inv_freq.to(device=x.device, dtype=x.dtype)",
            ),
            (
                "return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)",
                "return cos, sin",
            ),
        ],
        name="forward",
    )
    # The eager path's softmax is computed in f32 even for a f64 model.
    modeling.eager_attention_forward = patch_source(
        modeling,
        stock["eager_attention"],
        [
            (
                "nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)",
                "nn.functional.softmax(attn_weights, dim=-1)",
            ),
        ],
        name="eager_attention_forward",
    )

    # The delta rule's two pure-PyTorch bodies cast every activation to f32 at entry. Remove that
    # cast and the cast back: with them the chunked and recurrent forms differ at f32 precision,
    # which is nowhere near the 1e-10 the issue asks for.
    patch_source(
        modeling,
        stock["chunk"],
        [
            (
                "        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)\n"
                "        for x in (query, key, value, beta, decay)",
                "        x.transpose(1, 2)\n        for x in (query, key, value, beta, decay)",
            ),
            (
                "    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, "
                "memory_format=torch.contiguous_format)",
                "    core_attn_out = core_attn_out.transpose(1, 2).contiguous()",
            ),
        ],
        name="torch_chunk_gated_delta_rule",
    )
    patch_source(
        modeling,
        stock["recurrent"],
        [
            (
                "        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)\n"
                "        for x in (query, key, value, beta, decay)",
                "        x.transpose(1, 2)\n        for x in (query, key, value, beta, decay)",
            ),
            (
                "    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)",
                "    core_attn_out = core_attn_out.transpose(1, 2).contiguous()",
            ),
        ],
        name="torch_recurrent_gated_delta_rule",
    )
    # The decay projection casts A_log and a to f32 before exp/softplus.
    gdn = patch_source(
        modeling,
        stock["gdn_forward"],
        [
            ("self.A_log.float().exp()", "self.A_log.exp()"),
            ("F.softplus(a.float() + self.dt_bias)", "F.softplus(a + self.dt_bias)"),
        ],
        name="forward",
    )
    modeling.Qwen3_5GatedDeltaNet.forward = gdn

    return stock


def prove_dtype_only(modeling, config, stock) -> dict[str, float]:
    """In f32, every override is the stock module to the bit.

    The reference runs in f64, so the proof is done in f32 where the stock casts are no-ops: if an
    override changed anything but the dtype, the two would part company here. The returned values
    are the largest absolute differences, all zero; the script prints them into the manifest.
    """
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    out: dict[str, float] = {}

    def f32(t):
        return t.to(torch.float32)

    # RMSNorm and RMSNormGated.
    norm = modeling.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(torch.float32)
    x = f32(torch.randn(2, 7, config.hidden_size, generator=gen))
    a = stock["rms_norm"](norm, x)
    b = modeling.Qwen3_5RMSNorm.forward(norm, x)
    out["rms_norm"] = float((a - b).detach().abs().max())

    gated = modeling.Qwen3_5RMSNormGated(
        config.linear_value_head_dim, eps=config.rms_norm_eps
    ).to(torch.float32)
    xg = f32(torch.randn(2, 7, config.linear_value_head_dim, generator=gen))
    g = f32(torch.randn(2, 7, config.linear_value_head_dim, generator=gen))
    a = stock["rms_norm_gated"](gated, xg, g)
    b = modeling.Qwen3_5RMSNormGated.forward(gated, xg, g)
    out["rms_norm_gated"] = float((a - b).detach().abs().max())

    # Rotary: both read the same f32 inv_freq buffer, so this is exact.
    rot = modeling.Qwen3_5TextRotaryEmbedding(config).to(torch.float32)
    position_ids = torch.arange(2 * 7).reshape(2, 7)
    cos_a, sin_a = stock["rotary"](rot, x, position_ids)
    cos_b, sin_b = modeling.Qwen3_5TextRotaryEmbedding.forward(rot, x, position_ids)
    out["rotary"] = float(max((cos_a - cos_b).detach().abs().max(), (sin_a - sin_b).detach().abs().max()))

    # Eager attention. The stock function takes the projection module for num_key_value_groups and
    # training; a plain namespace is enough.
    import types

    module = types.SimpleNamespace(num_key_value_groups=1, training=False)
    heads, head_dim = 2, 4
    q = f32(torch.randn(2, heads, 7, head_dim, generator=gen))
    k = f32(torch.randn(2, heads, 7, head_dim, generator=gen))
    v = f32(torch.randn(2, heads, 7, head_dim, generator=gen))
    mask = torch.zeros(2, 1, 7, 7, dtype=torch.float32)
    mask = mask.masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), torch.finfo(torch.float32).min)
    a, _ = stock["eager_attention"](module, q, k, v, mask, head_dim**-0.5)
    b, _ = modeling.eager_attention_forward(module, q, k, v, mask, head_dim**-0.5)
    out["eager_attention"] = float((a - b).detach().abs().max())

    # The decay projection's f32 casts in the gated delta net's forward.
    gdn = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(torch.float32)
    xh = f32(torch.randn(2, 7, config.hidden_size, generator=gen))
    a = stock["gdn_forward"](gdn, xh, cache_params=None, attention_mask=None)
    b = modeling.Qwen3_5GatedDeltaNet.forward(gdn, xh, cache_params=None, attention_mask=None)
    out["gdn_forward"] = float((a - b).detach().abs().max())

    # Both delta-rule bodies, with their own f32 inputs.
    batch, seq, k_heads, v_heads, kd, vd = 2, 9, 3, 3, 8, 6
    q = f32(torch.randn(batch, seq, k_heads, kd, generator=gen))
    k = f32(torch.randn(batch, seq, k_heads, kd, generator=gen))
    v = f32(torch.randn(batch, seq, v_heads, vd, generator=gen))
    g = -F.softplus(f32(torch.randn(batch, seq, v_heads, generator=gen)))
    beta = torch.sigmoid(f32(torch.randn(batch, seq, v_heads, generator=gen)))
    for name in ("chunk", "recurrent"):
        a, _ = stock[name](q, k, v, g, beta, output_final_state=False)
        b, _ = getattr(modeling, f"torch_{name}_gated_delta_rule")(
            q, k, v, g, beta, output_final_state=False
        )
        out[f"delta_{name}"] = float((a - b).detach().abs().max())

    for key, value in out.items():
        if value != 0.0:
            raise SystemExit(f"oracle: the {key} override differs from the stock module in f32: {value}")
    return out


def load_config(checkpoint: Path):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    raw = json.loads((checkpoint / "config.json").read_text())
    # The checkpoint is the multimodal release's text tower: `model.language_model.…` names and
    # an architecture with no vision configuration. Build the text configuration from the same
    # keys and change no value. `architectures` and `model_type` are container-level and the text
    # config is strict about unknown keys; every text key is passed through untouched.
    fields = Qwen3_5TextConfig.__dataclass_fields__
    text = {k: v for k, v in raw.items() if k in fields}
    config = Qwen3_5TextConfig(**text)
    config._attn_implementation = "eager"
    return config


def map_checkpoint_names(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """`model.language_model.X` -> `model.X`; values are cast, names are only moved."""
    out = {}
    prefix = "model.language_model."
    for name, value in state_dict.items():
        if name.startswith(prefix):
            name = "model." + name[len(prefix) :]
        out[name] = value.to(torch.float64)
    return out


def build_model(checkpoint: Path):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = load_config(checkpoint)
    torch.set_default_dtype(torch.float64)
    model = Qwen3_5ForCausalLM(config)
    model = model.to(torch.float64)
    state = safetensors.torch.load_file(str(checkpoint / "model.safetensors"))
    missing, unexpected = model.load_state_dict(map_checkpoint_names(state), strict=False)
    if unexpected:
        raise SystemExit(f"oracle: checkpoint has tensors the text model does not use: {unexpected}")
    if missing:
        raise SystemExit(f"oracle: text model has tensors the checkpoint does not provide: {missing}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config, modeling


def prepare_f64_constants(model, modeling, config) -> None:
    """Put the rotary frequencies in f64, recomputed from the same theta.

    transformers builds `inv_freq` with `dtype=torch.float` explicitly, so a model instantiated in
    f64 still carries an f32 table. This is not an activation cast; it is a constant that only
    needs to be computed in the reference's dtype.
    """
    rope = model.model.rotary_emb
    dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
    theta = config.rope_parameters["rope_theta"]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    rope.inv_freq = inv_freq
    rope.original_inv_freq = inv_freq.clone()


class LoRALinear(torch.nn.Module):
    """y = W x + (alpha / r) B A x. The base weight stays frozen; A and B are leaves."""

    def __init__(self, base: torch.nn.Linear, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Parameter(lora_a)
        self.lora_B = torch.nn.Parameter(lora_b)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


LORA_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def add_lora(model, rank: int, alpha: float, generator: torch.Generator):
    """Wrap every target linear, seeded, A and B both non-zero.

    PEFT's default B is zero so a freshly loaded adapter changes nothing; the reference needs a
    non-zero adapter so the gradient of every A and B is exercised. A is scaled like PEFT's
    kaiming A and B like a non-zero PEFT B, both in f64.
    """
    scale = alpha / rank
    modules = []
    for layer_idx, layer in enumerate(model.model.layers):
        for target in LORA_TARGETS:
            *path, leaf = target.split(".")
            parent = layer
            missing = False
            for part in path:
                if not hasattr(parent, part):
                    missing = True
                    break
                parent = getattr(parent, part)
            if missing or not hasattr(parent, leaf):
                continue
            base = getattr(parent, leaf)
            if not isinstance(base, torch.nn.Linear):
                continue
            name = f"model.layers.{layer_idx}.{target}"
            a = torch.randn(rank, base.in_features, generator=generator, dtype=torch.float64)
            a *= base.in_features**-0.5
            b = torch.randn(base.out_features, rank, generator=generator, dtype=torch.float64)
            b *= rank**-0.5
            setattr(parent, leaf, LoRALinear(base, a, b, scale))
            modules.append((name, parent, leaf, rank))
    return modules


def write_adapter(out_dir: Path, modules, rank: int, alpha: float, base_name: str) -> None:
    adir = out_dir / "adapter"
    adir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, parent, leaf, _ in modules:
        module = getattr(parent, leaf)
        # PEFT's on-disk spelling: the full path plus `.lora_A.weight` / `.lora_B.weight`.
        # PEFT files hold f32 adapter weights; radiance's loader accepts BF16, F16 and F32 only.
        tensors[f"base_model.model.{name}.lora_A.weight"] = (
            module.lora_A.detach().to(torch.float32).contiguous()
        )
        tensors[f"base_model.model.{name}.lora_B.weight"] = (
            module.lora_B.detach().to(torch.float32).contiguous()
        )
    safetensors.torch.save_file(tensors, str(adir / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "r": rank,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "use_rslora": False,
    }
    (adir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")


def scored_loss(logits: torch.Tensor, input_ids: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """include/r97/batch.h: mean over score == 1 of -log p(token i | logits at i - 1)."""
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    nll = -logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    weights = score[:, 1:].to(nll.dtype)
    return (nll * weights).sum() / weights.sum()


def per_sequence_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, score: torch.Tensor
) -> torch.Tensor:
    """The same loss restricted to each sequence: what a solo run of that sequence would loss."""
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    nll = -logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    weights = score[:, 1:].to(nll.dtype)
    return torch.stack([(nll[b] * weights[b]).sum() / weights[b].sum() for b in range(nll.shape[0])])


def batch_inputs(lengths: list[int], gen: torch.Generator, vocab: int):
    max_len = max(lengths)
    ids = torch.zeros(len(lengths), max_len, dtype=torch.long)
    score = torch.zeros(len(lengths), max_len, dtype=torch.long)
    mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
    positions = torch.zeros(len(lengths), max_len, dtype=torch.long)
    for b, length in enumerate(lengths):
        ids[b, :length] = torch.randint(1, vocab, (length,), generator=gen)
        # Score all but the first token, as a prefill loss usually does; every row has some.
        score[b, 1:length] = 1
        mask[b, :length] = 1
        positions[b, :length] = torch.arange(length)
    return ids, score, mask, positions


def causal_mask(modeling, model, config, embeds, mask, position_ids):
    return modeling.create_causal_mask(
        config, embeds, mask, None, position_ids, None
    )


def run_model(model, modeling, config, ids, score, mask, positions, keep_grad):
    """Run the text model and return (loss, per-sequence loss, logits, boundaries, boundary_grads)."""
    boundaries = []

    def remember(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        # The embedding is frozen and its input is integer ids, so boundary 0 is a leaf that does
        # not require grad even though the adapter downstream does. It is still a residual
        # boundary: make it one so autograd can return the cotangent of the loss there.
        if keep_grad and not output.requires_grad:
            output.requires_grad_(True)
        boundaries.append(output)

    handles = [model.model.embed_tokens.register_forward_hook(remember)]
    handles += [layer.register_forward_hook(remember) for layer in model.model.layers]
    try:
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    logits = outputs.logits
    loss = scored_loss(logits, ids, score)
    per_seq = per_sequence_loss(logits, ids, score)
    grads = None
    if keep_grad:
        # backward() fills .grad on every adapter leaf; the second pass returns the cotangent at
        # each residual boundary. One graph, two reads of it.
        loss.backward(retain_graph=True)
        grads = torch.autograd.grad(loss, boundaries, retain_graph=False, allow_unused=False)
    return loss, per_seq, logits, boundaries, grads


def block_cases(model, modeling, config, gen):
    """The three block cases, each for one sequence and for a padded batch of unequal lengths."""
    cases = {}
    layer_gdn = 0
    layer_attn = max(
        i for i, t in enumerate(config.layer_types) if t == "full_attention"
    )
    layer_mlp = 1
    for tag, lengths in (("single", [23]), ("batch", [17, 11, 6])):
        max_len = max(lengths)
        hidden = config.hidden_size
        # A residual and its cotangent are the input/output of each sublayer case.
        x = torch.randn(len(lengths), max_len, hidden, generator=gen, dtype=torch.float64)
        for b, length in enumerate(lengths):
            x[b, length:] = 0.0
        mask2d = torch.zeros(len(lengths), max_len, dtype=torch.long)
        for b, length in enumerate(lengths):
            mask2d[b, :length] = 1
        positions = torch.zeros(len(lengths), max_len, dtype=torch.long)
        for b, length in enumerate(lengths):
            positions[b, :length] = torch.arange(length)

        def finish(name, inp, out, lengths=lengths):
            cot = torch.randn(out.shape, generator=gen, dtype=torch.float64)
            (in_cot,) = torch.autograd.grad(out, inp, cot, retain_graph=False)
            cases[f"{name}.{tag}.input"] = inp.detach()
            cases[f"{name}.{tag}.output"] = out.detach()
            cases[f"{name}.{tag}.cotangent"] = cot.detach()
            cases[f"{name}.{tag}.input_cotangent"] = in_cot.detach()
            cases[f"{name}.{tag}.lengths"] = torch.tensor(lengths, dtype=torch.long)
            cases[f"{name}.{tag}.mask"] = mask2d.clone()

        # gdn_mixer: input_layernorm's output -> linear_attn's output, before the residual add.
        inp = x.clone().requires_grad_(True)
        attn = model.model.layers[layer_gdn].linear_attn
        normed = model.model.layers[layer_gdn].input_layernorm(inp)
        mixer_out = attn(normed, cache_params=None, attention_mask=mask2d)
        finish("gdn_mixer", inp, mixer_out)

        # attention_sublayer: residual -> residual + self_attn(input_layernorm(residual)).
        inp = x.clone().requires_grad_(True)
        layer = model.model.layers[layer_attn]
        position_ids = positions[None].expand(4, -1, -1)
        text_positions, rope_positions = position_ids[0], position_ids[1:]
        embeds = inp
        attn_mask = causal_mask(modeling, model, config, embeds, mask2d, positions)
        cos, sin = model.model.rotary_emb(embeds, rope_positions)
        attn_out, _ = layer.self_attn(
            layer.input_layernorm(inp),
            position_embeddings=(cos, sin),
            attention_mask=attn_mask,
            position_ids=text_positions,
        )
        finish("attention_sublayer", inp, inp + attn_out)

        # mlp_sublayer: residual -> residual + mlp(post_attention_layernorm(residual)).
        inp = x.clone().requires_grad_(True)
        layer = model.model.layers[layer_mlp]
        finish("mlp_sublayer", inp, inp + layer.mlp(layer.post_attention_layernorm(inp)))
    return cases


def check_gradcheck(model, modeling, config, gen) -> None:
    """gradcheck in f64 on the three block cases, on the single-sequence shape each uses."""
    layer_gdn, layer_mlp = 0, 1
    layer_attn = max(i for i, t in enumerate(config.layer_types) if t == "full_attention")
    length = 7
    hidden = config.hidden_size
    x = torch.randn(1, length, hidden, generator=gen, dtype=torch.float64)

    def gdn(inp):
        attn = model.model.layers[layer_gdn].linear_attn
        normed = model.model.layers[layer_gdn].input_layernorm(inp)
        return attn(normed, cache_params=None, attention_mask=None)

    layer = model.model.layers[layer_attn]
    positions = torch.arange(length)[None]
    position_ids = positions[None].expand(4, -1, -1)
    text_positions, rope_positions = position_ids[0], position_ids[1:]
    attn_mask = causal_mask(modeling, model, config, x, None, positions)
    cos, sin = model.model.rotary_emb(x, rope_positions)

    def attention(inp):
        out, _ = layer.self_attn(
            layer.input_layernorm(inp),
            position_embeddings=(cos, sin),
            attention_mask=attn_mask,
            position_ids=text_positions,
        )
        return inp + out

    mlp_layer = model.model.layers[layer_mlp]

    def mlp(inp):
        return inp + mlp_layer.mlp(mlp_layer.post_attention_layernorm(inp))

    for name, fn in (("gdn_mixer", gdn), ("attention_sublayer", attention), ("mlp_sublayer", mlp)):
        ok = torch.autograd.gradcheck(fn, (x.clone().requires_grad_(True),), eps=1e-6, atol=1e-6, rtol=1e-5)
        if not ok:
            raise SystemExit(f"oracle: gradcheck failed for {name}")


def check_delta_agreement(modeling, gen) -> float:
    """The chunked and recurrent delta rule, in f64, on the reference's own kind of input."""
    batch, seq, k_heads, v_heads, kd, vd = 2, 40, 4, 4, 16, 16
    q = torch.randn(batch, seq, k_heads, kd, generator=gen, dtype=torch.float64)
    k = torch.randn(batch, seq, k_heads, kd, generator=gen, dtype=torch.float64)
    v = torch.randn(batch, seq, v_heads, vd, generator=gen, dtype=torch.float64)
    g = -F.softplus(torch.randn(batch, seq, v_heads, generator=gen, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(batch, seq, v_heads, generator=gen, dtype=torch.float64))
    chunk, _ = modeling.torch_chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=False)
    recurrent, _ = modeling.torch_recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=False)
    denom = recurrent.abs().clamp_min(1e-30)
    rel = ((chunk - recurrent).abs() / denom).max().item()
    if rel > 1e-10:
        raise SystemExit(f"oracle: chunked and recurrent delta rule differ by {rel:g} relative")
    return rel


def tensor_case_names(cases: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    out = {}
    for name, tensor in cases.items():
        out[name] = {"shape": list(tensor.shape), "dtype": "f64" if tensor.is_floating_point() else "i64"}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="the tiny bf16 fixture directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "tests/fixtures/oracle")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--check-only", action="store_true", help="run the checks, write nothing")
    args = parser.parse_args()

    if not (args.checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"oracle: {args.checkpoint} has no model.safetensors; run r97_write_fixture first")

    started = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    assert_pure_pytorch(modeling)
    model, config, modeling = build_model(args.checkpoint)
    stock = patch_dtype_casts(modeling)
    prepare_f64_constants(model, modeling, config)
    dtype_proof = prove_dtype_only(modeling, config, stock)
    chunk_rel = check_delta_agreement(modeling, gen)
    check_gradcheck(model, modeling, config, gen)
    print(f"oracle: dtype overrides match stock in f32; delta rules agree to {chunk_rel:.3g} relative")

    fixture_sha = sha256_file(args.checkpoint / "model.safetensors")
    lengths = [140, 73, 31]
    ids, score, mask, positions = batch_inputs(lengths, gen, config.vocab_size)

    with torch.no_grad():
        _, _, logits_before, _, _ = run_model(
            model, modeling, config, ids, score, mask, positions, keep_grad=False
        )
    modules = add_lora(model, args.lora_rank, args.lora_alpha, gen)
    loss, per_seq, logits_after, boundaries, grads = run_model(
        model, modeling, config, ids, score, mask, positions, keep_grad=True
    )
    if not loss.requires_grad:
        raise SystemExit("oracle: the loss with the adapter carries no gradient")

    tensors: dict[str, torch.Tensor] = {}
    for b, length in enumerate(lengths):
        tensors[f"seq{b}/input_ids"] = ids[b, :length].detach()
        tensors[f"seq{b}/score"] = score[b, :length].detach()
        tensors[f"seq{b}/logits_without_adapter"] = logits_before[b, :length].detach()
        tensors[f"seq{b}/loss_with_adapter"] = per_seq[b].detach().reshape(())
    tensors["with_adapter/loss"] = loss.detach().reshape(())
    tensors["with_adapter/lengths"] = torch.tensor(lengths, dtype=torch.long)
    # Per sequence, without the padding: the residual at every layer boundary and the cotangent of
    # the loss there. Boundary 0 is the embedding output, boundary i the output of layer i - 1.
    for b, length in enumerate(lengths):
        for i, (hidden, grad) in enumerate(zip(boundaries, grads)):
            tensors[f"seq{b}/residual.{i}"] = hidden[b, :length].detach()
            tensors[f"seq{b}/residual_grad.{i}"] = grad[b, :length].detach()
    adapter_grads = {}
    for name, parent, leaf, _ in modules:
        module = getattr(parent, leaf)
        adapter_grads[f"lora_grad/{name}.lora_A"] = module.lora_A.grad.detach()
        adapter_grads[f"lora_grad/{name}.lora_B"] = module.lora_B.grad.detach()
    tensors.update(adapter_grads)

    tensors.update(block_cases(model, modeling, config, gen))

    # A batch of unequal lengths has padding; check it costs nothing: the scored loss is the
    # token-count-weighted mean of the per-sequence losses, so no padded tail leaks in.
    weights = score[:, 1:].sum(dim=1).to(loss.dtype)
    weighted = (per_seq * weights).sum() / weights.sum()
    if not torch.isfinite(loss).item() or abs(float((loss - weighted).detach())) > 1e-12:
        raise SystemExit("oracle: the per-sequence losses do not add up to the batch loss")

    manifest = {
        "generator": "tools/oracle/make_reference.py",
        "packages": package_versions(),
        "seed": args.seed,
        "torch_threads": 1,
        "fixture": {
            "file": "model.safetensors",
            "sha256": fixture_sha,
            "config_sha256": sha256_file(args.checkpoint / "config.json"),
        },
        "dtype_overrides": dtype_proof,
        "delta_rule_relative_difference": chunk_rel,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "targets": list(LORA_TARGETS)},
        "cases": {
            "sequences": {
                "lengths": lengths,
                "longest_chunks": [length // DEFAULT_CHUNK + (length % DEFAULT_CHUNK != 0) for length in lengths],
                "names": ["seq0", "seq1", "seq2"],
            },
            "blocks": ["gdn_mixer", "attention_sublayer", "mlp_sublayer"],
        },
        "tensors": tensor_case_names(tensors),
    }

    if not args.check_only:
        args.out.mkdir(parents=True, exist_ok=True)
        ref = args.out / "reference.safetensors"
        tmp = ref.with_suffix(".safetensors.tmp")
        safetensors.torch.save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
        os.replace(tmp, ref)
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # A stable name, not the temporary checkpoint path: the adapter is an artifact of this
        # reference and its bytes must not depend on where the writer put the checkpoint.
        write_adapter(args.out, modules, args.lora_rank, args.lora_alpha, "r97-tiny-qwen35-bf16")
        size_mb = ref.stat().st_size / 1e6
    else:
        size_mb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6

    elapsed = time.time() - started
    print(f"oracle: {len(tensors)} tensors, {size_mb:.2f} MB, {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
