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


BLOCK_NAMES = ("gdn_mixer", "attention_sublayer", "mlp_sublayer")


def block_layers(config) -> dict[str, int]:
    """The layer each block case is taken from; the manifest records these.

    `gdn_mixer` and `mlp_sublayer` use layer 0, the first gated delta net layer, and
    `attention_sublayer` the last full-attention layer, which on this fixture is the only one
    (layer 3). One function rather than a literal in each place: a doc line and a script literal
    had already drifted apart over which layer `mlp_sublayer` uses, and nothing recorded it.
    """
    full = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
    if not full:
        raise SystemExit("oracle: the fixture has no full-attention layer")
    return {"gdn_mixer": 0, "attention_sublayer": max(full), "mlp_sublayer": 0}


def assert_no_lora(model, where: str) -> None:
    """Fail unless every LoRA target still holds a plain torch.nn.Linear.

    The block cases are the plain trunk. The first version of this script computed them after
    add_lora(), so each stored output and cotangent was `W x + (alpha / r) B A x` rather than the
    block the issue defines, and a test differentiating the trunk's block could not use them. The
    guard sits here so that cannot come back unnoticed.
    """
    for layer in model.model.layers:
        for target in LORA_TARGETS:
            module = layer
            for part in target.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if isinstance(module, LoRALinear):
                raise SystemExit(f"oracle: {where}: {target} is LoRA-wrapped")


def block_output(name, model, modeling, config, x, mask2d, positions):
    """One block case's output, on the trunk as it is.

    `x` is the residual for the two sublayer cases and the already-normed hidden state for
    `gdn_mixer`, which is where the issue starts that case. The norms the other two cases include
    are inside them, as the issue defines them from residual to residual.
    """
    layers = block_layers(config)
    if name == "gdn_mixer":
        layer = model.model.layers[layers["gdn_mixer"]]
        return layer.linear_attn(x, cache_params=None, attention_mask=mask2d)
    if name == "attention_sublayer":
        layer = model.model.layers[layers["attention_sublayer"]]
        position_ids = positions[None].expand(4, -1, -1)
        attn_mask = causal_mask(modeling, model, config, x, mask2d, positions)
        cos, sin = model.model.rotary_emb(x, position_ids[1:])
        attn_out, _ = layer.self_attn(
            layer.input_layernorm(x),
            position_embeddings=(cos, sin),
            attention_mask=attn_mask,
            position_ids=position_ids[0],
        )
        return x + attn_out
    if name == "mlp_sublayer":
        layer = model.model.layers[layers["mlp_sublayer"]]
        return x + layer.mlp(layer.post_attention_layernorm(x))
    raise SystemExit(f"oracle: unknown block case {name}")


def block_cases(model, modeling, config, gen):
    """The three block cases, each for one sequence and for a padded batch of unequal lengths.

    Computed on the plain trunk, before add_lora() wraps anything: the issue defines `gdn_mixer` as
    the module `linear_attn` from the normed hidden state to its output, and the other two from the
    residual to the residual after the add. The layer each case comes from is in block_layers(),
    and the manifest records it per case.
    """
    assert_no_lora(model, "block_cases")
    cases = {}
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

        # gdn_mixer: the normed hidden state -> linear_attn's output, before the residual add.
        # The norm is applied first and its result is the leaf, so the case's input really is
        # input_layernorm's output: its cotangent is the mixer's own, not the composite's through
        # the norm, which is what the residual being the leaf would give.
        gdn = model.model.layers[block_layers(config)["gdn_mixer"]]
        normed = gdn.input_layernorm(x).detach().requires_grad_(True)
        finish(
            "gdn_mixer",
            normed,
            block_output("gdn_mixer", model, modeling, config, normed, mask2d, positions),
        )

        # attention_sublayer: residual -> residual + self_attn(input_layernorm(residual)).
        # mlp_sublayer: residual -> residual + mlp(post_attention_layernorm(residual)).
        for name in ("attention_sublayer", "mlp_sublayer"):
            inp = x.clone().requires_grad_(True)
            finish(name, inp, block_output(name, model, modeling, config, inp, mask2d, positions))
    return cases


def check_blocks_exclude_adapter(model, modeling, config, cases) -> float:
    """Recompute each stored block case on the adapter-wrapped model and require it to differ.

    A positive control for block_cases() having run before add_lora(): if those numbers had been
    taken from the adapter-wrapped trunk, this would find them unchanged. The returned number is
    the smallest of the six max-abs differences and goes into the manifest, so a consumer can see
    how far the adapter would move each case.
    """
    smallest = None
    with torch.no_grad():
        for name in BLOCK_NAMES:
            for tag in ("single", "batch"):
                inp = cases[f"{name}.{tag}.input"]
                mask2d = cases[f"{name}.{tag}.mask"]
                positions = torch.zeros_like(mask2d)
                for row in range(mask2d.shape[0]):
                    length = int(mask2d[row].sum())
                    positions[row, :length] = torch.arange(length)
                out = block_output(name, model, modeling, config, inp, mask2d, positions)
                diff = float((out - cases[f"{name}.{tag}.output"]).abs().max())
                if diff < 1e-6:
                    raise SystemExit(
                        f"oracle: {name}.{tag} is unchanged with the adapter wrapped in; the "
                        "stored case is not the plain trunk"
                    )
                smallest = diff if smallest is None else min(smallest, diff)
    return smallest


def check_gradcheck(model, modeling, config, gen) -> None:
    """gradcheck in f64 on the three block cases, on the single-sequence shape each uses."""
    assert_no_lora(model, "check_gradcheck")
    length = 7
    hidden = config.hidden_size
    x = torch.randn(1, length, hidden, generator=gen, dtype=torch.float64)
    positions = torch.arange(length)[None]
    for name in BLOCK_NAMES:
        def fn(inp, name=name):
            # The block case starts at the normed hidden state for gdn_mixer, so this is the mixer
            # on its own there and residual-to-residual for the other two.
            return block_output(name, model, modeling, config, inp, None, positions)

        ok = torch.autograd.gradcheck(fn, (x.clone().requires_grad_(True),), eps=1e-6, atol=1e-6, rtol=1e-5)
        if not ok:
            raise SystemExit(f"oracle: gradcheck failed for {name}")


DELTA_RULE_ARGUMENTS = ("query", "key", "value", "g", "beta")


def capture_delta_rule_inputs(model, modeling, config, ids, mask, positions):
    """Run the reference batch once and keep what the trunk hands the chunked delta rule.

    The agreement check has to be about the tensors the reference itself uses, and the sequence it
    uses has to cross chunk boundaries: the batch is padded to 140 tokens, three 64-token chunks of
    the scan. A shorter input would agree just as well and say nothing about the multi-chunk path,
    which is the reason the 140-token sequence is in the reference at all.

    The trunk resolves `torch_chunk_gated_delta_rule` in the modeling module's namespace at call
    time, so wrapping that name records every gated delta net's query, key, value, g and beta
    without re-deriving them here. Nothing is drawn from the generator, so the capture cannot move
    a reference tensor.
    """
    original = modeling.torch_chunk_gated_delta_rule
    calls = []

    def recording(*args, **kwargs):
        output = original(*args, **kwargs)
        calls.append((args, kwargs, output))
        return output

    modeling.torch_chunk_gated_delta_rule = recording
    try:
        with torch.no_grad():
            model.model(
                input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False
            )
    finally:
        modeling.torch_chunk_gated_delta_rule = original

    expected = sum(1 for kind in config.layer_types if kind == "linear_attention")
    if len(calls) != expected:
        raise SystemExit(
            f"oracle: the trunk called the chunked delta rule {len(calls)} times, expected {expected}"
        )
    return calls


def check_delta_agreement(modeling, calls) -> dict[str, object]:
    """The chunked and the recurrent delta rule, in f64, on the reference's own inputs.

    `calls` is what capture_delta_rule_inputs() recorded from the reference batch's forward pass:
    the exact query, key, value, g and beta of every gated delta net layer. The recurrent form is
    run on the same tensors and compared over the whole 140-token sequence, so the comparison
    covers the chunk boundaries the chunked form has to stitch together.
    """
    chunk_signature = inspect.signature(modeling.torch_chunk_gated_delta_rule)
    for name in DELTA_RULE_ARGUMENTS:
        if name not in chunk_signature.parameters:
            raise SystemExit(f"oracle: the chunked delta rule has no {name!r} parameter")

    worst = (0.0, None)
    worst_elementwise = 0.0
    shape = None
    chunk_size = None
    for layer, (args, kwargs, output) in enumerate(calls):
        bound = chunk_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        named = bound.arguments
        for name in DELTA_RULE_ARGUMENTS[:3]:
            if not bool(named[name].any()):
                raise SystemExit(f"oracle: the captured {name} of layer {layer} is all zero")
        chunk = output[0] if isinstance(output, tuple) else output
        recurrent, _ = modeling.torch_recurrent_gated_delta_rule(
            *(named[name] for name in DELTA_RULE_ARGUMENTS),
            initial_state=named.get("initial_state"),
            output_final_state=False,
            use_qk_l2norm_in_kernel=named.get("use_qk_l2norm_in_kernel", False),
            **named.get("kwargs", {}),
        )
        # Two zeros would agree perfectly and prove nothing.
        if not bool(chunk.any()) or not bool(recurrent.any()):
            raise SystemExit(f"oracle: the delta rule returned all zeros at layer {layer}")
        difference = (chunk - recurrent).abs()
        # Over the tensor, not elementwise: a single output near zero would otherwise decide the
        # number on its own (the same measure the host tests use for the logits).
        relative = float(difference.norm() / recurrent.norm().clamp_min(1e-30))
        if relative > worst[0]:
            worst = (relative, layer)
        elementwise = float((difference / recurrent.abs().clamp_min(1e-30)).max())
        worst_elementwise = max(worst_elementwise, elementwise)
        shape = list(named["query"].shape)
        chunk_size = int(named["chunk_size"])

    if shape is None:
        raise SystemExit("oracle: the reference batch called the delta rule no times")
    sequence_length = shape[1]
    chunks = -(-sequence_length // chunk_size)
    if chunks <= 2:
        raise SystemExit(
            f"oracle: the delta rule agreement check would not cross a chunk boundary: "
            f"{sequence_length} tokens in {chunk_size}-token chunks"
        )
    if max(worst[0], worst_elementwise) > 1e-10:
        raise SystemExit(
            f"oracle: chunked and recurrent delta rule differ by {worst[0]:g} relative "
            f"over the tensor and {worst_elementwise:g} elementwise (layer {worst[1]})"
        )
    return {
        "relative_difference": worst[0],
        "max_elementwise_relative_difference": worst_elementwise,
        "gdn_layer_of_worst": worst[1],
        "layers_checked": len(calls),
        "sequence_length": sequence_length,
        "chunk_size": chunk_size,
        "chunks": chunks,
        "shape": shape,
        "inputs": (
            "the reference batch's forward pass, without the adapter: query, key, value, g and "
            "beta at every gated delta net layer"
        ),
    }


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
    fixture_sha = sha256_file(args.checkpoint / "model.safetensors")
    lengths = [140, 73, 31]
    ids, score, mask, positions = batch_inputs(lengths, gen, config.vocab_size)

    delta_calls = capture_delta_rule_inputs(model, modeling, config, ids, mask, positions)
    delta_rule = check_delta_agreement(modeling, delta_calls)
    check_gradcheck(model, modeling, config, gen)
    # Before add_lora(): the block cases are the plain trunk (block_cases asserts it), and the
    # draws they take from the generator therefore come before the adapter's A and B.
    block = block_cases(model, modeling, config, gen)
    print(
        "oracle: dtype overrides match stock in f32; the chunked and the recurrent delta rule "
        f"agree to {delta_rule['relative_difference']:.3g} relative over {delta_rule['chunks']} chunks"
    )
    with torch.no_grad():
        _, _, logits_before, _, _ = run_model(
            model, modeling, config, ids, score, mask, positions, keep_grad=False
        )
    modules = add_lora(model, args.lora_rank, args.lora_alpha, gen)
    adapter_control = check_blocks_exclude_adapter(model, modeling, config, block)
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

    tensors.update(block)

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
        "delta_rule": delta_rule,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "targets": list(LORA_TARGETS)},
        "cases": {
            "sequences": {
                "lengths": lengths,
                "longest_chunks": [length // DEFAULT_CHUNK + (length % DEFAULT_CHUNK != 0) for length in lengths],
                "names": ["seq0", "seq1", "seq2"],
            },
            # The layer each case comes from, and that no adapter is inside it: the first version
            # of this reference stored adapter-wrapped blocks and said neither.
            "blocks": [
                {"name": name, "layer": block_layers(config)[name], "with_adapter": False}
                for name in BLOCK_NAMES
            ],
        },
        "block_adapter_control": {
            "smallest_output_change": adapter_control,
            "what": (
                "each stored block case recomputed on the adapter-wrapped model: the smallest "
                "max-abs difference from the stored output, which would be zero if the cases "
                "had been taken with the adapter in"
            ),
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
