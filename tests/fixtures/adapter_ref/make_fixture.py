#!/usr/bin/env python3
"""Build the reference adapter directory by hand, from the format docs/ADAPTER.md states.

The point of the fixture is to be independent of `src/adapter_io.cpp`: the bytes here are laid out
from the safetensors specification and PEFT's naming rule, not by calling the writer under test, so
tests/host/adapter_io.cpp comparing the writer's output to this directory says the writer follows
the document and not only itself.

The adapter is the tiny fixture container's (`tests/support/fixture.h`: hidden 64, ffn 128) one
module at rank 2, `model.layers.0.mlp.down_proj`, so shapes are A [2, 128], B [64, 2]. Values are
exact integers, so an f32 round trip is exact and a mismatch is a layout error, not a rounding one.

Run from this directory:  python3 make_fixture.py
"""

import json
import struct

MODULE = "model.layers.0.mlp.down_proj"
PREFIX = "base_model.model."
RANK = 2
K = 128   # down_proj's input: the ffn width
N = 64    # down_proj's output: the hidden width


def factors():
    """The hand-built values: A[r][k] = r * K + k, B[n][r] = -(n * RANK + r + 1)."""
    a = [float(r * K + k) for r in range(RANK) for k in range(K)]
    b = [float(-(n * RANK + r + 1)) for n in range(N) for r in range(RANK)]
    return a, b


def safetensors(tensors):
    """`tensors` is [(name, shape, values)]; dtype is f32, the file's names sorted."""
    tensors = sorted(tensors, key=lambda t: t[0])
    header, off = {}, 0
    for name, shape, values in tensors:
        nbytes = len(values) * 4
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-(8 + len(text)) % 8)          # every header starts on a multiple of 8
    out = struct.pack("<Q", len(text)) + text
    for _, _, values in tensors:
        out += struct.pack("<%df" % len(values), *values)
    return out


def main():
    a, b = factors()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "qwen3.5-tiny",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "r": RANK,
        "target_modules": ["down_proj"],
        "use_rslora": False,
    }
    with open("adapter_config.json", "w") as f:
        f.write(json.dumps(config, indent=2) + "\n")
    with open("adapter_model.safetensors", "wb") as f:
        f.write(safetensors([
            (PREFIX + MODULE + ".lora_A.weight", [RANK, K], a),
            (PREFIX + MODULE + ".lora_B.weight", [N, RANK], b),
        ]))


if __name__ == "__main__":
    main()
