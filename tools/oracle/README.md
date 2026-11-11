# The reference generator

`make_reference.py` writes `tests/fixtures/oracle/` from Hugging Face's Qwen3.5 in PyTorch. It is
run by hand, once per reference, from a virtual environment; the build and the tests never import
Python and never need PyTorch.

## A fresh environment

From the repository root (no `sudo`, nothing system-wide, nothing in `/opt/hf`):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r tools/oracle/requirements.txt
```

`requirements.txt` pins `torch==2.15.0+cpu` (the CPU wheel), `transformers==5.22.2`,
`safetensors==0.8.0` and `numpy==2.5.4`. The `+cpu` wheel comes from PyTorch's own index; the file
adds that index. No card is used.

## The checkpoint

The script's input is a checkpoint directory, not a container. `write_fixture.cpp` writes it
through the fixture's own helper, so the bytes are exactly the ones the host tests start from:

```sh
cmake -B build -G Ninja -DR97_WITH_HIP=OFF
cmake --build build --target r97_write_fixture
build/r97_write_fixture /tmp/tiny_qwen35_bf16
```

The directory is created if needed. It holds `config.json`, `model.safetensors`, `tokenizer.json`
and `tokenizer_config.json` (with the chat template), like any checkpoint.

## Generate

```sh
.venv/bin/python tools/oracle/make_reference.py --checkpoint /tmp/tiny_qwen35_bf16
```

It writes `reference.safetensors`, `manifest.json` and `adapter/` under `tests/fixtures/oracle/`.
`--out DIR` writes elsewhere, and `--check-only` runs every check without writing.

The script checks, in this order:

1. that `fla`, `causal_conv1d` and hub `kernels` are absent and the pure-PyTorch delta rule and
   convolution are the ones in use;
2. that every f64 override equals the stock module in f32 to the bit (see `docs/ORACLE.md`);
3. that the chunked and recurrent delta rules agree to 1e-10 relative on the reference's own
   inputs, the batch of three sequences, captured from its forward pass, so the comparison crosses
   three 64-token chunks;
4. that `torch.autograd.gradcheck` passes in f64 on the three block cases.

Then it writes the reference. On this machine the generator takes about 20 seconds (16.9 s and
21.7 s in the last two runs, 34 s under load); the fresh venv's `pip install` takes about three
minutes and building the writer about six. Two runs from the same environment produce byte-identical
files. The environment lives at `.venv` under the checkout, which `.gitignore` already covers.

## What not to do

Do not regenerate to make a failing host test pass. `tests/host/oracle_fixture_sha.cpp` fails when
the fixture changed, and `tests/host/oracle_prefill.cpp` fails when radiance's bf16 logits and the
f64 reference disagree; both are findings to understand and write into `docs/ORACLE.md` first.
