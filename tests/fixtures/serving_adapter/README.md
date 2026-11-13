# serving_adapter: one adapter, written by r97train, that the serving test loads.

`tests/host/serving_prefill.cpp` answers a prefill with these weights, so the directory is a
committed fixture and not a build product. The weights are the tiny int8 fixture's, trained for 48
steps by the default build's `r97train`; nothing here is hand-written.

To regenerate (from a worktree with the default tree built, `cmake -B build -G Ninja
-DR97_WITH_HIP=OFF && cmake --build build`):

```sh
mkdir -p /tmp/serving_fixture
build/r97_write_fixture /tmp/serving_fixture/ckpt
build/radiance-build/bin/rad-convert /tmp/serving_fixture/ckpt \
    -o /tmp/serving_fixture/tiny_qwen35_w8a8.rad \
    --kernels libavx,libref --home build/radiance-build/radiance_home \
    --recipe tests/fixtures/w8a8_fwht128.recipe
build/r97train train --model /tmp/serving_fixture/tiny_qwen35_w8a8.rad \
    --data tests/fixtures/chat_small.jsonl --out /tmp/serving_fixture/run \
    --delta.targets q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,gate_proj,up_proj,down_proj \
    --steps 48 --checkpoint_every 48 --lr 1e-2
cp /tmp/serving_fixture/run/checkpoint-000000048/adapter/* tests/fixtures/serving_adapter/
```

`adapter_config.json`'s `base_model_name_or_path` is rewritten from the temporary path above to
`tiny_qwen35_w8a8.rad`: it names the fixture container, and no loader validates it. Everything
else, including `r` and `lora_alpha`, is what the writer wrote.

The reference logits beside this directory, `serving_prefill_logits.bin`, are written by the same
test in the default build:

```sh
R97_WRITE_SERVING_GOLDEN=tests/fixtures/serving_prefill_logits.bin \
R97_TEST_ONLY=serving_prefill_with_the_fixture_adapter_matches_the_reference \
    build/r97_host_tests
```

The serving build runs the same test and must reproduce the file byte for byte. `docs/TESTING.md`
has the two-build check, and `docs/FIXTURES.md` the fixture the adapter was trained on.
