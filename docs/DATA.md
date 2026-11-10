# Data

`data/` is gitignored: the datasets it holds are fetched, not checked in.

## Placeholder dataset: UltraChat 200k

The first real training run uses `HuggingFaceH4/ultrachat_200k` as a
placeholder. The dataset is released under the MIT licence and is not gated,
so the fetch needs no login.

Each conversation is one JSON line of
`{"messages": [{"role": "user"|"assistant"|"system", "content": "..."}, ...]}`:

- `data/ultrachat/train.jsonl` — the dataset's `train_sft` split
- `data/ultrachat/heldout.jsonl` — the dataset's `test_sft` split

Fetch with:

```sh
python3 tools/fetch_ultrachat.py
```

The download goes through the `hf` CLI. The script is idempotent: it skips
splits whose output file already exists, prints the number of conversations
per file, and fails on an empty `messages` list or an unknown role. It
converts the parquet shards itself (pure stdlib), so no datasets/pyarrow
installation is needed. To re-fetch from scratch, delete `data/ultrachat/`.

This is a placeholder; it should be replaced by a real dataset file of the
same format before any serious run.
