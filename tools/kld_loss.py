#!/usr/bin/env python3
"""The held-out loss of a radiance KL record.

`radiance --kld-record DIR --kld-corpus FILE` writes the reference's next-token NLL for every
corpus position into DIR/rows.f32; this script reads that column and prints its mean and the
number of positions, the same two numbers `r97train loss` prints for the same token sequences.
"""
import argparse
import array
import json
import math
import os
import sys

FORMATS = {
    "radiance-kld-1": 2,       # rows.f32: NLL, entropy
    "radiance-kld-top-1": 3,   # rows.f32: NLL, entropy, tail mass
}


def read_rows(path, n_rows, width):
    with open(path, "rb") as f:
        raw = f.read()
    expected = n_rows * width * 4
    if len(raw) != expected:
        raise SystemExit(f"{path}: {len(raw)} bytes, expected {expected} for {n_rows} rows")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record_dir", help="the directory radiance wrote with --kld-record")
    args = parser.parse_args(argv)

    manifest_path = os.path.join(args.record_dir, "kld.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    form = manifest.get("format")
    if form not in FORMATS:
        raise SystemExit(f"{manifest_path}: unknown format {form!r}")
    width = FORMATS[form]
    n_rows = int(manifest["rows"])
    if n_rows <= 0:
        raise SystemExit(f"{manifest_path}: the record has no scored positions")

    values = read_rows(os.path.join(args.record_dir, "rows.f32"), n_rows, width)
    nll = values[0::width]
    for v in nll:
        if not math.isfinite(v):
            raise SystemExit(f"{manifest_path}: a row has a non-finite NLL")

    print(f"loss: {math.fsum(nll) / n_rows:.12g}")
    print(f"scored: {n_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
