#!/usr/bin/env python3
"""Fetch UltraChat 200k and convert it to plain chat JSONL.

Downloads HuggingFaceH4/ultrachat_200k (MIT licence, not gated) with the `hf`
CLI, then converts the parquet shards to chat JSONL using a small pure-python
parquet reader (stdlib only; no pyarrow/datasets needed):

    data/ultrachat/train.jsonl   <- split train_sft
    data/ultrachat/heldout.jsonl <- split test_sft

Each output line is one conversation: {"messages": [{"role": "...",
"content": "..."}, ...]} with roles limited to user/assistant/system.

The script is idempotent: when an output file already exists the download and
conversion for it are skipped (delete data/ultrachat/ to re-fetch).
"""
import fnmatch
import glob
import json
import mmap
import os
import struct
import subprocess
import sys

REPO = "HuggingFaceH4/ultrachat_200k"
OUT_DIR = os.path.join("data", "ultrachat")
RAW_DIR = os.path.join(OUT_DIR, "_raw")
SPLITS = [
    ("data/ultrachat/train.jsonl", "data/train_sft-*.parquet"),
    ("data/ultrachat/heldout.jsonl", "data/test_sft-*.parquet"),
]
ALLOWED_ROLES = {"user", "assistant", "system"}


class ParquetError(Exception):
    pass


# ---------------------------------------------------------------- thrift compact

class Reader:
    """Compact-protocol reader over a bytes-like object."""

    def __init__(self, buf, pos=0):
        self.buf, self.pos = buf, pos

    def varint(self):
        val = shift = 0
        while True:
            byte = self.buf[self.pos]
            self.pos += 1
            val |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return val
            shift += 7

    def zigzag(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def binary(self):
        n = self.varint()
        v = self.buf[self.pos:self.pos + n]
        self.pos += n
        return v


def parse_struct(r):
    """Thrift compact struct -> {field_id: (type_id, value)}. Stops at 0x00."""
    out = {}
    fid = 0
    while True:
        head = r.buf[r.pos]
        if head == 0:  # STOP: a bare byte, never carries a field id
            r.pos += 1
            return out
        delta, ttype = head >> 4, head & 0xF
        fid = r.varint() if delta == 0 else fid + delta
        r.pos += 1
        if ttype in (1, 2):
            out[fid] = (ttype, ttype == 1)
        elif ttype in (4, 5, 6):
            out[fid] = (ttype, r.zigzag())
        elif ttype == 8:
            out[fid] = (ttype, r.binary())
        elif ttype == 12:
            out[fid] = (ttype, parse_struct(r))
        elif ttype in (9, 10):
            lhead = r.buf[r.pos]
            r.pos += 1
            size, etype = lhead >> 4, lhead & 0xF
            if size == 15:
                size = r.varint()
            if etype == 12:
                out[fid] = (ttype, [parse_struct(r) for _ in range(size)])
            elif etype in (4, 5, 6):
                out[fid] = (ttype, [(etype, r.zigzag()) for _ in range(size)])
            elif etype == 8:
                out[fid] = (ttype, [(etype, r.binary()) for _ in range(size)])
            elif etype in (1, 2):
                out[fid] = (ttype, [(etype, None)] * size)
            else:
                raise ParquetError(f"unsupported list element type {etype}")
        else:
            raise ParquetError(f"unsupported thrift type {ttype}")


def parse_struct_at(buf, pos):
    r = Reader(buf, pos)
    return parse_struct(r), r.pos


# ---------------------------------------------------------------- snappy

def snappy_decompress(src):
    pos = 0
    want = shift = 0
    while True:
        byte = src[pos]
        pos += 1
        want |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    out = bytearray()
    n = len(src)
    while pos < n:
        tag = src[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:  # literal
            length = (tag >> 2) + 1
            if length > 60:
                extra = length - 60
                length = int.from_bytes(src[pos:pos + extra], "little") + 1
                pos += extra
            out += src[pos:pos + length]
            pos += length
        else:  # backward copy
            if kind == 1:
                length = ((tag >> 2) & 7) + 4
                offset = ((tag >> 5) << 8) | src[pos]
                pos += 1
            elif kind == 2:
                length = (tag >> 2) + 1
                offset = int.from_bytes(src[pos:pos + 2], "little")
                pos += 2
            else:
                length = (tag >> 2) + 1
                offset = int.from_bytes(src[pos:pos + 4], "little")
                pos += 4
            start = len(out) - offset
            if start < 0:
                raise ParquetError("bad snappy back-reference")
            if offset >= length:
                out += out[start:start + length]
            else:  # overlapping copy: tile the pattern
                pat = bytes(out[start:])
                full, rem = divmod(length, offset)
                out += pat * full + pat[:rem]
    if len(out) != want:
        raise ParquetError(f"snappy produced {len(out)} bytes, expected {want}")
    return bytes(out)


# ---------------------------------------------------------------- RLE / bit-packed

def read_varint(buf, pos):
    val = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        val |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return val, pos
        shift += 7


def rle_hybrid(data, width, count):
    """RLE / bit-packed hybrid run data (no length prefix, no width byte)."""
    out = []
    pos, n = 0, len(data)
    nbytes = (width + 7) // 8
    mask = (1 << width) - 1
    while len(out) < count and pos < n:
        header, pos = read_varint(data, pos)
        if header & 1:  # bit-packed: header>>1 groups of 8 values
            groups = header >> 1
            for k in range(groups * 8):
                if len(out) >= count:
                    break
                off = pos * 8 + k * width
                val = 0
                for j in range(width):
                    byte = data[(off + j) >> 3]
                    val |= ((byte >> ((off + j) & 7)) & 1) << j
                out.append(val)
            pos += (groups * 8 * width + 7) // 8
        else:  # rle run: header>>1 copies of one fixed-width value
            run = header >> 1
            val = int.from_bytes(data[pos:pos + nbytes], "little") & mask
            pos += nbytes
            out.extend([val] * min(run, count - len(out)))
    if len(out) < count:
        raise ParquetError(f"level stream short: {len(out)} < {count}")
    return out[:count]


def read_file_metadata(buf):
    """Returns (schema elements, row groups, num_rows) from the file footer."""
    if buf[:4] != b"PAR1" or buf[-4:] != b"PAR1":
        raise ParquetError("not a parquet file")
    meta_len = struct.unpack("<I", buf[-8:-4])[0]
    meta, _ = parse_struct_at(buf, len(buf) - 8 - meta_len)
    if 2 not in meta or meta[2][0] != 9:
        raise ParquetError("no schema in footer")
    schema = []
    for el in meta[2][1]:
        schema.append({
            "name": el[4][1].decode("utf-8"),
            "rep": el[3][1] if 3 in el else 0,  # 0 req, 1 opt, 2 rep
            "num_children": el[5][1] if 5 in el else 0,
        })
    if 4 not in meta or meta[4][0] != 9:
        raise ParquetError("no row groups in footer")
    return schema, meta[4][1], meta[3][1]


def leaf_depths(schema):
    """[(path, max_def, max_rep)] for every leaf, from repetition types.

    Element 0 of the schema is the root; every optional node on a path adds a
    definition level, every repeated node adds a definition AND repetition
    level.
    """
    leaves = []

    def dfs(d, r, idx=0, path=()):
        node = schema[idx]
        idx += 1
        if idx > 1:
            path = path + (node["name"],)
        if idx > 1:
            if node["rep"] == 2:
                d += 1
                r += 1
            elif node["rep"] == 1:
                d += 1
        if node["num_children"]:
            for _ in range(node["num_children"]):
                idx = dfs(d, r, idx, path)
            return idx
        leaves.append((path, d, r))
        return idx

    dfs(0, 0)
    return leaves


def leaf_spec(leaves, path):
    for lp, d, r in leaves:
        if lp == path:
            return d, r
    raise ParquetError(f"missing leaf {path}")


# ---------------------------------------------------------------- page decoding

def decode_column(buf, cm, max_def, max_rep):
    """Decode a column chunk; returns (defs, reps, values as bytes)."""
    num_values = cm[5][1]
    codec = cm[4][1]
    offsets = [cm[9][1]]
    if 11 in cm:
        offsets.append(cm[11][1])
    pos = min(offsets)
    dictionary = None
    defs, reps, values = [], [], []
    while len(defs) < num_values:
        header, end = parse_struct_at(buf, pos)
        page_type, comp_size = header[1][1], header[3][1]
        page = buf[end:end + comp_size]
        pos = end + comp_size
        raw = snappy_decompress(page) if codec == 1 else bytes(page)
        if page_type == 2:  # dictionary page (PLAIN byte arrays)
            count = header[7][1][1][1]
            dictionary = []
            p = 0
            for _ in range(count):
                length = int.from_bytes(raw[p:p + 4], "little")
                p += 4
                dictionary.append(raw[p:p + length])
                p += length
            continue
        if page_type != 0:
            raise ParquetError(f"unexpected page type {page_type}")
        page_hdr = header[5][1]
        count = page_hdr[1][1]
        encoding = page_hdr[2][1]
        p = 0
        if max_rep:
            length = int.from_bytes(raw[p:p + 4], "little")
            p += 4
            reps += rle_hybrid(raw[p:p + length], max_rep, count)
            p += length
        if max_def:
            length = int.from_bytes(raw[p:p + 4], "little")
            p += 4
            defs += rle_hybrid(raw[p:p + length], max_def, count)
            p += length
        else:
            defs += [0] * count
        present = sum(1 for d in defs[-count:] if d == max_def) if max_def else count
        if encoding == 8:  # RLE_DICTIONARY indices
            if dictionary is None:
                raise ParquetError("dictionary-encoded page without dictionary")
            width = raw[p]
            p += 1
            indices = rle_hybrid(raw[p:], width, present)
            values += [dictionary[i] for i in indices]
        elif encoding == 0:  # PLAIN byte arrays
            for _ in range(present):
                length = int.from_bytes(raw[p:p + 4], "little")
                p += 4
                values.append(raw[p:p + length])
                p += length
        else:
            raise ParquetError(f"unsupported encoding {encoding}")
    return defs, reps, values


# ---------------------------------------------------------------- conversations

def decode_conversations(buf):
    """Decode one shard into a list of {"messages": [{"role","content"},...]}."""
    schema, row_groups, _num_rows = read_file_metadata(buf)
    leaves = leaf_depths(schema)
    conversations = []
    for row_group in row_groups:
        chunks = {}
        for c in row_group[1][1]:
            cm = c[3][1]
            path = tuple(f[1].decode("utf-8") for f in cm[3][1])
            chunks[path] = cm

        def column(last):
            path = next(p for p in chunks if p[-1] == last)
            return chunks[path], leaf_spec(leaves, path)

        # scalar columns (one entry per row, optional)
        def scalar(last):
            chunk, (md, _mr) = column(last)
            defs, _reps, values = decode_column(buf, chunk, md, 0)
            vals = iter(values)
            return [next(vals) if d == md else None for d in defs]

        prompts = scalar("prompt")

        cchunk, (cmd, cmr) = column("content")
        rchunk, (rmd, rmr) = column("role")
        if (cmd, cmr) != (rmd, rmr):
            raise ParquetError("content/role column structure differs")
        ddefs, dreps, cvals = decode_column(buf, cchunk, cmd, cmr)
        rdefs, rreps, rvals = decode_column(buf, rchunk, rmd, rmr)
        if (ddefs, dreps) != (rdefs, rreps):
            raise ParquetError("content/role def levels differ")

        cvals, rvals = iter(cvals), iter(rvals)
        exists = cmd - 1  # def level at which the message group is present
        rows = []
        cur = None
        for d, rp in zip(ddefs, dreps):
            if rp == 0:
                cur = []
                rows.append(cur)
            msg_level = cmd - 1  # message dict exists (leaf may still be null)
            if d >= msg_level:
                entry = {"role": None, "content": None}
                if d > msg_level:
                    entry["role"] = next(rvals).decode("utf-8")
                    entry["content"] = next(cvals).decode("utf-8")
                cur.append(entry)
        if len(rows) != len(prompts):
            raise ParquetError(
                f"row mismatch: {len(rows)} conversations, {len(prompts)} prompts")
        conversations.extend(rows)
    return conversations


def validate(messages):
    """Fail on an empty messages list or an unknown role."""
    if not isinstance(messages, list) or not messages:
        raise ParquetError("conversation with an empty messages list")
    for msg in messages:
        if not isinstance(msg, dict):
            raise ParquetError(f"message is not an object: {msg!r}")
        if msg.get("role") not in ALLOWED_ROLES:
            raise ParquetError(f"unknown role {msg.get('role')!r}")
        if not isinstance(msg.get("content"), str):
            raise ParquetError(f"message with non-string content "
                               f"for role {msg.get('role')!r}")


# ---------------------------------------------------------------- main

def convert_shard(shard_path, out):
    with open(shard_path, "rb") as fh:
        buf = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
    convos = decode_conversations(buf)
    for msgs in convos:
        validate(msgs)
        out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    buf.close()
    return len(convos)


def count_lines(path):
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [(out, pat) for out, pat in SPLITS if not os.path.exists(out)]
    if todo:
        os.makedirs(RAW_DIR, exist_ok=True)
    for out_name, pattern in todo:
        cmd = ["hf", "download", REPO, "--repo-type", "dataset",
               "--include", pattern, "--local-dir", RAW_DIR]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    for out_name, pattern in SPLITS:
        if os.path.exists(out_name):
            print(f"{out_name}: {count_lines(out_name)} conversations (cached)")
            continue
        shards = sorted(glob.glob(os.path.join(RAW_DIR, pattern)))
        if not shards:
            raise ParquetError(f"no shards found for {pattern}")
        total = 0
        with open(out_name, "w", encoding="utf-8") as out:
            for shard in shards:
                n = convert_shard(shard, out)
                total += n
                print(f"  {os.path.basename(shard)}: {n} conversations",
                      flush=True)
                os.remove(shard)
        print(f"{out_name}: {total} conversations")
    try:
        os.rmdir(RAW_DIR)
    except OSError:  # still holds files of the other split
        pass


if __name__ == "__main__":
    try:
        main()
    except ParquetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
