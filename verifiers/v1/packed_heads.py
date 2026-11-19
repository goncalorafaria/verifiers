"""Pack sampler rows without expanding MC32 draws into Python objects.

Empty rows align prompts/tool observations with sampled completion rows. The
mode tag distinguishes Monte Carlo draws (duplicates allowed) from top-k rows.
"""

import numpy as np


def encode(counts, values, dtype, mode=b"topk"):
    if mode not in (b"topk", b"mc32"):
        raise ValueError("Unknown sampler mode")
    return {
        "mode": mode,
        "counts": np.asarray(counts, dtype=np.int32).tobytes(),
        "values": np.asarray(values, dtype=dtype).tobytes(),
    }


def arrays(rows, dtype):
    counts = np.frombuffer(rows["counts"], dtype=np.int32)
    values = np.frombuffer(rows["values"], dtype=dtype)
    if (counts < 0).any() or counts.sum(dtype=np.int64) != values.size:
        raise ValueError("Malformed packed sampler rows")
    return counts, values


def pack(rows, dtype, n=0):
    if rows is None:
        return encode(np.zeros(n, dtype=np.int32), [], dtype)
    if isinstance(rows, dict):
        arrays(rows, dtype)
        return rows
    counts = np.fromiter((len(row) for row in rows), dtype=np.int32, count=len(rows))
    values = (
        np.concatenate([np.asarray(row, dtype=dtype) for row in rows])
        if counts.sum()
        else []
    )
    return encode(counts, values, dtype)


def concat(parts):
    modes = {p.get("mode", b"topk") for p in parts if len(p["values"])}
    if len(modes) > 1:
        raise ValueError("Cannot mix top-k and MC32 samples")
    mode = next(iter(modes), parts[0].get("mode", b"topk") if parts else b"topk")
    return {
        "mode": mode,
        "counts": b"".join(p["counts"] for p in parts),
        "values": b"".join(p["values"] for p in parts),
    }


def align(rows, mask, dtype):
    mask = np.asarray(mask, dtype=np.bool_)
    if rows is None:
        return pack(None, dtype, len(mask))
    rows = pack(rows, dtype)
    counts, values = arrays(rows, dtype)
    if len(counts) != int(mask.sum()):
        raise ValueError("Sampler head rows must match sampled node tokens")
    aligned = np.zeros(len(mask), dtype=np.int32)
    aligned[mask] = counts
    return encode(aligned, values, dtype, rows.get("mode", b"topk"))
