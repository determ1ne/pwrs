# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np


def as_column(value) -> np.ndarray:
    return np.asarray(value).reshape(-1, 1)


def build_e2i_map(external_ids, *, start=1) -> dict[int, int]:
    ext = np.asarray(external_ids, dtype=int).reshape(-1)
    return {int(bus): idx for idx, bus in enumerate(ext, start=start)}


def map_e2i(mapping: dict[int, int], values) -> np.ndarray:
    vals = np.asarray(values, dtype=int).reshape(-1)
    return np.fromiter((mapping[int(v)] for v in vals), dtype=int, count=vals.size)


def as_scalar(value, cast=float):
    arr = np.asarray(value)
    if arr.size == 0:
        return cast(0)
    return cast(arr.reshape(-1)[0])


def get_nested(data, path, default=None):
    cur = data
    for key in path:
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        elif hasattr(cur, key):
            cur = getattr(cur, key)
        else:
            return default
    return cur
