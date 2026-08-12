# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
from scipy import sparse

from .get_reorder import get_reorder


def _state(value: Any) -> str:
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


def _dim_size(value: Any, dim: int) -> int:
    axis = dim - 1
    if sparse.issparse(value):
        if axis > 1:
            raise ValueError("e2i_data: sparse matrices only support dimensions 1 and 2")
        return value.shape[axis]
    arr = np.asarray(value, dtype=object if isinstance(value, list) else None)
    if arr.ndim == 0:
        return 1
    if axis >= arr.ndim:
        return 1
    return arr.shape[axis]


def _concat(parts: list[Any], dim: int) -> Any:
    if not parts:
        return np.array([])
    first = parts[0]
    if sparse.issparse(first):
        return sparse.vstack(parts) if dim == 1 else sparse.hstack(parts)
    return np.concatenate(parts, axis=dim - 1)


def e2i_data(mpc, val, ordering, dim=1):
    """Reorder arbitrary data from external to internal indexing.

    Uses the ordering metadata stored by ``ext2int`` to reorder a value along
    a specified dimension according to a MATPOWER ordering descriptor such as
    ``'bus'``, ``'gen'``, or a composite ordering list.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct in internal order, with ``order`` metadata.
    val : Any
        Data to reorder.
    ordering : str or sequence
        MATPOWER ordering descriptor.
    dim : int, optional
        One-based dimension along which to reorder.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    Any
        Reordered value in internal ordering.
    """
    if "order" not in mpc:
        raise ValueError(
            "e2i_data: mpc does not have the 'order' field required to convert from external to internal numbering."
        )
    o = mpc["order"]
    if _state(o["state"]) != "i":
        raise ValueError("e2i_data: mpc does not have internal ordering data available, call ext2int first")
    dim = int(np.asarray(dim).reshape(-1)[0])

    if isinstance(ordering, str):
        if ordering == "gen":
            idx = np.asarray(o[ordering]["status"]["on"]).reshape(-1)[
                np.asarray(o[ordering]["i2e"], dtype=np.int64).reshape(-1) - 1
            ]
        else:
            idx = np.asarray(o[ordering]["status"]["on"]).reshape(-1)
        return get_reorder(val, idx, dim)

    base = 0
    parts: list[Any] = []
    for item in ordering:
        n = _dim_size(o["ext"][item], 1)
        v = get_reorder(val, np.arange(base + 1, base + n + 1), dim)
        parts.append(e2i_data(mpc, v, item, dim))
        base += n
    n = _dim_size(val, dim)
    if n > base:
        parts.append(get_reorder(val, np.arange(base + 1, n + 1), dim))
    return _concat(parts, dim)
