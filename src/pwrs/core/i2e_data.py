# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
from scipy import sparse

from .get_reorder import get_reorder
from .set_reorder import set_reorder


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
            raise ValueError("i2e_data: sparse matrices only support dimensions 1 and 2")
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


def i2e_data(mpc, val, oldval, ordering, dim=1):
    """Reorder arbitrary data from internal to external indexing.

    Uses the ordering metadata stored by ``ext2int`` to map internal-order
    data back into the external layout, optionally filling into a previously
    saved external-order template ``oldval``.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct in internal order, with ``order`` metadata.
    val : Any
        Internal-order data to map back to external ordering.
    oldval : Any
        External-order template value to receive the reordered data.
    ordering : str or sequence
        MATPOWER ordering descriptor.
    dim : int, optional
        One-based dimension along which to reorder.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    Any
        Value mapped back to external ordering.
    """
    if "order" not in mpc:
        raise ValueError(
            "i2e_data: mpc does not have the 'order' field required for conversion back to external numbering."
        )
    o = mpc["order"]
    if _state(o["state"]) != "i":
        raise ValueError("i2e_data: mpc does not appear to be in internal order")
    dim = int(np.asarray(dim).reshape(-1)[0])

    if isinstance(ordering, str):
        if ordering == "gen":
            v = get_reorder(val, o[ordering]["e2i"], dim)
        else:
            v = val
        return set_reorder(oldval, v, o[ordering]["status"]["on"], dim)

    be = 0
    bi = 0
    parts: list[Any] = []
    for item in ordering:
        ne = _dim_size(o["ext"][item], 1)
        ni = _dim_size(mpc[item], 1)
        v = get_reorder(val, np.arange(bi + 1, bi + ni + 1), dim)
        oldv = get_reorder(oldval, np.arange(be + 1, be + ne + 1), dim)
        parts.append(i2e_data(mpc, v, oldv, item, dim))
        be += ne
        bi += ni
    ni = _dim_size(val, dim)
    if ni > bi:
        parts.append(get_reorder(val, np.arange(bi + 1, ni + 1), dim))
    return _concat(parts, dim)
