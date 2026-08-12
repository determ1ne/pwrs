# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
from scipy import sparse


def _normalize_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=object if isinstance(value, list) else None)


def _pad_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    ndim = max(len(shape_a), len(shape_b), 2)
    sa = shape_a + (1,) * (ndim - len(shape_a))
    sb = shape_b + (1,) * (ndim - len(shape_b))
    return tuple(max(a, b) for a, b in zip(sa, sb))


def set_reorder(A, B, idx, dim):
    """Insert reordered slices along a specified dimension.

    Mirrors MATPOWER's ``set_reorder`` helper by writing ``B`` into ``A`` at
    the one-based indices ``idx`` along the requested dimension, expanding the
    target array shape when necessary.

    Parameters
    ----------
    A : array_like
        Destination array-like value.
    B : array_like
        Source values to insert.
    idx : array_like
        One-based destination indices.
    dim : int
        One-based dimension along which to assign.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray
        Updated array with ``B`` inserted into ``A``.
    """
    axis = int(np.asarray(dim).reshape(-1)[0]) - 1
    sel = np.asarray(idx, dtype=np.int64).reshape(-1) - 1

    if axis < 0:
        raise ValueError("set_reorder: dim must be positive")
    if sparse.issparse(A):
        if axis not in {0, 1}:
            raise ValueError("set_reorder: sparse matrices only support dimensions 1 and 2")
        source = B if sparse.issparse(B) else sparse.csc_matrix(np.asarray(B))
        source = sparse.csc_matrix(source)
        source_shape = source.shape
        assert source_shape is not None
        if source_shape[axis] != sel.size:
            raise ValueError("set_reorder: indexed dimension of B must match idx")
        target_shape = list(A.shape)
        target_shape[axis] = max(target_shape[axis], int(sel.max()) + 1 if sel.size else 0)
        other_axis = 1 - axis
        target_shape[other_axis] = max(target_shape[other_axis], source_shape[other_axis])
        target = sparse.lil_matrix(tuple(target_shape), dtype=np.result_type(A.dtype, source.dtype))
        target[: A.shape[0], : A.shape[1]] = A
        if axis == 0:
            target[sel, : source_shape[1]] = source
        else:
            target[: source_shape[0], sel] = source
        return target.tocsc()

    arr_a = _normalize_array(A)
    arr_b = np.asarray(B.toarray()) if sparse.issparse(B) else _normalize_array(B)

    if arr_a.ndim == 0:
        return arr_a

    target_shape_list = list(_pad_shape(arr_a.shape, arr_b.shape))
    if axis >= len(target_shape_list):
        target_shape_list.extend([1] * (axis + 1 - len(target_shape_list)))
    target_shape_list[axis] = max(
        target_shape_list[axis], int(sel.max()) + 1 if sel.size else 0
    )
    target_shape = tuple(target_shape_list)
    if arr_a.shape != target_shape:
        if arr_a.dtype == object:
            padded = np.empty(target_shape, dtype=object)
            padded.fill(None)
        else:
            padded = np.zeros(target_shape, dtype=arr_a.dtype)
        arr_a_expanded = arr_a.reshape(arr_a.shape + (1,) * (len(target_shape) - arr_a.ndim))
        slices_a = tuple(slice(0, n) for n in arr_a_expanded.shape)
        padded[slices_a] = arr_a_expanded
        arr_a = padded

    ndim = arr_a.ndim
    arr_b = arr_b.reshape(arr_b.shape + (1,) * (ndim - arr_b.ndim))
    sb = arr_b.shape
    index = []
    for k in range(ndim):
        if k == axis:
            index.append(sel)
        else:
            if arr_a.shape[k] == sb[k]:
                index.append(slice(None))
            else:
                index.append(slice(0, sb[k]))
    arr_a[tuple(index)] = arr_b
    return arr_a
