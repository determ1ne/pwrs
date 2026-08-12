# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from scipy import sparse


def get_reorder(A, idx, dim):
    """Extract reordered slices along a specified dimension.

    Mirrors MATPOWER's ``get_reorder`` helper by selecting rows or columns
    from an array-like input using one-based indices along the requested
    dimension.

    Parameters
    ----------
    A : array_like or sparse matrix
        Input data to reorder.
    idx : array_like
        One-based indices to extract.
    dim : int
        One-based dimension along which to reorder.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    Any
        Reordered slice of ``A``.
    """
    axis = int(np.asarray(dim).reshape(-1)[0]) - 1
    sel = np.asarray(idx, dtype=np.int64).reshape(-1) - 1

    if sparse.issparse(A):
        if axis == 0:
            return A[sel, :]
        if axis == 1:
            return A[:, sel]
        raise ValueError("get_reorder: sparse matrices only support dimensions 1 and 2")

    arr = np.asarray(A, dtype=object if isinstance(A, list) else None)
    if arr.ndim == 0:
        return arr
    if axis >= arr.ndim:
        return arr
    return np.take(arr, sel, axis=axis)
