# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def _row_scale_(A, v):
    """Return ``diag(v) * A``"""
    v = np.asarray(v).reshape(-1)
    A.data = A.data * v[A.indices]
    return A



def _col_scale_(A, v):
    """Return ``A * diag(v)``"""
    v = np.asarray(v).reshape(-1)
    A.data = A.data * np.repeat(v, np.diff(A.indptr))
    return A


def _build_A_from_connection(cbr_idx, Ybr, mu):
    W = Ybr.getH().tocsr() 
    mu = np.asarray(mu).reshape(-1)
    W.data *= mu[W.indices]
    cbr_idx = np.asarray(cbr_idx, dtype=np.int32)
    W.indices = cbr_idx[W.indices]
    n_bus = W.shape[0]
    W._shape = (n_bus, n_bus)
    W.sum_duplicates()
    return W.tocsc()


def d2Sbr_dV2(Cbr, Ybr, V, mu, vcart=0, nargout=1):
    """Return 2nd derivatives of complex branch power flows.

    Computes the Hessian blocks of ``mu' * Sbr(V)`` for a branch-end power
    flow expression defined by connection matrix ``Cbr`` and admittance
    matrix ``Ybr``.

    Parameters
    ----------
    Cbr : array_like
        Zero-based endpoint bus indices for the constrained branch rows.
    Ybr : sparse matrix
        Branch admittance matrix for the corresponding terminal.
    V : array_like
        Complex bus voltage vector.
    mu : array_like
        Multiplier vector for the selected branch-end flow equations.
    vcart : int or bool, optional
        Voltage coordinate flag. Use polar coordinates when false and
        cartesian coordinates when true.
    nargout : int, optional
        MATLAB compatibility flag controlling how many Hessian blocks are
        returned.

    Returns
    -------
    scipy.sparse.csc_matrix or tuple of scipy.sparse.csc_matrix
        Hessian blocks ``(H11, H12, H21, H22)`` for the selected voltage
        coordinate system, or just ``H11`` when ``nargout == 1``.
    """
    if vcart is None:
        vcart = 0

    V = np.asarray(V).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    nb = len(V)
    Ybr = Ybr.tocsc(copy=False)

    A = _build_A_from_connection(Cbr, Ybr, mu)
    if vcart:
        H11 = A + A.T
        H12 = 1j * (A - A.T)
        H21 = -H12
        H22 = H11
    else:
        B = _row_scale_(_col_scale_(A.copy(), V), np.conjugate(V))
        D = sparse.diags((A @ V) * np.conjugate(V), offsets=0, shape=(nb, nb), format="csc")
        E = sparse.diags((A.T @ np.conjugate(V)) * V, offsets=0, shape=(nb, nb), format="csc")
        F = B + B.T
        invVm = np.ones(nb) / np.abs(V)

        H11 = F - D - E
        H21 = 1j * _row_scale_(B - B.T - D + E, invVm)
        H12 = H21.T
        H22 = _row_scale_(_col_scale_(F, invVm), invVm)

    outputs = (H11, H12, H21, H22)
    return outputs[:nargout] if nargout > 1 else H11
