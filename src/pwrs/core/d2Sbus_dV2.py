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


def _row_scale(A, v):
    """Return ``diag(v) * A`` without explicitly forming the diagonal."""
    v = np.asarray(v).reshape(-1)
    if sparse.isspmatrix_csc(A):
        out = A.copy()
        out.data = out.data * v[out.indices]
        return out
    A = A.tocsc(copy=False)
    out = A.copy()
    out.data = out.data * v[out.indices]
    return out


def _col_scale(A, v):
    """Return ``A * diag(v)`` without explicitly forming the diagonal."""
    v = np.asarray(v).reshape(-1)
    if sparse.isspmatrix_csc(A):
        out = A.copy()
        out.data = out.data * np.repeat(v, np.diff(out.indptr))
        return out
    A = A.tocsc(copy=False)
    out = A.copy()
    out.data = out.data * np.repeat(v, np.diff(out.indptr))
    return out


def d2Sbus_dV2(Ybus, V, lam, vcart=0, nargout=1):
    """Return 2nd derivatives of the power balance equations.

    Computes the Hessian blocks of ``lam' * Sbus(V)`` with respect to the
    voltage state variables. The coordinate representation follows MATPOWER:
    polar when ``vcart`` is false and cartesian when ``vcart`` is true.

    Parameters
    ----------
    Ybus : sparse matrix
        Bus admittance matrix.
    V : array_like
        Complex bus voltage vector.
    lam : array_like
        Multiplier vector for the bus power balance equations.
    vcart : int or bool, optional
        Voltage coordinate flag. Use polar coordinates when false and
        cartesian coordinates when true.
    nargout : int, optional
        MATLAB compatibility flag controlling how many Hessian blocks are
        returned.

    Returns
    -------
    scipy.sparse.csc_matrix or tuple of scipy.sparse.csc_matrix
        Hessian blocks ``(G11, G12, G21, G22)`` for the selected voltage
        coordinate system, or just ``G11`` when ``nargout == 1``.
    """
    if vcart is None:
        vcart = 0

    V = np.asarray(V).reshape(-1)
    lam = np.asarray(lam).reshape(-1)
    n = len(V)
    Ybus = Ybus.tocsc(copy=False)

    if vcart:
        E = _row_scale(Ybus.conjugate(), lam)
        F = E + E.T
        G = 1j * (E - E.T)

        G11 = F
        G21 = G
        G12 = G21.T
        G22 = G11
    else:
        Ibus = Ybus @ V
        B = _col_scale(Ybus, V)
        C = _row_scale(B.conjugate(), lam * V)
        D = _col_scale(Ybus.conjugate().T, V)
        E2 = sparse.diags(V.conjugate() * (D @ lam), offsets=0, shape=(n, n), format="csc")
        E = _row_scale_(D, V.conjugate()) @ sparse.diags(lam, offsets=0, shape=(n, n), format="csc")
        E = E - E2
        F = C - sparse.diags(lam * V * Ibus.conjugate(), offsets=0, shape=(n, n), format="csc")
        inv_abs_V = np.ones(n) / np.abs(V)

        G11 = E + F
        G21 = 1j * _row_scale_(E - F, inv_abs_V)
        G12 = G21.T
        G22 = _col_scale_(_row_scale_(C + C.T, inv_abs_V), inv_abs_V)

    outputs = (G11, G12, G21, G22)
    return outputs[:nargout] if nargout > 1 else G11
