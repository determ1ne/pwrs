# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_brch import F_BUS, T_BUS


def _row_scale(A, v):
    """Return ``diag(v) * A`` without forming the diagonal matrix."""
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
    """Return ``A * diag(v)`` without forming the diagonal matrix."""
    v = np.asarray(v).reshape(-1)
    if sparse.isspmatrix_csc(A):
        out = A.copy()
        out.data = out.data * np.repeat(v, np.diff(out.indptr))
        return out
    A = A.tocsc(copy=False)
    out = A.copy()
    out.data = out.data * np.repeat(v, np.diff(out.indptr))
    return out


def dSbr_dV(branch, Yf, Yt, V, vcart=0, *, nargout=None):
    """Compute partial derivatives of branch power flows w.r.t. voltage.

    Parameters
    ----------
    branch : ndarray
        Branch matrix.
    Yf, Yt : array_like or sparse matrix
        Branch admittance matrices for the from and to ends.
    V : array_like
        Complex bus voltage vector.
    vcart : int, optional
        Coordinate selector. ``0`` uses polar derivatives with respect to
        voltage angle and magnitude, ``1`` uses cartesian derivatives with
        respect to real and imaginary voltage parts.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple
        ``(dSf_dV1, dSf_dV2, dSt_dV1, dSt_dV2, Sf, St)`` in MATLAB-compatible
        form.
    """
    branch = np.asarray(branch)
    V = np.asarray(V).reshape(-1)

    f = np.asarray(branch[:, F_BUS - 1], dtype=int).reshape(-1) - 1
    t = np.asarray(branch[:, T_BUS - 1], dtype=int).reshape(-1) - 1
    nl = len(f)
    nb = len(V)

    Yfc = Yf.conjugate()
    Ytc = Yt.conjugate()
    Vc = V.conjugate()
    Ifc = Yfc @ Vc
    Itc = Ytc @ Vc

    if sparse.issparse(Yf):
        Yf = Yf.tocsc(copy=False)
        Yt = Yt.tocsc(copy=False)
        Yfc = Yfc.tocsc(copy=False)
        Ytc = Ytc.tocsc(copy=False)
        if not vcart:
            Vnorm = V / np.abs(V)
    else:
        Yf = np.asarray(Yf)
        Yt = np.asarray(Yt)
        Yfc = np.asarray(Yfc)
        Ytc = np.asarray(Ytc)
        diagVf = np.diag(V[f])
        diagVt = np.diag(V[t])
        diagIfc = np.diag(Ifc)
        diagItc = np.diag(Itc)
        if not vcart:
            Vnorm = V / np.abs(V)
            diagVc = np.diag(Vc)
            diagVnorm = np.diag(Vnorm)
            CVf = sparse.csc_matrix((V[f], (np.arange(nl), f)), shape=(nl, nb)).toarray()
            CVnf = sparse.csc_matrix((Vnorm[f], (np.arange(nl), f)), shape=(nl, nb)).toarray()
            CVt = sparse.csc_matrix((V[t], (np.arange(nl), t)), shape=(nl, nb)).toarray()
            CVnt = sparse.csc_matrix((Vnorm[t], (np.arange(nl), t)), shape=(nl, nb)).toarray()

    if vcart:
        if not sparse.issparse(Yf):
            Cf = sparse.csc_matrix((np.ones(nl), (np.arange(nl), f)), shape=(nl, nb)).toarray()
            Ct = sparse.csc_matrix((np.ones(nl), (np.arange(nl), t)), shape=(nl, nb)).toarray()
            Af = diagIfc @ Cf
            Bf = diagVf @ Yfc
            At = diagItc @ Ct
            Bt = diagVt @ Ytc
        else:
            Af = sparse.csc_matrix((Ifc, (np.arange(nl), f)), shape=(nl, nb))
            Bf = _row_scale(Yfc, V[f])
            At = sparse.csc_matrix((Itc, (np.arange(nl), t)), shape=(nl, nb))
            Bt = _row_scale(Ytc, V[t])
        dSf_dV1 = Af + Bf
        dSf_dV2 = 1j * (Af - Bf)
        dSt_dV1 = At + Bt
        dSt_dV2 = 1j * (At - Bt)
    else:
        if sparse.issparse(Yf):
            dSf1a = sparse.csc_matrix((Ifc * V[f], (np.arange(nl), f)), shape=(nl, nb))
            dSf2b = sparse.csc_matrix((Ifc * Vnorm[f], (np.arange(nl), f)), shape=(nl, nb))
            dSt1a = sparse.csc_matrix((Itc * V[t], (np.arange(nl), t)), shape=(nl, nb))
            dSt2b = sparse.csc_matrix((Itc * Vnorm[t], (np.arange(nl), t)), shape=(nl, nb))
            dSf_dV1 = 1j * (dSf1a - _row_scale(_col_scale(Yfc, Vc), V[f]))
            dSf_dV2 = _row_scale(_col_scale(Yf, Vnorm).conjugate(), V[f]) + dSf2b
            dSt_dV1 = 1j * (dSt1a - _row_scale(_col_scale(Ytc, Vc), V[t]))
            dSt_dV2 = _row_scale(_col_scale(Yt, Vnorm).conjugate(), V[t]) + dSt2b
        else:
            dSf_dV1 = 1j * (diagIfc @ CVf - diagVf @ Yfc @ diagVc)
            dSf_dV2 = diagVf @ (Yf @ diagVnorm).conjugate() + diagIfc @ CVnf
            dSt_dV1 = 1j * (diagItc @ CVt - diagVt @ Ytc @ diagVc)
            dSt_dV2 = diagVt @ (Yt @ diagVnorm).conjugate() + diagItc @ CVnt

    Sf = V[f] * Ifc
    St = V[t] * Itc

    return dSf_dV1, dSf_dV2, dSt_dV1, dSt_dV2, Sf, St
