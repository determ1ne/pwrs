# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def _row_scale(A, v):
    """Return ``diag(v) * A``"""
    v = np.asarray(v).reshape(-1)
    out = A.copy()
    out.data = out.data * v[out.indices]
    return out


def _as_csc(A):
    if sparse.isspmatrix_csc(A):
        return A
    return sparse.csc_matrix(A)


def d2Abr_dV2(d2F_dV2, dF_dV1, dF_dV2, F, V, mu, nargout=1):
    """Return 2nd derivatives of squared branch flow magnitudes.

    Given first- and second-derivative data for a branch flow quantity
    ``F(V)``, computes the Hessian blocks of the corresponding squared
    magnitude expression used in OPF flow-limit constraints.

    Parameters
    ----------
    d2F_dV2 : callable
        Function returning the four Hessian blocks of ``mu' * F(V)``.
    dF_dV1 : sparse matrix
        First derivative of ``F`` with respect to the first voltage block.
    dF_dV2 : sparse matrix
        First derivative of ``F`` with respect to the second voltage block.
    F : array_like
        Complex branch flow vector.
    V : array_like
        Complex bus voltage vector.
    mu : array_like
        Multiplier vector for the branch flow limit terms.
    nargout : int, optional
        MATLAB compatibility flag controlling how many Hessian blocks are
        returned.

    Returns
    -------
    scipy.sparse.csc_matrix or tuple of scipy.sparse.csc_matrix
        Real Hessian blocks ``(H11, H12, H21, H22)`` for the squared flow
        expression, or just ``H11`` when ``nargout == 1``.
    """
    mu = np.asarray(mu).reshape(-1)
    dF_dV1 = dF_dV1.tocsc(copy=False)
    dF_dV2 = dF_dV2.tocsc(copy=False)

    F11, F12, F21, F22 = d2F_dV2(V, np.conjugate(np.asarray(F).reshape(-1)) * mu)
    G1 = _row_scale(np.conjugate(dF_dV1), mu)
    G2 = _row_scale(np.conjugate(dF_dV2), mu)

    H11 = 2 * np.real(F11 + dF_dV1.T @ G1)
    H21 = 2 * np.real(F21 + dF_dV2.T @ G1)
    H12 = 2 * np.real(F12 + dF_dV1.T @ G2)
    H22 = 2 * np.real(F22 + dF_dV2.T @ G2)

    outputs = (_as_csc(H11), _as_csc(H12), _as_csc(H21), _as_csc(H22))
    return outputs[:nargout] if nargout > 1 else outputs[0]


def d2Abr_dV2_full(d2F_dV2, dF_dV1, dF_dV2, F, V, mu):
    """Return all four Hessian blocks."""
    return d2Abr_dV2(d2F_dV2, dF_dV1, dF_dV2, F, V, mu, nargout=4)
