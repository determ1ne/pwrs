# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def d2Sbr_dV2(Cbr, Ybr, V, mu, vcart=0, nargout=1):
    """Return 2nd derivatives of complex branch power flows.

    Computes the Hessian blocks of ``mu' * Sbr(V)`` for a branch-end power
    flow expression defined by connection matrix ``Cbr`` and admittance
    matrix ``Ybr``.

    Parameters
    ----------
    Cbr : sparse matrix
        Branch connection matrix selecting the branch-end terminal buses.
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
    nl = len(mu)
    nb = len(V)

    A = Ybr.conjugate().T @ sparse.diags(mu, offsets=0, shape=(nl, nl), format="csc") @ Cbr
    if vcart:
        H11 = A + A.T
        H12 = 1j * (A - A.T)
        H21 = -H12
        H22 = H11
    else:
        diagV = sparse.diags(V, offsets=0, shape=(nb, nb), format="csc")

        B = diagV.conjugate() @ A @ diagV
        D = sparse.diags((A @ V) * np.conjugate(V), offsets=0, shape=(nb, nb), format="csc")
        E = sparse.diags((A.T @ np.conjugate(V)) * V, offsets=0, shape=(nb, nb), format="csc")
        F = B + B.T
        G = sparse.diags(np.ones(nb) / np.abs(V), offsets=0, shape=(nb, nb), format="csc")

        H11 = F - D - E
        H21 = 1j * G @ (B - B.T - D + E)
        H12 = H21.T
        H22 = G @ F @ G

    outputs = (H11, H12, H21, H22)
    return outputs[:nargout] if nargout > 1 else H11
