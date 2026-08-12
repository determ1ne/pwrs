# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def d2Ibr_dV2(Ybr, V, mu, vcart=0, nargout=1):
    """Return 2nd derivatives of complex branch currents.

    Computes the Hessian blocks of ``mu' * Ibr(V)`` for current flow
    expressions formed from the branch admittance matrix ``Ybr``.

    Parameters
    ----------
    Ybr : sparse matrix
        Branch admittance matrix for the corresponding terminal.
    V : array_like
        Complex bus voltage vector.
    mu : array_like
        Multiplier vector for the selected branch current equations.
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

    if vcart:
        H11 = sparse.csc_matrix((nb, nb))
        H12 = H11
        H21 = H11
        H22 = H11
    else:
        diagInvVm = sparse.diags(np.ones(nb) / np.abs(V), offsets=0, shape=(nb, nb), format="csc")
        H11 = sparse.diags(-(Ybr.T @ mu) * V, offsets=0, shape=(nb, nb), format="csc")
        H21 = -1j * H11 @ diagInvVm
        H12 = H21
        H22 = sparse.csc_matrix((nb, nb))

    outputs = (H11, H12, H21, H22)
    return outputs[:nargout] if nargout > 1 else H11


def d2Ibr_dV2_full(Ybr, V, mu, vcart=0):
    """Return all four current-flow Hessian blocks."""
    return d2Ibr_dV2(Ybr, V, mu, vcart, nargout=4)
