# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def d2Imis_dV2(Sbus, Ybus, V, lam, vcart=0, nargout=1):
    """Return 2nd derivatives of current mismatch equations.

    Computes the Hessian blocks of ``lam' * Imis(V, Sg)`` with respect to the
    voltage state variables. The mismatch is represented in polar or
    cartesian voltage coordinates according to ``vcart``.

    Parameters
    ----------
    Sbus : array_like
        Complex bus injection vector.
    Ybus : sparse matrix
        Bus admittance matrix.
    V : array_like
        Complex bus voltage vector.
    lam : array_like
        Multiplier vector for the current balance equations.
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

    Sbus = np.asarray(Sbus).reshape(-1)
    V = np.asarray(V).reshape(-1)
    lam = np.asarray(lam).reshape(-1)
    nb = len(V)

    if vcart:
        C = 2 * lam * np.conjugate(Sbus / (V**3))

        G22 = sparse.diags(C, offsets=0, shape=(nb, nb), format="csc")
        G11 = -G22
        G12 = 1j * G22
        G21 = G12
    else:
        absV = np.abs(V)
        diagV = sparse.diags(V, offsets=0, shape=(nb, nb), format="csc")
        diagV1 = sparse.diags(1 / V, offsets=0, shape=(nb, nb), format="csc")
        diagVmV1 = sparse.diags(1 / (V * absV), offsets=0, shape=(nb, nb), format="csc")
        diagVmV2 = sparse.diags(1 / (V * (absV**2)), offsets=0, shape=(nb, nb), format="csc")
        diagLamS = sparse.diags(lam * Sbus, offsets=0, shape=(nb, nb), format="csc")
        diagYlam = sparse.diags(Ybus.T @ lam, offsets=0, shape=(nb, nb), format="csc")
        diagE = sparse.diags(V / absV, offsets=0, shape=(nb, nb), format="csc")

        G11 = -diagYlam @ diagV + (diagLamS @ diagV1).conjugate()
        G22 = -2 * (diagLamS @ diagVmV2).conjugate()
        G21 = 1j * (diagYlam @ diagE + (diagLamS @ diagVmV1).conjugate())
        G12 = G21.T

    outputs = (G11, G12, G21, G22)
    return outputs[:nargout] if nargout > 1 else G11
