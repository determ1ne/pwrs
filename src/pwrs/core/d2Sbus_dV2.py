# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


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

    diaglam = sparse.diags(lam, offsets=0, shape=(n, n), format="csc")
    if vcart:
        E = diaglam @ Ybus.conjugate()
        F = E + E.T
        G = 1j * (E - E.T)

        G11 = F
        G21 = G
        G12 = G21.T
        G22 = G11
    else:
        Ibus = Ybus @ V
        diagV = sparse.diags(V, offsets=0, shape=(n, n), format="csc")

        A = sparse.diags(lam * V, offsets=0, shape=(n, n), format="csc")
        B = Ybus @ diagV
        C = A @ B.conjugate()
        D = Ybus.conjugate().T @ diagV
        E = diagV.conjugate() @ (D @ diaglam - sparse.diags(D @ lam, offsets=0, shape=(n, n), format="csc"))
        F = C - A @ sparse.diags(Ibus.conjugate(), offsets=0, shape=(n, n), format="csc")
        G = sparse.diags(np.ones(n) / np.abs(V), offsets=0, shape=(n, n), format="csc")

        G11 = E + F
        G21 = 1j * G @ (E - F)
        G12 = G21.T
        G22 = G @ (C + C.T) @ G

    outputs = (G11, G12, G21, G22)
    return outputs[:nargout] if nargout > 1 else G11
