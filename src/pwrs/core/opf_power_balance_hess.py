# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig
from .d2Sbus_dV2 import d2Sbus_dV2_full
from .makeSdzip import makeSdzip


def _diag_sparse(v, n):
    """Return a CSC diagonal matrix with diagonal entries ``v``."""
    v = np.asarray(v).reshape(-1)
    idx = np.arange(n + 1, dtype=np.int32)
    return sparse.csc_matrix((v, idx[:-1], idx), shape=(n, n))


def opf_power_balance_hess(x, lambda_, mpc, Ybus, mpopt: MatpowerConfig, nargout=1):
    """Return Hessian of AC OPF power balance constraints.

    Forms the Hessian of the Lagrangian contribution from the power balance
    equality constraints, using the multipliers in ``lambda_`` and the
    current OPF state ``x``.

    Parameters
    ----------
    x : sequence of array_like
        OPF state blocks. The layout matches MATPOWER: ``(Va, Vm, Pg, Qg)``
        in polar form or ``(Vr, Vi, Pg, Qg)`` in cartesian form.
    lambda_ : array_like
        Multipliers for the stacked real and reactive power balance
        constraints.
    mpc : dict
        Internal MATPOWER case struct.
    Ybus : sparse matrix
        Bus admittance matrix.
    mpopt : MatpowerConfig
        Typed MATPOWER options configuration.
    nargout : int, optional
        MATLAB compatibility flag controlling how many outputs are returned.

    Returns
    -------
    scipy.sparse.csc_matrix
        Hessian block for the power balance constraint contribution to the
        OPF Lagrangian.
    """
    if mpopt.opf.v_cartesian:
        Vr, Vi, Pg, Qg = [np.asarray(v).reshape(-1) for v in x]
        V = Vr + 1j * Vi
    else:
        Va, Vm, Pg, Qg = [np.asarray(v).reshape(-1) for v in x]
        V = Vm * np.exp(1j * Va)

    nb = len(V)
    ng = len(Pg)

    lambda_ = np.asarray(lambda_).reshape(-1)
    nlam = len(lambda_) // 2
    lamP = lambda_[:nlam]
    lamQ = lambda_[nlam : nlam + nlam]

    Gp11, Gp12, Gp21, Gp22 = d2Sbus_dV2_full(Ybus, V, lamP, mpopt.opf.v_cartesian)
    Gq11, Gq12, Gq21, Gq22 = d2Sbus_dV2_full(Ybus, V, lamQ, mpopt.opf.v_cartesian)

    if not mpopt.opf.v_cartesian:
        Sd = makeSdzip(mpc["baseMVA"], mpc["bus"], mpopt)
        Gp22 = Gp22 + _diag_sparse(2 * lamP * np.asarray(Sd["z"]).reshape(-1), nb)

    H11 = np.real(Gp11) + np.imag(Gq11)
    H12 = np.real(Gp12) + np.imag(Gq12)
    H21 = np.real(Gp21) + np.imag(Gq21)
    H22 = np.real(Gp22) + np.imag(Gq22)
    top_left = sparse.bmat([[H11, H12], [H21, H22]], format="csc")
    top_right = sparse.csc_matrix((2 * nb, 2 * ng))
    bottom = sparse.csc_matrix((2 * ng, 2 * nb + 2 * ng))
    d2G = sparse.vstack(
        [
            sparse.hstack([top_left, top_right], format="csc"),
            bottom,
        ],
        format="csc",
    )

    outputs = (d2G,)
    return outputs[:nargout] if nargout > 1 else d2G
