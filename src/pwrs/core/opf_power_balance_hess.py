# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .d2Sbus_dV2 import d2Sbus_dV2
from .makeSdzip import makeSdzip


def opf_power_balance_hess(x, lambda_, mpc, Ybus, mpopt, nargout=1):
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
    mpopt : dict
        MATPOWER options struct.
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

    Gp11, Gp12, Gp21, Gp22 = d2Sbus_dV2(Ybus, V, lamP, mpopt.opf.v_cartesian, nargout=4)
    Gq11, Gq12, Gq21, Gq22 = d2Sbus_dV2(Ybus, V, lamQ, mpopt.opf.v_cartesian, nargout=4)

    if not mpopt.opf.v_cartesian:
        diaglam = sparse.diags(lamP, offsets=0, shape=(nb, nb), format="csc")
        Sd = makeSdzip(mpc["baseMVA"], mpc["bus"], mpopt)
        diagSdz = sparse.diags(np.asarray(Sd["z"]).reshape(-1), offsets=0, shape=(nb, nb), format="csc")
        Gp22 = Gp22 + 2 * diaglam @ diagSdz

    top_left = np.real(sparse.bmat([[Gp11, Gp12], [Gp21, Gp22]], format="csc")) + np.imag(
        sparse.bmat([[Gq11, Gq12], [Gq21, Gq22]], format="csc")
    )
    d2G = sparse.vstack(
        [
            sparse.hstack([top_left, sparse.csc_matrix((2 * nb, 2 * ng))], format="csc"),
            sparse.csc_matrix((2 * ng, 2 * nb + 2 * ng)),
        ],
        format="csc",
    )

    outputs = (d2G,)
    return outputs[:nargout] if nargout > 1 else d2G
