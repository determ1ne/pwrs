# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .d2Imis_dV2 import d2Imis_dV2
from .d2Imis_dVdSg import d2Imis_dVdSg
from .idx_gen import GEN_BUS, PG, QG
from .makeSbus import makeSbus, makeSbus_value


def opf_current_balance_hess(x, lambda_, mpc, Ybus, mpopt, nargout=1):
    """Return Hessian of AC OPF current balance constraints.

    Forms the Hessian of the Lagrangian contribution from the current balance
    equality constraints, using the multipliers in ``lambda_`` and the
    current OPF state ``x``.

    Parameters
    ----------
    x : sequence of array_like
        OPF state blocks. The layout matches MATPOWER: ``(Va, Vm, Pg, Qg)``
        in polar form or ``(Vr, Vi, Pg, Qg)`` in cartesian form.
    lambda_ : array_like
        Multipliers for the stacked real and imaginary current balance
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
        Hessian block for the current balance constraint contribution to the
        OPF Lagrangian.
    """
    baseMVA = mpc["baseMVA"]
    bus = mpc["bus"]
    gen = mpc["gen"].copy()
    if mpopt["opf"]["v_cartesian"]:
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
    Cg = sparse.csc_matrix((np.ones(ng), (gen[:, GEN_BUS - 1].astype(int) - 1, np.arange(ng))), shape=(nb, ng))

    gen[:, PG - 1] = Pg * baseMVA
    gen[:, QG - 1] = Qg * baseMVA
    Sbus = np.asarray(makeSbus_value(baseMVA, bus, gen)).reshape(-1)

    Gr11, Gr12, Gr21, Gr22 = d2Imis_dV2(Sbus, Ybus, V, lamP, mpopt["opf"]["v_cartesian"], nargout=4)
    Gi11, Gi12, Gi21, Gi22 = d2Imis_dV2(Sbus, Ybus, V, lamQ, mpopt["opf"]["v_cartesian"], nargout=4)
    Gr_sv = d2Imis_dVdSg(Cg, V, lamP, mpopt["opf"]["v_cartesian"], nargout=1)
    Gi_sv = d2Imis_dVdSg(Cg, V, lamQ, mpopt["opf"]["v_cartesian"], nargout=1)

    d2G = sparse.bmat(
        [
            [
                np.real(sparse.bmat([[Gr11, Gr12], [Gr21, Gr22]], format="csc"))
                + np.imag(sparse.bmat([[Gi11, Gi12], [Gi21, Gi22]], format="csc")),
                np.real(Gr_sv.T) + np.imag(Gi_sv.T),
            ],
            [np.real(Gr_sv) + np.imag(Gi_sv), sparse.csc_matrix((2 * ng, 2 * ng))],
        ],
        format="csc",
    )

    outputs = (d2G,)
    return outputs[:nargout] if nargout > 1 else d2G
