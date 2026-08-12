# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig, add_matrices, as_csc_matrix, matrix_imag, matrix_real
from .d2Imis_dV2 import d2Imis_dV2_full
from .d2Imis_dVdSg import d2Imis_dVdSg
from .idx_gen import GEN_BUS, PG, QG
from .makeSbus import makeSbus_value


def opf_current_balance_hess(x, lambda_, mpc, Ybus, mpopt: MatpowerConfig, nargout=1):
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
    mpopt : MatpowerConfig
        Typed MATPOWER options configuration.
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
    Cg = sparse.csc_matrix((np.ones(ng), (gen[:, GEN_BUS].astype(int) - 1, np.arange(ng))), shape=(nb, ng))

    gen[:, PG] = Pg * baseMVA
    gen[:, QG] = Qg * baseMVA
    Sbus = np.asarray(makeSbus_value(baseMVA, bus, gen)).reshape(-1)

    Gr11, Gr12, Gr21, Gr22 = d2Imis_dV2_full(Sbus, Ybus, V, lamP, mpopt.opf.v_cartesian)
    Gi11, Gi12, Gi21, Gi22 = d2Imis_dV2_full(Sbus, Ybus, V, lamQ, mpopt.opf.v_cartesian)
    Gr_sv = d2Imis_dVdSg(Cg, V, lamP, mpopt.opf.v_cartesian)
    Gi_sv = d2Imis_dVdSg(Cg, V, lamQ, mpopt.opf.v_cartesian)
    if isinstance(Gr_sv, tuple) or isinstance(Gi_sv, tuple):
        raise TypeError("opf_current_balance_hess: mixed derivative shim returned multiple outputs")

    Gr_vv = as_csc_matrix(sparse.bmat([[Gr11, Gr12], [Gr21, Gr22]], format="csc"))
    Gi_vv = as_csc_matrix(sparse.bmat([[Gi11, Gi12], [Gi21, Gi22]], format="csc"))
    Gr_sv = as_csc_matrix(Gr_sv)
    Gi_sv = as_csc_matrix(Gi_sv)

    d2G = sparse.bmat(
        [
            [
                add_matrices(matrix_real(Gr_vv), matrix_imag(Gi_vv)),
                add_matrices(matrix_real(Gr_sv.T), matrix_imag(Gi_sv.T)),
            ],
            [
                add_matrices(matrix_real(Gr_sv), matrix_imag(Gi_sv)),
                sparse.csc_matrix((2 * ng, 2 * ng)),
            ],
        ],
        format="csc",
    )

    outputs = (d2G,)
    return outputs[:nargout] if nargout > 1 else d2G
