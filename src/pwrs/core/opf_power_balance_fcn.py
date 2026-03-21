# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .dSbus_dV import dSbus_dV
from .idx_gen import GEN_BUS, PG, QG
from .makeSbus import makeSbus


def opf_power_balance_fcn(x, mpc, Ybus, mpopt, nargout=1):
    """Evaluate AC OPF power balance constraints and Jacobian.

    Computes the nonlinear equality constraints enforcing real and reactive
    power balance at each bus for the current OPF state. When requested, it
    also returns the corresponding Jacobian with respect to the OPF decision
    variables.

    Parameters
    ----------
    x : sequence of array_like
        OPF state blocks. The layout matches MATPOWER: ``(Va, Vm, Pg, Qg)``
        in polar form or ``(Vr, Vi, Pg, Qg)`` in cartesian form.
    mpc : dict
        Internal MATPOWER case struct.
    Ybus : sparse matrix
        Bus admittance matrix.
    mpopt : dict
        MATPOWER options struct.
    nargout : int, optional
        MATLAB compatibility flag controlling whether the Jacobian is
        returned.

    Returns
    -------
    numpy.ndarray or tuple
        Constraint vector ``g`` containing stacked real and reactive power
        mismatches, and optionally its Jacobian ``dg``.
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

    gen[:, PG - 1] = Pg * baseMVA
    gen[:, QG - 1] = Qg * baseMVA

    if mpopt.opf.v_cartesian:
        Sbus = makeSbus(baseMVA, bus, gen, nargout=1)
    else:
        Sbus = makeSbus(baseMVA, bus, gen, mpopt, Vm, nargout=1)
    Sbus = np.asarray(Sbus).reshape(-1)

    mis = V * np.conjugate(Ybus @ V) - Sbus
    g = np.r_[np.real(mis), np.imag(mis)]

    if nargout > 1:
        dSbus_dV1, dSbus_dV2 = dSbus_dV(Ybus, V, mpopt.opf.v_cartesian)
        neg_Cg = sparse.csc_matrix((-np.ones(ng), (gen[:, GEN_BUS - 1].astype(int) - 1, np.arange(ng))), shape=(nb, ng))
        if not mpopt.opf.v_cartesian:
            _, neg_dSd_dVm = makeSbus(baseMVA, bus, gen, mpopt, Vm, nargout=2)
            dSbus_dV2 = dSbus_dV2 - neg_dSd_dVm
        dg = sparse.vstack(
            [
                sparse.hstack(
                    [np.real(sparse.hstack([dSbus_dV1, dSbus_dV2])), neg_Cg, sparse.csc_matrix((nb, ng))], format="csc"
                ),
                sparse.hstack(
                    [np.imag(sparse.hstack([dSbus_dV1, dSbus_dV2])), sparse.csc_matrix((nb, ng)), neg_Cg], format="csc"
                ),
            ],
            format="csc",
        )
        outputs = (g, dg)
    else:
        outputs = (g,)
    return outputs[:nargout] if nargout > 1 else g
