# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .dImis_dV import dImis_dV
from .idx_gen import GEN_BUS, PG, QG
from .makeSbus import makeSbus


def opf_current_balance_fcn(x, mpc, Ybus, mpopt, nargout=1):
    """Evaluate AC OPF current balance constraints and Jacobian.

    Computes the nonlinear equality constraints enforcing real and imaginary
    current balance at each bus for the current OPF state. When requested, it
    also returns the Jacobian with respect to the OPF decision variables.

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
        Constraint vector ``g`` containing stacked real and imaginary current
        mismatches, and optionally its Jacobian ``dg``.
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

    gen[:, PG - 1] = Pg * baseMVA
    gen[:, QG - 1] = Qg * baseMVA

    Sbus = np.asarray(makeSbus(baseMVA, bus, gen, nargout=1)).reshape(-1)
    mis = Ybus @ V - np.conjugate(Sbus / V)
    g = np.r_[np.real(mis), np.imag(mis)]

    if nargout > 1:
        Cg = sparse.csc_matrix((np.ones(ng), (gen[:, GEN_BUS - 1].astype(int) - 1, np.arange(ng))), shape=(nb, ng))
        InvConjV = sparse.diags(1 / np.conjugate(V), offsets=0, shape=(nb, nb), format="csc")
        dImis_dPg = -InvConjV @ Cg
        dImis_dQg = -1j * dImis_dPg
        dImis_dV1, dImis_dV2 = dImis_dV(Sbus, Ybus, V, mpopt["opf"]["v_cartesian"])
        dg = sparse.vstack(
            [
                np.real(sparse.hstack([dImis_dV1, dImis_dV2, dImis_dPg, dImis_dQg], format="csc")),
                np.imag(sparse.hstack([dImis_dV1, dImis_dV2, dImis_dPg, dImis_dQg], format="csc")),
            ],
            format="csc",
        )
        outputs = (g, dg)
    else:
        outputs = (g,)
    return outputs[:nargout] if nargout > 1 else g
