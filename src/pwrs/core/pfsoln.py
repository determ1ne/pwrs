# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_brch import BR_STATUS, F_BUS, PF, PT, QF, QT, T_BUS
from .idx_bus import BUS_TYPE, PQ, VA, VM
from .idx_gen import GEN_BUS, GEN_STATUS, PG, QG, QMAX, QMIN
from .mpoption import mpoption
from .total_load import total_load_p, total_load_pq


def pfsoln(baseMVA, bus0, gen0, branch0, Ybus, Yf, Yt, V, ref, pv, pq, mpopt=None):
    """Update bus, gen and branch matrices to match a power flow solution.

    Parameters
    ----------
    baseMVA : float
        System base MVA.
    bus0 : array_like
        Original MATPOWER bus matrix.
    gen0 : array_like
        Original MATPOWER generator matrix.
    branch0 : array_like
        Original MATPOWER branch matrix.
    Ybus : array_like or sparse matrix
        Full system admittance matrix.
    Yf : array_like or sparse matrix
        Branch admittance matrix for currents injected at the from end.
    Yt : array_like or sparse matrix
        Branch admittance matrix for currents injected at the to end.
    V : array_like
        Solved complex bus voltage vector.
    ref : array_like
        Reference bus index vector.
    pv : array_like
        PV bus index vector.
    pq : array_like
        PQ bus index vector.
    mpopt : dict, optional
        MATPOWER options dict.
    Returns
    -------
    tuple
        ``(bus, gen, branch)`` updated to reflect the solved voltage,
        generator dispatch and branch flow results.
    """
    if mpopt is None:
        mpopt = mpoption()

    bus = bus0.copy()
    gen = gen0.copy()
    branch = branch0.copy()
    V = V.reshape(-1)

    bus[:, VM - 1] = np.abs(V)
    bus[:, VA - 1] = np.angle(V) * 180.0 / np.pi

    gen_bus = gen[:, GEN_BUS - 1].astype(int) - 1
    bus_type = bus[gen_bus, BUS_TYPE - 1]
    on = np.flatnonzero((gen[:, GEN_STATUS - 1] > 0) & (bus_type != PQ))
    off = np.flatnonzero(gen[:, GEN_STATUS - 1] <= 0)
    gbus = gen[on, GEN_BUS - 1].astype(int)
    gbus_idx = gbus - 1

    Sbus = V[gbus_idx] * np.conj(Ybus[gbus_idx, :] @ V)

    gen[off, QG - 1] = np.zeros(off.size)
    if on.size:
        _, Qd_gbus = total_load_pq(bus[gbus_idx, :], None, "bus", None, mpopt)
        Qd_gbus = np.ravel(Qd_gbus)
        gen[on, QG - 1] = np.imag(Sbus) * baseMVA + Qd_gbus

    if on.size > 1:
        nb = bus.shape[0]
        ngon = on.size
        Cg = sparse.csc_matrix((np.ones(ngon), (np.arange(ngon), gbus_idx)), shape=(ngon, nb))

        ngg = np.ravel(Cg @ np.ravel(Cg.sum(axis=0)))
        gen[on, QG - 1] = gen[on, QG - 1] / ngg

        Qmin = gen[on, QMIN - 1].copy()
        Qmax = gen[on, QMAX - 1].copy()
        M = np.abs(gen[on, QG - 1])
        finite_qmax = ~np.isinf(Qmax)
        finite_qmin = ~np.isinf(Qmin)
        M[finite_qmax] = M[finite_qmax] + np.abs(Qmax[finite_qmax])
        M[finite_qmin] = M[finite_qmin] + np.abs(Qmin[finite_qmin])
        M = np.ravel(Cg @ (Cg.T @ M))

        Qmin[Qmin == np.inf] = M[Qmin == np.inf]
        Qmin[Qmin == -np.inf] = -M[Qmin == -np.inf]
        Qmax[Qmax == np.inf] = M[Qmax == np.inf]
        Qmax[Qmax == -np.inf] = -M[Qmax == -np.inf]

        Cmin = sparse.csc_matrix((Qmin, (np.arange(ngon), gbus_idx)), shape=(ngon, nb))
        Cmax = sparse.csc_matrix((Qmax, (np.arange(ngon), gbus_idx)), shape=(ngon, nb))
        Qg_tot = np.ravel(Cg.T @ gen[on, QG - 1])
        Qg_min = np.ravel(Cmin.sum(axis=0))
        Qg_max = np.ravel(Cmax.sum(axis=0))
        gen[on, QG - 1] = Qmin + (
            np.ravel(Cg @ ((Qg_tot - Qg_min) / (Qg_max - Qg_min + np.finfo(float).eps))) * (Qmax - Qmin)
        )

        ig = np.flatnonzero(np.abs(np.ravel(Cg @ (Qg_min - Qg_max))) < 10 * np.finfo(float).eps)
        if ig.size:
            ib = np.flatnonzero(np.ravel(Cg[ig, :].sum(axis=0)))
            counts = np.ravel(Cg[:, ib].sum(axis=0))
            mis = np.zeros(nb)
            mis[ib] = (Qg_tot[ib] - Qg_min[ib]) / counts
            gen[on[ig], QG - 1] = Qmin[ig] + np.ravel(Cg[ig, :] @ mis)

    for ref_k in ref:
        refgen = np.flatnonzero(gbus == ref_k)
        if refgen.size == 0:
            continue
        Pd_refk = np.ravel(total_load_p(bus[ref_k - 1 : ref_k, :], None, "bus", None, mpopt))[0]
        first = on[refgen[0]]
        gen[first, PG - 1] = np.real(Sbus[refgen[0]]) * baseMVA + Pd_refk
        if refgen.size > 1:
            gen[first, PG - 1] = gen[first, PG - 1] - np.sum(gen[on[refgen[1:]], PG - 1])

    out = np.flatnonzero(branch[:, BR_STATUS - 1] == 0)
    br = np.flatnonzero(branch[:, BR_STATUS - 1] != 0)
    if br.size:
        f_idx = branch[br, F_BUS - 1].astype(int) - 1
        t_idx = branch[br, T_BUS - 1].astype(int) - 1
        Sf = V[f_idx] * np.conj(Yf[br, :] @ V) * baseMVA
        St = V[t_idx] * np.conj(Yt[br, :] @ V) * baseMVA
        flow_columns = np.array([PF - 1, QF - 1, PT - 1, QT - 1], dtype=int)
        branch[np.ix_(br, flow_columns)] = np.column_stack(
            [np.real(Sf), np.imag(Sf), np.real(St), np.imag(St)]
        )
    if out.size:
        branch[np.ix_(out, np.array([PF - 1, QF - 1, PT - 1, QT - 1], dtype=int))] = 0.0

    return bus, gen, branch
