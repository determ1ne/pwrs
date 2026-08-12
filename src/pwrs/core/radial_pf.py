# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig
from .calc_v_i_sum import calc_v_i_sum
from .calc_v_pq_sum import calc_v_pq_sum
from .calc_v_y_sum import calc_v_y_sum
from .idx_brch import BR_B, BR_R, BR_X, F_BUS, PF, PT, QF, QT, T_BUS, TAP
from .idx_bus import BS, BUS_I, GS, PD, QD, VA, VM
from .idx_gen import GEN_BUS, PG, QG, VG
from .make_vcorr import make_vcorr
from .make_zpv import make_zpv
from .mpoption import get_zip_weights, mpoption
from .order_radial import order_radial


def _get_zip_weights(mpopt):
    pw, qw = get_zip_weights(mpopt)
    return np.asarray(pw).reshape(-1), np.asarray(qw).reshape(-1)


def _radial_init(Sd, pv, Pg, mpopt):
    tol = float(mpopt.pf.tol)
    iter_max = int(mpopt.pf.radial.max_it)
    vcorr = int(mpopt.pf.radial.vcorr) == 1
    Sd = Sd.copy()
    if pv.size:
        Sd[pv] = Sd[pv] - Pg
    pw, qw = _get_zip_weights(mpopt)
    Sdz = np.real(Sd) * pw[2] + 1j * np.imag(Sd) * qw[2]
    Sdi = np.real(Sd) * pw[1] + 1j * np.imag(Sd) * qw[1]
    Sdp = np.real(Sd) * pw[0] + 1j * np.imag(Sd) * qw[0]
    return tol, iter_max, vcorr, Sdz, Sdi, Sdp


def _calc_v_pq_sum(Vslack, nb, nl, f, Zb, Ybf, Ybt, Yd, Sd, pv, Pg, Vg, mpopt):
    tol, iter_max, vcorr, Sdz, Sdi, Sdp = _radial_init(Sd, pv, Pg, mpopt)
    V = Vslack * np.ones(nb, dtype=complex)
    Vold = V.copy()
    iter_count = 0
    success = 0.0

    f = np.r_[0, f]
    Zb = np.r_[0.0 + 0.0j, Zb]
    nl = nl + 1

    if pv.size:
        Zpv = make_zpv(pv, nb, nl, f, Zb, Yd)
        Bpv = np.linalg.inv(np.imag(Zpv))
    else:
        Bpv = np.zeros((0, 0))
    Qpv = np.zeros(pv.size)
    St = np.zeros(nl, dtype=complex)
    Sf = np.zeros(nl, dtype=complex)

    while success == 0.0 and iter_count < iter_max:
        iter_count += 1
        Vm = np.abs(V)
        S = Sdp + Sdi * Vm + Sdz * Vm**2 + np.conj(Yd) * Vm**2
        St = S.copy()
        Sf = St.copy()
        for k in range(nl - 1, 0, -1):
            i = f[k]
            Sf[k] = St[k] + Zb[k] * np.abs(St[k] / V[k]) ** 2
            St[i] = St[i] + Sf[k]
        for k in range(1, nl):
            i = f[k]
            V[k] = V[i] - Zb[k] * np.conj(Sf[k] / V[i])
        DU = np.abs(V - Vold)
        DU[np.isnan(DU)] = np.inf
        if np.max(DU) > tol:
            Vold = V.copy()
            if pv.size:
                DE = (Vg / np.abs(V[pv]) - 1.0) * np.real(V[pv])
                DD = Bpv @ DE
                if vcorr:
                    DC = DD * np.imag(V[pv]) / np.real(V[pv])
                    V = V + make_vcorr(DC + 1j * DD, pv, nb, nl, f, Zb)
                DQ = DD * np.abs(V[pv]) ** 2 / np.real(V[pv])
                Qpv = Qpv + DQ
                Sdp[pv] = Sdp[pv] - 1j * DQ
        else:
            success = 1.0

    Sslack = St[0]
    Sf = Sf[1:]
    St = St[1:]
    f = f[1:]
    Sf = Sf + np.conj(Ybf) * np.abs(V[f]) ** 2
    St = St - np.conj(Ybt) * np.abs(V[1:]) ** 2
    return V, Qpv.reshape(-1, 1), Sf, St, Sslack, float(iter_count), success


def _calc_v_i_sum(Vslack, nb, nl, f, Zb, Ybf, Ybt, Yd, Sd, pv, Pg, Vg, mpopt):
    tol, iter_max, vcorr, Sdz, Sdi, Sdp = _radial_init(Sd, pv, Pg, mpopt)
    V = Vslack * np.ones(nb, dtype=complex)
    Vold = V.copy()
    iter_count = 0
    success = 0.0

    f = np.r_[0, f]
    Zb = np.r_[0.0 + 0.0j, Zb]
    nl = nl + 1

    if pv.size:
        Zpv = make_zpv(pv, nb, nl, f, Zb, Yd)
        Bpv = np.linalg.inv(np.imag(Zpv))
    else:
        Bpv = np.zeros((0, 0))
    Qpv = np.zeros(pv.size)
    I = np.zeros(nl, dtype=complex)

    while success == 0.0 and iter_count < iter_max:
        iter_count += 1
        Vm = np.abs(V)
        S = Sdp + Sdi * Vm + Sdz * Vm**2 + np.conj(Yd) * Vm**2
        I = np.conj(S / V)
        for k in range(nl - 1, 0, -1):
            i = f[k]
            I[i] = I[i] + I[k]
        for k in range(1, nl):
            i = f[k]
            V[k] = V[i] - Zb[k] * I[k]
        DU = np.abs(V - Vold)
        DU[np.isnan(DU)] = np.inf
        if np.max(DU) > tol:
            Vold = V.copy()
            if pv.size:
                DE = (Vg / np.abs(V[pv]) - 1.0) * np.real(V[pv])
                DD = Bpv @ DE
                if vcorr:
                    DC = DD * np.imag(V[pv]) / np.real(V[pv])
                    V = V + make_vcorr(DC + 1j * DD, pv, nb, nl, f, Zb)
                DQ = DD * np.abs(V[pv]) ** 2 / np.real(V[pv])
                Qpv = Qpv + DQ
                Sdp[pv] = Sdp[pv] - 1j * DQ
        else:
            success = 1.0

    Sslack = V[0] * np.conj(I[0])
    I = I[1:]
    f = f[1:]
    Sf = V[f] * np.conj(I) + np.conj(Ybf) * np.abs(V[f]) ** 2
    St = V[1:] * np.conj(I) - np.conj(Ybt) * np.abs(V[1:]) ** 2
    return V, Qpv.reshape(-1, 1), Sf, St, Sslack, float(iter_count), success


def _calc_v_y_sum(Vslack, nb, nl, f, Zb, Ybf, Ybt, Yd, Sd, pv, Pg, Vg, mpopt):
    tol, iter_max, vcorr, Sdz, Sdi, Sdp = _radial_init(Sd, pv, Pg, mpopt)
    V = Vslack * np.ones(nb, dtype=complex)
    Vold = V.copy()
    iter_count = 0
    success = 0.0

    f = np.r_[0, f]
    Zb = np.r_[0.0 + 0.0j, Zb]
    nl = nl + 1

    if pv.size:
        Zpv = make_zpv(pv, nb, nl, f, Zb, Yd)
        Bpv = np.linalg.inv(np.imag(Zpv))
    else:
        Bpv = np.zeros((0, 0))
    Qpv = np.zeros(pv.size)
    Je = np.zeros(nl, dtype=complex)

    Ye = np.conj(Sdz) + Yd
    D = np.zeros(nl, dtype=complex)
    for k in range(nl - 1, 0, -1):
        D[k] = 1.0 / (1.0 + Zb[k] * Ye[k])
        i = f[k]
        Ye[i] = Ye[i] + D[k] * Ye[k]

    while success == 0.0 and iter_count < iter_max:
        iter_count += 1
        Vm = np.abs(V)
        S = Sdp + Sdi * Vm
        Je = np.conj(S / V)
        for k in range(nl - 1, 0, -1):
            i = f[k]
            Je[i] = Je[i] + D[k] * Je[k]
        for k in range(1, nl):
            i = f[k]
            V[k] = D[k] * (V[i] - Zb[k] * Je[k])
        DU = np.abs(V - Vold)
        DU[np.isnan(DU)] = np.inf
        if np.max(DU) > tol:
            Vold = V.copy()
            if pv.size:
                DE = (Vg / np.abs(V[pv]) - 1.0) * np.real(V[pv])
                DD = Bpv @ DE
                if vcorr:
                    DC = DD * np.imag(V[pv]) / np.real(V[pv])
                    V = V + make_vcorr(DC + 1j * DD, pv, nb, nl, f, Zb)
                DQ = DD * np.abs(V[pv]) ** 2 / np.real(V[pv])
                Qpv = Qpv + DQ
                Sdp[pv] = Sdp[pv] - 1j * DQ
        else:
            success = 1.0

    Sslack = V[0] * np.conj(Je[0]) + np.conj(Ye[0]) * np.abs(V[0]) ** 2
    f = f[1:]
    Zb = Zb[1:]
    I = (V[f] - V[1:]) / Zb
    Sf = V[f] * np.conj(I) + np.conj(Ybf) * np.abs(V[f]) ** 2
    St = V[1:] * np.conj(I) - np.conj(Ybt) * np.abs(V[1:]) ** 2
    return V, Qpv.reshape(-1, 1), Sf, St, Sslack, float(iter_count), success


def radial_pf(mpc, mpopt: MatpowerConfig | dict | None = None):
    """Solve a power flow using a backward-forward sweep method.

    Parameters
    ----------
    mpc : dict
        MATPOWER case dict using internal bus numbering.
    mpopt : MatpowerConfig or dict, optional
        MATPOWER options configuration. A legacy dictionary is accepted at
        the compatibility boundary. It can be used to select the radial solver
        algorithm, output options, termination tolerances and related
        settings.

    Returns
    -------
    tuple
        ``(mpc, success, iterations)`` where ``mpc`` is the updated results
        dict, ``success`` is a success flag and ``iterations`` is the number
        of sweep iterations performed.

    Notes
    -----
    This routine supports only radial networks. It orders the case for the
    sweep solution, dispatches to the selected radial algorithm, then maps
    the solved results back to the original bus and branch ordering.

    See Also
    --------
    caseformat, loadcase, mpoption
    """
    if mpopt is None:
        mpopt = mpoption()
    elif not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)

    mpc = order_radial(mpc)
    loop = np.asarray(mpc.get("loop", np.array([]))).reshape(-1)
    if loop.size:
        raise ValueError(f"radial_pf: power flow algorithm {mpopt.pf.alg} can only handle radial networks.")

    branch = np.atleast_2d(np.array(mpc["branch"], dtype=float, copy=True))
    bus = np.atleast_2d(np.array(mpc["bus"], dtype=float, copy=True))
    gen = np.atleast_2d(np.array(mpc["gen"], dtype=float, copy=True))
    if branch.shape[1] <= QT:
        branch = np.concatenate([branch, np.zeros((branch.shape[0], QT + 1 - branch.shape[1]))], axis=1)
    baseMVA = mpc["baseMVA"]

    f = branch[:, F_BUS].astype(int) - 1
    t = branch[:, T_BUS].astype(int) - 1
    Zb = branch[:, BR_R] + 1j * branch[:, BR_X]
    Yb = 1j * branch[:, BR_B]
    Sd = bus[:, PD] + 1j * bus[:, QD]
    Ysh = bus[:, GS] + 1j * bus[:, BS]
    nl = branch.shape[0]
    nb = bus.shape[0]
    Sd = Sd / baseMVA
    Ysh = Ysh / baseMVA
    tap = np.ones(nl)
    nonzero_tap = np.flatnonzero(branch[:, TAP])
    tap[nonzero_tap] = branch[nonzero_tap, TAP]
    Ybf = Yb / 2.0 + (1.0 / tap) * (1.0 / tap - 1.0) / Zb
    Ybt = Yb / 2.0 + (1.0 - 1.0 / tap) / Zb
    br_reverse = np.asarray(mpc["br_reverse"]).reshape(-1).astype(bool)
    Ybf_rev = Ybf.copy()
    Ybt_rev = Ybt.copy()
    Ybf[br_reverse] = Ybt_rev[br_reverse]
    Ybt[br_reverse] = Ybf_rev[br_reverse]
    Zb = Zb * tap
    Yd = Ysh + (
        sparse.csc_matrix((Ybf, (f, f)), shape=(nb, nb)) + sparse.csc_matrix((Ybt, (t, t)), shape=(nb, nb))
    ) @ np.ones(nb)

    pv = gen[1:, GEN_BUS].astype(int) - 1
    Pg = gen[1:, PG] / baseMVA
    Vg = gen[1:, VG]

    Vslack = gen[0, VG]
    alg = mpopt.pf.alg.upper()
    if alg == "PQSUM":
        V, Qpv, Sf, St, Sslack, iterations, success = calc_v_pq_sum(
            Vslack, nb, nl, f + 1, Zb, Ybf, Ybt, Yd, Sd, pv + 1, Pg, Vg, mpopt
        )
    elif alg == "ISUM":
        V, Qpv, Sf, St, Sslack, iterations, success = calc_v_i_sum(
            Vslack, nb, nl, f + 1, Zb, Ybf, Ybt, Yd, Sd, pv + 1, Pg, Vg, mpopt
        )
    elif alg == "YSUM":
        V, Qpv, Sf, St, Sslack, iterations, success = calc_v_y_sum(
            Vslack, nb, nl, f + 1, Zb, Ybf, Ybt, Yd, Sd, pv + 1, Pg, Vg, mpopt
        )
    else:
        raise ValueError(f"radial_pf: unsupported algorithm {alg}")

    mpc["success"] = success
    V = np.asarray(V).reshape(-1)
    Qpv = np.asarray(Qpv).reshape(-1)
    Sf = np.asarray(Sf).reshape(-1)
    St = np.asarray(St).reshape(-1)
    bus[:, VM] = np.abs(V)
    bus[:, VA] = np.angle(V) / np.pi * 180.0
    branch[:, PF] = np.real(Sf) * baseMVA
    branch[:, QF] = np.imag(Sf) * baseMVA
    branch[:, PT] = -np.real(St) * baseMVA
    branch[:, QT] = -np.imag(St) * baseMVA
    pf_rev = branch[br_reverse, PF].copy()
    pt_rev = branch[br_reverse, PT].copy()
    qf_rev = branch[br_reverse, QF].copy()
    qt_rev = branch[br_reverse, QT].copy()
    branch[br_reverse, PF] = pt_rev
    branch[br_reverse, PT] = pf_rev
    branch[br_reverse, QF] = qt_rev
    branch[br_reverse, QT] = qf_rev
    gen[0, PG] = np.real(Sslack) * baseMVA
    gen[0, QG] = np.imag(Sslack) * baseMVA
    if pv.size:
        gen[1:, QG] = np.asarray(Qpv).reshape(-1) * baseMVA

    bus_order_inv = np.asarray(mpc["bus_order_inv"]).reshape(-1).astype(int)
    branch_order_inv = np.asarray(mpc["branch_order_inv"]).reshape(-1).astype(int)
    bus_order = np.asarray(mpc["bus_order"]).reshape(-1).astype(int)
    _branch_order = np.asarray(mpc["branch_order"]).reshape(-1).astype(int)

    bus = bus[bus_order_inv[1:] - 1, :]
    bus[:, BUS_I] = bus_order[bus[:, BUS_I].astype(int) - 1]
    f = branch[:, F_BUS].astype(int)
    t = branch[:, T_BUS].astype(int)
    tmp = f[br_reverse].copy()
    f[br_reverse] = t[br_reverse]
    t[br_reverse] = tmp
    branch[:, [F_BUS, T_BUS]] = np.column_stack([bus_order[f - 1], bus_order[t - 1]])
    branch = branch[branch_order_inv[1:] - 1, :]
    gen[:, GEN_BUS] = bus_order[gen[:, GEN_BUS].astype(int) - 1]

    mpc["bus"] = bus
    mpc["branch"] = branch
    mpc["gen"] = gen
    return mpc, success, iterations
