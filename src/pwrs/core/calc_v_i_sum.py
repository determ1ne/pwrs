# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .make_vcorr import make_vcorr
from .make_zpv import make_zpv


def calc_v_i_sum(Vslack, nb, nl, f, Zb, Ybf, Ybt, Yd, Sd, pv, Pg, Vg, mpopt):
    """Solve radial power flow by the current summation method.

    Mirrors MATPOWER's ``calc_v_i_sum`` radial helper. It performs iterative
    backward/forward sweeps using branch currents, optionally applies PV-bus
    voltage correction, and returns voltages, flows, slack power, iteration
    count, and convergence status.

    Parameters
    ----------
    Vslack : complex
        Slack-bus voltage.
    nb : int
        Number of buses.
    nl : int
        Number of branches.
    f : array_like
        One-based parent-bus indices for each branch.
    Zb : array_like
        Branch series impedances.
    Ybf : array_like
        Branch shunt admittances at the from end.
    Ybt : array_like
        Branch shunt admittances at the to end.
    Yd : array_like
        Bus shunt admittances.
    Sd : array_like
        Complex bus power demands.
    pv : array_like
        One-based PV bus indices.
    Pg : array_like
        Real generation at PV buses.
    Vg : array_like
        Target voltage magnitudes at PV buses.
    mpopt : dict
        MATPOWER options struct.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    tuple
        ``(V, Qpv, Sf, St, Sslack, iterations, success)``.
    """
    tol = mpopt.pf.tol
    iter_max = mpopt.pf.radial.max_it
    vcorr = mpopt.pf.radial.vcorr == 1.0

    nb = int(np.asarray(nb).reshape(-1)[0])
    nl = int(np.asarray(nl).reshape(-1)[0])
    f = np.asarray(f, dtype=int).reshape(-1)
    Zb = np.asarray(Zb, dtype=complex).reshape(-1)
    Ybf = np.asarray(Ybf, dtype=complex).reshape(-1)
    Ybt = np.asarray(Ybt, dtype=complex).reshape(-1)
    Yd = np.asarray(Yd, dtype=complex).reshape(-1)
    Sd = np.asarray(Sd, dtype=complex).reshape(-1).copy()
    pv = np.asarray(pv, dtype=int).reshape(-1)
    Pg = np.asarray(Pg, dtype=float).reshape(-1)
    Vg = np.asarray(Vg, dtype=float).reshape(-1)

    f = f - 1
    pv = pv - 1

    Sd[pv] = Sd[pv] - Pg
    V = Vslack * np.ones(nb, dtype=complex)
    Vold = V.copy()
    iter_count = 0
    success = 0.0

    pw = mpopt.exp.sys_wide_zip_loads_pw
    qw = mpopt.exp.sys_wide_zip_loads_qw
    if pw is None:
        pw = np.array([1.0, 0.0, 0.0])
    if qw is None:
        qw = pw.copy()
    Sdz = np.real(Sd) * pw[2] + 1j * np.imag(Sd) * qw[2]
    Sdi = np.real(Sd) * pw[1] + 1j * np.imag(Sd) * qw[1]
    Sdp = np.real(Sd) * pw[0] + 1j * np.imag(Sd) * qw[0]

    f = np.r_[0, f]
    Zb = np.r_[0.0 + 0.0j, Zb]
    nl = nl + 1

    if pv.size:
        Zpv = make_zpv(pv + 1, nb, nl, f + 1, Zb, Yd)
        Bpv = np.linalg.inv(np.imag(Zpv))
    npv = pv.size
    Qpv = np.zeros(npv)

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
                    V = V + np.asarray(make_vcorr(DC + 1j * DD, pv + 1, nb, nl, f + 1, Zb)).reshape(-1)
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

    return (
        V.reshape(-1, 1),
        Qpv.reshape(-1, 1),
        Sf.reshape(-1, 1),
        St.reshape(-1, 1),
        Sslack,
        float(iter_count),
        success,
    )
