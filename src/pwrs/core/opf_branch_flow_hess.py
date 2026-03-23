# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .d2Abr_dV2 import d2Abr_dV2
from .d2Ibr_dV2 import d2Ibr_dV2
from .d2Sbr_dV2 import d2Sbr_dV2
from .dIbr_dV import dIbr_dV
from .dSbr_dV import dSbr_dV
from .idx_brch import F_BUS, T_BUS


def opf_branch_flow_hess(x, lambda_, mpc, Yf, Yt, il, mpopt, nargout=1):
    """Return Hessian of AC OPF branch flow limit constraints.

    Forms the Hessian of the Lagrangian contribution from the branch flow
    inequality constraints selected by ``il``. The exact flow model follows
    MATPOWER's ``flow_lim`` option for current, active-power, or apparent
    power limits.

    Parameters
    ----------
    x : sequence of array_like
        Voltage-state blocks. The layout matches MATPOWER: ``(Va, Vm)`` in
        polar form or ``(Vr, Vi)`` in cartesian form.
    lambda_ : array_like
        Multipliers for the stacked "from" and "to" branch flow constraints.
    mpc : dict
        Internal MATPOWER case struct.
    Yf : sparse matrix
        Admittance matrix mapping bus voltages to branch "from" currents.
    Yt : sparse matrix
        Admittance matrix mapping bus voltages to branch "to" currents.
    il : array_like
        One-based indices of branches with active flow limits.
    mpopt : dict
        MATPOWER options struct.
    nargout : int, optional
        MATLAB compatibility flag controlling how many outputs are returned.

    Returns
    -------
    scipy.sparse.csc_matrix
        Hessian block for the branch flow constraint contribution to the OPF
        Lagrangian.
    """
    opf_opt = mpopt.opf
    lim_type = opf_opt.flow_lim.upper()
    vcart = opf_opt.v_cartesian
    if vcart:
        Vr, Vi = [np.asarray(v).reshape(-1) for v in x]
        V = Vr + 1j * Vi
    else:
        Va, Vm = [np.asarray(v).reshape(-1) for v in x]
        V = Vm * np.exp(1j * Va)

    nb = len(V)
    il = np.asarray(il).reshape(-1).astype(int)
    nl2 = len(il)
    lambda_ = np.asarray(lambda_).reshape(-1)
    branch_il = mpc["branch"][il - 1, :]

    nmu = len(lambda_) // 2
    if nmu:
        muF = lambda_[:nmu]
        muT = lambda_[nmu : nmu + nmu]
    else:
        muF = np.zeros(0)
        muT = np.zeros(0)

    if lim_type == "I":
        dIf_dV1, dIf_dV2, dIt_dV1, dIt_dV2, If, It = dIbr_dV(
            branch_il, Yf, Yt, V, vcart, nargout=6
        )
        d2If_dV2 = lambda Vv, muv: d2Ibr_dV2(Yf, Vv, muv, vcart, nargout=4)
        d2It_dV2 = lambda Vv, muv: d2Ibr_dV2(Yt, Vv, muv, vcart, nargout=4)
        Hf11, Hf12, Hf21, Hf22 = d2Abr_dV2(d2If_dV2, dIf_dV1, dIf_dV2, If, V, muF, nargout=4)
        Ht11, Ht12, Ht21, Ht22 = d2Abr_dV2(d2It_dV2, dIt_dV1, dIt_dV2, It, V, muT, nargout=4)
    else:
        f_idx = branch_il[:, F_BUS - 1].astype(int) - 1
        t_idx = branch_il[:, T_BUS - 1].astype(int) - 1
        dSf_dV1, dSf_dV2, dSt_dV1, dSt_dV2, Sf, St = dSbr_dV(
            branch_il, Yf, Yt, V, vcart, nargout=6
        )
        d2Sf_dV2 = lambda Vv, muv: d2Sbr_dV2(f_idx, Yf, Vv, muv, vcart, nargout=4)
        d2St_dV2 = lambda Vv, muv: d2Sbr_dV2(t_idx, Yt, Vv, muv, vcart, nargout=4)
        if lim_type == "2":
            Hf11, Hf12, Hf21, Hf22 = d2Abr_dV2(
                d2Sf_dV2, np.real(dSf_dV1), np.real(dSf_dV2), np.real(Sf), V, muF, nargout=4
            )
            Ht11, Ht12, Ht21, Ht22 = d2Abr_dV2(
                d2St_dV2, np.real(dSt_dV1), np.real(dSt_dV2), np.real(St), V, muT, nargout=4
            )
        elif lim_type == "P":
            Hf11, Hf12, Hf21, Hf22 = d2Sf_dV2(V, muF)
            Ht11, Ht12, Ht21, Ht22 = d2St_dV2(V, muT)
            Hf11, Hf12, Hf21, Hf22 = np.real(Hf11), np.real(Hf12), np.real(Hf21), np.real(Hf22)
            Ht11, Ht12, Ht21, Ht22 = np.real(Ht11), np.real(Ht12), np.real(Ht21), np.real(Ht22)
        else:
            Hf11, Hf12, Hf21, Hf22 = d2Abr_dV2(d2Sf_dV2, dSf_dV1, dSf_dV2, Sf, V, muF, nargout=4)
            Ht11, Ht12, Ht21, Ht22 = d2Abr_dV2(d2St_dV2, dSt_dV1, dSt_dV2, St, V, muT, nargout=4)

    H11 = Hf11 + Ht11
    H12 = Hf12 + Ht12
    H21 = Hf21 + Ht21
    H22 = Hf22 + Ht22
    d2H = sparse.bmat([[H11, H12], [H21, H22]], format="csc")
    outputs = (d2H,)
    return outputs[:nargout] if nargout > 1 else d2H
