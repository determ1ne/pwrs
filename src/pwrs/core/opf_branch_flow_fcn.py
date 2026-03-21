# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .dAbr_dV import dAbr_dV
from .dIbr_dV import dIbr_dV
from .dSbr_dV import dSbr_dV
from .idx_brch import F_BUS, RATE_A, T_BUS


def opf_branch_flow_fcn(x, mpc, Yf, Yt, il, mpopt, nargout=1):
    """Evaluate AC OPF branch flow limit constraints and Jacobian.

    Computes the nonlinear inequality constraints for branch flow limits on
    the selected branches ``il``. The exact form depends on
    ``mpopt['opf']['flow_lim']`` and matches MATPOWER's handling of current,
    active-power, or apparent-power limits.

    Parameters
    ----------
    x : sequence of array_like
        Voltage-state blocks. The layout matches MATPOWER: ``(Va, Vm)`` in
        polar form or ``(Vr, Vi)`` in cartesian form.
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
        MATLAB compatibility flag controlling whether the Jacobian is
        returned.

    Returns
    -------
    numpy.ndarray or tuple
        Inequality constraint vector ``h`` for the selected branch limits,
        and optionally its Jacobian ``dh``.
    """
    lim_type = mpopt.opf.flow_lim.upper()
    branch = mpc["branch"]
    if mpopt.opf.v_cartesian:
        Vr, Vi = [np.asarray(v).reshape(-1) for v in x]
        V = Vr + 1j * Vi
    else:
        Va, Vm = [np.asarray(v).reshape(-1) for v in x]
        V = Vm * np.exp(1j * Va)

    nb = len(V)
    il = np.asarray(il).reshape(-1).astype(int)
    nl2 = len(il)

    if nl2 > 0:
        flow_max = branch[il - 1, RATE_A - 1] / mpc["baseMVA"]
        if lim_type != "P":
            flow_max = flow_max**2
        if lim_type == "I":
            If = Yf @ V
            It = Yt @ V
            h = np.r_[np.real(If * np.conjugate(If) - flow_max), np.real(It * np.conjugate(It) - flow_max)]
        else:
            Sf = V[branch[il - 1, F_BUS - 1].astype(int) - 1] * np.conjugate(Yf @ V)
            St = V[branch[il - 1, T_BUS - 1].astype(int) - 1] * np.conjugate(Yt @ V)
            if lim_type == "2":
                h = np.r_[np.real(Sf) ** 2 - flow_max, np.real(St) ** 2 - flow_max]
            elif lim_type == "P":
                h = np.r_[np.real(Sf) - flow_max, np.real(St) - flow_max]
            else:
                h = np.r_[np.real(Sf * np.conjugate(Sf) - flow_max), np.real(St * np.conjugate(St) - flow_max)]
    else:
        h = np.zeros(0)

    if nargout > 1:
        if nl2 > 0:
            if lim_type == "I":
                dFf_dV1, dFf_dV2, dFt_dV1, dFt_dV2, Ff, Ft = dIbr_dV(
                    branch[il - 1, :], Yf, Yt, V, mpopt.opf.v_cartesian, nargout=6
                )
            else:
                dFf_dV1, dFf_dV2, dFt_dV1, dFt_dV2, Ff, Ft = dSbr_dV(
                    branch[il - 1, :], Yf, Yt, V, mpopt.opf.v_cartesian, nargout=6
                )
            if lim_type in ("P", "2"):
                dFf_dV1 = np.real(dFf_dV1)
                dFf_dV2 = np.real(dFf_dV2)
                dFt_dV1 = np.real(dFt_dV1)
                dFt_dV2 = np.real(dFt_dV2)
                Ff = np.real(Ff)
                Ft = np.real(Ft)

            if lim_type == "P":
                df_dV1, df_dV2, dt_dV1, dt_dV2 = dFf_dV1, dFf_dV2, dFt_dV1, dFt_dV2
            else:
                df_dV1, df_dV2, dt_dV1, dt_dV2 = dAbr_dV(dFf_dV1, dFf_dV2, dFt_dV1, dFt_dV2, Ff, Ft, nargout=4)

            dh = sparse.vstack(
                [
                    sparse.hstack([df_dV1, df_dV2], format="csc"),
                    sparse.hstack([dt_dV1, dt_dV2], format="csc"),
                ],
                format="csc",
            )
        else:
            dh = sparse.csc_matrix((0, 2 * nb))
        outputs = (h, dh)
    else:
        outputs = (h,)
    return outputs[:nargout] if nargout > 1 else h
