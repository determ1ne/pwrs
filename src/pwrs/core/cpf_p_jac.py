# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def cpf_p_jac(parameterization, z, V, lam, Vprv, lamprv, pv, pq):
    """Compute partial derivatives of the CPF parameterization function.

    Parameters
    ----------
    parameterization : int
        CPF parameterization mode.
    z : array_like
        Normalized tangent prediction vector from the previous step.
    V : array_like
        Complex bus voltage vector at the current solution.
    lam : float
        Current continuation parameter value.
    Vprv : array_like
        Complex bus voltage vector at the previous solution.
    lamprv : float
        Previous continuation parameter value.
    pv, pq : array_like
        PV and PQ bus index vectors.
    Returns
    -------
    tuple
        ``(dP_dV, dP_dlam)``.
    """
    parameterization = int(np.asarray(parameterization).reshape(-1)[0])
    z = np.asarray(z).reshape(-1)
    V = np.asarray(V).reshape(-1)
    lam = float(np.asarray(lam).reshape(-1)[0])
    Vprv = np.asarray(Vprv).reshape(-1)
    lamprv = float(np.asarray(lamprv).reshape(-1)[0])
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1

    if parameterization == 1:
        npv = len(pv)
        npq = len(pq)
        dP_dV = np.zeros((1, npv + 2 * npq))
        dP_dlam = 1.0 if lam >= lamprv else -1.0
    elif parameterization == 2:
        Va = np.angle(V)
        Vm = np.abs(V)
        Vaprv = np.angle(Vprv)
        Vmprv = np.abs(Vprv)
        dP_dV = 2 * (np.r_[Va[np.r_[pv, pq]], Vm[pq]] - np.r_[Vaprv[np.r_[pv, pq]], Vmprv[pq]]).reshape(1, -1)
        dP_dlam = 1.0 if lam == lamprv else 2 * (lam - lamprv)
    elif parameterization == 3:
        nb = len(V)
        dP_dV = z[np.r_[pv, pq, nb + pq]].reshape(1, -1)
        dP_dlam = z[2 * nb]
    else:
        raise ValueError("cpf_p_jac: invalid parameterization")
    return dP_dV, float(dP_dlam)
