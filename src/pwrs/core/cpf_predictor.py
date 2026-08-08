# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def cpf_predictor(V, lam, z, step, pv, pq):
    """Perform the predictor step for continuation power flow.

    Parameters
    ----------
    V : array_like
        Complex bus voltage vector at the current solution.
    lam : float
        Current continuation parameter value.
    z : array_like
        Normalized tangent prediction vector from the previous step.
    step : float
        Continuation step length.
    pv, pq : array_like
        PV and PQ bus index vectors.
    Returns
    -------
    tuple
        ``(V_hat, lam_hat)`` containing the predicted voltage vector and
        continuation parameter.
    """
    V = np.asarray(V).reshape(-1)
    lam = float(np.asarray(lam).reshape(-1)[0])
    z = np.asarray(z).reshape(-1)
    step = float(np.asarray(step).reshape(-1)[0])
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1

    nb = len(V)
    Va = np.angle(V)
    Vm = np.abs(V)
    Va_hat = Va.copy()
    Vm_hat = Vm.copy()

    pvpq = np.r_[pv, pq]
    Va_hat[pvpq] = Va[pvpq] + step * z[pvpq]
    Vm_hat[pq] = Vm[pq] + step * z[nb + pq]
    lam_hat = lam + step * z[2 * nb]
    V_hat = Vm_hat * np.exp(1j * Va_hat)
    return V_hat.reshape(-1, 1), float(lam_hat)
