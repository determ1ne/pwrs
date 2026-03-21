# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def cpf_p(parameterization, step, z, V, lam, Vprv, lamprv, pv, pq, *, nargout=None):
    """Compute the continuation power flow parameterization value.

    Parameters
    ----------
    parameterization : int
        CPF parameterization mode.
    step : float
        Continuation step size.
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
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    float
        Value of the parameterization function at the current point.
    """
    parameterization = int(np.asarray(parameterization).reshape(-1)[0])
    step = float(np.asarray(step).reshape(-1)[0])
    z = np.asarray(z).reshape(-1)
    V = np.asarray(V).reshape(-1)
    lam = float(np.asarray(lam).reshape(-1)[0])
    Vprv = np.asarray(Vprv).reshape(-1)
    lamprv = float(np.asarray(lamprv).reshape(-1)[0])
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1

    if parameterization == 1:
        if lam >= lamprv:
            P = lam - lamprv - step
        else:
            P = lamprv - lam - step
    elif parameterization == 2:
        Va = np.angle(V)
        Vm = np.abs(V)
        Vaprv = np.angle(Vprv)
        Vmprv = np.abs(Vprv)
        P = (
            np.sum((np.r_[Va[np.r_[pv, pq]], Vm[pq], lam] - np.r_[Vaprv[np.r_[pv, pq]], Vmprv[pq], lamprv]) ** 2)
            - step**2
        )
    elif parameterization == 3:
        nb = len(V)
        Va = np.angle(V)
        Vm = np.abs(V)
        Vaprv = np.angle(Vprv)
        Vmprv = np.abs(Vprv)
        P = (
            z[np.r_[pv, pq, nb + pq, 2 * nb]].T
            @ (np.r_[Va[np.r_[pv, pq]], Vm[pq], lam] - np.r_[Vaprv[np.r_[pv, pq]], Vmprv[pq], lamprv])
            - step
        )
    else:
        raise ValueError("cpf_p: invalid parameterization")
    return float(np.asarray(P).reshape(-1)[0])
