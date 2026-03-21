# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..utils import get_nested


def _scalar(value):
    arr = np.asarray(value)
    return arr.reshape(-1)[0]


def gausspf(Ybus, Sbus, V0, ref, pv, pq, mpopt=None, *, nargout=None):
    """Solve an AC power flow using the Gauss-Seidel method.

    Mirrors MATPOWER's ``gausspf`` solver by iteratively updating PQ-bus and
    PV-bus voltages with Gauss-Seidel sweeps until the power mismatch meets
    the configured tolerance.

    Parameters
    ----------
    Ybus : array_like or sparse matrix
        Bus admittance matrix.
    Sbus : array_like
        Complex bus power injections.
    V0 : array_like
        Initial complex bus voltage vector.
    ref : array_like
        One-based reference bus indices.
    pv : array_like
        One-based PV bus indices.
    pq : array_like
        One-based PQ bus indices.
    mpopt : dict, optional
        MATPOWER options struct.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    tuple
        ``(V, converged, iterations)``.
    """
    if mpopt is None:
        mpopt = {}

    tol = float(get_nested(mpopt, ["pf", "tol"], 1e-8))
    max_it = int(get_nested(mpopt, ["pf", "gs", "max_it"], 1000))

    if sparse.issparse(Ybus):
        Ybus = Ybus.tocsr()
    else:
        Ybus = np.asarray(Ybus, dtype=complex)

    V = np.asarray(V0).reshape(-1).astype(complex, copy=True)
    Sbus = np.asarray(Sbus).reshape(-1).astype(complex, copy=True)
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1

    converged = 0.0
    i = 0.0
    Vm = np.abs(V)

    mis = V * np.conj(Ybus @ V) - Sbus
    F = np.r_[np.real(mis[np.r_[pv, pq]]), np.imag(mis[pq])]
    normF = np.linalg.norm(F, np.inf) if F.size else 0.0
    if normF < tol:
        converged = 1.0

    while not converged and i < max_it:
        i += 1.0

        for k in pq:
            Yk = _scalar(Ybus.getrow(k) @ V) if sparse.issparse(Ybus) else _scalar(Ybus[k, :] @ V)
            Ykk = _scalar(Ybus[k, k])
            V[k] = V[k] + (np.conj(Sbus[k] / V[k]) - Yk) / Ykk

        if pv.size:
            for k in pv:
                Yk = _scalar(Ybus.getrow(k) @ V) if sparse.issparse(Ybus) else _scalar(Ybus[k, :] @ V)
                Ykk = _scalar(Ybus[k, k])
                Sbus[k] = np.real(Sbus[k]) + 1j * np.imag(V[k] * np.conj(Yk))
                V[k] = V[k] + (np.conj(Sbus[k] / V[k]) - Yk) / Ykk
            V[pv] = Vm[pv] * V[pv] / np.abs(V[pv])

        mis = V * np.conj(Ybus @ V) - Sbus
        F = np.r_[np.real(mis[pv]), np.real(mis[pq]), np.imag(mis[pq])]
        normF = np.linalg.norm(F, np.inf) if F.size else 0.0
        if normF < tol:
            converged = 1.0

    return V.reshape(-1, 1), converged, i
