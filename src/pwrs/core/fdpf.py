# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..utils import get_nested


def _evaluate_sbus(Sbus: Callable[[np.ndarray], Any], Vm: np.ndarray):
    result = Sbus(Vm)
    if isinstance(result, tuple):
        if len(result) == 0:
            raise ValueError("fdpf: Sbus callable returned no outputs")
        return np.asarray(result[0]).reshape(-1)
    return np.asarray(result).reshape(-1)


def _solve_lu(lu, rhs):
    out = lu.solve(np.asarray(rhs).reshape(-1, 1))
    return np.asarray(out).reshape(-1)


def fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, mpopt=None, *, nargout=None):
    """Solve an AC power flow using the fast-decoupled method.

    Mirrors MATPOWER's ``fdpf`` solver by alternating voltage-angle and
    voltage-magnitude updates using the reduced ``Bp`` and ``Bpp`` matrices
    until the power mismatch satisfies the configured tolerance.

    Parameters
    ----------
    Ybus : array_like or sparse matrix
        Bus admittance matrix.
    Sbus : callable
        Callable returning complex bus injections for a given voltage
        magnitude vector.
    V0 : array_like
        Initial complex bus voltage vector.
    Bp : array_like or sparse matrix
        Fast-decoupled Jacobian approximation for active-power updates.
    Bpp : array_like or sparse matrix
        Fast-decoupled Jacobian approximation for reactive-power updates.
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
    if not callable(Sbus):
        raise TypeError("fdpf: Sbus must be callable in the Python API")

    if mpopt is None:
        mpopt = {}

    tol = float(get_nested(mpopt, ["pf", "tol"], 1e-8))
    max_it = int(get_nested(mpopt, ["pf", "fd", "max_it"], 30))
    verbose = int(get_nested(mpopt, ["verbose"], 0))

    if sparse.issparse(Ybus):
        Ybus = Ybus.tocsc()
    else:
        Ybus = np.asarray(Ybus, dtype=complex)

    V = np.asarray(V0).reshape(-1).astype(complex, copy=True)
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1

    converged = 0.0
    i = 0.0
    Va = np.angle(V)
    Vm = np.abs(V)

    pvpq = np.r_[pv, pq]
    mis = (V * np.conj(Ybus @ V) - _evaluate_sbus(Sbus, Vm)) / Vm
    P = np.real(mis[pvpq])
    Q = np.imag(mis[pq])

    normP = np.linalg.norm(P, np.inf) if P.size else 0.0
    normQ = np.linalg.norm(Q, np.inf) if Q.size else 0.0
    if normP < tol and normQ < tol:
        converged = 1.0

    Bp = Bp[pvpq][:, pvpq]
    Bpp = Bpp[pq][:, pq]
    Bp_lu = splu(Bp.tocsc() if sparse.issparse(Bp) else sparse.csc_matrix(Bp))
    Bpp_lu = splu(Bpp.tocsc() if sparse.issparse(Bpp) else sparse.csc_matrix(Bpp))

    while not converged and i < max_it:
        i += 1.0

        dVa = -_solve_lu(Bp_lu, P)
        Va[pvpq] = Va[pvpq] + dVa
        V = Vm * np.exp(1j * Va)

        mis = (V * np.conj(Ybus @ V) - _evaluate_sbus(Sbus, Vm)) / Vm
        P = np.real(mis[pvpq])
        Q = np.imag(mis[pq])
        normP = np.linalg.norm(P, np.inf) if P.size else 0.0
        normQ = np.linalg.norm(Q, np.inf) if Q.size else 0.0
        if normP < tol and normQ < tol:
            converged = 1.0
            break

        dVm = -_solve_lu(Bpp_lu, Q)
        Vm[pq] = Vm[pq] + dVm
        V = Vm * np.exp(1j * Va)

        mis = (V * np.conj(Ybus @ V) - _evaluate_sbus(Sbus, Vm)) / Vm
        P = np.real(mis[pvpq])
        Q = np.imag(mis[pq])
        normP = np.linalg.norm(P, np.inf) if P.size else 0.0
        normQ = np.linalg.norm(Q, np.inf) if Q.size else 0.0
        if normP < tol and normQ < tol:
            converged = 1.0
            break

    if verbose and not converged:
        pass

    return V.reshape(-1, 1), converged, i
