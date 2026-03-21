# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..mips.mplinsolve import mplinsolve
from ..utils import get_nested
from .dImis_dV import dImis_dV
from .newtonpf import _evaluate_sbus


def newtonpf_I_polar(Ybus, Sbus, V0, ref, pv, pq, mpopt=None, *, nargout=None):
    """Solve a power flow using full Newton's method (current/polar).

    Parameters
    ----------
    Ybus : array_like or sparse matrix
        Full system admittance matrix for all buses.
    Sbus : callable
        Callable returning the complex bus power injection vector for all
        buses as a function of bus voltage magnitudes.
    V0 : array_like
        Initial complex bus voltage vector.
    ref : array_like
        Reference bus index vector. Included for MATLAB interface
        compatibility.
    pv : array_like
        Bus indices for PV buses.
    pq : array_like
        Bus indices for PQ buses.
    mpopt : dict, optional
        MATPOWER options dict controlling tolerance, maximum Newton
        iterations and linear solver selection.
    nargout : int, optional
        MATLAB-compatibility placeholder. Ignored.

    Returns
    -------
    tuple
        ``(V, converged, i)`` containing the final complex bus voltages, a
        convergence flag, and the iteration count.

    Notes
    -----
    This variant uses nodal current balance equations with a polar voltage
    representation.

    See Also
    --------
    runpf, newtonpf, newtonpf_S_cart, newtonpf_I_cart
    """
    if mpopt is None:
        mpopt = {}
    tol = float(get_nested(mpopt, ["pf", "tol"], 1e-8))
    max_it = int(get_nested(mpopt, ["pf", "nr", "max_it"], 10))
    lin_solver = str(get_nested(mpopt, ["pf", "nr", "lin_solver"], ""))
    V = np.asarray(V0).reshape(-1).astype(complex, copy=True)
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1
    converged = 0
    i = 0
    Va = np.angle(V)
    Vm = np.abs(V)
    n = V.size
    npv = pv.size
    npq = pq.size
    j1, j2 = 0, npv
    j3, j4 = j2, j2 + npq
    j5, j6 = j4, j4 + npv
    j7, j8 = j6, j6 + npq

    Sb, _ = _evaluate_sbus(Sbus, Vm)
    if npv:
        Sb[pv] = np.real(Sb[pv]) + 1j * np.imag(V[pv] * np.conj(Ybus[pv, :] @ V))
    pvpq = np.r_[pv, pq]
    mis = Ybus @ V - np.conj(Sb / V)
    F = np.r_[np.real(mis[pvpq]), np.imag(mis[pvpq])]
    if np.linalg.norm(F, np.inf) < tol:
        converged = 1
    if not lin_solver:
        lin_solver = "\\" if F.size <= 10 else "LU3"

    while not converged and i < max_it:
        i += 1
        dImis_dQ = sparse.csc_matrix((1j / np.conj(V[pv]), (pv, pv)), shape=(n, n))
        dImis_dVa, dImis_dVm = dImis_dV(Sb, Ybus, V, 0, nargout=2)
        dImis_dVm = dImis_dVm.tocsc() if sparse.issparse(dImis_dVm) else sparse.csc_matrix(dImis_dVm)
        if npv:
            dImis_dVm[:, pv] = dImis_dQ[:, pv]
        j11 = np.real(dImis_dVa[pvpq][:, pvpq])
        j12 = np.real(dImis_dVm[pvpq][:, pvpq])
        j21 = np.imag(dImis_dVa[pvpq][:, pvpq])
        j22 = np.imag(dImis_dVm[pvpq][:, pvpq])
        J = sparse.bmat([[j11, j12], [j21, j22]], format="csc")
        dx = np.asarray(mplinsolve(J, -F, lin_solver, None)).reshape(-1)
        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
            Sb[pv] = np.real(Sb[pv]) + 1j * (np.imag(Sb[pv]) + dx[j5:j6])
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j7:j8]
        V = Vm * np.exp(1j * Va)
        Vm = np.abs(V)
        Va = np.angle(V)
        mis = Ybus @ V - np.conj(Sb / V)
        F = np.r_[np.real(mis[pvpq]), np.imag(mis[pvpq])]
        if np.linalg.norm(F, np.inf) < tol:
            converged = 1
    return V.reshape(-1, 1), float(converged), float(i)
