# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig
from ..mips.mplinsolve import mplinsolve
from .mpoption import mpoption
from .dImis_dV import dImis_dV
from .newtonpf import _evaluate_sbus


def newtonpf_I_hybrid(Ybus, Sbus, V0, ref, pv, pq, mpopt=None):
    """Solve a power flow using full Newton's method (current/hybrid).

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
    Returns
    -------
    tuple
        ``(V, converged, i)`` containing the final complex bus voltages, a
        convergence flag, and the iteration count.

    Notes
    -----
    This variant uses nodal current balance equations with a hybrid voltage
    representation, where a polar update is computed using a cartesian
    Jacobian.

    See Also
    --------
    runpf, newtonpf, newtonpf_S_cart, newtonpf_I_polar
    """
    if mpopt is None:
        mpopt = mpoption()
    elif not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)
    tol = float(mpopt.pf.tol)
    max_it = int(mpopt.pf.nr.max_it)
    lin_solver = str(mpopt.pf.nr.lin_solver)
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
        dImis_dQ = sparse.csc_matrix((1j / np.conj(V), (np.arange(n), np.arange(n))), shape=(n, n))
        dImis_dVr, dImis_dVi = dImis_dV(Sb, Ybus, V, 1)
        dImis_dVr = dImis_dVr.tocsc() if sparse.issparse(dImis_dVr) else sparse.csc_matrix(dImis_dVr)
        dImis_dVi = dImis_dVi.tocsc() if sparse.issparse(dImis_dVi) else sparse.csc_matrix(dImis_dVi)
        if npv:
            rv = sparse.diags(np.real(V[pv]), offsets=0, shape=(npv, npv), format="csc")
            iv = sparse.diags(np.imag(V[pv]), offsets=0, shape=(npv, npv), format="csc")
            dImis_dVi[:, pv] = dImis_dVi[:, pv] @ rv - dImis_dVr[:, pv] @ iv
            dImis_dVr[:, pv] = dImis_dQ[:, pv]
        j11 = np.real(dImis_dVr[pvpq][:, pvpq])
        j12 = np.real(dImis_dVi[pvpq][:, pvpq])
        j21 = np.imag(dImis_dVr[pvpq][:, pvpq])
        j22 = np.imag(dImis_dVi[pvpq][:, pvpq])
        J = sparse.bmat([[j11, j12], [j21, j22]], format="csc")
        dx = np.asarray(mplinsolve(J, -F, lin_solver, None)).reshape(-1)
        if npv:
            Va[pv] = Va[pv] + dx[j5:j6]
            Sb[pv] = np.real(Sb[pv]) + 1j * (np.imag(Sb[pv]) + dx[j1:j2])
        if npq:
            Vm[pq] = Vm[pq] + (np.real(V[pq]) / Vm[pq]) * dx[j3:j4] + (np.imag(V[pq]) / Vm[pq]) * dx[j7:j8]
            Va[pq] = (
                Va[pq] + (np.real(V[pq]) / (Vm[pq] ** 2)) * dx[j7:j8] - (np.imag(V[pq]) / (Vm[pq] ** 2)) * dx[j3:j4]
            )
        V = Vm * np.exp(1j * Va)
        Vm = np.abs(V)
        Va = np.angle(V)
        mis = Ybus @ V - np.conj(Sb / V)
        F = np.r_[np.real(mis[pvpq]), np.imag(mis[pvpq])]
        if np.linalg.norm(F, np.inf) < tol:
            converged = 1
    return V.reshape(-1, 1), float(converged), float(i)
