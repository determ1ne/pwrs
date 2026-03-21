# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..mips.mplinsolve import mplinsolve
from ..utils import get_nested
from .dSbus_dV import dSbus_dV
from .newtonpf import _evaluate_sbus


def newtonpf_S_hybrid(Ybus, Sbus, V0, ref, pv, pq, mpopt=None, *, nargout=None):
    """Solve a power flow using full Newton's method (power/hybrid).

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
    This variant uses nodal power balance equations with a hybrid voltage
    representation, where a polar update is computed using a cartesian
    Jacobian.

    See Also
    --------
    runpf, newtonpf, newtonpf_I_polar, newtonpf_I_cart
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
    npv = pv.size
    npq = pq.size
    j1, j2 = 0, npq
    j3, j4 = j2, j2 + npv
    j5, j6 = j4, j4 + npq

    Sbus_val, _ = _evaluate_sbus(Sbus, Vm)
    mis = V * np.conj(Ybus @ V) - Sbus_val
    pqpv = np.r_[pq, pv]
    F = np.r_[np.real(mis[pqpv]), np.imag(mis[pq])]
    if np.linalg.norm(F, np.inf) < tol:
        converged = 1
    if not lin_solver:
        lin_solver = "\\" if F.size <= 10 else "LU3"

    while not converged and i < max_it:
        i += 1
        dSbus_dVr, dSbus_dVi = dSbus_dV(Ybus, V, 1, nargout=2)
        if npv:
            rv = sparse.diags(np.real(V[pv]), offsets=0, shape=(npv, npv), format="csc")
            iv = sparse.diags(np.imag(V[pv]), offsets=0, shape=(npv, npv), format="csc")
            dSbus_dVi = dSbus_dVi.tocsc() if sparse.issparse(dSbus_dVi) else sparse.csc_matrix(dSbus_dVi)
            dSbus_dVr = dSbus_dVr.tocsc() if sparse.issparse(dSbus_dVr) else sparse.csc_matrix(dSbus_dVr)
            dSbus_dVi[:, pv] = dSbus_dVi[:, pv] @ rv - dSbus_dVr[:, pv] @ iv
        j11 = np.real(dSbus_dVr[pqpv][:, pq])
        j12 = np.real(dSbus_dVi[pqpv][:, np.r_[pv, pq]])
        j21 = np.imag(dSbus_dVr[pq][:, pq])
        j22 = np.imag(dSbus_dVi[pq][:, np.r_[pv, pq]])
        J = sparse.bmat([[j11, j12], [j21, j22]], format="csc")
        dx = np.asarray(mplinsolve(J, -F, lin_solver, None)).reshape(-1)
        if npv:
            Va[pv] = Va[pv] + dx[j3:j4]
        if npq:
            Vm[pq] = Vm[pq] + (np.real(V[pq]) / Vm[pq]) * dx[j1:j2] + (np.imag(V[pq]) / Vm[pq]) * dx[j5:j6]
            Va[pq] = (
                Va[pq] + (np.real(V[pq]) / (Vm[pq] ** 2)) * dx[j5:j6] - (np.imag(V[pq]) / (Vm[pq] ** 2)) * dx[j1:j2]
            )
        V = Vm * np.exp(1j * Va)
        Vm = np.abs(V)
        Va = np.angle(V)
        Sbus_val, _ = _evaluate_sbus(Sbus, Vm)
        mis = V * np.conj(Ybus @ V) - Sbus_val
        F = np.r_[np.real(mis[pqpv]), np.imag(mis[pq])]
        if np.linalg.norm(F, np.inf) < tol:
            converged = 1
    return V.reshape(-1, 1), float(converged), float(i)
