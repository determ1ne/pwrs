# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig
from .cpf_p import cpf_p
from .cpf_p_jac import cpf_p_jac
from .dSbus_dV import dSbus_dV
from .mpoption import mpoption


def _eval_sbus(Sbus, Vm):
    result = Sbus(Vm)
    if isinstance(result, tuple):
        if len(result) == 1:
            return np.asarray(result[0]).reshape(-1), None
        return np.asarray(result[0]).reshape(-1), result[1]
    return np.asarray(result).reshape(-1), None


def cpf_corrector(Ybus, Sbusb, V_hat, ref, pv, pq, lam_hat, Sbust, Vprv, lamprv, z, step, parameterization, mpopt=None):
    """Solve the corrector step of a continuation power flow.

    Parameters
    ----------
    Ybus : array_like or sparse matrix
        Bus admittance matrix.
    Sbusb, Sbust : callable
        Callables returning base and target complex injections in per unit
        and, optionally, their derivatives with respect to voltage
        magnitude.
    V_hat : array_like
        Predicted complex bus voltage vector.
    ref, pv, pq : array_like
        REF, PV, and PQ bus index vectors.
    lam_hat : float
        Predicted continuation parameter.
    Vprv : array_like
        Previous complex bus voltage vector.
    lamprv : float
        Previous continuation parameter value.
    z : array_like
        Normalized tangent prediction vector.
    step : float
        Continuation step size.
    parameterization : int
        CPF parameterization mode.
    mpopt : dict, optional
        MATPOWER options dict.
    Returns
    -------
    tuple
        ``(V, converged, i, lam)`` containing the corrected voltage vector,
        convergence flag, Newton iteration count, and continuation
        parameter.
    """
    if mpopt is None:
        mpopt = mpoption()
    elif not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)

    tol = float(mpopt.pf.tol)
    max_it = int(mpopt.pf.nr.max_it)
    verbose = int(mpopt.verbose)

    V = np.asarray(V_hat).reshape(-1).astype(complex, copy=True)
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1
    Vprv = np.asarray(Vprv).reshape(-1)
    z = np.asarray(z).reshape(-1)
    lam = float(np.asarray(lam_hat).reshape(-1)[0])
    lamprv = float(np.asarray(lamprv).reshape(-1)[0])
    step = float(np.asarray(step).reshape(-1)[0])
    parameterization = int(np.asarray(parameterization).reshape(-1)[0])

    converged = 0
    i = 0
    Va = np.angle(V)
    Vm = np.abs(V)

    npv = len(pv)
    npq = len(pq)
    j1 = 0
    j2 = npv
    j3 = j2
    j4 = j2 + npq
    j5 = j4
    j6 = j4 + npq
    j7 = j6
    j8 = j6 + 1
    pvpq = np.r_[pv, pq]

    Sb = _eval_sbus(Sbusb, Vm)[0]
    St = _eval_sbus(Sbust, Vm)[0]
    mis = V * np.conj(Ybus @ V) - Sb - lam * (St - Sb)
    F = np.r_[np.real(mis[pvpq]), np.imag(mis[pq])]
    P = cpf_p(parameterization, step, z, V, lam, Vprv, lamprv, pv + 1, pq + 1)
    F = np.r_[F, P]

    normF = np.linalg.norm(F, np.inf)
    if normF < tol:
        converged = 1

    while not converged and i < max_it:
        i += 1
        dSbus_dVa, dSbus_dVm = dSbus_dV(Ybus, V)
        _, neg_dSdb_dVm = _eval_sbus(Sbusb, Vm)
        _, neg_dSdt_dVm = _eval_sbus(Sbust, Vm)
        dSbus_dVm = dSbus_dVm - neg_dSdb_dVm - lam * (neg_dSdt_dVm - neg_dSdb_dVm)

        j11 = np.real(dSbus_dVa[pvpq][:, pvpq])
        j12 = np.real(dSbus_dVm[pvpq][:, pq])
        j21 = np.imag(dSbus_dVa[pq][:, pvpq])
        j22 = np.imag(dSbus_dVm[pq][:, pq])
        J = (
            sparse.bmat([[j11, j12], [j21, j22]], format="csc")
            if any(sparse.issparse(x) for x in (j11, j12, j21, j22))
            else np.block([[j11, j12], [j21, j22]])
        )

        Sxf = St - Sb
        dF_dlam = -np.r_[np.real(Sxf[pvpq]), np.imag(Sxf[pq])].reshape(-1, 1)
        dP_dV, dP_dlam = cpf_p_jac(parameterization, z, V, lam, Vprv, lamprv, pv + 1, pq + 1)
        J = (
            sparse.vstack(
                [sparse.hstack([J, sparse.csc_matrix(dF_dlam)]), sparse.csc_matrix(np.c_[dP_dV, dP_dlam])], format="csc"
            )
            if sparse.issparse(J)
            else np.block([[J, dF_dlam], [dP_dV, np.array([[dP_dlam]])]])
        )

        rhs = -F.reshape(-1, 1)
        if sparse.issparse(J):
            dx = sparse.linalg.spsolve(J, rhs).reshape(-1)
        else:
            dx = np.linalg.solve(J, rhs).reshape(-1)

        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        V = Vm * np.exp(1j * Va)
        Vm = np.abs(V)
        Va = np.angle(V)
        lam = lam + dx[j7:j8][0]

        Sb = _eval_sbus(Sbusb, Vm)[0]
        St = _eval_sbus(Sbust, Vm)[0]
        mis = V * np.conj(Ybus @ V) - Sb - lam * (St - Sb)
        F = np.r_[np.real(mis[pv]), np.real(mis[pq]), np.imag(mis[pq])]
        P = cpf_p(parameterization, step, z, V, lam, Vprv, lamprv, pv + 1, pq + 1)
        F = np.r_[F, P]

        normF = np.linalg.norm(F, np.inf)
        if normF < tol:
            converged = 1

    if verbose and not converged:
        pass

    return V.reshape(-1, 1), float(converged), float(i), float(lam)
