# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from ..corex import as_dense_matrix, matrix_imag, matrix_real, subtract_matrices
from .cpf_p_jac import cpf_p_jac
from .dSbus_dV import dSbus_dV


def _eval_sbus(Sbus, Vm):
    result = Sbus(Vm)
    if isinstance(result, tuple):
        if len(result) == 1:
            return np.asarray(result[0]).reshape(-1), None
        return np.asarray(result[0]).reshape(-1), result[1]
    return np.asarray(result).reshape(-1), None


def cpf_tangent(V, lam, Ybus, Sbusb, Sbust, pv, pq, zprv, Vprv, lamprv, parameterization, direction):
    """Compute the normalized tangent predictor for continuation power flow.

    Parameters
    ----------
    V : array_like
        Complex bus voltage vector at the current solution.
    lam : float
        Current continuation parameter value.
    Ybus : array_like or sparse matrix
        Bus admittance matrix.
    Sbusb, Sbust : callable
        Callables returning base and target complex injections in per unit
        and, optionally, their derivatives with respect to voltage
        magnitude.
    pv, pq : array_like
        PV and PQ bus index vectors.
    zprv : array_like
        Normalized tangent prediction vector from the previous step.
    Vprv : array_like
        Previous complex bus voltage vector.
    lamprv : float
        Previous continuation parameter value.
    parameterization : int
        CPF parameterization mode.
    direction : float
        Continuation direction.
    Returns
    -------
    ndarray
        Normalized tangent prediction vector.
    """
    V = np.asarray(V).reshape(-1)
    lam = float(np.asarray(lam).reshape(-1)[0])
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1
    zprv = np.asarray(zprv).reshape(-1)
    Vprv = np.asarray(Vprv).reshape(-1)
    lamprv = float(np.asarray(lamprv).reshape(-1)[0])
    parameterization = int(np.asarray(parameterization).reshape(-1)[0])
    direction = float(np.asarray(direction).reshape(-1)[0])

    nb = len(V)
    npv = len(pv)
    npq = len(pq)
    Vm = np.abs(V)
    pvpq = np.r_[pv, pq]

    dSbus_dVa, dSbus_dVm = dSbus_dV(Ybus, V)
    _, neg_dSdb_dVm = _eval_sbus(Sbusb, Vm)
    _, neg_dSdt_dVm = _eval_sbus(Sbust, Vm)
    if neg_dSdb_dVm is None or neg_dSdt_dVm is None:
        raise ValueError("cpf_tangent: Sbus callables must return voltage-magnitude derivatives")
    transfer_derivative = subtract_matrices(neg_dSdt_dVm, neg_dSdb_dVm)
    dSbus_dVm = subtract_matrices(dSbus_dVm, neg_dSdb_dVm)
    dSbus_dVm = subtract_matrices(dSbus_dVm, lam * transfer_derivative)

    j11 = matrix_real(dSbus_dVa[pvpq][:, pvpq])
    j12 = matrix_real(dSbus_dVm[pvpq][:, pq])
    j21 = matrix_imag(dSbus_dVa[pq][:, pvpq])
    j22 = matrix_imag(dSbus_dVm[pq][:, pq])
    J = (
        sparse.bmat([[j11, j12], [j21, j22]], format="csc")
        if any(sparse.issparse(x) for x in (j11, j12, j21, j22))
        else np.block(
            [
                [as_dense_matrix(j11), as_dense_matrix(j12)],
                [as_dense_matrix(j21), as_dense_matrix(j22)],
            ]
        )
    )

    Sxf = _eval_sbus(Sbust, Vm)[0] - _eval_sbus(Sbusb, Vm)[0]
    dF_dlam = -np.r_[np.real(Sxf[pvpq]), np.imag(Sxf[pq])].reshape(-1, 1)
    dP_dV, dP_dlam = cpf_p_jac(parameterization, zprv, V, lam, Vprv, lamprv, pv + 1, pq + 1)

    J = (
        sparse.vstack(
            [sparse.hstack([J, sparse.csc_matrix(dF_dlam)]), sparse.csc_matrix(np.c_[dP_dV, dP_dlam])], format="csc"
        )
        if sparse.issparse(J)
        else np.block([[J, dF_dlam], [dP_dV, np.array([[dP_dlam]])]])
    )

    z = zprv.copy()
    s = np.zeros(npv + 2 * npq + 1)
    s[-1] = np.sign(direction)
    rhs = s.reshape(-1, 1)
    if sparse.issparse(J):
        sol = spla.spsolve(J, rhs).reshape(-1)
    else:
        sol = np.linalg.solve(np.asarray(J), rhs).reshape(-1)
    z[np.r_[pvpq, nb + pq, 2 * nb]] = sol
    z = z / np.linalg.norm(z)
    return z.reshape(-1, 1)
