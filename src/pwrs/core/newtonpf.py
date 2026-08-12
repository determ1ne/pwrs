# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
from scipy import sparse

from ..corex import (
    ComplexArray,
    IntArray,
    MatpowerConfig,
    Matrix,
    SbusFunction,
    as_complex_vector,
    as_dense_matrix,
    as_index_vector,
    complex_matvec,
    matrix_imag,
    matrix_real,
    subtract_matrices,
)
from ..mips.mplinsolve import mplinsolve
from .dSbus_dV import dSbus_dV
from .mpoption import mpoption


def _evaluate_sbus(Sbus: SbusFunction, Vm: np.ndarray) -> tuple[ComplexArray, Matrix | None]:
    result = Sbus(Vm)
    if isinstance(result, tuple):
        if len(result) == 0:
            raise ValueError("newtonpf: Sbus callable returned no outputs")
        if len(result) == 1:
            return np.asarray(result[0]).reshape(-1), None
        return np.asarray(result[0]).reshape(-1), result[1]
    return np.asarray(result).reshape(-1), None


def newtonpf(
    Ybus: Matrix,
    Sbus: SbusFunction,
    V0: ComplexArray,
    ref: IntArray,
    pv: IntArray,
    pq: IntArray,
    mpopt: MatpowerConfig | dict[str, Any] | None = None,
) -> tuple[ComplexArray, int, int]:
    """Solve a power flow using full Newton's method (power/polar).

    Parameters
    ----------
    Ybus : array_like or sparse matrix
        Full system admittance matrix for all buses.
    Sbus : callable
        Callable returning the complex bus power injection vector for all
        buses as a function of bus voltage magnitudes. It may optionally
        also return the negative derivative of the voltage-dependent load
        term with respect to ``Vm``.
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
    This variant uses nodal power balance equations with a polar voltage
    representation.

    See Also
    --------
    runpf, newtonpf_S_cart, newtonpf_I_polar, newtonpf_I_cart
    """
    if mpopt is None:
        mpopt = mpoption()
    elif not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)

    ref = as_index_vector(ref, name="ref") - 1
    pv = as_index_vector(pv, name="pv") - 1
    pq = as_index_vector(pq, name="pq") - 1

    tol = mpopt.pf.tol
    max_it = mpopt.pf.nr.max_it
    lin_solver = mpopt.pf.nr.lin_solver

    converged = 0
    i = 0
    V = as_complex_vector(V0, name="V0")
    Va = np.angle(V)
    Vm = np.abs(V)

    npv = pv.size
    npq = pq.size
    j1 = 0
    j2 = npv
    j3 = j2
    j4 = j2 + npq
    j5 = j4
    j6 = j4 + npq

    Sbus_val, neg_dSd_dVm = _evaluate_sbus(Sbus, Vm)
    mis = V * np.conj(complex_matvec(Ybus, V)) - Sbus_val
    pvpq = np.r_[pv, pq]
    F = np.r_[np.real(mis[pvpq]), np.imag(mis[pq])]

    normF = np.linalg.norm(F, np.inf)
    if mpopt.verbose > 1:
        print("\n it    max P & Q mismatch (p.u.)", end="")
        print("\n----  ---------------------------", end="")
        print(f"\n{i:3d}        {normF:10.3e}", end="")
    if normF < tol:
        converged = 1
        if mpopt.verbose > 1:
            print("\nConverged!")

    if not lin_solver:
        nx = F.size
        lin_solver = "\\" if nx <= 10 else "LU3"

    while not converged and i < max_it:
        i += 1

        dSbus_dVa, dSbus_dVm = dSbus_dV(Ybus, V)

        if neg_dSd_dVm is None:
            raise ValueError("newtonpf: Sbus callable must return its voltage-magnitude derivative")
        dSbus_dVm = subtract_matrices(dSbus_dVm, neg_dSd_dVm)

        j11 = matrix_real(dSbus_dVa[pvpq][:, pvpq])
        j12 = matrix_real(dSbus_dVm[pvpq][:, pq])
        j21 = matrix_imag(dSbus_dVa[pq][:, pvpq])
        j22 = matrix_imag(dSbus_dVm[pq][:, pq])

        if sparse.issparse(j11) or sparse.issparse(j12) or sparse.issparse(j21) or sparse.issparse(j22):
            J = sparse.bmat([[j11, j12], [j21, j22]], format="csc")
        else:
            J = np.block(
                [
                    [as_dense_matrix(j11), as_dense_matrix(j12)],
                    [as_dense_matrix(j21), as_dense_matrix(j22)],
                ]
            )

        dx = mplinsolve(J, -F, lin_solver)

        if npv:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        V = Vm * np.exp(1j * Va)
        Vm = np.abs(V)
        Va = np.angle(V)

        Sbus_val, neg_dSd_dVm = _evaluate_sbus(Sbus, Vm)
        mis = V * np.conj(complex_matvec(Ybus, V)) - Sbus_val
        F = np.r_[np.real(mis[pvpq]), np.imag(mis[pq])]

        normF = np.linalg.norm(F, np.inf)
        if mpopt.verbose > 1:
            print(f"\n{i:3d}        {normF:10.3e}", end="")
        if normF < tol:
            converged = 1
            if mpopt.verbose:
                print(f"\nNewton's method power flow (power balance, polar) converged in {i:d} iterations.")

    if mpopt.verbose and not converged:
        print(f"\nNewton's method power flow (power balance, polar) did not converge in {i:d} iterations.")

    return V, converged, i
