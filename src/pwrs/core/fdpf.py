# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..corex import ComplexArray, IntArray, MatpowerConfig, Matrix, SbusFunction, as_csc_matrix, complex_matvec
from .mpoption import mpoption


def _evaluate_sbus(Sbus: SbusFunction, Vm: np.ndarray) -> ComplexArray:
    result = Sbus(Vm)
    if isinstance(result, tuple):
        if len(result) == 0:
            raise ValueError("fdpf: Sbus callable returned no outputs")
        return np.asarray(result[0]).reshape(-1)
    return np.asarray(result).reshape(-1)


def _solve_lu(lu, rhs):
    out = lu.solve(np.asarray(rhs).reshape(-1, 1))
    return np.asarray(out).reshape(-1)


def fdpf(
    Ybus: Matrix,
    Sbus: SbusFunction,
    V0: ComplexArray,
    Bp: Matrix,
    Bpp: Matrix,
    ref: IntArray,
    pv: IntArray,
    pq: IntArray,
    mpopt: MatpowerConfig | dict[str, object] | None = None,
) -> tuple[ComplexArray, float, float]:
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
    Returns
    -------
    tuple
        ``(V, converged, iterations)``.
    """
    if not callable(Sbus):
        raise TypeError("fdpf: Sbus must be callable in the Python API")

    if mpopt is None:
        mpopt = mpoption()
    elif not isinstance(mpopt, MatpowerConfig):
        mpopt = mpoption(mpopt)

    tol = float(mpopt.pf.tol)
    max_it = int(mpopt.pf.fd.max_it)
    verbose = int(mpopt.verbose)

    if sparse.issparse(Ybus):
        Ybus = as_csc_matrix(Ybus)
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
    mis = (V * np.conj(complex_matvec(Ybus, V)) - _evaluate_sbus(Sbus, Vm)) / Vm
    P = np.real(mis[pvpq])
    Q = np.imag(mis[pq])

    normP = np.linalg.norm(P, np.inf) if P.size else 0.0
    normQ = np.linalg.norm(Q, np.inf) if Q.size else 0.0
    if normP < tol and normQ < tol:
        converged = 1.0

    Bp = Bp[pvpq][:, pvpq]
    Bpp = Bpp[pq][:, pq]
    Bp_lu = splu(sparse.csc_matrix(Bp))
    Bpp_lu = splu(sparse.csc_matrix(Bpp))

    while not converged and i < max_it:
        i += 1.0

        dVa = -_solve_lu(Bp_lu, P)
        Va[pvpq] = Va[pvpq] + dVa
        V = Vm * np.exp(1j * Va)

        mis = (V * np.conj(complex_matvec(Ybus, V)) - _evaluate_sbus(Sbus, Vm)) / Vm
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

        mis = (V * np.conj(complex_matvec(Ybus, V)) - _evaluate_sbus(Sbus, Vm)) / Vm
        P = np.real(mis[pvpq])
        Q = np.imag(mis[pq])
        normP = np.linalg.norm(P, np.inf) if P.size else 0.0
        normQ = np.linalg.norm(Q, np.inf) if Q.size else 0.0
        if normP < tol and normQ < tol:
            converged = 1.0
            break

    if verbose and not converged:
        # TODO(core): align the non-convergence diagnostic with MATPOWER's
        # iteration report; currently the solver only exposes the flag.
        print(f"\nFast-decoupled power flow did not converge in {i:d} iterations.")

    return V.reshape(-1, 1), converged, i
