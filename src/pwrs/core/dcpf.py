# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def dcpf(B, Pbus, Va0, ref, pv, pq):
    """Solve the linear DC power flow equations.

    solves for the bus voltage angles at all but the reference bus,
    given the full system B matrix and the vector of bus real power injections,
    the initial vector of bus voltage angles (in radians), and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. Returns a vector of bus voltage angles in radians.

    Parameters
    ----------
    B : array_like or sparse matrix
        DC power flow susceptance matrix.
    Pbus : array_like
        Real bus power injections.
    Va0 : array_like
        Initial/reference bus voltage angles.
    ref : array_like
        One-based reference bus indices.
    pv : array_like
        One-based PV bus indices.
    pq : array_like
        One-based PQ bus indices.
    Returns
    -------
    tuple
        ``(Va, success)`` with solved bus voltage angles and a convergence flag.
    """
    va_threshold = 1e5

    if sparse.issparse(B):
        B = B.tocsc()
    else:
        B = np.asarray(B)

    Pbus = np.asarray(Pbus, dtype=float).reshape(-1)
    Va0 = np.asarray(Va0, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=int).reshape(-1) - 1
    pv = np.asarray(pv, dtype=int).reshape(-1) - 1
    pq = np.asarray(pq, dtype=int).reshape(-1) - 1

    Va = Va0.copy()
    success = 1.0
    pvpq = np.r_[pv, pq]

    try:
        rhs = Pbus[pvpq] - B[np.ix_(pvpq, ref)] @ Va0[ref]
        submatrix = B[np.ix_(pvpq, pvpq)]
        if sparse.issparse(submatrix):
            Va[pvpq] = spsolve(submatrix, rhs)
        else:
            Va[pvpq] = np.linalg.solve(submatrix, rhs)
    except Exception:
        success = 0.0

    if np.max(np.abs(Va)) > va_threshold:
        success = 0.0

    return Va.reshape(-1, 1), success
