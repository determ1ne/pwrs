# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Callable
from importlib import import_module

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

type LinearSolution = np.ndarray
type SparseSolver = Callable[[sparse.csc_matrix, np.ndarray], LinearSolution]

_SUPERLU_NAMES = {"LU3", "LU3A", "LU3M", "LU3AM", "SUPERLU"}
_UMFPACK_NAMES = {"LU", "LU4", "LU4M", "LU5", "LU5M", "UMFPACK"}


def _as_rhs(b: object) -> np.ndarray:
    return np.asarray(b, dtype=float)


def _as_csc(A: object) -> sparse.csc_matrix:
    return sparse.csc_matrix(A, dtype=float)


def _solve_default(A: object, b: object) -> LinearSolution:
    rhs = _as_rhs(b)
    if sparse.issparse(A):
        return np.asarray(spsolve(A, rhs), dtype=float)
    return np.asarray(np.linalg.solve(np.asarray(A, dtype=float), rhs), dtype=float)


def _solve_superlu(A: object, b: object) -> LinearSolution:
    """Solve with SuperLU using a minimum-degree ordering for LU3 compatibility."""
    matrix = _as_csc(A)
    factor = splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=1.0)
    return np.asarray(factor.solve(_as_rhs(b)), dtype=float)


def _solve_columns(solver: SparseSolver, matrix: sparse.csc_matrix, rhs: np.ndarray) -> LinearSolution:
    if rhs.ndim == 1:
        return np.asarray(solver(matrix, rhs), dtype=float)
    return np.column_stack([solver(matrix, rhs[:, column]) for column in range(rhs.shape[1])])


def _solve_umfpack(A: object, b: object) -> LinearSolution:
    """Solve with SuiteSparse UMFPACK through scikit-umfpack."""
    umfpack_spsolve = import_module("scikits.umfpack").spsolve
    matrix = _as_csc(A)
    rhs = _as_rhs(b)
    return _solve_columns(umfpack_spsolve, matrix, rhs)


def _solve_pardiso(A: object, b: object) -> LinearSolution:
    """Solve with Intel MKL PARDISO through PyPardiso."""
    pardiso_spsolve = import_module("pypardiso").spsolve
    return np.asarray(pardiso_spsolve(sparse.csr_matrix(A, dtype=float), _as_rhs(b)), dtype=float)


def _solve_klu(A: object, b: object) -> LinearSolution:
    """Solve with SuiteSparse KLU through nbklu."""
    import nbklu

    matrix = _as_csc(A)
    factor = nbklu.KLUSolver()
    factor.common.btf = 0
    factor.analyze(matrix.indptr.size - 1, matrix.indptr, matrix.indices)
    factor.factor(matrix.data)
    return np.asarray(factor.solve(_as_rhs(b)), dtype=float)


def _fallback(solver: str, package: str, A: object, b: object) -> LinearSolution:
    warnings.warn(
        f"mplinsolve: {solver} requires {package}, falling back to SciPy SuperLU",
        RuntimeWarning,
        stacklevel=2,
    )
    return _solve_superlu(A, b)


def mplinsolve(A: object, b: object, solver: str = "", opt: object | None = None) -> LinearSolution:
    """Solve ``A @ x = b`` using a MATPOWER-compatible backend name.

    ``LU3`` aliases select SuperLU. ``LU``, ``LU4`` and ``LU5`` aliases
    select UMFPACK; pwrs intentionally does not distinguish the MATLAB
    four- and five-output LU forms. ``PARDISO`` and ``KLU`` select their
    corresponding optional sparse direct solvers.

    ``opt`` is retained only for MATPOWER call compatibility.
    """
    del opt
    normalized = solver.upper()
    if normalized in {"", "\\"}:
        return _solve_default(A, b)
    if normalized in _SUPERLU_NAMES:
        return _solve_superlu(A, b)
    if normalized in _UMFPACK_NAMES:
        try:
            return _solve_umfpack(A, b)
        except (ImportError, OSError):
            return _fallback(normalized, "scikit-umfpack", A, b)
    if normalized == "PARDISO":
        try:
            return _solve_pardiso(A, b)
        except (ImportError, OSError):
            return _fallback(normalized, "pypardiso", A, b)
    if normalized == "KLU":
        try:
            return _solve_klu(A, b)
        except (ImportError, OSError):
            return _fallback(normalized, "nbklu", A, b)

    warnings.warn(
        f"mplinsolve: unrecognized solver {solver!r}, falling back to SciPy SuperLU",
        RuntimeWarning,
        stacklevel=2,
    )
    return _solve_superlu(A, b)
