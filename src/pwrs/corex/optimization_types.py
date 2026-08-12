# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NotRequired, Protocol, TypedDict, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .mpoption import MipsConfig

type SolverOptions = dict[str, object]
type FloatVector = npt.NDArray[np.float64]


class NlpOptions(TypedDict):
    """Normalized options accepted by the common nonlinear solver layer."""

    alg: str
    verbose: int
    mips_opt: NotRequired[MipsConfig]
    fmincon_opt: NotRequired[SolverOptions]
    ipopt_opt: NotRequired[SolverOptions]
    knitro_opt: NotRequired[SolverOptions]
    x0: NotRequired[np.ndarray]


@dataclass(frozen=True)
class QpProblem:
    """Validated, fully populated quadratic-program input."""

    H: sparse.csc_matrix
    c: FloatVector
    A: sparse.csc_matrix
    l: FloatVector
    u: FloatVector
    xmin: FloatVector
    xmax: FloatVector
    x0: FloatVector
    options: SolverOptions
    nx: int
    n_constraints: int


class SolverOutput(TypedDict, total=False):
    alg: str
    status: str | int
    status_msg: str
    term: str
    iterations: int | None
    message: str
    hist: list[dict[str, float]]


class QpMultipliers(TypedDict):
    mu_l: FloatVector
    mu_u: FloatVector
    lower: FloatVector
    upper: FloatVector


type QpResult = tuple[FloatVector, float, int, SolverOutput, QpMultipliers]


class NlpMultipliers(QpMultipliers, total=False):
    eqnonlin: FloatVector
    ineqnonlin: FloatVector


type NlpResult = tuple[FloatVector, float, int, SolverOutput, NlpMultipliers]


@runtime_checkable
class _SparseInput(Protocol):
    shape: tuple[int, int]
    nnz: int
    data: npt.NDArray[np.number]


def _is_empty(value: object | None) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return not value
    if isinstance(value, _SparseInput):
        return value.shape[0] == 0 or value.shape[1] == 0
    return np.asarray(value).size == 0


def _is_zero_matrix(value: object | None) -> bool:
    if _is_empty(value):
        return True
    if isinstance(value, _SparseInput):
        return value.nnz == 0 or not np.any(value.data)
    return not bool(np.any(np.asarray(value)))


def _vector(value: object | None, default: float, size: int, name: str) -> FloatVector:
    if _is_empty(value):
        return np.full(size, default, dtype=float)
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must contain {size} elements, got {vector.size}")
    return vector


def _matrix_shape(value: object) -> tuple[int, int]:
    shape = value.shape if isinstance(value, _SparseInput) else np.asarray(value).shape
    if len(shape) != 2:
        raise ValueError("A must be a two-dimensional matrix")
    return int(shape[0]), int(shape[1])


def normalize_qp_problem(
    H: object,
    c: object | None = None,
    A: object | None = None,
    l: object | None = None,
    u: object | None = None,
    xmin: object | None = None,
    xmax: object | None = None,
    x0: object | None = None,
    opt: Mapping[str, object] | MipsConfig | None = None,
    *,
    solver_name: str,
    linear_only: bool = False,
) -> QpProblem:
    """Normalize a MATPOWER-style QP call into one typed representation."""
    if isinstance(H, Mapping):
        problem = H
        H = problem.get("H")
        c = problem.get("c")
        A = problem.get("A")
        l = problem.get("l")
        u = problem.get("u")
        xmin = problem.get("xmin")
        xmax = problem.get("xmax")
        x0 = problem.get("x0")
        configured_options = problem.get("opt")
        opt = configured_options if isinstance(configured_options, (Mapping, MipsConfig)) else None

    zero_hessian = _is_zero_matrix(H)
    if zero_hessian:
        if _is_empty(A) and _is_empty(xmin) and _is_empty(xmax):
            raise ValueError(f"{solver_name}: LP problem must include constraints or variable bounds")
        if not _is_empty(A):
            assert A is not None
            _, nx = _matrix_shape(A)
        elif not _is_empty(xmin):
            assert xmin is not None
            nx = int(np.asarray(xmin).size)
        else:
            assert xmax is not None
            nx = int(np.asarray(xmax).size)
        hessian = sparse.csc_matrix((nx, nx), dtype=float)
    else:
        if linear_only:
            raise ValueError(f"{solver_name}: GLPK handles only LP problems, not QP problems")
        if H is None:
            raise RuntimeError("non-zero Hessian unexpectedly missing")
        h_rows, h_columns = _matrix_shape(H)
        if h_rows != h_columns:
            raise ValueError("H must be a square matrix")
        hessian = sparse.csc_matrix(H, dtype=float)
        nx = h_rows

    c_vector = _vector(c, 0.0, nx, "c")
    no_active_linear_constraints = _is_empty(A) or (
        (_is_empty(l) or np.all(np.asarray(l) == -np.inf))
        and (_is_empty(u) or np.all(np.asarray(u) == np.inf))
    )
    if no_active_linear_constraints:
        linear = sparse.csc_matrix((0, nx), dtype=float)
        n_constraints = 0
    else:
        if A is None:
            raise RuntimeError("active linear constraint matrix unexpectedly missing")
        n_constraints, columns = _matrix_shape(A)
        if columns != nx:
            raise ValueError(f"A must have {nx} columns, got {columns}")
        linear = sparse.csc_matrix(A, dtype=float)

    options = opt.to_dict() if isinstance(opt, MipsConfig) else ({} if opt is None else dict(opt))
    return QpProblem(
        H=hessian,
        c=c_vector,
        A=linear,
        l=_vector(l, -np.inf, n_constraints, "l"),
        u=_vector(u, np.inf, n_constraints, "u"),
        xmin=_vector(xmin, -np.inf, nx, "xmin"),
        xmax=_vector(xmax, np.inf, nx, "xmax"),
        x0=_vector(x0, 0.0, nx, "x0"),
        options=options,
        nx=nx,
        n_constraints=n_constraints,
    )
