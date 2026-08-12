# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import sparse

from ..corex import QpMultipliers, QpResult, SolverOutput, normalize_qp_problem
from .glpk_options import glpk_options
from .have_feature_glpk import have_feature_glpk


def qps_glpk_full(
    H: object,
    c: object | None = None,
    A: object | None = None,
    l: object | None = None,
    u: object | None = None,
    xmin: object | None = None,
    xmax: object | None = None,
    x0: object | None = None,
    opt: Mapping[str, object] | None = None,
) -> QpResult:
    """Solve an LP with GLPK and return all five solver outputs."""
    import pyomo.environ as pyo

    problem = normalize_qp_problem(
        H, c, A, l, u, xmin, xmax, x0, opt, solver_name="qps_glpk", linear_only=True
    )
    A_matrix = sparse.csr_matrix(problem.A)
    nx = problem.nx
    nA = problem.n_constraints

    ieq = np.where(np.abs(problem.u - problem.l) <= np.finfo(float).eps)[0]
    igt = np.where((problem.u >= 1e10) & (problem.l > -1e10))[0]
    ilt = np.where((problem.l <= -1e10) & (problem.u < 1e10))[0]
    ibx = np.where(
        (np.abs(problem.u - problem.l) > np.finfo(float).eps)
        & (problem.u < 1e10)
        & (problem.l > -1e10)
    )[0]
    stacked = sparse.csr_matrix(
        sparse.vstack(
            [A_matrix[ieq, :], A_matrix[ilt, :], -A_matrix[igt, :], A_matrix[ibx, :], -A_matrix[ibx, :]],
            format="csr",
        )
    )
    bounds = np.r_[problem.u[ieq], problem.u[ilt], -problem.l[igt], problem.u[ibx], -problem.l[ibx]]

    nlt = len(ilt)
    ngt = len(igt)
    nbx = len(ibx)
    neq = len(ieq)
    nie = nlt + ngt + 2 * nbx

    if not have_feature_glpk():
        raise NotImplementedError("qps_glpk requires Pyomo with an available GLPK solver")

    native_options = problem.options.get("glpk_opt")
    glpk_opt = glpk_options(native_options if isinstance(native_options, Mapping) else None)
    verbose_value = problem.options.get("verbose", 0)
    verbose = int(verbose_value) if isinstance(verbose_value, (int, float)) else 0
    glpk_opt["msglev"] = verbose

    # Pyomo creates model components dynamically, so its model and solver are
    # intentionally isolated as Any at this third-party boundary.
    def dynamic(value: object) -> Any:
        return value

    def required_float(value: Any) -> float:
        if value is None:
            raise RuntimeError("GLPK did not return a required primal or objective value")
        return float(value)

    model = dynamic(pyo.ConcreteModel())
    model.I = pyo.RangeSet(0, nx - 1)
    variables = dynamic(pyo.Var(model.I, domain=pyo.Reals))
    model.x = variables
    for i in range(nx):
        variables[i].setlb(None if np.isneginf(problem.xmin[i]) else float(problem.xmin[i]))
        variables[i].setub(None if np.isposinf(problem.xmax[i]) else float(problem.xmax[i]))

    model.obj = pyo.Objective(expr=sum(float(problem.c[i]) * variables[i] for i in range(nx)))
    constraint_list = dynamic(pyo.ConstraintList())
    model.cons = constraint_list
    rows: list[object] = []
    for k in range(bounds.size):
        row = stacked.getrow(k)
        expression = sum(float(value) * variables[int(j)] for j, value in zip(row.indices, row.data))
        constraint = constraint_list.add(expression == float(bounds[k])) if k < neq else constraint_list.add(
            expression <= float(bounds[k])
        )
        rows.append(constraint)

    dual_suffix = dynamic(pyo.Suffix(direction=pyo.Suffix.IMPORT))
    reduced_cost_suffix = dynamic(pyo.Suffix(direction=pyo.Suffix.IMPORT))
    model.dual = dual_suffix
    model.rc = reduced_cost_suffix
    solver = dynamic(pyo.SolverFactory("glpk"))
    results = solver.solve(model, tee=verbose > 1)

    term = str(results.solver.termination_condition).lower()
    status = str(results.solver.status).lower()
    eflag = 1 if "optimal" in term else 0
    output: SolverOutput = {"status": status, "term": term}

    solution = np.array([required_float(pyo.value(variables[i])) for i in range(nx)], dtype=float)
    objective = required_float(pyo.value(model.obj))
    lam_eqlin = np.zeros(neq, dtype=float)
    lam_ineqlin = np.zeros(nie, dtype=float)
    for k, constraint in enumerate(rows):
        dual = required_float(dual_suffix.get(constraint, 0.0))
        if k < neq:
            lam_eqlin[k] = dual
        else:
            lam_ineqlin[k - neq] = dual
    reduced_costs = np.array(
        [required_float(reduced_cost_suffix.get(variables[i], 0.0)) for i in range(nx)], dtype=float
    )
    lower = np.maximum(reduced_costs, 0)
    upper = np.maximum(-reduced_costs, 0)

    kl = np.where(lam_eqlin > 0)[0]
    ku = np.where(lam_eqlin < 0)[0]
    mu_l = np.zeros(nA, dtype=float)
    mu_l[ieq[kl]] = lam_eqlin[kl]
    mu_l[igt] = -lam_ineqlin[nlt + np.arange(ngt)]
    mu_l[ibx] = -lam_ineqlin[nlt + ngt + nbx + np.arange(nbx)]
    mu_u = np.zeros(nA, dtype=float)
    mu_u[ieq[ku]] = -lam_eqlin[ku]
    mu_u[ilt] = -lam_ineqlin[np.arange(nlt)]
    mu_u[ibx] = -lam_ineqlin[nlt + ngt + np.arange(nbx)]

    multipliers: QpMultipliers = {"mu_l": mu_l, "mu_u": mu_u, "lower": lower, "upper": upper}
    return solution, objective, eflag, output, multipliers


def qps_glpk(
    H: object,
    c: object | None = None,
    A: object | None = None,
    l: object | None = None,
    u: object | None = None,
    xmin: object | None = None,
    xmax: object | None = None,
    x0: object | None = None,
    opt: Mapping[str, object] | None = None,
    nargout: int = 1,
) -> object:
    """MATPOWER-compatible GLPK entry point; use ``qps_glpk_full`` in typed code."""
    result = qps_glpk_full(H, c, A, l, u, xmin, xmax, x0, opt)
    return result[:nargout] if nargout > 1 else result[0]
