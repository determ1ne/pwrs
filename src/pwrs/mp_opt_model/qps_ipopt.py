# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..corex import QpMultipliers, QpResult, SolverOutput, normalize_qp_problem
from .have_feature_ipopt import have_feature_ipopt
from .ipopt_options import ipopt_options


def qps_ipopt_full(
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
    """Solve a QP with IPOPT and return all five solver outputs."""
    import pyomo.environ as pyo

    problem = normalize_qp_problem(H, c, A, l, u, xmin, xmax, x0, opt, solver_name="qps_ipopt")
    if not have_feature_ipopt():
        raise NotImplementedError("qps_ipopt requires an available IPOPT backend")

    H_matrix = problem.H.tocoo()
    from scipy import sparse

    A_matrix = sparse.csr_matrix(problem.A)
    nx = problem.nx
    nA = problem.n_constraints
    options = problem.options
    verbose_value = options.get("verbose", 0)
    verbose = int(verbose_value) if isinstance(verbose_value, (int, float)) else 0
    native_options = options.get("ipopt_opt")
    ipopt_opt = ipopt_options(native_options if isinstance(native_options, Mapping) else None)
    ipopt_opt["print_level"] = min(12, verbose * 2 + 1) if verbose else 0

    # Pyomo creates model components dynamically, so its model and solver are
    # intentionally isolated as Any at this third-party boundary.
    def dynamic(value: object) -> Any:
        return value

    def required_float(value: Any) -> float:
        if value is None:
            raise RuntimeError("IPOPT did not return a required primal or objective value")
        return float(value)

    model = dynamic(pyo.ConcreteModel())
    model.I = pyo.RangeSet(0, nx - 1)
    variables = dynamic(pyo.Var(model.I, domain=pyo.Reals, initialize={i: float(problem.x0[i]) for i in range(nx)}))
    model.x = variables
    for i in range(nx):
        variables[i].setlb(None if np.isneginf(problem.xmin[i]) else float(problem.xmin[i]))
        variables[i].setub(None if np.isposinf(problem.xmax[i]) else float(problem.xmax[i]))

    expression = sum(float(problem.c[i]) * variables[i] for i in range(nx))
    h_rows = np.asarray(H_matrix.row if H_matrix.row is not None else [], dtype=int)
    h_columns = np.asarray(H_matrix.col if H_matrix.col is not None else [], dtype=int)
    for i, j, value in zip(h_rows, h_columns, H_matrix.data):
        if j < i:
            continue
        coefficient = float(value)
        if i == j:
            expression += 0.5 * coefficient * variables[int(i)] * variables[int(j)]
        else:
            expression += coefficient * variables[int(i)] * variables[int(j)]
    model.obj = pyo.Objective(expr=expression)

    constraint_list = dynamic(pyo.ConstraintList())
    model.cons = constraint_list
    constraints: list[tuple[str, int, object]] = []
    for k in range(nA):
        row = A_matrix.getrow(k)
        row_expression = sum(float(value) * variables[int(j)] for j, value in zip(row.indices, row.data))
        if abs(problem.u[k] - problem.l[k]) <= np.finfo(float).eps:
            constraint = constraint_list.add(row_expression == float(problem.u[k]))
            constraints.append(("e", k, constraint))
        else:
            if problem.l[k] > -np.inf:
                constraints.append(("l", k, constraint_list.add(row_expression >= float(problem.l[k]))))
            if problem.u[k] < np.inf:
                constraints.append(("u", k, constraint_list.add(row_expression <= float(problem.u[k]))))

    dual_suffix = dynamic(pyo.Suffix(direction=pyo.Suffix.IMPORT))
    lower_suffix = dynamic(pyo.Suffix(direction=pyo.Suffix.IMPORT))
    upper_suffix = dynamic(pyo.Suffix(direction=pyo.Suffix.IMPORT))
    model.dual = dual_suffix
    model.ipopt_zL_out = lower_suffix
    model.ipopt_zU_out = upper_suffix
    solver = dynamic(pyo.SolverFactory("ipopt"))
    for key, value in ipopt_opt.items():
        solver.options[key] = value
    print_level = ipopt_opt.get("print_level", 0)
    results = solver.solve(model, tee=isinstance(print_level, (int, float)) and print_level > 1)

    term = str(results.solver.termination_condition).lower()
    status = str(results.solver.status).lower()
    eflag = 1 if "optimal" in term or "locallyoptimal" in term or "feasible" in term else 0
    solution = np.array([required_float(pyo.value(variables[i])) for i in range(nx)], dtype=float)
    objective = required_float(pyo.value(model.obj))
    iterations_value = getattr(results.solver, "iterations", None)
    iterations = int(iterations_value) if isinstance(iterations_value, (int, float)) else None
    output: SolverOutput = {"status": status, "term": term, "iterations": iterations}

    mu_l = np.zeros(nA, dtype=float)
    mu_u = np.zeros(nA, dtype=float)
    for kind, index, constraint in constraints:
        dual = required_float(dual_suffix.get(constraint, 0.0))
        if kind == "e":
            if dual < 0:
                mu_l[index] = -dual
            elif dual > 0:
                mu_u[index] = dual
        elif kind == "l":
            mu_l[index] = max(mu_l[index], dual)
        else:
            mu_u[index] = max(mu_u[index], -dual)

    lower = np.array([required_float(lower_suffix.get(variables[i], np.nan)) for i in range(nx)], dtype=float)
    upper = np.array([required_float(upper_suffix.get(variables[i], np.nan)) for i in range(nx)], dtype=float)
    multipliers: QpMultipliers = {"mu_l": mu_l, "mu_u": mu_u, "lower": lower, "upper": upper}
    return solution, objective, eflag, output, multipliers


def qps_ipopt(
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
    """MATPOWER-compatible IPOPT QP entry point.

    Native Python callers should use :func:`qps_ipopt_full` for a statically
    typed fixed-shape result.
    """
    result = qps_ipopt_full(H, c, A, l, u, xmin, xmax, x0, opt)
    return result[:nargout] if nargout > 1 else result[0]
