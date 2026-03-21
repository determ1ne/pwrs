# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .have_feature_ipopt import have_feature_ipopt
from .ipopt_options import ipopt_options


def _is_empty(value):
    return (
        value is None
        or (isinstance(value, (list, tuple)) and len(value) == 0)
        or (hasattr(value, "size") and value.size == 0)
    )


def _as_array(value, default, size=None):
    if _is_empty(value):
        if size is None:
            return np.asarray(default, dtype=float)
        return np.full(size, default, dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def qps_ipopt(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None, x0=None, opt=None, nargout=1):
    """Quadratic-program solver wrapper based on IPOPT.

    Solves ``min 0.5 * x' * H * x + c' * x`` subject to linear constraints
    and variable bounds.

    Parameters
    ----------
    H : array_like or dict
        Quadratic cost matrix or a problem dict containing the full solver
        inputs.
    c : array_like, optional
        Linear cost vector.
    A, l, u : array_like, optional
        Linear constraints ``l <= A*x <= u``.
    xmin, xmax : array_like, optional
        Variable bounds.
    x0 : array_like, optional
        Initial point.
    opt : dict, optional
        Solver options dict containing ``ipopt_opt`` and verbosity controls.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or ndarray
        Returns ``(x, f, eflag, output, lambda_)`` or the leading subset
        requested by ``nargout``.
    """
    import pyomo.environ as pyo

    if isinstance(H, dict):
        p = H
        opt = p.get("opt", [])
        x0 = p.get("x0", [])
        xmax = p.get("xmax", [])
        xmin = p.get("xmin", [])
        u = p.get("u", [])
        l = p.get("l", [])
        A = p.get("A", [])
        c = p.get("c", [])
        H = p.get("H", [])
    else:
        if opt is None:
            opt = []
        if x0 is None:
            x0 = []
        if xmax is None:
            xmax = []
        if xmin is None:
            xmin = []
        if u is None:
            u = []
        if l is None:
            l = []
        if A is None:
            A = []
        if c is None:
            c = []

    if _is_empty(H) or not np.any(np.asarray(H)):
        if _is_empty(A) and _is_empty(xmin) and _is_empty(xmax):
            raise ValueError("qps_ipopt: LP problem must include constraints or variable bounds")
        if not _is_empty(A):
            nx = A.shape[1]
        elif not _is_empty(xmin):
            nx = len(np.asarray(xmin).reshape(-1))
        else:
            nx = len(np.asarray(xmax).reshape(-1))
        H = sparse.csc_matrix((nx, nx))
    else:
        if sparse.issparse(H):
            H = H.tocsr().astype(float)
        else:
            H = sparse.csc_matrix(np.asarray(H, dtype=float))
        nx = H.shape[0]

    if not have_feature_ipopt():
        raise NotImplementedError("qps_ipopt requires an available IPOPT backend")

    c = _as_array(c, 0.0, nx)
    if _is_empty(A) or (
        (_is_empty(l) or np.all(np.asarray(l) == -np.inf)) and (_is_empty(u) or np.all(np.asarray(u) == np.inf))
    ):
        A = sparse.csc_matrix((0, nx))
    elif not sparse.issparse(A):
        A = sparse.csc_matrix(np.asarray(A, dtype=float))
    else:
        A = A.tocsr().astype(float)
    nA = A.shape[0]
    u = _as_array(u, np.inf, nA)
    l = _as_array(l, -np.inf, nA)
    xmin = _as_array(xmin, -np.inf, nx)
    xmax = _as_array(xmax, np.inf, nx)
    x0 = _as_array(x0, 0.0, nx)

    verbose = int(opt.get("verbose", 0)) if isinstance(opt, dict) else 0
    ipopt_opt = ipopt_options(opt.get("ipopt_opt") if isinstance(opt, dict) else None, nargout=1)
    if verbose:
        ipopt_opt["print_level"] = min(12, verbose * 2 + 1)
    else:
        ipopt_opt["print_level"] = 0

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, nx - 1)
    model.x = pyo.Var(model.I, domain=pyo.Reals, initialize={i: float(x0[i]) for i in range(nx)})
    for i in range(nx):
        model.x[i].setlb(None if np.isneginf(xmin[i]) else float(xmin[i]))
        model.x[i].setub(None if np.isposinf(xmax[i]) else float(xmax[i]))

    H = H.tocoo()
    expr = sum(float(c[i]) * model.x[i] for i in range(nx))
    for i, j, val in zip(H.row, H.col, H.data):
        if j < i:
            continue
        coeff = float(val)
        if i == j:
            expr += 0.5 * coeff * model.x[int(i)] * model.x[int(j)]
        else:
            expr += coeff * model.x[int(i)] * model.x[int(j)]
    model.obj = pyo.Objective(expr=expr)

    model.cons = pyo.ConstraintList()
    cons = []
    A = A.tocsr()
    for k in range(nA):
        row = A.getrow(k)
        row_expr = sum(float(val) * model.x[int(j)] for j, val in zip(row.indices, row.data))
        if abs(u[k] - l[k]) <= np.finfo(float).eps:
            con = model.cons.add(row_expr == float(u[k]))
        else:
            if l[k] > -np.inf:
                cons.append(("l", k, model.cons.add(row_expr >= float(l[k]))))
            if u[k] < np.inf:
                cons.append(("u", k, model.cons.add(row_expr <= float(u[k]))))
            continue
        cons.append(("e", k, con))

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("ipopt")
    for key, value in ipopt_opt.items():
        solver.options[key] = value
    results = solver.solve(model, tee=ipopt_opt.get("print_level", 0) > 1)

    term = str(results.solver.termination_condition).lower()
    status = str(results.solver.status).lower()
    eflag = 1 if "optimal" in term or "locallyoptimal" in term or "feasible" in term else 0
    x = np.array([pyo.value(model.x[i]) for i in range(nx)], dtype=float)
    f = float(pyo.value(model.obj))
    output = {"status": status, "term": term, "iterations": getattr(results.solver, "iterations", None)}

    mu_l = np.zeros(nA, dtype=float)
    mu_u = np.zeros(nA, dtype=float)
    for kind, idx, con in cons:
        dual = float(model.dual.get(con, 0.0))
        if kind == "e":
            if dual < 0:
                mu_l[idx] = -dual
            elif dual > 0:
                mu_u[idx] = dual
        elif kind == "l":
            mu_l[idx] = max(mu_l[idx], dual)
        else:
            mu_u[idx] = max(mu_u[idx], -dual)

    zl = np.array([float(model.ipopt_zL_out.get(model.x[i], np.nan)) for i in range(nx)], dtype=float)
    zu = np.array([float(model.ipopt_zU_out.get(model.x[i], np.nan)) for i in range(nx)], dtype=float)
    lambda_ = {"mu_l": mu_l, "mu_u": mu_u, "lower": zl, "upper": zu}
    outputs = (x, f, eflag, output, lambda_)
    return outputs[:nargout] if nargout > 1 else x
