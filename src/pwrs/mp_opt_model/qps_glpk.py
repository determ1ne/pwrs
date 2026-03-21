# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .glpk_options import glpk_options
from .have_feature_glpk import have_feature_glpk


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


def qps_glpk(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None, x0=None, opt=None, nargout=1):
    """Linear-program solver wrapper based on GLPK.

    Solves ``min c' * x`` subject to linear constraints and variable
    bounds. ``H`` is accepted for API compatibility but must be empty or
    zero because GLPK is used only for LP problems in this port.

    Parameters
    ----------
    H : array_like or dict
        Dummy quadratic cost matrix or a problem dict containing the full
        solver inputs.
    c : array_like, optional
        Linear cost vector.
    A, l, u : array_like, optional
        Linear constraints ``l <= A*x <= u``.
    xmin, xmax : array_like, optional
        Variable bounds.
    x0 : array_like, optional
        Initial point. Accepted for interface compatibility and ignored by
        the GLPK backend.
    opt : dict, optional
        Solver options dict containing ``glpk_opt`` and verbosity controls.
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
            raise ValueError("qps_glpk: LP problem must include constraints or variable bounds")
        if not _is_empty(A):
            nx = A.shape[1]
        elif not _is_empty(xmin):
            nx = len(np.asarray(xmin).reshape(-1))
        else:
            nx = len(np.asarray(xmax).reshape(-1))
    else:
        raise ValueError("qps_glpk: GLPK handles only LP problems, not QP problems")

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

    ieq = np.where(np.abs(u - l) <= np.finfo(float).eps)[0]
    igt = np.where((u >= 1e10) & (l > -1e10))[0]
    ilt = np.where((l <= -1e10) & (u < 1e10))[0]
    ibx = np.where((np.abs(u - l) > np.finfo(float).eps) & (u < 1e10) & (l > -1e10))[0]
    AA = sparse.vstack([A[ieq, :], A[ilt, :], -A[igt, :], A[ibx, :], -A[ibx, :]], format="csc")
    bb = np.r_[u[ieq], u[ilt], -l[igt], u[ibx], -l[ibx]]

    nlt = len(ilt)
    ngt = len(igt)
    nbx = len(ibx)
    neq = len(ieq)
    nie = nlt + ngt + 2 * nbx

    if not have_feature_glpk():
        raise NotImplementedError("qps_glpk requires Pyomo with an available GLPK solver")

    glpk_opt = glpk_options(opt.get("glpk_opt") if isinstance(opt, dict) else None, nargout=1)
    verbose = int(opt.get("verbose", 0)) if isinstance(opt, dict) else 0
    glpk_opt["msglev"] = verbose

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, nx - 1)
    model.x = pyo.Var(model.I, domain=pyo.Reals)
    for i in range(nx):
        model.x[i].setlb(None if np.isneginf(xmin[i]) else float(xmin[i]))
        model.x[i].setub(None if np.isposinf(xmax[i]) else float(xmax[i]))

    cvec = c.copy()
    model.obj = pyo.Objective(expr=sum(float(cvec[i]) * model.x[i] for i in range(nx)))
    model.cons = pyo.ConstraintList()
    rows = []
    AA = AA.tocsr()
    for k in range(AA.shape[0]):
        row = AA.getrow(k)
        expr = sum(float(val) * model.x[int(j)] for j, val in zip(row.indices, row.data))
        if k < neq:
            con = model.cons.add(expr == float(bb[k]))
        else:
            con = model.cons.add(expr <= float(bb[k]))
        rows.append(con)

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=verbose > 1)

    term = str(results.solver.termination_condition).lower()
    status = str(results.solver.status).lower()
    optimal = "optimal" in term
    eflag = 1 if optimal else 0
    output = {"status": status, "term": term}

    x = np.array([pyo.value(model.x[i]) for i in range(nx)], dtype=float)
    f = float(pyo.value(model.obj))

    lam_eqlin = np.zeros(neq, dtype=float)
    lam_ineqlin = np.zeros(nie, dtype=float)
    for k, con in enumerate(rows):
        dual = float(model.dual.get(con, 0.0))
        if k < neq:
            lam_eqlin[k] = dual
        else:
            lam_ineqlin[k - neq] = dual
    redcosts = np.array([float(model.rc.get(model.x[i], 0.0)) for i in range(nx)], dtype=float)
    lam_lower = redcosts.copy()
    lam_upper = -redcosts.copy()
    lam_lower[lam_lower < 0] = 0
    lam_upper[lam_upper < 0] = 0

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

    lambda_ = {"mu_l": mu_l, "mu_u": mu_u, "lower": lam_lower, "upper": lam_upper}
    outputs = (x, f, eflag, output, lambda_)
    return outputs[:nargout] if nargout > 1 else x
