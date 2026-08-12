# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import ctypes
import os
import shutil
from typing import Any, cast

import numpy as np
from scipy import sparse

from ..corex import NlpMultipliers, NlpResult, SolverOutput
from .have_feature_ipopt import have_feature_ipopt
from .ipopt_options import ipopt_options


def _preload_ipopt_library():
    candidates: list[str] = []
    ipopt_bin = shutil.which("ipopt")
    if ipopt_bin:
        bindir = os.path.dirname(os.path.realpath(ipopt_bin))
        candidates.extend(
            [
                os.path.join(os.path.dirname(bindir), "lib", "libipopt.so.3"),
                os.path.join(os.path.dirname(bindir), "lib", "libipopt.so"),
            ]
        )
    for lib in candidates:
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            return


def _import_cyipopt():
    _preload_ipopt_library()
    from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
    from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptProblemInterface, cyipopt_available

    if not cyipopt_available:
        raise NotImplementedError("nlps_ipopt requires the Python cyipopt binding")
    return CyIpoptProblemInterface, CyIpoptSolver


def _is_empty(value):
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return not value
    if sparse.issparse(value):
        return value.shape[0] == 0 or value.shape[1] == 0
    return np.asarray(value).size == 0


def _to_sparse(mat, shape=None):
    if _is_empty(mat):
        return sparse.csc_matrix(shape or (0, 0), dtype=float)
    if sparse.issparse(mat):
        return mat.tocsr().astype(float)
    arr = np.asarray(mat, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return sparse.csc_matrix(arr)


def _to_vector(value, default, size):
    if _is_empty(value):
        return np.full(size, default, dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def _values_in_structure(mat, rows, cols):
    coo = mat.tocoo()
    vals = {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}
    return np.array([vals.get((int(r), int(c)), 0.0) for r, c in zip(rows, cols)], dtype=float)


def nlps_ipopt_full(
    f_fcn, x0=None, A=None, l=None, u=None, xmin=None, xmax=None, gh_fcn=None, hess_fcn=None, opt=None
) -> NlpResult:
    """Nonlinear-program solver wrapper based on IPOPT.

    Solves ``min F(X)`` subject to nonlinear equalities/inequalities,
    linear constraints, and variable bounds.

    Parameters
    ----------
    f_fcn : callable or dict
        Objective callback ``[f, df, d2f] = f_fcn(x)`` or a problem dict
        containing the full solver inputs.
    x0 : array_like, optional
        Initial point.
    A, l, u : array_like, optional
        Linear constraints ``l <= A*x <= u``.
    xmin, xmax : array_like, optional
        Variable bounds.
    gh_fcn : callable, optional
        Nonlinear constraint callback ``[h, g, dh, dg] = gh_fcn(x)``.
    hess_fcn : callable, optional
        Lagrangian Hessian callback ``Lxx = hess_fcn(x, lam)``.
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
    if isinstance(f_fcn, dict):
        p = f_fcn
        f_fcn = p["f_fcn"]
        x0 = p["x0"]
        nx = np.size(x0)
        opt = p.get("opt", [])
        hess_fcn = p.get("hess_fcn", "")
        gh_fcn = p.get("gh_fcn", "")
        xmax = p.get("xmax", [])
        xmin = p.get("xmin", [])
        u = p.get("u", [])
        l = p.get("l", [])
        A = p.get("A", sparse.csc_matrix((0, nx)))
    else:
        if x0 is None:
            raise ValueError("nlps_ipopt: x0 is required")
        nx = np.size(x0)
        if A is None:
            A = sparse.csc_matrix((0, nx))
        if opt is None:
            opt = []
        if hess_fcn is None:
            hess_fcn = ""
        if gh_fcn is None:
            gh_fcn = ""
        if xmax is None:
            xmax = []
        if xmin is None:
            xmin = []
        if u is None:
            u = []
        if l is None:
            l = []

    if not have_feature_ipopt():
        raise NotImplementedError("nlps_ipopt requires an available IPOPT backend")

    CyIpoptProblemInterface, CyIpoptSolver = _import_cyipopt()

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    nx = x0.size

    A = _to_sparse(A, (0, nx))
    matrix_shape = np.asarray(A.shape, dtype=int)
    if matrix_shape[1] != nx:
        A = sparse.csc_matrix(A, shape=(int(matrix_shape[0]), nx))
        matrix_shape = np.asarray(A.shape, dtype=int)
    if matrix_shape[0] == 0 or (
        (_is_empty(l) or np.all(np.asarray(l) == -np.inf)) and (_is_empty(u) or np.all(np.asarray(u) == np.inf))
    ):
        A = sparse.csc_matrix((0, nx), dtype=float)
    nA = int(np.asarray(A.shape)[0])

    u = _to_vector(u, np.inf, nA)
    l = _to_vector(l, -np.inf, nA)
    xmin = _to_vector(xmin, -np.inf, nx)
    xmax = _to_vector(xmax, np.inf, nx)
    nonlinear = not _is_empty(gh_fcn) and gh_fcn != ""

    kk = np.where(xmin == xmax)[0]
    nk = len(kk)
    if nk:
        A = sparse.vstack([A, sparse.csc_matrix((np.ones(nk), (np.arange(nk), kk)), shape=(nk, nx))], format="csc")
        l = np.r_[l, xmin[kk]]
        u = np.r_[u, xmax[kk]]
        xmin = xmin.copy()
        xmax = xmax.copy()
        xmin[kk] = -np.inf
        xmax[kk] = np.inf
        nA = int(np.asarray(A.shape)[0])

    randx = np.random.rand(*x0.shape)
    nonz = 1e-20
    if nonlinear:
        constraint_fcn = cast(Any, gh_fcn)
        h, g, dhs, dgs = constraint_fcn(randx)
        h = np.asarray(h, dtype=float).reshape(-1)
        g = np.asarray(g, dtype=float).reshape(-1)
        dhs = _to_sparse(dhs, (nx, len(h)))
        dgs = _to_sparse(dgs, (nx, len(g)))
        dhs = dhs.copy()
        dgs = dgs.copy()
        if dhs.nnz:
            dhs.data[:] = nonz
        if dgs.nnz:
            dgs.data[:] = nonz
        Js = sparse.vstack([dgs.T, dhs.T, A], format="csc")
    else:
        constraint_fcn = None
        h = np.array([], dtype=float)
        g = np.array([], dtype=float)
        dhs = sparse.csc_matrix((nx, 0), dtype=float)
        dgs = sparse.csc_matrix((nx, 0), dtype=float)
        Js = A.tocsr()
    neq = len(g)
    niq = len(h)

    if not hess_fcn:
        _, _, Hs = f_fcn(randx)
        Hs = _to_sparse(Hs, (nx, nx))
    else:
        hessian_fcn = cast(Any, hess_fcn)
        lam = {"eqnonlin": np.random.rand(neq), "ineqnonlin": np.random.rand(niq)}
        Hs = _to_sparse(hessian_fcn(randx, lam, 1), (nx, nx))
    Hs = sparse.tril(Hs.tocsr())
    Hs = Hs.copy()
    if Hs.nnz:
        Hs.data[:] = nonz

    jrows, jcols = Js.tocoo().row.astype(np.int32), Js.tocoo().col.astype(np.int32)
    hrows, hcols = Hs.tocoo().row.astype(np.int32), Hs.tocoo().col.astype(np.int32)

    ipopt_opt = ipopt_options(opt.get("ipopt_opt") if isinstance(opt, dict) else None)
    verbose = int(opt.get("verbose", 0)) if isinstance(opt, dict) else 0
    if verbose:
        ipopt_opt["print_level"] = min(12, verbose * 2 + 1)
    else:
        ipopt_opt["print_level"] = 0

    class _Problem(CyIpoptProblemInterface):
        def __init__(self):
            super().__init__()

        def x_init(self):
            return x0.copy()

        def x_lb(self):
            return xmin.copy()

        def x_ub(self):
            return xmax.copy()

        def g_lb(self):
            return np.r_[np.zeros(neq), -np.inf * np.ones(niq), l]

        def g_ub(self):
            return np.r_[np.zeros(neq), np.zeros(niq), u]

        def scaling_factors(self):
            return (None, None, None)

        def objective(self, x):
            return float(f_fcn(x)[0])

        def gradient(self, x):
            return np.asarray(f_fcn(x)[1], dtype=float).reshape(-1)

        def constraints(self, x):
            if nonlinear:
                assert constraint_fcn is not None
                h_, g_ = constraint_fcn(x)[:2]
                h_ = np.asarray(h_, dtype=float).reshape(-1)
                g_ = np.asarray(g_, dtype=float).reshape(-1)
                return np.r_[g_, h_, A @ np.asarray(x, dtype=float).reshape(-1)]
            return np.asarray(A @ np.asarray(x, dtype=float).reshape(-1), dtype=float).reshape(-1)

        def jacobianstructure(self):
            return jrows, jcols

        def jacobian(self, x):
            if nonlinear:
                assert constraint_fcn is not None
                h_, g_, dh, dg = constraint_fcn(x)
                dh = _to_sparse(dh, (nx, len(np.asarray(h_).reshape(-1))))
                dg = _to_sparse(dg, (nx, len(np.asarray(g_).reshape(-1))))
                J = sparse.vstack([(dg + dgs).T, (dh + dhs).T, A], format="csc")
            else:
                J = A
            return _values_in_structure(J, jrows, jcols)

        def hessianstructure(self):
            return hrows, hcols

        def hessian(self, x, y, obj_factor):
            if not hess_fcn:
                H = _to_sparse(f_fcn(x)[2], (nx, nx))
                H = sparse.tril(H * obj_factor)
            else:
                hessian_fcn = cast(Any, hess_fcn)
                lam = {
                    "eqnonlin": np.asarray(y[:neq], dtype=float).reshape(-1),
                    "ineqnonlin": np.asarray(y[neq : neq + niq], dtype=float).reshape(-1),
                }
                H = _to_sparse(hessian_fcn(x, lam, obj_factor), (nx, nx))
                H = sparse.tril(H + Hs)
            return _values_in_structure(H, hrows, hcols)

    problem = _Problem()
    solver = CyIpoptSolver(problem, options=ipopt_opt)
    x, info = solver.solve(x0=x0, tee=bool(verbose))
    x = np.asarray(x, dtype=float).reshape(-1)

    status = int(info.get("status", -1))
    if status in (0, 1):
        eflag = 1
    else:
        eflag = 0
    status_message_value = info.get("status_msg", b"")
    status_message = (
        status_message_value.decode() if isinstance(status_message_value, (bytes, bytearray)) else str(status_message_value)
    )
    iterations_value = info.get("iter")
    output: SolverOutput = {
        "status": status,
        "status_msg": status_message,
        "iterations": int(iterations_value) if isinstance(iterations_value, (int, float)) else None,
    }

    f = float(info.get("obj_val", f_fcn(x)[0]))
    lam_all = np.asarray(info.get("mult_g", np.full(neq + niq + nA, np.nan)), dtype=float).reshape(-1)
    zl = np.asarray(info.get("mult_x_L", np.full(nx, np.nan)), dtype=float).reshape(-1)
    zu = np.asarray(info.get("mult_x_U", np.full(nx, np.nan)), dtype=float).reshape(-1)

    if nk:
        offset = neq + niq + nA - nk
        lam_tmp = lam_all[offset : offset + nk]
        kl = np.where(lam_tmp < 0)[0]
        ku = np.where(lam_tmp > 0)[0]
        zl[kk[kl]] = -lam_tmp[kl]
        zu[kk[ku]] = lam_tmp[ku]
        lam_all = np.r_[lam_all[:offset], lam_all[offset + nk :]]
        nA -= nk

    lam_lin = lam_all[neq + niq : neq + niq + nA]
    kl = np.where(lam_lin < 0)[0]
    ku = np.where(lam_lin > 0)[0]
    mu_l = np.zeros(nA, dtype=float)
    mu_l[kl] = -lam_lin[kl]
    mu_u = np.zeros(nA, dtype=float)
    mu_u[ku] = lam_lin[ku]

    lambda_: NlpMultipliers = {
        "lower": zl,
        "upper": zu,
        "eqnonlin": lam_all[:neq],
        "ineqnonlin": lam_all[neq : neq + niq],
        "mu_l": mu_l,
        "mu_u": mu_u,
    }

    return x, f, eflag, output, lambda_


def nlps_ipopt(
    f_fcn,
    x0=None,
    A=None,
    l=None,
    u=None,
    xmin=None,
    xmax=None,
    gh_fcn=None,
    hess_fcn=None,
    opt=None,
    nargout: int = 1,
):
    """MATPOWER-compatible IPOPT NLP entry point; use ``nlps_ipopt_full`` in typed code."""
    result = nlps_ipopt_full(f_fcn, x0, A, l, u, xmin, xmax, gh_fcn, hess_fcn, opt)
    return result[:nargout] if nargout > 1 else result[0]
