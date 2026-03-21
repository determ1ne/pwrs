# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
from scipy import sparse

from ..corex.mpoption import MipsConfig, MipsScConfig
from .mipsver import mipsver
from .mplinsolve import mplinsolve


def _norm_inf(vec: np.ndarray) -> float:
    return np.linalg.norm(vec, np.inf)


def _normalize_opt(opt: Any) -> MipsConfig:
    if opt is None:
        cfg = MipsConfig(verbose=0)
    elif isinstance(opt, MipsConfig):
        cfg = opt
    elif isinstance(opt, dict):
        cfg = MipsConfig(**opt)
    else:
        cfg = MipsConfig(**opt.to_dict())

    if isinstance(cfg.sc, dict):
        cfg.sc = MipsScConfig(**cfg.sc)
    if cfg.linsolver is None:
        cfg.linsolver = ""
    if cfg.feastol in (None, 0):
        cfg.feastol = 1e-6
    if cfg.gradtol in (None, 0):
        cfg.gradtol = 1e-6
    if cfg.comptol in (None, 0):
        cfg.comptol = 1e-6
    if cfg.costtol in (None, 0):
        cfg.costtol = 1e-6
    if cfg.max_it is None:
        cfg.max_it = 150
    if cfg.sc is None:
        cfg.sc = MipsScConfig()
    if cfg.sc.red_it is None:
        cfg.sc.red_it = 20
    if cfg.step_control is None:
        cfg.step_control = 0
    if cfg.cost_mult is None:
        cfg.cost_mult = 1
    if cfg.verbose is None:
        cfg.verbose = 0
    if cfg.xi is None:
        cfg.xi = 0.99995
    if cfg.sigma is None:
        cfg.sigma = 0.1
    if cfg.z0 is None:
        cfg.z0 = 1
    if cfg.alpha_min is None:
        cfg.alpha_min = 1e-8
    if cfg.rho_min is None:
        cfg.rho_min = 0.95
    if cfg.rho_max is None:
        cfg.rho_max = 1.05
    if cfg.mu_threshold is None:
        cfg.mu_threshold = 1e-5
    if cfg.max_stepsize is None:
        cfg.max_stepsize = 1e10
    return cfg


def mips(f_fcn, x0=None, A=None, l=None, u=None, xmin=None, xmax=None, gh_fcn=None, hess_fcn=None, opt=None, nargout=1):
    """Pwrs Interior Point Solver.

    Primal-dual interior point method for nonlinear programming. Solves
    ``min F(X)`` subject to nonlinear equalities/inequalities, linear
    constraints, and variable bounds.

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
        MIPS options dict controlling tolerances, linear solver choice,
        step control, and barrier parameters.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or ndarray
        Returns ``(x, f, eflag, output, lambda_)`` or the leading subset
        requested by ``nargout``.
    """
    if isinstance(f_fcn, dict):
        p = dict(f_fcn)
        f_fcn = p["f_fcn"]
        x0 = p["x0"]
        nx = np.size(x0)
        opt = p.get("opt", None)
        hess_fcn = p.get("hess_fcn", "")
        gh_fcn = p.get("gh_fcn", "")
        xmax = p.get("xmax", [])
        xmin = p.get("xmin", [])
        u = p.get("u", [])
        l = p.get("l", [])
        A = p.get("A", sparse.csc_matrix((0, nx)))
    else:
        nx = np.size(x0)
        hess_fcn = "" if hess_fcn is None else hess_fcn
        gh_fcn = "" if gh_fcn is None else gh_fcn
        xmax = [] if xmax is None else xmax
        xmin = [] if xmin is None else xmin
        u = [] if u is None else u
        l = [] if l is None else l
        A = sparse.csc_matrix((0, nx)) if A is None else A

    if A.shape[0] == 0 or (
        (np.size(l) == 0 or np.all(np.asarray(l) == -np.inf)) and (np.size(u) == 0 or np.all(np.asarray(u) == np.inf))
    ):
        A = sparse.csc_matrix((0, nx))
    nA = A.shape[0]
    u = np.full(nA, np.inf) if np.size(u) == 0 else u
    l = np.full(nA, -np.inf) if np.size(l) == 0 else l
    xmin = np.full(nx, -np.inf) if np.size(xmin) == 0 else xmin
    xmax = np.full(nx, np.inf) if np.size(xmax) == 0 else xmax

    nonlinear = gh_fcn not in ("", None)
    gn = np.array([], dtype=float)
    hn = np.array([], dtype=float)

    opt = _normalize_opt(opt)

    xi = float(opt.xi)
    sigma = float(opt.sigma)
    z0 = float(opt.z0)
    alpha_min = float(opt.alpha_min)
    rho_min = float(opt.rho_min)
    rho_max = float(opt.rho_max)
    mu_threshold = float(opt.mu_threshold)
    max_stepsize = float(opt.max_stepsize)
    if xi >= 1 or xi < 0.5:
        raise ValueError(f"mips: opt.xi ({opt.xi:g}) must be a number slightly less than 1")
    if sigma > 1 or sigma <= 0:
        raise ValueError(f"mips: opt.sigma ({opt.sigma:g}) must be a number between 0 and 1")

    i = 0
    converged = 0
    eflag = 0

    AA = sparse.vstack([sparse.eye(nx, format="csc"), A], format="csc")
    ll = np.r_[xmin, l]
    uu = np.r_[xmax, u]

    ieq = np.flatnonzero(np.abs(uu - ll) <= np.finfo(float).eps)
    igt = np.flatnonzero((uu >= 1e10) & (ll > -1e10))
    ilt = np.flatnonzero((ll <= -1e10) & (uu < 1e10))
    ibx = np.flatnonzero((np.abs(uu - ll) > np.finfo(float).eps) & (uu < 1e10) & (ll > -1e10))
    Ae = AA[ieq, :]
    be = uu[ieq]
    Ai = sparse.vstack([AA[ilt, :], -AA[igt, :], AA[ibx, :], -AA[ibx, :]], format="csc")
    bi = np.r_[uu[ilt], -ll[igt], uu[ibx], -ll[ibx]]

    x = x0.copy()
    f, df, _ = f_fcn(x)
    f *= opt.cost_mult
    df *= opt.cost_mult
    if nonlinear:
        hn, gn, dhn, dgn = gh_fcn(x)
        h = np.r_[hn, Ai @ x - bi]
        g = np.r_[gn, Ae @ x - be]
        dh = sparse.hstack([dhn, Ai.T], format="csc")
        dg = sparse.hstack([dgn, Ae.T], format="csc")
    else:
        h = Ai @ x - bi
        g = Ae @ x - be
        dh = Ai.T.tocsr()
        dg = Ae.T.tocsr()

    neq = g.size
    niq = h.size
    neqnln = gn.size
    niqnln = hn.size
    nlt = ilt.size
    ngt = igt.size
    nbx = ibx.size

    gamma = 1.0
    lam = np.zeros(neq)
    z = z0 * np.ones(niq)
    mu = z.copy()
    k = np.flatnonzero(h < -z0)
    z[k] = -h[k]
    k = np.flatnonzero(gamma / z > z0) if niq else np.array([], dtype=int)
    if k.size:
        mu[k] = gamma / z[k]
    e = np.ones(niq)

    f0 = f
    if opt.step_control:
        L = f + lam @ g + mu @ (h + z) - gamma * np.sum(np.log(z))
    Lx = df + dg @ lam + dh @ mu
    maxh = 0.0 if h.size == 0 else float(np.max(h))
    feascond = max(_norm_inf(g), maxh) / (1 + max(_norm_inf(x), _norm_inf(z)))
    gradcond = _norm_inf(Lx) / (1 + max(_norm_inf(lam), _norm_inf(mu)))
    compcond = float(z @ mu) / (1 + _norm_inf(x))
    costcond = abs(f - f0) / (1 + abs(f0))
    hist = [
        {
            "feascond": feascond,
            "gradcond": gradcond,
            "compcond": compcond,
            "costcond": costcond,
            "gamma": gamma,
            "stepsize": 0.0,
            "obj": f / opt.cost_mult,
            "alphap": 0.0,
            "alphad": 0.0,
        }
    ]

    if opt.verbose:
        if opt.step_control:
            suffix = "-sc"
        else:
            suffix = ""
        v = mipsver("all", nargout=1)
        print(
            f"Pwrs Interior Point Solver -- PIPS{suffix}, Version {v['Version']}, {v['Date']}\n"
            " (using built-in linear solver)",
            end="",
        )
        if opt.verbose > 1:
            print("\n it    objective   step size   feascond     gradcond     compcond     costcond  ", end="")
            print("\n----  ------------ --------- ------------ ------------ ------------ ------------", end="")
            print(
                f"\n{i:3d}  {f / opt.cost_mult:12.8g} {'':>10s} {feascond:12g} {gradcond:12g} {compcond:12g} {costcond:12g}",
                end="",
            )
    if feascond < opt.feastol and gradcond < opt.gradtol and compcond < opt.comptol and costcond < opt.costtol:
        converged = 1
        if opt.verbose:
            print("\nConverged!")

    while (not converged) and i < opt.max_it:
        i += 1
        lambda_ = {"eqnonlin": lam[:neqnln], "ineqnonlin": mu[:niqnln]}
        if nonlinear:
            if hess_fcn in ("", None):
                raise NotImplementedError(
                    "mips: Hessian evaluation via finite differences not yet implemented. Please provide your own hessian evaluation function."
                )
            Lxx = hess_fcn(x, lambda_, opt.cost_mult)
        else:
            _, _, d2f = f_fcn(x)
            Lxx = d2f * opt.cost_mult
        zinvdiag = sparse.diags(1.0 / z, format="csc") if niq else sparse.csc_matrix((0, 0))
        mudiag = sparse.diags(mu, format="csc") if niq else sparse.csc_matrix((0, 0))
        dh_zinv = dh @ zinvdiag if niq else sparse.csc_matrix((nx, 0))
        M = Lxx + (dh_zinv @ mudiag @ dh.T if niq else sparse.csc_matrix((nx, nx)))
        N = Lx + (dh_zinv @ (mudiag @ h + gamma * e) if niq else np.zeros(nx))
        KKT = sparse.bmat([[M, dg], [dg.T, None]], format="csc")
        rhs = np.r_[-N, -g]
        dxdlam = mplinsolve(KKT, rhs, opt.linsolver, None)
        if np.any(np.isnan(dxdlam)) or np.linalg.norm(dxdlam) > max_stepsize:
            if opt.verbose:
                print("\nNumerically Failed")
            eflag = -1
            break
        dx = dxdlam[:nx]
        dlam = dxdlam[nx : nx + neq]
        dz = -h - z - dh.T @ dx
        dmu = -mu + (zinvdiag @ (gamma * e - mudiag @ dz) if niq else np.array([], dtype=float))

        sc = 0
        if opt.step_control:
            x1 = x + dx
            f1, df1 = f_fcn(x1)
            f1 *= opt.cost_mult
            df1 *= opt.cost_mult
            if nonlinear:
                hn1, gn1, dhn1, dgn1 = gh_fcn(x1)
                h1 = np.r_[hn1, Ai @ x1 - bi]
                g1 = np.r_[gn1, Ae @ x1 - be]
                dh1 = sparse.hstack([dhn1, Ai.T], format="csc")
                dg1 = sparse.hstack([dgn1, Ae.T], format="csc")
            else:
                h1 = Ai @ x1 - bi
                g1 = Ae @ x1 - be
                dh1 = dh
                dg1 = dg

            Lx1 = df1 + dg1 @ lam + dh1 @ mu
            maxh1 = 0.0 if h1.size == 0 else float(np.max(h1))
            feascond1 = max(_norm_inf(g1), maxh1) / (1 + max(_norm_inf(x1), _norm_inf(z)))
            gradcond1 = _norm_inf(Lx1) / (1 + max(_norm_inf(lam), _norm_inf(mu)))
            if feascond1 > feascond and gradcond1 > gradcond:
                sc = 1

        if sc:
            alpha = 1.0
            for j in range(1, int(opt.sc.red_it) + 1):
                dx1 = alpha * dx
                x1 = x + dx1
                f1, _, _ = f_fcn(x1)
                f1 = f1 * opt.cost_mult
                if nonlinear:
                    hn1, gn1, _, _ = gh_fcn(x1)
                    h1 = np.r_[hn1, Ai @ x1 - bi]
                    g1 = np.r_[gn1, Ae @ x1 - be]
                else:
                    h1 = Ai @ x1 - bi
                    g1 = Ae @ x1 - be
                L1 = f1 + lam @ g1 + mu @ (h1 + z) - gamma * np.sum(np.log(z))
                if opt.verbose > 2:
                    print(f"\n   {-j:3d}            {np.linalg.norm(dx1):10g}", end="")
                denom = float(Lx @ dx1 + 0.5 * dx1 @ (Lxx @ dx1))
                rho = (L1 - L) / denom if denom != 0 else np.inf
                if rho > rho_min and rho < rho_max:
                    break
                alpha /= 2.0
            dx = alpha * dx
            dz = alpha * dz
            dlam = alpha * dlam
            dmu = alpha * dmu

        k = np.flatnonzero(dz < 0)
        alphap = min(float(xi * np.min(z[k] / -dz[k])), 1.0) if k.size else 1.0
        k = np.flatnonzero(dmu < 0)
        alphad = min(float(xi * np.min(mu[k] / -dmu[k])), 1.0) if k.size else 1.0
        x = x + alphap * dx
        z = z + alphap * dz
        lam = lam + alphad * dlam
        mu = mu + alphad * dmu
        if niq > 0:
            gamma = sigma * float(z @ mu) / niq

        f, df, _ = f_fcn(x)
        f *= opt.cost_mult
        df *= opt.cost_mult
        if nonlinear:
            hn, gn, dhn, dgn = gh_fcn(x)
            h = np.r_[hn, Ai @ x - bi]
            g = np.r_[gn, Ae @ x - be]
            dh = sparse.hstack([dhn, Ai.T], format="csc")
            dg = sparse.hstack([dgn, Ae.T], format="csc")
        else:
            h = Ai @ x - bi
            g = Ae @ x - be
        Lx = df + dg @ lam + dh @ mu
        maxh = 0.0 if h.size == 0 else float(np.max(h))
        feascond = max(_norm_inf(g), maxh) / (1 + max(_norm_inf(x), _norm_inf(z)))
        gradcond = _norm_inf(Lx) / (1 + max(_norm_inf(lam), _norm_inf(mu)))
        compcond = float(z @ mu) / (1 + _norm_inf(x))
        costcond = abs(f - f0) / (1 + abs(f0))
        hist.append(
            {
                "feascond": feascond,
                "gradcond": gradcond,
                "compcond": compcond,
                "costcond": costcond,
                "gamma": gamma,
                "stepsize": float(np.linalg.norm(dx)),
                "obj": f / opt.cost_mult,
                "alphap": alphap,
                "alphad": alphad,
            }
        )

        if opt.verbose > 1:
            print(
                f"\n{i:3d}  {f / opt.cost_mult:12.8g} {np.linalg.norm(dx):10.5g} {feascond:12g} {gradcond:12g} {compcond:12g} {costcond:12g}",
                end="",
            )
        if feascond < opt.feastol and gradcond < opt.gradtol and compcond < opt.comptol and costcond < opt.costtol:
            converged = 1
            if opt.verbose:
                print("\nConverged!")
        else:
            if (
                np.any(np.isnan(x))
                or alphap < alpha_min
                or alphad < alpha_min
                or gamma < np.finfo(float).eps
                or gamma > 1 / np.finfo(float).eps
            ):
                if opt.verbose:
                    print("\nNumerically Failed")
                eflag = -1
                break
            f0 = f
            if opt.step_control:
                L = f + lam @ g + mu @ (h + z) - gamma * np.sum(np.log(z))

    if opt.verbose and not converged:
        print(f"\nDid not converge in {i} iterations.")

    if eflag != -1:
        eflag = converged
    output = {"iterations": i, "hist": hist, "message": ""}
    if eflag == 0:
        output["message"] = "Did not converge"
    elif eflag == 1:
        output["message"] = "Converged"
    elif eflag == -1:
        output["message"] = "Numerically failed"
    else:
        output["message"] = "Please hang up and dial again"

    if mu.size:
        mask = (h < -opt.feastol) & (mu < mu_threshold)
        mu[mask] = 0.0

    f = f / opt.cost_mult
    lam = lam / opt.cost_mult
    mu = mu / opt.cost_mult

    lam_lin = lam[neqnln:neq]
    mu_lin = mu[niqnln:niq]
    kl = np.flatnonzero(lam_lin < 0)
    ku = np.flatnonzero(lam_lin > 0)

    mu_l = np.zeros(nx + nA)
    mu_l[ieq[kl]] = -lam_lin[kl]
    mu_l[igt] = mu_lin[nlt : nlt + ngt]
    mu_l[ibx] = mu_lin[nlt + ngt + nbx : nlt + ngt + 2 * nbx]

    mu_u = np.zeros(nx + nA)
    mu_u[ieq[ku]] = lam_lin[ku]
    mu_u[ilt] = mu_lin[:nlt]
    mu_u[ibx] = mu_lin[nlt + ngt : nlt + ngt + nbx]

    fields = {
        "mu_l": mu_l[nx:],
        "mu_u": mu_u[nx:],
        "lower": mu_l[:nx],
        "upper": mu_u[:nx],
    }
    if niqnln > 0:
        fields["ineqnonlin"] = mu[:niqnln]
    if neqnln > 0:
        fields["eqnonlin"] = lam[:neqnln]
    outputs = (x, f, eflag, output, fields)
    return outputs[:nargout] if nargout > 1 else x
