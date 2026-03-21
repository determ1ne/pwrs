# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..mips.mips import mips
from .nlps_ipopt import nlps_ipopt


def nlps_master(
    f_fcn, x0=None, A=None, l=None, u=None, xmin=None, xmax=None, gh_fcn=None, hess_fcn=None, opt=None, nargout=1
):
    """Nonlinear programming solver wrapper.

    Solves the NLP

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
        Solver options dict. ``opt["alg"]`` selects ``MIPS`` or ``IPOPT``
        in the current Python port.
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
        opt = p.get("opt", {})
        hess_fcn = p.get("hess_fcn", "")
        gh_fcn = p.get("gh_fcn", "")
        xmax = p.get("xmax", [])
        xmin = p.get("xmin", [])
        u = p.get("u", [])
        l = p.get("l", [])
        A = p.get("A", sparse.csc_matrix((0, nx)))
    else:
        nx = np.size(x0)
        if opt is None:
            opt = {}
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
        if A is None:
            A = sparse.csc_matrix((0, nx))

    alg = str(opt.get("alg", "DEFAULT")).upper() if opt else "DEFAULT"
    verbose = opt.get("verbose", 0) if opt else 0
    if alg == "DEFAULT":
        alg = "MIPS"

    if alg == "MIPS":
        mips_opt = opt["mips_opt"]
        mips_opt.verbose = verbose
        outputs = mips(f_fcn, x0, A, l, u, xmin, xmax, gh_fcn, hess_fcn, mips_opt, nargout=5)
    elif alg == "FMINCON":
        raise NotImplementedError("nlps_master FMINCON not yet implemented")
    elif alg == "IPOPT":
        outputs = nlps_ipopt(f_fcn, x0, A, l, u, xmin, xmax, gh_fcn, hess_fcn, opt, nargout=5)
    elif alg == "KNITRO":
        raise NotImplementedError("nlps_master KNITRO not yet implemented")
    else:
        raise ValueError(f"nlps_master: '{alg}' is not a valid algorithm code")

    x, f, eflag, output, lambda_ = outputs
    if not output.get("alg"):
        output["alg"] = alg
    ret = (x, f, eflag, output, lambda_)
    return ret[:nargout] if nargout > 1 else x
