# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np

from ..mips.qps_mips import qps_mips
from .have_feature_glpk import have_feature_glpk
from .qps_glpk import qps_glpk
from .qps_ipopt import qps_ipopt


def qps_master(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None, x0=None, opt=None, nargout=1):
    """Quadratic-program solver wrapper.

    Solves the QP

    ``min 0.5 * x' * H * x + c' * x``

    subject to linear constraints and variable bounds.

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
        Solver options dict. ``opt["alg"]`` selects ``MIPS``, ``GLPK``,
        or ``IPOPT`` in the current Python port.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or ndarray
        Returns ``(x, f, eflag, output, lambda_)`` or the leading subset
        requested by ``nargout``.
    """
    if isinstance(H, dict):
        p = H
        H = p.get("H")
        c = p.get("c")
        A = p.get("A")
        l = p.get("l")
        u = p.get("u")
        xmin = p.get("xmin")
        xmax = p.get("xmax")
        x0 = p.get("x0")
        opt = p.get("opt")
    opt = {} if opt is None else dict(opt)
    alg = opt.get("alg", "DEFAULT")
    if not isinstance(alg, str):
        if alg == 0:
            alg = "DEFAULT"
        elif alg == 200:
            alg = "MIPS"
            opt.setdefault("mips_opt", {})["step_control"] = 0
        elif alg == 250:
            alg = "MIPS"
            opt.setdefault("mips_opt", {})["step_control"] = 1
        elif alg == 400:
            alg = "IPOPT"
        else:
            raise ValueError(f"qps_master: {alg!r} is not a valid algorithm code")
    else:
        alg = alg.upper()
    if alg == "DEFAULT":
        if H is None or np.size(H) == 0 or not np.any(np.asarray(H)):
            alg = "GLPK" if have_feature_glpk() else "MIPS"
        else:
            alg = "MIPS"
    if alg == "MIPS":
        outputs = qps_mips(H, c, A, l, u, xmin, xmax, x0, opt.get("mips_opt", opt), nargout=5)
    elif alg == "GLPK":
        outputs = qps_glpk(H, c, A, l, u, xmin, xmax, x0, opt, nargout=5)
    elif alg == "IPOPT":
        outputs = qps_ipopt(H, c, A, l, u, xmin, xmax, x0, opt, nargout=5)
    else:
        raise NotImplementedError(f"qps_master solver {alg!r} not yet implemented")
    x, f, eflag, output, lambda_ = outputs
    if not output.get("alg"):
        output["alg"] = alg
    result = (x, f, eflag, output, lambda_)
    return result[:nargout] if nargout > 1 else result[0]
