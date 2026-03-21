# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
from scipy import sparse

from .mips import mips


def qps_mips(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None, x0=None, opt=None, nargout=1):
    """Quadratic-program solver wrapper based on MIPS.

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
        MIPS options dict.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or ndarray
        Returns ``(x, f, eflag, output, lambda_)`` or the leading subset
        requested by ``nargout``.
    """
    if isinstance(H, dict):
        p = dict(H)
    else:
        p = {"H": H, "c": c, "A": A, "l": l, "u": u, "xmin": xmin, "xmax": xmax, "x0": x0, "opt": opt}

    if (
        "H" not in p
        or p["H"] is None
        or np.size(p["H"]) == 0
        or (sparse.issparse(p["H"]) and p["H"].nnz == 0)
        or (not sparse.issparse(p["H"]) and not np.any(p["H"]))
    ):
        if (
            ("A" not in p or p["A"] is None or np.size(p["A"]) == 0)
            and ("xmin" not in p or p["xmin"] is None or np.size(p["xmin"]) == 0)
            and ("xmax" not in p or p["xmax"] is None or np.size(p["xmax"]) == 0)
        ):
            raise ValueError("qps_mips: LP problem must include constraints or variable bounds")
        if "A" in p and p["A"] is not None and np.size(p["A"]):
            nx = p["A"].shape[1]
        elif "xmin" in p and p["xmin"] is not None and np.size(p["xmin"]):
            nx = len(np.asarray(p["xmin"]).reshape(-1))
        else:  # if isfield(p, 'xmax') && ~isempty(p.xmax)
            nx = len(np.asarray(p["xmax"]).reshape(-1))
        p["H"] = sparse.csc_matrix((nx, nx))
    else:
        p["H"] = sparse.csc_matrix(p["H"])
        nx = p["H"].shape[0]

    if "c" not in p or p["c"] is None or np.size(p["c"]) == 0:
        p["c"] = np.zeros(nx)
    else:
        p["c"] = np.asarray(p["c"], dtype=float).reshape(-1)
    if "x0" not in p or p["x0"] is None or np.size(p["x0"]) == 0:
        p["x0"] = np.zeros(nx)
    else:
        p["x0"] = np.asarray(p["x0"], dtype=float).reshape(-1)

    def qp_f(x):
        f = 0.5 * x @ (p["H"] @ x) + p["c"] @ x
        df = p["H"] @ x + p["c"]
        return f, df, p["H"]

    p["f_fcn"] = qp_f
    outputs = mips(p, nargout=5)
    return outputs[:nargout] if nargout > 1 else outputs[0]
