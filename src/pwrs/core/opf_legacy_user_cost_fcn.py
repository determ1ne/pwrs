# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def opf_legacy_user_cost_fcn(x, cp, nargout=1):
    """Evaluate legacy user-defined costs and derivatives.

    Parameters
    ----------
    x : array_like
        Optimization variable vector.
    cp : dict
        Legacy user-defined cost parameter dict such as returned by
        ``OPFModel.get_cost_params()``.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    float or tuple
        Returns the scalar cost and, when requested, its gradient and Hessian.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    N = sparse.csc_matrix(cp["N"])
    Cw = np.asarray(cp["Cw"], dtype=float).reshape(-1)
    H = sparse.csc_matrix(cp.get("H", sparse.csc_matrix((len(Cw), len(Cw)))))
    dd = np.asarray(cp.get("dd", np.ones_like(Cw)), dtype=float).reshape(-1)
    rh = np.asarray(cp.get("rh", np.zeros_like(Cw)), dtype=float).reshape(-1)
    kk = np.asarray(cp.get("kk", np.zeros_like(Cw)), dtype=float).reshape(-1)
    mm = np.asarray(cp.get("mm", np.ones_like(Cw)), dtype=float).reshape(-1)

    nx = x.size
    if Cw.size == 0:
        outputs = (0.0, np.zeros(nx), sparse.csc_matrix((nx, nx)))
        return outputs[:nargout] if nargout > 1 else outputs[0]

    r = np.asarray(N @ x - rh, dtype=float).reshape(-1)
    below = r < -kk
    no_dead_zone_origin = (r == 0) & (kk == 0)
    above = r > kk
    active = below | no_dead_zone_origin | above

    kbar = np.zeros_like(r)
    kbar[below] = kk[below]
    kbar[above] = -kk[above]
    rr = r + kbar

    M = sparse.diags(mm * active, format="csc")
    LL = sparse.diags((dd == 1).astype(float), format="csc")
    QQ = sparse.diags((dd == 2).astype(float), format="csc")
    diagrr = sparse.diags(rr, format="csc")
    w = np.asarray(M @ (LL + QQ @ diagrr) @ rr, dtype=float).reshape(-1)
    f = float(0.5 * w @ (H @ w) + Cw @ w)
    if nargout == 1:
        return f

    HwC = np.asarray(H @ w + Cw, dtype=float).reshape(-1)
    AA = sparse.csc_matrix(N.T @ M @ (LL + 2 * QQ @ diagrr))
    df = np.asarray(AA @ HwC, dtype=float).reshape(-1)
    if nargout == 2:
        return f, df

    d2f = AA @ H @ AA.T + 2 * N.T @ M @ QQ @ sparse.diags(HwC, format="csc") @ N
    outputs = (f, df, sparse.csc_matrix(d2f))
    return outputs[:nargout]


def opf_legacy_user_cost_fcn_full(x, cp):
    """Request the full legacy user-cost contract.

    This semantic helper keeps internal callers independent of ``nargout``.
    """
    return opf_legacy_user_cost_fcn(x, cp, nargout=3)
