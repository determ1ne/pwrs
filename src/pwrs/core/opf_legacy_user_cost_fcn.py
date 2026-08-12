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
        Returns the scalar cost and, in the full MATLAB implementation,
        optionally its gradient and Hessian. The current Python port
        implements only the scalar cost path.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    N = sparse.csc_matrix(cp["N"])
    Cw = np.asarray(cp["Cw"], dtype=float).reshape(-1)
    H = sparse.csc_matrix(cp.get("H", sparse.csc_matrix((len(Cw), len(Cw)))))
    dd = np.asarray(cp.get("dd", np.ones_like(Cw)), dtype=float).reshape(-1)
    rh = np.asarray(cp.get("rh", np.zeros_like(Cw)), dtype=float).reshape(-1)
    kk = np.asarray(cp.get("kk", np.zeros_like(Cw)), dtype=float).reshape(-1)
    mm = np.asarray(cp.get("mm", np.ones_like(Cw)), dtype=float).reshape(-1)

    R = N @ x - rh
    K = np.zeros_like(R)
    K[R < -kk] = kk[R < -kk]
    K[R > kk] = -kk[R > kk]
    RR = R + K
    U = ((R < -kk) | (R > kk)).astype(float)
    Dl = mm * U * (dd == 1)
    Dq = mm * U * (dd == 2)
    w = (Dl + Dq * RR) * RR
    f = float(0.5 * w @ (H @ w) + Cw @ w)
    if nargout == 1:
        return f
    # TODO(core): port the piecewise/quadratic derivative path, including
    # active-set handling for the dead-zone terms K, U and their Hessian.
    raise NotImplementedError("opf_legacy_user_cost_fcn derivatives not yet implemented")


def opf_legacy_user_cost_fcn_full(x, cp):
    """Request the full legacy user-cost contract.

    The derivative path is intentionally explicit here so callers do not
    encode output meaning with ``nargout``. It currently raises the same
    ``NotImplementedError`` until the legacy derivative formulas are ported.
    """
    return opf_legacy_user_cost_fcn(x, cp, nargout=3)
