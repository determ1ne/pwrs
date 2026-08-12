# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from ..corex import MatpowerConfig


def opf_vlim_hess(x, lambda_, mpc, idx, mpopt: MatpowerConfig, nargout=1):
    """Return Hessian of cartesian voltage magnitude limit constraints.

    Forms the Hessian of the Lagrangian contribution from the lower and upper
    voltage magnitude inequality constraints in the cartesian OPF
    formulation.

    Parameters
    ----------
    x : sequence of array_like
        Cartesian voltage state blocks ``(Vr, Vi)``.
    lambda_ : array_like
        Multipliers for the stacked lower and upper voltage magnitude
        constraints.
    mpc : dict
        Internal MATPOWER case struct.
    idx : array_like
        One-based indices of buses with active voltage magnitude limits.
    mpopt : dict
        MATPOWER options struct.
    nargout : int, optional
        MATLAB compatibility flag controlling how many outputs are returned.

    Returns
    -------
    scipy.sparse.csc_matrix
        Hessian block for the voltage magnitude limit constraint
        contribution to the OPF Lagrangian.
    """
    _, Vi = [np.asarray(v).reshape(-1) for v in x]
    idx = np.asarray(idx).reshape(-1).astype(int)
    nb = len(Vi)

    lambda_ = np.asarray(lambda_).reshape(-1)
    nlam = len(lambda_) // 2
    if nlam:
        lam = lambda_[nlam : 2 * nlam] - lambda_[:nlam]
    else:
        lam = np.zeros(0)

    ii = idx - 1
    dlam = sparse.csc_matrix((2 * lam, (ii, ii)), shape=(nb, nb))
    zz = sparse.csc_matrix((nb, nb))
    d2Vlims = sparse.bmat([[dlam, zz], [zz, dlam]], format="csc")
    outputs = (d2Vlims,)
    return outputs[:nargout] if nargout > 1 else d2Vlims
