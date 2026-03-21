# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def opf_veq_hess(x, lambda_, mpc, idx, mpopt, nargout=1):
    """Return Hessian of cartesian fixed-voltage equality constraints.

    Forms the Hessian of the Lagrangian contribution from the fixed-voltage
    magnitude equality constraints in the cartesian OPF formulation.

    Parameters
    ----------
    x : sequence of array_like
        Cartesian voltage state blocks ``(Vr, Vi)``.
    lambda_ : array_like
        Multipliers for the fixed-voltage equality constraints.
    mpc : dict
        Internal MATPOWER case struct.
    idx : array_like
        One-based indices of buses with fixed voltage magnitude constraints.
    mpopt : dict
        MATPOWER options struct.
    nargout : int, optional
        MATLAB compatibility flag controlling how many outputs are returned.

    Returns
    -------
    scipy.sparse.csc_matrix
        Hessian block for the fixed-voltage equality constraint contribution
        to the OPF Lagrangian.
    """
    _, Vi = [np.asarray(v).reshape(-1) for v in x]
    idx = np.asarray(idx).reshape(-1).astype(int)
    nb = len(Vi)
    ii = idx - 1

    dlam = sparse.csc_matrix((2 * np.asarray(lambda_).reshape(-1), (ii, ii)), shape=(nb, nb))
    zz = sparse.csc_matrix((nb, nb))
    d2Veq = sparse.bmat([[dlam, zz], [zz, dlam]], format="csc")
    outputs = (d2Veq,)
    return outputs[:nargout] if nargout > 1 else d2Veq
