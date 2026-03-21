# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def opf_branch_ang_hess(x, lambda_, Aang, lang, uang, nargout=1):
    """Return Hessian of branch angle difference constraints.

    Forms the Hessian of the Lagrangian contribution from the branch
    angle-difference inequality constraints for the cartesian-voltage OPF
    formulation.

    Parameters
    ----------
    x : sequence of array_like
        Cartesian voltage state blocks ``(Vr, Vi)``.
    lambda_ : array_like
        Multipliers for the stacked lower and upper angle-difference
        constraints.
    Aang : sparse matrix
        Linear operator mapping voltage angles to branch angle differences.
    lang : array_like
        Lower angle-difference limits.
    uang : array_like
        Upper angle-difference limits.
    nargout : int, optional
        MATLAB compatibility flag controlling how many outputs are returned.

    Returns
    -------
    scipy.sparse.csc_matrix
        Hessian block for the branch angle-difference constraint
        contribution to the OPF Lagrangian.
    """
    Vr, Vi = [np.asarray(v).reshape(-1) for v in x]
    nb = len(Vr)

    lambda_ = np.asarray(lambda_).reshape(-1)
    nlam = len(lambda_) // 2
    if nlam:
        lam = lambda_[nlam : 2 * nlam] - lambda_[:nlam]
    else:
        lam = np.zeros(0)

    Vr2 = Vr**2
    Vi2 = Vi**2

    lam_Vm4 = (Aang.T @ lam) / (Vr2 + Vi2) ** 2
    VaDif_rr = sparse.diags(2 * lam_Vm4 * Vr * Vi, offsets=0, shape=(nb, nb), format="csc")
    VaDif_ri = sparse.diags(lam_Vm4 * (Vi2 - Vr2), offsets=0, shape=(nb, nb), format="csc")
    VaDif_ir = VaDif_ri
    VaDif_ii = -VaDif_rr

    d2VaDif = sparse.bmat([[VaDif_rr, VaDif_ri], [VaDif_ir, VaDif_ii]], format="csc")
    outputs = (d2VaDif,)
    return outputs[:nargout] if nargout > 1 else d2VaDif
