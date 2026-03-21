# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def opf_branch_ang_fcn(x, Aang, lang, uang, nargout=1):
    """Evaluate branch angle difference constraints and Jacobian.

    Computes the nonlinear inequality constraints on branch voltage angle
    differences for the cartesian-voltage OPF formulation. When requested, it
    also returns the Jacobian with respect to ``Vr`` and ``Vi``.

    Parameters
    ----------
    x : sequence of array_like
        Cartesian voltage state blocks ``(Vr, Vi)``.
    Aang : sparse matrix
        Linear operator mapping voltage angles to branch angle differences.
    lang : array_like
        Lower angle-difference limits.
    uang : array_like
        Upper angle-difference limits.
    nargout : int, optional
        MATLAB compatibility flag controlling whether the Jacobian is
        returned.

    Returns
    -------
    numpy.ndarray or tuple
        Constraint vector containing lower and upper angle-difference
        violations, and optionally its Jacobian.
    """
    Vr, Vi = [np.asarray(v).reshape(-1) for v in x]
    nb = len(Vi)

    Va = np.angle(Vr + 1j * Vi)
    Ax = Aang @ Va
    VaDif = np.r_[lang - Ax, Ax - uang]

    if nargout > 1:
        Vm2 = Vr**2 + Vi**2
        AangdVa_dVr = Aang @ sparse.diags(-Vi / Vm2, offsets=0, shape=(nb, nb), format="csc")
        AangdVa_dVi = Aang @ sparse.diags(Vr / Vm2, offsets=0, shape=(nb, nb), format="csc")
        dVaDif = sparse.vstack(
            [
                sparse.hstack([-AangdVa_dVr, -AangdVa_dVi], format="csc"),
                sparse.hstack([AangdVa_dVr, AangdVa_dVi], format="csc"),
            ],
            format="csc",
        )
        outputs = (VaDif, dVaDif)
    else:
        outputs = (VaDif,)
    return outputs[:nargout] if nargout > 1 else VaDif
