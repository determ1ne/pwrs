# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse


def opf_vref_hess(x, lam, mpc, refs, mpopt, nargout=1):
    """Return Hessian of reference angle constraints.

    Forms the Hessian of the Lagrangian contribution from the cartesian
    reference-angle equality constraints.

    Parameters
    ----------
    x : sequence of array_like
        Cartesian voltage state blocks ``(Vr, Vi)``.
    lam : array_like
        Multipliers for the reference-angle constraints.
    mpc : dict
        Internal MATPOWER case struct.
    refs : array_like
        One-based indices of reference buses.
    mpopt : dict
        MATPOWER options struct.
    nargout : int, optional
        MATLAB compatibility flag controlling how many outputs are returned.

    Returns
    -------
    scipy.sparse.csc_matrix
        Hessian block for the reference-angle constraint contribution to the
        OPF Lagrangian.
    """
    Vr, Vi = [np.asarray(v).reshape(-1) for v in x]
    refs = np.asarray(refs).reshape(-1).astype(int)
    nb = len(Vr)

    ref_idx = refs - 1
    VVr = Vr[ref_idx]
    VVi = Vi[ref_idx]
    VVr2 = VVr**2
    VVi2 = VVi**2
    lamVm4 = np.asarray(lam).reshape(-1) / (VVr2 + VVi2) ** 2

    d2Vref_rr = sparse.csc_matrix((2 * lamVm4 * VVr * VVi, (ref_idx, ref_idx)), shape=(nb, nb))
    d2Vref_ri = sparse.csc_matrix((lamVm4 * (VVi2 - VVr2), (ref_idx, ref_idx)), shape=(nb, nb))
    d2Vref_ir = d2Vref_ri
    d2Vref_ii = -d2Vref_rr

    d2Vref = sparse.bmat([[d2Vref_rr, d2Vref_ri], [d2Vref_ir, d2Vref_ii]], format="csc")
    outputs = (d2Vref,)
    return outputs[:nargout] if nargout > 1 else d2Vref
