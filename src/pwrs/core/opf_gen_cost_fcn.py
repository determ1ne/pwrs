# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_cost import MODEL, POLYNOMIAL
from .polycost import polycost
from .totcost import totcost


def opf_gen_cost_fcn(x, baseMVA, gencost, ig=None, mpopt=None, nargout=1):
    """Evaluate polynomial generator costs and derivatives.

    Parameters
    ----------
    x : sequence
        Single-element sequence containing the active or reactive dispatch
        vector in per unit.
    baseMVA : float
        System power base.
    gencost : ndarray
        Standard MATPOWER ``gencost`` matrix corresponding to the dispatch
        provided in ``x``.
    ig : array_like, optional
        Generator indices of interest. Defaults to all polynomial-cost rows.
    mpopt : dict, optional
        MATPOWER options dict. Accepted for API compatibility.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    float or tuple
        Returns total cost ``f`` and, when requested, the gradient ``df``
        and Hessian ``d2f``.
    """
    if ig is None or len(np.asarray(ig).reshape(-1)) == 0:
        ig = np.flatnonzero(gencost[:, MODEL] == POLYNOMIAL)
    else:
        ig = np.asarray(ig).reshape(-1)

    PQg = np.asarray(x[0]).reshape(-1)
    ng = len(PQg)

    xx = PQg[ig] * baseMVA
    f = np.sum(totcost(gencost[ig, :], xx))

    if nargout > 1:
        df = np.zeros(ng)
        df[ig] = baseMVA * polycost(gencost[ig, :], xx, 1)

        if nargout > 2:
            d2f = sparse.csc_matrix((baseMVA**2 * polycost(gencost[ig, :], xx, 2), (ig, ig)), shape=(ng, ng))
            outputs = (f, df, d2f)
        else:
            outputs = (f, df)
    else:
        outputs = (f,)
    return outputs[:nargout] if nargout > 1 else f


def opf_gen_cost_fcn_full(x, baseMVA, gencost, ig=None, mpopt=None):
    """Return generator cost, gradient and Hessian."""
    return opf_gen_cost_fcn(x, baseMVA, gencost, ig, mpopt, nargout=3)
