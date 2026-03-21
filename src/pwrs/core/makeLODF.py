# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_brch import F_BUS, T_BUS


def makeLODF(branch, PTDF, *, nargout=None):
    """Build the DC line outage distribution factor matrix.

    Parameters
    ----------
    branch : array_like
        MATPOWER branch matrix.
    PTDF : array_like
        DC PTDF matrix, typically produced by :func:`makePTDF`.
    nargout : int, optional
        MATLAB-compatibility placeholder. Ignored.

    Returns
    -------
    numpy.ndarray
        ``nbr x nbr`` line outage distribution factor matrix, where ``nbr``
        is the number of branches.

    Notes
    -----
    This is the MATPOWER DC LODF construction corresponding to
    ``makeLODF(branch, H)`` in MATLAB.

    See Also
    --------
    makePTDF
    """
    branch = np.asarray(branch, dtype=float)
    PTDF = np.asarray(PTDF, dtype=float)
    nl, nb = PTDF.shape
    f = branch[:, F_BUS - 1].astype(int)
    t = branch[:, T_BUS - 1].astype(int)
    rows = np.r_[f, t] - 1
    cols = np.r_[np.arange(nl), np.arange(nl)]
    data = np.r_[np.ones(nl), -np.ones(nl)]
    Cft = sparse.csc_matrix((data, (rows, cols)), shape=(nb, nl))

    H = PTDF @ Cft
    h = np.diag(H)
    LODF = H / (np.ones((nl, nl)) - np.ones((nl, 1)) @ h.reshape(1, -1))
    LODF = LODF - np.diag(np.diag(LODF)) - np.eye(nl)
    return LODF
