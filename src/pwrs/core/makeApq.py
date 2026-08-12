# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .hasPQcap import hasPQcap
from .idx_gen import PC1, PC2, QC1MAX, QC1MIN, QC2MAX, QC2MIN


def makeApq(baseMVA, gen, nargout=1):
    """Construct generator capability-curve linear constraints.

    Builds the linear constraints ``Apqh * [Pg; Qg] <= ubpqh`` and
    ``Apql * [Pg; Qg] <= ubpql`` for trapezoidal generator capability
    curves.

    Parameters
    ----------
    baseMVA : float
        System power base.
    gen : ndarray
        Generator matrix.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or sparse matrix
        Returns ``(Apqh, ubpqh, Apql, ubpql, data)`` or the leading subset
        requested by ``nargout``.
    """
    ng = gen.shape[0]
    ipqh = np.flatnonzero(hasPQcap(gen, "U")) + 1
    ipql = np.flatnonzero(hasPQcap(gen, "L")) + 1
    npqh = ipqh.shape[0]
    npql = ipql.shape[0]

    data = {}
    if npqh > 0:
        rows = ipqh - 1
        data["h"] = np.c_[gen[rows, QC1MAX] - gen[rows, QC2MAX], gen[rows, PC2] - gen[rows, PC1]]
        ubpqh = data["h"][:, 0] * gen[rows, PC1] + data["h"][:, 1] * gen[rows, QC1MAX]
        for i in range(npqh):
            tmp = np.linalg.norm(data["h"][i, :])
            data["h"][i, :] = data["h"][i, :] / tmp
            ubpqh[i] = ubpqh[i] / tmp
        ii = np.r_[np.arange(1, npqh + 1), np.arange(1, npqh + 1)] - 1
        jj = np.r_[ipqh, ipqh + ng] - 1
        Apqh = sparse.csc_matrix((data["h"].reshape(-1, order="F"), (ii, jj)), shape=(npqh, 2 * ng))
        ubpqh = ubpqh / baseMVA
    else:
        data["h"] = np.array([])
        Apqh = sparse.csc_matrix((0, 2 * ng))
        ubpqh = np.array([])

    if npql > 0:
        rows = ipql - 1
        data["l"] = np.c_[gen[rows, QC2MIN] - gen[rows, QC1MIN], gen[rows, PC1] - gen[rows, PC2]]
        ubpql = data["l"][:, 0] * gen[rows, PC1] + data["l"][:, 1] * gen[rows, QC1MIN]
        for i in range(npql):
            tmp = np.linalg.norm(data["l"][i, :])
            data["l"][i, :] = data["l"][i, :] / tmp
            ubpql[i] = ubpql[i] / tmp
        ii = np.r_[np.arange(1, npql + 1), np.arange(1, npql + 1)] - 1
        jj = np.r_[ipql, ipql + ng] - 1
        Apql = sparse.csc_matrix((data["l"].reshape(-1, order="F"), (ii, jj)), shape=(npql, 2 * ng))
        ubpql = ubpql / baseMVA
    else:
        data["l"] = np.array([])
        Apql = sparse.csc_matrix((0, 2 * ng))
        ubpql = np.array([])

    data["ipql"] = ipql
    data["ipqh"] = ipqh
    outputs = (Apqh, ubpqh, Apql, ubpql, data)
    return outputs[:nargout] if nargout > 1 else Apqh


def makeApq_full(baseMVA, gen):
    """Return all generator capability-curve constraint outputs."""
    return makeApq(baseMVA, gen, nargout=5)
