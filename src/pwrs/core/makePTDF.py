# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .idx_bus import BUS_I, BUS_TYPE, REF
from .makeBdc import makeBdc


def makePTDF(baseMVA, bus=None, branch=None, slack=None, bus_idx=None, *, nargout=None):
    """Build the DC PTDF matrix for a given choice of slack.

    Parameters
    ----------
    baseMVA : float or dict
        System base MVA, or an MATPOWER case dict containing ``baseMVA``,
        ``bus`` and ``branch``.
    bus : array_like, optional
        MATPOWER bus matrix, or the slack specification when ``baseMVA`` is
        a case dict.
    branch : array_like, optional
        MATPOWER branch matrix, or ``bus_idx`` / transfer specification when
        ``baseMVA`` is a case dict.
    slack : int or array_like, optional
        Slack specification. A scalar selects a single slack bus. A length
        ``nb`` vector specifies distributed slack weights. For the full PTDF
        case only, an ``nb x nb`` matrix may be supplied to define a
        bus-specific slack distribution for each PTDF column.
    bus_idx : array_like, optional
        Either a vector of bus indices identifying specific PTDF columns to
        compute, or a transfer matrix whose columns each sum to zero.
    nargout : int, optional
        MATLAB-compatibility placeholder. Ignored.

    Returns
    -------
    numpy.ndarray
        DC PTDF matrix. The default shape is ``nbr x nb``. If a transfer
        matrix or subset of buses is supplied, the number of columns matches
        the requested transfers or selected buses.

    Notes
    -----
    Bus numbers must use internal ordering, i.e. be consecutive starting at
    1. If ``bus_idx`` is a transfer matrix, each column defines a slack-
    independent transfer with zero column sum.

    See Also
    --------
    makeLODF, makeBdc
    """
    if isinstance(baseMVA, dict):
        mpc = baseMVA
        if branch is None:
            if bus is None:
                slack = []
            else:
                slack = bus
            bus_idx = []
        else:
            slack = bus
            bus_idx = branch
        baseMVA = mpc["baseMVA"]
        bus = mpc["bus"]
        branch = mpc["branch"]
    else:
        if bus_idx is None:
            bus_idx = []
        if slack is None:
            slack = []

    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)

    if np.size(slack) == 0:
        refs = np.flatnonzero(bus[:, BUS_TYPE - 1] == REF) + 1
        slack = refs[0]

    nb = bus.shape[0]
    nbr = branch.shape[0]
    txfr = False
    if np.size(bus_idx) != 0:
        bus_idx = np.asarray(bus_idx)
        compute_full_H = False
        if bus_idx.shape[0] == nb and np.sum(bus_idx) == 0:
            txfr = True
            dP = np.asarray(bus_idx, dtype=float)
    else:
        compute_full_H = True

    slack_arr = np.asarray(slack).reshape(-1)
    if slack_arr.size == 1:
        slack_bus = slack_arr
    else:
        slack_bus = 1

    noref = np.arange(2, nb + 1, dtype=int)
    noslack = np.flatnonzero((np.arange(1, nb + 1) != slack_bus)) + 1

    if np.any(bus[:, BUS_I - 1] != np.arange(1, nb + 1)):
        raise ValueError(
            "makePTDF: buses must be numbered consecutively in bus matrix; use ext2int() to convert to internal ordering"
        )

    Bbus, Bf, _, _ = makeBdc(baseMVA, bus, branch)

    if compute_full_H:
        nbi = nb
        dP = sparse.eye(nb, nb, format="csc")
    else:
        if txfr:
            nbi = dP.shape[1]
        else:
            nbi0 = len(np.asarray(bus_idx).reshape(-1))
            bidx = np.asarray(bus_idx, dtype=int).reshape(-1)
            if slack_arr.size != 1 and np.asarray(slack).ndim >= 1 and np.asarray(slack).shape[-1] == 1:
                slacks = np.flatnonzero(np.asarray(slack).reshape(-1)) + 1
                missing = slacks[~np.isin(slacks, bidx)]
                if missing.size:
                    bidx = np.r_[bidx, missing]
            nbi = len(bidx)
            rows = bidx - 1
            cols = np.arange(nbi)
            dP = sparse.csc_matrix((np.ones(nbi), (rows, cols)), shape=(nb, nbi))

    dTheta = np.zeros((nb, nbi))
    A = Bbus[np.ix_(noslack - 1, noref - 1)]
    rhs = dP[noslack - 1, :]
    if sparse.issparse(rhs):
        rhs = rhs.toarray()
    dTheta[noref - 1, :] = spsolve(A, rhs) if sparse.issparse(A) else np.linalg.solve(A, rhs)
    H = Bf @ dTheta

    if slack_arr.size != 1 and not txfr:
        slack_mat = np.asarray(slack, dtype=float)
        if slack_mat.ndim == 1 or (slack_mat.ndim == 2 and slack_mat.shape[1] == 1):
            slack_vec = slack_mat.reshape(-1)
            slack_vec = slack_vec / np.sum(slack_vec)
            if compute_full_H:
                v = H @ slack_vec
                H = H - v[:, None]
            else:
                v = H @ slack_vec[bidx - 1]
                H = H - v[:, None]
                H = H[:, :nbi0]
        else:
            H = H @ slack_mat

    return H
