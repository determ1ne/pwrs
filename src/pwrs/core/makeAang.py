# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_brch import ANGMAX, ANGMIN, F_BUS, T_BUS


def makeAang(baseMVA, branch, nb, mpopt, nargout=1):
    """Construct branch angle-difference limit constraints.

    Builds the linear constraint ``lang <= Aang * Va <= uang``, where
    ``Va`` is the vector of bus voltage angles and ``nb`` is the number of
    buses.

    Parameters
    ----------
    baseMVA : float
        System power base. Accepted for MATLAB interface compatibility.
    branch : ndarray
        Branch matrix.
    nb : int
        Number of buses.
    mpopt : dict
        MATPOWER options dict. ``opf.ignore_angle_lim`` controls whether
        angle limits are enforced.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or sparse matrix
        Returns ``(Aang, lang, uang, iang)`` or the leading subset
        requested by ``nargout``.
    """
    if mpopt.opf.ignore_angle_lim:
        out = (sparse.csc_matrix((0, nb)), np.array([]), np.array([]), np.array([], dtype=int))
    else:
        iang = np.flatnonzero(
            ((branch[:, ANGMIN - 1] != 0) & (branch[:, ANGMIN - 1] > -360))
            | ((branch[:, ANGMAX - 1] != 0) & (branch[:, ANGMAX - 1] < 360))
            | ((branch[:, ANGMIN - 1] != 0) & (branch[:, ANGMAX - 1] == 0))
            | ((branch[:, ANGMIN - 1] == 0) & (branch[:, ANGMAX - 1] != 0))
        )
        nang = len(iang)
        if nang:
            ii = np.r_[np.arange(nang), np.arange(nang)]
            jj = np.r_[branch[iang, F_BUS - 1].astype(int) - 1, branch[iang, T_BUS - 1].astype(int) - 1]
            vv = np.r_[np.ones(nang), -np.ones(nang)]
            Aang = sparse.csc_matrix((vv, (ii, jj)), shape=(nang, nb))
            lang = branch[iang, ANGMIN - 1].astype(float).copy()
            uang = branch[iang, ANGMAX - 1].astype(float).copy()
            lang[lang < -360] = -np.inf
            uang[uang > 360] = np.inf
            out = (Aang, lang * np.pi / 180.0, uang * np.pi / 180.0, iang + 1)
        else:
            out = (sparse.csc_matrix((0, nb)), np.array([]), np.array([]), np.array([], dtype=int))
    return out[:nargout] if nargout > 1 else out[0]
