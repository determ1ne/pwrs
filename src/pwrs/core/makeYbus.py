# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, SHIFT, T_BUS, TAP
from .idx_bus import BS, BUS_I, GS


def makeYbus(baseMVA, bus=None, branch=None, *, nargout=None):
    """Build the bus admittance matrix and branch admittance matrices.

    Parameters
    ----------
    baseMVA : float or dict
        System base MVA, or an MATPOWER case dict containing ``baseMVA``,
        ``bus`` and ``branch``.
    bus : array_like, optional
        MATPOWER bus matrix when ``baseMVA`` is numeric.
    branch : array_like, optional
        MATPOWER branch matrix when ``baseMVA`` is numeric.
    nargout : int, optional
        MATLAB-compatibility output selector. If ``nargout == 1``, only
        ``Ybus`` is returned.

    Returns
    -------
    scipy.sparse.csc_matrix or tuple
        Returns ``Ybus`` alone when ``nargout == 1``. Otherwise returns
        ``(Ybus, Yf, Yt)``, where ``Yf`` and ``Yt`` map bus voltages to branch
        currents injected at the from and to ends of each branch.

    Notes
    -----
    Inputs may be supplied either as an MATPOWER case dict or as separate
    ``baseMVA``, ``bus`` and ``branch`` arguments. Bus numbers must use
    internal ordering, i.e. be consecutive starting at 1.

    See Also
    --------
    makeJac, makeSbus, ext2int
    """
    if branch is None:
        mpc = baseMVA
        baseMVA = mpc["baseMVA"]
        bus = mpc["bus"]
        branch = mpc["branch"]

    nb = bus.shape[0]
    nl = branch.shape[0]

    if np.any(bus[:, BUS_I - 1] != np.arange(1, nb + 1)):
        raise ValueError(
            "makeYbus: buses must be numbered consecutively in bus matrix; use ext2int() to convert to internal ordering"
        )

    stat = branch[:, BR_STATUS - 1]
    Ys = stat / (branch[:, BR_R - 1] + 1j * branch[:, BR_X - 1])
    Bc = stat * branch[:, BR_B - 1]
    tap = np.ones(nl, dtype=complex)
    nonzero_tap = np.nonzero(branch[:, TAP - 1])[0]
    tap[nonzero_tap] = branch[nonzero_tap, TAP - 1]
    tap = tap * np.exp(1j * np.pi / 180.0 * branch[:, SHIFT - 1])
    Ytt = Ys + 1j * Bc / 2.0
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -Ys / np.conj(tap)
    Ytf = -Ys / tap

    Ysh = (bus[:, GS - 1] + 1j * bus[:, BS - 1]) / baseMVA

    f = branch[:, F_BUS - 1].astype(int) - 1
    t = branch[:, T_BUS - 1].astype(int) - 1

    # the triplet approach is always faster
    if True:
        rows = np.concatenate([np.arange(nl), np.arange(nl)])
        cols = np.concatenate([f, t])
        Yf = sparse.csc_matrix((np.concatenate([Yff, Yft]), (rows, cols)), shape=(nl, nb))
        Yt = sparse.csc_matrix((np.concatenate([Ytf, Ytt]), (rows, cols)), shape=(nl, nb))

        ybus_rows = np.concatenate([f, f, t, t])
        ybus_cols = np.concatenate([f, t, f, t])
        ybus_data = np.concatenate([Yff, Yft, Ytf, Ytt])
        Ybus = sparse.csc_matrix((ybus_data, (ybus_rows, ybus_cols)), shape=(nb, nb))
        Ybus = Ybus + sparse.diags(Ysh, offsets=0, shape=(nb, nb), format="csc")
    else:
        Cf = sparse.csc_matrix((np.ones(nl), (np.arange(nl), f)), shape=(nl, nb))
        Ct = sparse.csc_matrix((np.ones(nl), (np.arange(nl), t)), shape=(nl, nb))
        Yf = (
            sparse.diags(Yff, offsets=0, shape=(nl, nl), format="csc") @ Cf
            + sparse.diags(Yft, offsets=0, shape=(nl, nl), format="csc") @ Ct
        )
        Yt = (
            sparse.diags(Ytf, offsets=0, shape=(nl, nl), format="csc") @ Cf
            + sparse.diags(Ytt, offsets=0, shape=(nl, nl), format="csc") @ Ct
        )
        Ybus = Cf.T @ Yf + Ct.T @ Yt + sparse.diags(Ysh, offsets=0, shape=(nb, nb), format="csc")

    if nargout == 1:
        return Ybus
    return Ybus, Yf, Yt
