# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_brch import BR_STATUS, F_BUS, T_BUS
from .idx_bus import BUS_I, VA


def calc_branch_angle(mpc):
    """Compute branch voltage angle differences.

    Mirrors MATPOWER's ``calc_branch_angle`` helper by forming the active
    branch incidence matrix and multiplying it by the bus voltage angle
    vector.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray
        Column vector of branch angle differences in degrees.
    """
    status = np.asarray(mpc["branch"][:, BR_STATUS - 1], dtype=float)
    nl = mpc["branch"].shape[0]
    nb = mpc["bus"].shape[0]
    max_bus_num = int(np.max(mpc["bus"][:, BUS_I - 1]))
    e2i = np.zeros(max_bus_num, dtype=int)
    e2i[mpc["bus"][:, BUS_I - 1].astype(int) - 1] = np.arange(1, nb + 1)
    bf = e2i[mpc["branch"][:, F_BUS - 1].astype(int) - 1]
    bt = e2i[mpc["branch"][:, T_BUS - 1].astype(int) - 1]

    A = sparse.csc_matrix(
        (
            np.concatenate([status, -status]),
            (
                np.concatenate([np.arange(nl), np.arange(nl)]),
                np.concatenate([bf.astype(int) - 1, bt.astype(int) - 1]),
            ),
        ),
        shape=(nl, nb),
    )
    delta = A @ mpc["bus"][:, VA - 1]
    return delta.reshape(-1, 1)
