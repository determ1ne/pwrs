# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .connected_components import connected_components
from .idx_brch import BR_STATUS, F_BUS, T_BUS
from .idx_bus import BUS_I


def find_islands(mpc, *, nargout=None):
    """Find electrical islands in a MATPOWER case.

    Builds the active-branch incidence matrix for the case and uses
    ``connected_components`` to identify connected bus groups and, when
    requested, isolated buses.

    Parameters
    ----------
    mpc : dict
        MATPOWER case struct.
    nargout : int, optional
        MATLAB compatibility flag controlling whether isolated buses are also
        returned.

    Returns
    -------
    list or tuple
        List of island bus groups, and optionally isolated bus indices.
    """
    nb = mpc["bus"].shape[0]
    nl = mpc["branch"].shape[0]
    bus_i = np.asarray(mpc["bus"][:, BUS_I - 1], dtype=int).reshape(-1)
    e2i = np.zeros(int(np.max(bus_i)) + 1, dtype=int)
    e2i[bus_i] = np.arange(1, nb + 1, dtype=int)
    f = e2i[np.asarray(mpc["branch"][:, F_BUS - 1], dtype=int)]
    t = e2i[np.asarray(mpc["branch"][:, T_BUS - 1], dtype=int)]
    status = np.asarray(mpc["branch"][:, BR_STATUS - 1], dtype=float).reshape(-1)
    C_on = sparse.csc_matrix((-status, (np.arange(nl), f - 1)), shape=(nl, nb)) + sparse.csc_matrix(
        (status, (np.arange(nl), t - 1)), shape=(nl, nb)
    )
    if C_on.nnz:
        groups, isolated = connected_components(C_on, nargout=2)
    else:
        groups = []
        isolated = np.arange(1, nb + 1, dtype=int).reshape(-1, 1)
    if nargout == 1 or nargout is None:
        return groups
    return groups, isolated
