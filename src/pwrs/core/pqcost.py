# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def pqcost(gencost, ng, on=None):
    """Split generator costs into active and reactive components.

    Mirrors MATPOWER's ``pqcost`` helper by separating a combined generator
    cost matrix into the active-power cost block and, when present, the
    reactive-power cost block.

    Parameters
    ----------
    gencost : array_like
        Generator cost matrix.
    ng : int
        Number of generators.
    on : array_like, optional
        One-based generator indices to extract.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    tuple
        ``(pcost, qcost)`` cost matrices for the selected generators.
    """
    gencost = np.asarray(gencost, dtype=float)
    ng = int(ng)
    if on is None:
        on = np.arange(ng)
    else:
        on = np.asarray(on).reshape(-1).astype(int) - 1

    if gencost.shape[0] == ng:
        pcost = gencost[on, :]
        qcost = np.array([], dtype=float)
    elif gencost.shape[0] == 2 * ng:
        pcost = gencost[on, :]
        qcost = gencost[on + ng, :]
    else:
        raise ValueError("pqcost: gencost has wrong number of rows")

    return pcost, qcost
