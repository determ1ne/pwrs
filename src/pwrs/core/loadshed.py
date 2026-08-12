# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .idx_gen import PG, PMIN
from .isload import isload


def loadshed(gen, ild=None):
    """Compute shed dispatchable load at selected generators.

    Mirrors MATPOWER's ``loadshed`` helper by measuring how much load has
    been curtailed for dispatchable loads represented as negative
    generators.

    Parameters
    ----------
    gen : array_like
        Generator matrix.
    ild : array_like, optional
        One-based indices of dispatchable-load generator rows. Defaults to
        all rows identified by ``isload``.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray
        Column vector of shed load quantities.
    """
    gen = np.atleast_2d(np.asarray(gen, dtype=float))
    if ild is None:
        ild = np.flatnonzero(isload(gen)) + 1
    ild = np.asarray(ild).reshape(-1).astype(int) - 1

    tol = 1e-5
    shed = gen[ild, PG - 1] - gen[ild, PMIN - 1]
    shed[shed < tol] = 0
    return shed.reshape(-1, 1)
