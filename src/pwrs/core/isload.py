# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from .idx_gen import PMAX, PMIN


def isload(gen):
    """Checks for dispatchable loads.
       isload(gen) returns a column vector of 1's and 0's. The 1's
       correspond to rows of the GEN matrix which represent dispatchable loads.
       The current test is Pmin < 0 AND Pmax == 0.
       This may need to be revised to allow sensible specification
       of both elastic demand and pumped storage units.

    Parameters
    ----------
    gen : array_like
        Generator matrix.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating which rows correspond to dispatchable loads.
    """
    return (gen[:, PMIN - 1] < 0) & (gen[:, PMAX - 1] == 0)
