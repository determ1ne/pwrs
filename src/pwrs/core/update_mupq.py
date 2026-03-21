# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .idx_gen import MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN


def update_mupq(baseMVA, gen, mu_PQh, mu_PQl, data, nargout=1):
    """Update generator limit shadow prices from capability-curve slopes.

    Parameters
    ----------
    baseMVA : float
        System power base.
    gen : ndarray
        Generator matrix to update.
    mu_PQh : array_like
        Shadow prices on the upper sloped portion of generator capability
        curves.
    mu_PQl : array_like
        Shadow prices on the lower sloped portion of generator capability
        curves.
    data : dict
        ``makeApq`` data dict containing the sloped-constraint index and
        coefficient information.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    ndarray
        Updated generator matrix with revised ``MU_PMIN``, ``MU_PMAX``,
        ``MU_QMIN``, and ``MU_QMAX`` entries.
    """
    ipqh = np.asarray(data["ipqh"]).reshape(-1).astype(int)
    ipql = np.asarray(data["ipql"]).reshape(-1).astype(int)
    Apqhdata = np.asarray(data["h"])
    Apqldata = np.asarray(data["l"])

    gen = np.array(gen, copy=True)
    mu_PQh = np.asarray(mu_PQh).reshape(-1)
    mu_PQl = np.asarray(mu_PQl).reshape(-1)

    muP = gen[:, MU_PMAX - 1] - gen[:, MU_PMIN - 1]
    muQ = gen[:, MU_QMAX - 1] - gen[:, MU_QMIN - 1]

    if ipqh.size:
        rows = ipqh - 1
        muP[rows] = muP[rows] - mu_PQh * Apqhdata[:, 0] / baseMVA
        muQ[rows] = muQ[rows] - mu_PQh * Apqhdata[:, 1] / baseMVA

    if ipql.size:
        rows = ipql - 1
        muP[rows] = muP[rows] - mu_PQl * Apqldata[:, 0] / baseMVA
        muQ[rows] = muQ[rows] - mu_PQl * Apqldata[:, 1] / baseMVA

    gen[:, MU_PMAX - 1] = (muP > 0) * muP
    gen[:, MU_PMIN - 1] = (muP < 0) * -muP
    gen[:, MU_QMAX - 1] = (muQ > 0) * muQ
    gen[:, MU_QMIN - 1] = (muQ < 0) * -muQ

    outputs = (gen,)
    return outputs[:nargout] if nargout > 1 else gen
