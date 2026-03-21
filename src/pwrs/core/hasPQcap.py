# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .idx_gen import PC1, PC2, PMAX, PMIN, QC1MAX, QC1MIN, QC2MAX, QC2MIN, QMAX, QMIN


def hasPQcap(gen, hilo="B", *, nargout=None):
    """Check whether generators have non-trivial PQ capability curves.

    Mirrors MATPOWER's ``hasPQcap`` helper by testing whether each generator
    has an active upper, lower, or either-side reactive capability curve
    constraint over its active power range.

    Parameters
    ----------
    gen : array_like
        Generator matrix.
    hilo : str, optional
        Capability side selector: ``'U'`` for upper, ``'L'`` for lower, or
        ``'B'`` for both.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray
        Column vector indicating whether each generator has the requested PQ
        capability constraint.
    """
    gen = np.atleast_2d(np.asarray(gen, dtype=float))

    k = np.flatnonzero((gen[:, PC1 - 1] != 0) | (gen[:, PC2 - 1] != 0))
    ng = gen.shape[0]

    if k.size == 0:
        return np.zeros((ng, 1))

    kk = np.flatnonzero(
        (gen[k, QMIN - 1] == gen[k, QMAX - 1])
        & (gen[k, QMIN - 1] == gen[k, QC1MAX - 1])
        & (gen[k, QMIN - 1] == gen[k, QC1MIN - 1])
        & (gen[k, QMIN - 1] == gen[k, QC2MAX - 1])
        & (gen[k, QMIN - 1] == gen[k, QC2MIN - 1])
    )
    k = np.delete(k, kk)

    if np.any(gen[k, PC1 - 1] >= gen[k, PC2 - 1]):
        raise ValueError("hasPQcap: must have Pc1 < Pc2")
    if np.any((gen[k, QC2MAX - 1] <= gen[k, QC2MIN - 1]) & (gen[k, QC1MAX - 1] <= gen[k, QC1MIN - 1])):
        raise ValueError("hasPQcap: capability curve defines an empty set")

    k = np.flatnonzero(gen[:, PC1 - 1] != gen[:, PC2 - 1])
    L = np.zeros(ng, dtype=bool)
    U = np.zeros(ng, dtype=bool)
    dPc = gen[k, PC2 - 1] - gen[k, PC1 - 1]

    if hilo != "U":
        dQc = gen[k, QC2MIN - 1] - gen[k, QC1MIN - 1]
        qmin_at_pmin = gen[k, QC1MIN - 1] + (gen[k, PMIN - 1] - gen[k, PC1 - 1]) * dQc / dPc
        qmin_at_pmax = gen[k, QC1MIN - 1] + (gen[k, PMAX - 1] - gen[k, PC1 - 1]) * dQc / dPc
        L[k] = (qmin_at_pmin > gen[k, QMIN - 1]) | (qmin_at_pmax > gen[k, QMIN - 1])

    if hilo != "L":
        dQc = gen[k, QC2MAX - 1] - gen[k, QC1MAX - 1]
        qmax_at_pmin = gen[k, QC1MAX - 1] + (gen[k, PMIN - 1] - gen[k, PC1 - 1]) * dQc / dPc
        qmax_at_pmax = gen[k, QC1MAX - 1] + (gen[k, PMAX - 1] - gen[k, PC1 - 1]) * dQc / dPc
        U[k] = (qmax_at_pmin < gen[k, QMAX - 1]) | (qmax_at_pmax < gen[k, QMAX - 1])

    return (L | U).astype(float).reshape(-1, 1)
