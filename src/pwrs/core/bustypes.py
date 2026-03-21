# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .idx_bus import BUS_TYPE, PQ, PV, REF
from .idx_gen import GEN_BUS, GEN_STATUS


def bustypes(
    bus: npt.NDArray[np.float64], gen: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Classify buses into REF, PV, and PQ sets.

    Combine the declared bus type with online generator status to
    determine the reference, PV, and PQ bus index sets used by
    power flow and OPF routines.

    Parameters
    ----------
    bus : npt.NDArray[np.float64]
        Bus matrix.
    gen : npt.NDArray[np.float64]
        Generator matrix.

    Returns
    -------
    tuple
        Column vectors ``(ref, pv, pq)`` of one-based bus indices.
    """
    nb = bus.shape[0]
    ng = gen.shape[0]
    gen_bus = gen[:, GEN_BUS - 1].astype(int) - 1
    data = (gen[:, GEN_STATUS - 1] > 0).astype(float)
    Cg = sparse.csc_matrix((data, (gen_bus, np.arange(ng))), shape=(nb, ng))
    bus_gen_status = np.asarray(Cg @ np.ones(ng)).reshape(-1)

    ref = np.flatnonzero((bus[:, BUS_TYPE - 1] == REF) & (bus_gen_status != 0)) + 1
    pv = np.flatnonzero((bus[:, BUS_TYPE - 1] == PV) & (bus_gen_status != 0)) + 1
    pq = np.flatnonzero((bus[:, BUS_TYPE - 1] == PQ) | (bus_gen_status == 0)) + 1

    if ref.size == 0 and pv.size:
        ref = pv[:1]
        pv = pv[1:]

    return ref, pv, pq
