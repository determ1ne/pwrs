# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import copy

import numpy as np

from ..corex import MatpowerConfig
from .idx_bus import PD, QD
from .idx_gen import PG
from .pfsoln import pfsoln


def cpf_current_mpc(mpc, mpct, Ybus, Yf, Yt, ref, pv, pq, V, lam, mpopt: MatpowerConfig):
    """Build the current CPF case at loading level ``lam``.

    Forms the intermediate MATPOWER case corresponding to the present
    continuation parameter value by interpolating between the base case
    ``mpc`` and target case ``mpct``, then updates the solved bus, generator,
    and branch quantities using ``pfsoln``.

    Parameters
    ----------
    mpc : dict
        Internal MATPOWER base case struct.
    mpct : dict
        Internal MATPOWER target case struct.
    Ybus : sparse matrix
        Bus admittance matrix.
    Yf : sparse matrix
        Branch "from" admittance matrix.
    Yt : sparse matrix
        Branch "to" admittance matrix.
    ref : array_like
        Reference bus index set.
    pv : array_like
        PV bus index set.
    pq : array_like
        PQ bus index set.
    V : array_like
        Complex bus voltage solution at the current CPF step.
    lam : array_like or float
        Continuation parameter value.
    mpopt : dict
        MATPOWER options struct.

    Returns
    -------
    dict
        Updated MATPOWER case struct for the current continuation point.
    """
    mpc = copy.deepcopy(mpc)
    lam = float(np.asarray(lam).reshape(-1)[0])

    mpc["bus"][:, PD] = mpc["bus"][:, PD] + lam * (mpct["bus"][:, PD] - mpc["bus"][:, PD])
    mpc["bus"][:, QD] = mpc["bus"][:, QD] + lam * (mpct["bus"][:, QD] - mpc["bus"][:, QD])
    mpc["gen"][:, PG] = mpc["gen"][:, PG] + lam * (mpct["gen"][:, PG] - mpc["gen"][:, PG])

    mpc["bus"], mpc["gen"], mpc["branch"] = pfsoln(
        mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], Ybus, Yf, Yt, V, ref, pv, pq, mpopt
    )
    return mpc
